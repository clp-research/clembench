import ast
import os
import json
import glob
from io import BytesIO
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import textwrap

# Init Config
DATA_ROOT = 'escaperoom/in/instances.json'
INTERACTIONS_PATTERN = os.path.join(
    'results', '*', 'escape_room', '*', 'episode_*', 'interactions.json'
)
ROBOT_PATH = 'engine/resources/robot.png'
ORACLE_PATH = 'engine/resources/oracle.png'
NODE_IMG_SIZE = (125, 125)
ROBOT_IMG_SIZE = (110, 110)      # increased explorer/robot size
ORACLE_IMG_SIZE = (110, 110)     # increased guide/oracle size
GRAPH_ZOOM = 0.75         # zoom in a bit

# To change the font, substitute a TTF path and size here:
try:
    FONT = ImageFont.truetype("arial.ttf", 12)
except IOError:
    FONT = ImageFont.load_default()

PADDING = 20                   # margin padding around elements
LINE_SPACING = 4               # reduced vertical space between lines
TEXT_WIDTH = 42               # narrower text wrap for clarity

# Direction mapping according to GymEnv
dir_map = {
    'north': (0, -1), 'south': (0, 1),
    'east': (1, 0),  'west': (-1, 0)
}


def fetch_and_resize_image(url, size):
    resp = requests.get(url)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert('RGBA')
    return img.resize(size, resample=Image.LANCZOS)


def render_graph_image(positions, edges, node_imgs,
                       robot_img, current_node, target_node):
    """
    Render the graph to a PIL image via Matplotlib, marking the target and current nodes.
    Robot icon is offset to the right of its node, except if the node is in the rightmost column,
    then offset to the left.
    """
    # Determine max x-coordinate to identify rightmost column
    max_x = max(x for x, y in positions.values())

    fig, ax = plt.subplots()
    ax.set_axis_off()

    # Draw edges
    for u, v in edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.plot([x1, x2], [y1, y2], linewidth=1, color='gray')

    # Draw nodes with appropriate borders
    for node, (x, y) in positions.items():
        img = node_imgs[node]
        if node == target_node:
            edge_color, lw = 'yellow', 3
        elif node == current_node:
            edge_color, lw = 'blue', 3
        else:
            edge_color, lw = 'black', 1
        im = OffsetImage(img, zoom=GRAPH_ZOOM)
        ab = AnnotationBbox(
            im, (x, y), frameon=True, box_alignment=(0.5, 0.5),
            bboxprops=dict(edgecolor=edge_color, linewidth=lw)
        )
        ax.add_artist(ab)

    # Compute robot offset: right of node unless node.x == max_x
    x0, y0 = positions[current_node]
    offset = 0.2
    if x0 >= max_x:
        robot_pos = (x0 - offset, y0)
    else:
        robot_pos = (x0 + offset, y0)

    rb = OffsetImage(robot_img, zoom=GRAPH_ZOOM)
    ab_r = AnnotationBbox(rb, robot_pos, frameon=False)
    ax.add_artist(ab_r)

    # Export to PIL
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def create_combined(graph_img, oracle_img, robot_img,
                    last_guide_msg, last_explorer_msg):
    gw, gh = graph_img.size
    total_w = 2 * gw + PADDING
    combined = Image.new('RGB', (total_w, gh), (255, 255, 255))
    combined.paste(graph_img, (0, 0))
    draw = ImageDraw.Draw(combined)
    start_x = gw + PADDING
    mid_x = start_x + gw // 2
    draw.line([(mid_x, 0), (mid_x, gh)], fill='black', width=2)
    # Guide panel
    if last_guide_msg:
        lines = textwrap.wrap(last_guide_msg, width=TEXT_WIDTH)
        y = PADDING
        combined.paste(oracle_img, (gw + PADDING, y), oracle_img)
        x0 = gw + PADDING + ORACLE_IMG_SIZE[0] + PADDING
        for line in lines:
            bbox = draw.textbbox((x0, y), line, font=FONT)
            h = bbox[3] - bbox[1]
            draw.text((x0, y), line, font=FONT, fill='black')
            y += h + LINE_SPACING
    # Explorer panel
    if last_explorer_msg:
        lines = textwrap.wrap(last_explorer_msg, width=TEXT_WIDTH)
        y = PADDING
        x_col = mid_x + PADDING
        combined.paste(robot_img, (x_col, y), robot_img)
        x0 = x_col + ROBOT_IMG_SIZE[0] + PADDING
        for line in lines:
            bbox = draw.textbbox((x0, y), line, font=FONT)
            h = bbox[3] - bbox[1]
            draw.text((x0, y), line, font=FONT, fill='black')
            y += h + LINE_SPACING
    return combined


def process_interactions():
    with open(DATA_ROOT) as f:
        data = json.load(f)
    lookup = {
        exp['name']: {inst['game_id']: inst for inst in exp['game_instances']}
        for exp in data['experiments']
    }
    for path in tqdm(glob.glob(INTERACTIONS_PATTERN)):
        with open(path) as f:
            inter = json.load(f)
        meta = inter['meta']
        md = lookup[meta['experiment_name']][meta['game_id']]
        positions, node_imgs, edges = {}, {}, []
        for coord_str, url in md['node_to_image'].items():
            coord = tuple(ast.literal_eval(coord_str))
            positions[coord_str] = (coord[0], -coord[1])
            node_imgs[coord_str] = fetch_and_resize_image(url, NODE_IMG_SIZE)
        for src, dst in md.get('unnamed_edges', []):
            edges.append((str(ast.literal_eval(src)), str(ast.literal_eval(dst))))
        current = md['start_node']
        target = md.get('target_node')
        robot_img = Image.open(ROBOT_PATH).convert('RGBA')
        robot_img = robot_img.resize(ROBOT_IMG_SIZE, resample=Image.LANCZOS)
        oracle_img = Image.open(ORACLE_PATH).convert('RGBA')
        oracle_img = oracle_img.resize(ORACLE_IMG_SIZE, resample=Image.LANCZOS)
        last_guide, last_explorer = None, None
        out_dir = os.path.join(os.path.dirname(path), 'combined_graphs')
        if os.path.exists(out_dir):
            continue
        os.makedirs(out_dir, exist_ok=True)
        for idx, turn in enumerate(inter['turns']):
            updated = False
            for i, act in enumerate(turn):
                frm, to = act['from'], act['to']
                typ = act['action']['type']
                cnt = act['action']['content']
                if frm.startswith('Player 2') and typ == 'get message' and cnt.startswith('MOVE:'):
                    if i + 1 < len(turn) and turn[i+1]['action']['type'] == 'move':
                        res = turn[i+1]['action']['content']
                        if res in ('valid', 'efficient', 'inefficient'):
                            direction = cnt.split('MOVE:')[1].strip()
                            dx, dy = dir_map.get(direction, (0, 0))
                            x0, y0 = eval(current)
                            new = (x0 + dx, y0 + dy)
                            if str(new) in positions:
                                current = str(new)
                            updated = True
                if frm.startswith('Player 1') and to == 'GM' and typ == 'get message':
                    last_guide = cnt; updated = True
                if frm.startswith('Player 2') and to == 'GM' and typ == 'get message':
                    last_explorer = cnt; updated = True
            if updated:
                graph_img = render_graph_image(
                    positions, edges, node_imgs,
                    robot_img, current, target
                )
                combined = create_combined(
                    graph_img, oracle_img, robot_img,
                    last_guide, last_explorer
                )
                out = os.path.join(out_dir, f'combined_{idx}.png')
                combined.save(out)

if __name__ == '__main__':
    process_interactions()