import os
import ast
import json
from collections import defaultdict, Counter

import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def fetch_and_resize_image(url, size=(200, 200)):
    """Fetches an image from a URL, resizes it to 'size', and returns a PIL Image."""
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGBA")
    return img.resize(size, resample=Image.LANCZOS)


def draw_graph(metadata, output_path, base_img_size=200, img_zoom=0.6, margin=0.1):
    """
    Draws and saves a graph based on metadata, with dynamic sizing and tight layout.

    - metadata: dict containing node_to_image, node_to_category, unnamed_edges, start_node, target_node
    - output_path: path to save the generated figure
    - base_img_size: the pixel size for image resizing (increases node size)
    - img_zoom: scaling factor when placing images
    - margin: fraction of extra space around the extents
    """
    # Build graph and positions
    G = nx.Graph()
    positions = {}
    xs, ys = [], []
    for coord_str, url in metadata['node_to_image'].items():
        x, y = ast.literal_eval(coord_str)
        positions[coord_str] = (x, -y)
        xs.append(x)
        ys.append(-y)
        G.add_node(coord_str)

    # Edges
    for src, dst in metadata.get('unnamed_edges', []):
        src_s = str(ast.literal_eval(src))
        dst_s = str(ast.literal_eval(dst))
        if src_s in G and dst_s in G:
            G.add_edge(src_s, dst_s)

    # Compute grid extents and counts
    x_vals = sorted(set(xs))
    y_vals = sorted(set(ys))
    num_cols = len(x_vals)
    num_rows = len(y_vals)

    cell_size = 6
    fig_w = num_cols * cell_size
    fig_h = num_rows * cell_size
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    # Draw edges
    for u, v in G.edges():
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.plot([x1, x2], [y1, y2], lw=7.0, color='gray')

    # Prepare labels
    categories = list(metadata.get('node_to_category', {}).values())
    total_per_cat = Counter(categories)
    inst_counter = defaultdict(int)

    start = metadata.get('start_node')
    target = metadata.get('target_node')

    # Place nodes
    for node, (x, y) in positions.items():
        url = metadata['node_to_image'][node]
        try:
            img = fetch_and_resize_image(url, size=(base_img_size, base_img_size))
            im = OffsetImage(img, zoom=img_zoom)
            # Highlight start/target
            boxprops = None
            if node == start:
                boxprops = dict(edgecolor='blue', linewidth=10, facecolor='none')
            elif node == target:
                boxprops = dict(edgecolor='yellow', linewidth=10, facecolor='none')

            ab = AnnotationBbox(
                im,
                (x, y),
                frameon=bool(boxprops),
                bboxprops=boxprops,
                box_alignment=(0.5, 0.5)
            )
            ax.add_artist(ab)

            # Labeling
            cat = metadata['node_to_category'].get(node, 'Unknown')
            inst_counter[cat] += 1
            cnt = inst_counter[cat]
            label = f"{cat}{cnt}" if total_per_cat[cat] > 1 else cat
            # y-offset above or below
            yspan = max(ys) - min(ys)
            if num_rows == 3:
                offset = yspan * 0.235
            elif num_rows == 2:
                offset = yspan * 0.48
            elif num_rows == 4:
                offset = yspan * 0.170
            elif num_rows == 5:
                offset = yspan * 0.122
            else:
                offset = 0
            # if near top, label below
            # y_off = -offset if y > (min(ys) + 0.8 * yspan) else offset
            y_off = offset
            ax.text(x, y + y_off, label, ha='center', va='center', fontsize=42)

        except Exception as e:
            print(f"Error loading image {url}: {e}")

    # Tight layout: set limits with margin
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    plt.tight_layout()

    # Ensure output dir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    fig.savefig(
        output_path,
        format='pdf',
        dpi=1200,
        bbox_inches='tight',
        pad_inches=0
    )
    plt.close(fig)


def main():
    base_dir = os.path.join('escaperoom', 'in')
    instances_file = os.path.join(base_dir, 'instances.json')
    out_dir = os.path.join(base_dir, 'output_graphs')
    os.makedirs(out_dir, exist_ok=True)

    with open(instances_file) as f:
        data = json.load(f)

    for exp in data['experiments']:
        exp_name = exp['name']
        print(exp_name)
        for meta in exp['game_instances']:
            gid = meta['game_id']
            subdir = os.path.join(out_dir, exp_name)
            os.makedirs(subdir, exist_ok=True)
            out_path = os.path.join(subdir, f"{gid}.pdf")
            if not os.path.exists(out_path):
                draw_graph(meta, out_path, base_img_size=500)



if __name__ == '__main__':
    main()