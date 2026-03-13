import matplotlib.pyplot as plt
import numpy as np

def plot_screw(ax, x, y, factor=3, color="b", cell_size=1, scale=1.0):
    if color == "y":
        color = "orange"

    center_x = x + cell_size / 2
    center_y = y + cell_size / 2

    base_radius = cell_size / factor
    radius = base_radius * scale   # <-- apply scale here

    circle = plt.Circle(
        (center_x, center_y),
        radius,
        edgecolor="k",
        facecolor=color,
    )
    ax.add_patch(circle)
    return


def plot_washer(ax, x, y, color="r", cell_size=1, scale=1.0):
    if color == "y":
        color = "orange"

    cx = x + cell_size / 2
    cy = y + cell_size / 2

    # Original diamond vertices
    points = [
        (cx, y),
        (x + cell_size, cy),
        (cx, y + cell_size),
        (x, cy),
    ]

    # Scale vertices around center
    scaled_points = []
    for px, py in points:
        sx = cx + (px - cx) * scale
        sy = cy + (py - cy) * scale
        scaled_points.append((sx, sy))

    diamond = plt.Polygon(
        scaled_points,
        linewidth=1,
        closed=True,
        edgecolor="k",
        facecolor=color,
    )

    ax.add_patch(diamond)


def plot_nut(ax, x, y, factor=1.2, color="g", cell_size=1, scale=1.0):
    if color == "y":
        color = "orange"

    # Base size of the square
    base_size = cell_size / factor
    size = base_size * scale   # <-- apply scale here

    # Center of the cell
    cx = x + cell_size / 2
    cy = y + cell_size / 2

    # Bottom-left corner so square stays centered
    bottom_left_x = cx - size / 2
    bottom_left_y = cy - size / 2

    square = plt.Rectangle(
        (bottom_left_x, bottom_left_y),
        size,
        size,
        linewidth=1,
        edgecolor="k",
        facecolor=color,
    )

    ax.add_patch(square)
    return


def plot_bridge_h(ax, x, y, factor=1.6, color="g", cell_size=1, scale=1.0):
    if color == "y":
        color = "orange"

    # Bridge spans 2 columns (length), but is thin (thickness)
    length = 2 * cell_size
    base_thickness = cell_size / factor
    thickness = base_thickness * scale  # scale only thickness

    # Center vertically within the cell row
    y0 = y + (cell_size / 2) - (thickness / 2)

    bridge = plt.Rectangle(
        (x, y0),          # start at left cell's left edge
        length,           # spans 2 cells
        thickness,
        linewidth=1,
        edgecolor="k",
        facecolor=color,
    )
    ax.add_patch(bridge)
    return


def plot_bridge_v(ax, x, y, factor=1.6, color="g", cell_size=1, scale=1.0):
    if color == "y":
        color = "orange"

    length = 2 * cell_size
    base_thickness = cell_size / factor
    thickness = base_thickness * scale

    # Center horizontally within the column
    x0 = x + (cell_size / 2) - (thickness / 2)

    bridge = plt.Rectangle(
        (x0, y),
        thickness,
        length,
        linewidth=1,
        edgecolor="k",
        facecolor=color,
    )
    ax.add_patch(bridge)
    return


def set_up_board_plot(rows, cols, cell_size):
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Draw grid first (low zorder)
    ax.grid(True, zorder=0, which='both', linewidth=0.3, alpha=0.2)    

    # Loop through rows and columns to create the grid
    for row in range(rows):
        for col in range(cols):
            # Define the coordinates and size of each rectangle
            x = col
            y = row
            width = 1
            height = 1

            # Create a rectangle for each cell in the grid
            rect = plt.Rectangle(
                (x, y), width, height, linewidth=1, edgecolor="k", facecolor="w"
            )
            ax.add_patch(rect)

    # Set axis limits to match the grid size
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)

    # ---- Put ticks at cell centers and use them as labels ----
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_yticks(np.arange(rows) + 0.5)

    ax.set_xticklabels(range(1, cols + 1))
    ax.set_yticklabels(range(1, rows + 1))

    #ax.tick_params(axis="both", length=0, pad=5)
    ax.tick_params(axis="both", length=0, pad=5, labelsize=15)  # <-- small font

    # Column numbers on top
    ax.xaxis.tick_top()

    # Axis titles
    ax.set_xlabel("Columns →", labelpad=2, fontsize=15)
    ax.xaxis.set_label_position('top')

    ax.set_ylabel("Rows →", labelpad=2, fontsize=15)    
    ax.yaxis.set_label_position('left')

    ax.set_aspect("equal", adjustable="box")

    # invert y, to move 0,0 to top left
    plt.gca().invert_yaxis()

    return fig, ax


def plot_board(board, filename=None):
    max_height, depth, rows, cols = board.shape

    fig, ax = set_up_board_plot(rows, cols, cell_size=1)
    H, depth, rows, cols = board.shape
    for h in range(H):
        obj = board[h, 0].T#this_layer[0].T
        clr = board[h, 1].T#this_layer[1].T

        # higher layers slightly smaller so you can see what's under
        scale = 1.0 - 0.12*h                # tweak 0.12 as you like
        if scale <= 0: 
            continue

        for r in range(obj.shape[1]):
            for c in range(obj.shape[0]):
                if obj[r, c] == "S":
                    plot_screw(ax, r, c, color=clr[r, c], scale=scale)
                if obj[r, c] == "W":
                    plot_washer(ax, r, c, color=clr[r, c], scale=scale)
                if obj[r, c] == "N":
                    plot_nut(ax, r, c, color=clr[r, c], scale=scale)
                if obj[r, c] == "L":
                    plot_bridge_h(ax, r, c, color=clr[r, c], scale=scale)
                if obj[r, c] == "T":
                    plot_bridge_v(ax, r, c, color=clr[r, c], scale=scale)
    if filename:
        plt.savefig(filename, dpi=300)
    plt.close()


long_to_short = {
    "washer": "W",
    "nut": "N",
    "screw": "S",
    "bridge-h": "L",
    "bridge-v": "T",
}
short_to_long = {"W": "washer", "N": "nut", "S": "screw", "L": "bridge-h", "R": "bridge-h", "T": "bridge-v", "B": "bridge-v"}
long_to_short_color = {"red": "r", "green": "g", "blue": "b", "yellow": "y"}
short_to_long_color = {"r": "red", "g": "green", "b": "blue", "y": "yellow"}


# custom exceptions
class SameShapeStackingError(Exception):
    "Raised when same shapes are stacked on top of each other"
    pass


class SameShapeAtAlternateLevels(Exception):
    "Raised when same shapes are stacked at alternate levels"
    pass


class SameColorAtAlternateLevels(Exception):
    "Raised when same colors are stacked at alternate levels"
    pass


class SameColorStackingError(Exception):
    "Raised when same colors are stacked on top of each other"
    pass


class NotOnTopOfScrewError(Exception):
    "Raised when a shape is placed on top of a screw"
    pass


class DepthMismatchError(Exception):
    "Raised when the depth of the new shape does not match the depth of existing shapes"
    pass


class BridgePlacementError(Exception):
    "Raised when a bridge is placed at levels greater than 2"
    pass


class DimensionsMismatchError(Exception):
    "Raised when the dimensions of the board do not match the dimensions of input x,y"
    pass


def get_top_layer(board, x, y):
    this_stack = board[:, 0, x, y]
    top_layer = np.where(this_stack == "0")[0]
    if top_layer.size > 0:
        top_layer = top_layer[0]
    else:
        #raise (ValueError("Placement not possible"))
        raise (ValueError(f"No more space to place shapes at this location ({x+1},{y+1})"))
    return top_layer

def get_total_occupied_layers(board, x, y):
    this_stack = board[:, 0, x, y]
    occupied_layers = np.where(this_stack != "0")[0]
    return occupied_layers.size


def check_for_errors(top_layer, board, shape, color, x, y):
    if board[top_layer - 1, 0, x, y] == "S":
        #raise (NotOnTopOfScrewError("Placement not possible"))
        raise (NotOnTopOfScrewError("Cannot place a shape on top of a screw"))

    start_layer = top_layer - 1
    while(start_layer >= 0):
        if board[start_layer, 0, x, y] == long_to_short[shape]:
            #raise (SameShapeStackingError("Placement not possible"))
            raise (SameShapeStackingError(f"Stacking same shapes is not allowed. Error at location ({x+1},{y+1})"))
        start_layer -= 1

    if shape == "bridge-h":
        if board[top_layer - 1, 0, x, y + 1] == "S":
            #raise (NotOnTopOfScrewError("Placement not possible"))
            raise (NotOnTopOfScrewError("Cannot place a bridge on top of a screw"))

        if (
            board[top_layer - 1, 1, x, y] == long_to_short_color[color]
            or board[top_layer - 1, 1, x, y + 1] == long_to_short_color[color]
        ):
            #raise (SameColorStackingError("Placement not possible"))
            raise (SameColorStackingError(f"Stacking shapes of the same color is not allowed. Error at location ({x+1},{y+1})"))

    elif shape == "bridge-v":
        if board[top_layer - 1, 0, x + 1, y] == "S":
            #raise (NotOnTopOfScrewError("Placement not possible"))
            raise (NotOnTopOfScrewError("Cannot place a bridge on top of a screw"))

        if (
            board[top_layer - 1, 1, x, y] == long_to_short_color[color]
            or board[top_layer - 1, 1, x + 1, y] == long_to_short_color[color]
        ):
            #raise (SameColorStackingError("Placement not possible"))
            raise (SameColorStackingError(f"Stacking shapes of the same color is not allowed. Error at location ({x+1},{y+1})"))
    else:
        if board[top_layer - 1, 1, x, y] == long_to_short_color[color]:
            #raise (SameColorStackingError("Placement not possible"))
            raise (SameColorStackingError(f"Stacking shapes of the same color is not allowed. Error at location ({x+1},{y+1})"))
    '''
    if top_layer > 1:
        # check if same shape is placed at alternate levels
        if board[top_layer - 2, 0, x, y] == long_to_short[shape]:
            raise (SameShapeAtAlternateLevels("Placement not possible"))

        # check if same color is placed at alternate levels
        if board[top_layer - 2, 1, x, y] == long_to_short_color[color]:
            raise (SameColorAtAlternateLevels("Placement not possible"))
    '''

# TODO: operate on copy of board, so that original board
# can be returned if placement not possible? (rather than
# raising an exception)
def put(board, shape, color, x, y):
    x_ = (x - 1)
    y_ = (y - 1)    
    if x_ >= board.shape[2] or y_ >= board.shape[3] or x_ < 0 or y_ < 0:
        #raise (DimensionsMismatchError("Placement not possible"))
        raise (DimensionsMismatchError(f"Given location ({x}, {y}) values are out of bounds (exceeding ({board.shape[2]}, {board.shape[3]})) while placing shape {shape} with color {color}. Board dimensions: ({board.shape[2]}, {board.shape[3]})."))
    
    if shape not in long_to_short:
        if shape in ["bridge-h-left", "bridge-h-right", "bridge-v-top", "bridge-v-bottom"]:
            raise (ValueError("Do not use 'bridge-h-left', 'bridge-h-right', 'bridge-v-top', or 'bridge-v-bottom'. Only use 'bridge-h' or 'bridge-v' for shape placement."))
        else:
            raise (ValueError("Unknown shape used in put(). Use only screw, washer, nut, bridge-h, and bridge-v shapes"))
        
    if color not in long_to_short_color:
        #raise (ValueError("Placement not possible"))
        raise (ValueError("Unknown color used in put(). Use only red, green, blue, and yellow colors"))

    top_layer = get_top_layer(board, x_, y_)

    if shape == "bridge-h":
        if y_ + 1 >= board.shape[3]:
            #raise (ValueError("Placement not possible"))
            raise (ValueError(f"Placing bridge-h in column ({y}) exceeds the board dimensions: ({board.shape[2]}x{board.shape[3]}).)."))

        if top_layer >= 2:
            #raise (BridgePlacementError("Placement not possible"))
            raise (BridgePlacementError("bridge-h can only be placed at levels 1 and 2"))
        

        if top_layer > 0:
            check_for_errors(top_layer, board, shape, color, x_, y_)

        top_layer_adjacent = get_top_layer(board, x_, y_ + 1)
        if top_layer != top_layer_adjacent:
            #raise (DepthMismatchError("Placement not possible"))
            raise (DepthMismatchError(f"Depth mismatch while placing bridge-h: adjacent cell({x_+1},{y_+1+1}) has a different depth"))

        board[top_layer, 0, x_, y_] = "L"
        board[top_layer, 1, x_, y_] = color

        board[top_layer, 0, x_, y_ + 1] = "R"
        board[top_layer, 1, x_, y_ + 1] = color
    elif shape == "bridge-v":
        if x_ + 1 >= board.shape[2]:
            #raise (ValueError("Placement not possible"))
            raise (ValueError(f"Placing bridge-v in row({x}) exceeds the board dimensions: ({board.shape[2]}x{board.shape[3]})."))
        

        if top_layer >= 2:
            #raise (BridgePlacementError("Placement not possible"))
            raise (BridgePlacementError("bridge-v can only be placed at levels 1 and 2"))

        if top_layer > 0:
            check_for_errors(top_layer, board, shape, color, x_, y_)

        top_layer_adjacent = get_top_layer(board, x_ + 1, y_)
        if top_layer != top_layer_adjacent:
            #raise (DepthMismatchError("Placement not possible"))
            raise (DepthMismatchError(f"Depth mismatch while placing bridge-v: adjacent cell({x_+1+1},{y_+1}) has a different depth"))

        board[top_layer, 0, x_, y_] = "T"
        board[top_layer, 1, x_, y_] = color

        board[top_layer, 0, x_ + 1, y_] = "B"
        board[top_layer, 1, x_ + 1, y_] = color
    else:
        # check if it is being placed on top of another screw
        if top_layer > 0:
            check_for_errors(top_layer, board, shape, color, x_, y_)

        board[top_layer, 0, x_, y_] = long_to_short[shape]
        board[top_layer, 1, x_, y_] = color
    # check whether resulting board is legal
    return


def init_board(rows=6, cols=6, max_height=4, depth=2):
    # the board is represented via stacked matrices:
    # there are max_height layers (by default 4), representing the stacking
    # each layer has depth channels (by default 2), one of which will
    # hold the shape information, the other the colour information
    # each layer is a 2d matrix with dimensions as given by rows and cols
    return np.full((max_height, depth, rows, cols), "0", dtype=str)


def place_on_board(board, obj_board, x, y):
    # TODO: check if x, y to be decreased by 1 for 0 indexing
    board[
        :, :, slice(x, x + obj_board.shape[2]), slice(y, y + obj_board.shape[3])
    ] = obj_board
    return board


def board_rot90(board):
    board_r = np.rot90(board, axes=(2, 3))
    bridge_rot_dict = {
        "R": "T",
        "L": "B",
        "T": "L",
        "B": "R",
        "W": "W",
        "N": "N",
        "S": "S",
        "0": "0",
    }
    board_r[:, 0] = np.vectorize(bridge_rot_dict.get)(board_r[:, 0])
    return board_r


def _validate_shapes(board, x, y, shapes_list, start_range, end_range):
    assert all(isinstance(s, tuple) and len(s)==2 for s in shapes_list)
    assert 0 <= start_range < end_range <= board.shape[0]
    assert end_range == get_total_occupied_layers(board, x, y)
    assert end_range - start_range == len(shapes_list)

    for layer_offset, (shape, color) in enumerate(shapes_list):
        layer = start_range + layer_offset
        if layer >= board.shape[0]:
            raise (ValueError(f"Not enough shapes at location({x+1},{y+1}) to move"))

        if color.lower() not in long_to_short_color:
            raise (ValueError(f"Unknown color ({color})."))


        board_shape = board[layer, 0, x, y]
        board_color = board[layer, 1, x, y]


        if board_shape in ["R", "B"]:
            raise (ValueError(f"Non-anchor token found (perhaps a bridge was partially removed) at location({x+1},{y+1}). Please note that the bridge should be removed as a whole, and only the topmost shape can be removed at a time."))
        
        if board_shape == "0" or board_shape not in short_to_long:
            raise (ValueError(f"Unknown shape (shape:{board_shape}, color:{short_to_long_color[board_color]}) data at location({x+1},{y+1})."))
        

        color_formatted = long_to_short_color[color.lower()]

        #print(f"Validating shape at layer {layer}, expected: ({shape}, {color_formatted}), found: ({short_to_long[board_shape]}, {board_color})")

        if short_to_long[board_shape] != shape or board_color != color_formatted:
            raise (ValueError(f"Shape ({short_to_long[board_shape]}), color ({short_to_long_color[board_color]}) at location ({x+1}, {y+1}) does not match with the input data:({shape}, {color_formatted})."))

def _clear_cell_shapes(board, x, y, undo_list):
    for shape, color, layer in undo_list:
        board[layer, 0, x, y] = "0"
        board[layer, 1, x, y] = "0"

        if shape == "bridge-h":
            board[layer, 0, x, y + 1] = "0"
            board[layer, 1, x, y + 1] = "0"
        elif shape == "bridge-v":
            board[layer, 0, x + 1, y] = "0"
            board[layer, 1, x + 1, y] = "0"
    return


def oldmove(board, x1, y1, x2, y2, shapes_list=None):

    x1_ = x1 - 1
    y1_ = y1 - 1
    x2_ = x2 - 1
    y2_ = y2 - 1

    if x1_ < 0 or x1_ >= board.shape[2] or y1_ < 0 or y1_ >= board.shape[3]:
        raise (DimensionsMismatchError("Source location out of bounds"))
    if x2_ < 0 or x2_ >= board.shape[2] or y2_ < 0 or y2_ >= board.shape[3]:
        raise (DimensionsMismatchError("Destination location out of bounds"))
    
    if x1_ == x2_ and y1_ == y2_:
        raise (ValueError("Source and destination locations are the same"))
    
    if shapes_list is not None and len(shapes_list) == 0:
        raise (ValueError("Shapes list cannot be empty"))

    if shapes_list is not None:
        new_layer_len = len(shapes_list)
        cur_max_top_layer = get_total_occupied_layers(board, x1_, y1_)
        if cur_max_top_layer < new_layer_len:
            raise (ValueError("Not enough shapes at source location to move"))
        start_range = cur_max_top_layer - new_layer_len
        end_range = cur_max_top_layer
        _validate_shapes(board, x1_, y1_, shapes_list, start_range, end_range)
    else:
        cur_max_top_layer = get_total_occupied_layers(board, x1_, y1_)
        new_layer_len = 1
        if cur_max_top_layer < new_layer_len:
            raise (ValueError("Not enough shapes at source location to move"))

        start_range = cur_max_top_layer-1
        end_range = start_range+1

    new_location_top_layer = get_total_occupied_layers(board, x2_, y2_)
    #print(f"new_layer_len: {new_layer_len}, cur_max_top_layer: {cur_max_top_layer}, new_location_top_layer: {new_location_top_layer}")
    
    if new_location_top_layer+new_layer_len > board.shape[0]:
        raise (ValueError(f"Not enough space at destination location ({x2_+1}, {y2_+1}), number of new layer to move {new_layer_len} current top layer at destination {new_location_top_layer} max height {board.shape[0]}"))

    
    move_shapes_list = []

    for layer in range(start_range, end_range):
        shape = short_to_long[board[layer, 0, x1_, y1_]]
        color = board[layer, 1, x1_, y1_]
        move_shapes_list.append((shape, color, layer))

    source_list = []
    dest_list = []
    dest_layer = new_location_top_layer

    for shape, color, layer in move_shapes_list:

        try:
            put(board, shape, color, x2_, y2_)
            # Tracking shape from the source and destination locations for later clearing
            source_list.append((shape, color, layer))
            dest_list.append((shape, color, dest_layer))
            dest_layer += 1

        except Exception as e:
            # If placement fails, clear the objects at the destination location
            _clear_cell_shapes(board, x2_, y2_, dest_list[::-1])
            raise e
    # If all placements are successful, clear the objects at the source location
    _clear_cell_shapes(board, x1_, y1_, source_list)
    return board


def move(board, x1, y1, x2, y2, shapes_list=None):

    x1_ = x1 - 1
    y1_ = y1 - 1
    x2_ = x2 - 1
    y2_ = y2 - 1

    if x1_ < 0 or x1_ >= board.shape[2] or y1_ < 0 or y1_ >= board.shape[3]:
        raise (DimensionsMismatchError(f"Source location ({x1_+1}, {y1_+1}) out of bounds"))
    if x2_ < 0 or x2_ >= board.shape[2] or y2_ < 0 or y2_ >= board.shape[3]:
        raise (DimensionsMismatchError(f"Destination location ({x2_+1}, {y2_+1}) out of bounds"))
    
    if x1_ == x2_ and y1_ == y2_:
        raise (ValueError("Source and destination locations are the same"))
    
    if shapes_list is not None and len(shapes_list) == 0:
        raise (ValueError("Shapes list cannot be empty"))

    if shapes_list is not None:
        new_layer_len = len(shapes_list)
        cur_max_top_layer = get_total_occupied_layers(board, x1_, y1_)
        if cur_max_top_layer < new_layer_len:
            raise (ValueError(f"Not enough shapes at location ({x1_+1}, {y1_+1}) to move"))
        start_range = cur_max_top_layer - new_layer_len
        end_range = cur_max_top_layer

        try:
            _validate_shapes(board, x1_, y1_, shapes_list, start_range, end_range)
        except Exception as e:
            raise ValueError(f"Validation failed for shapes at location ({x1_+1}, {y1_+1}). Expected format: [('shape', 'color'), ...]") from e
    else:
        cur_max_top_layer = get_total_occupied_layers(board, x1_, y1_)
        new_layer_len = 1
        if cur_max_top_layer < new_layer_len:
            raise (ValueError(f"Not enough shapes at location ({x1_+1}, {y1_+1}) to move"))

        start_range = cur_max_top_layer-1
        end_range = start_range+1

    new_location_top_layer = get_total_occupied_layers(board, x2_, y2_)
    #print(f"new_layer_len: {new_layer_len}, cur_max_top_layer: {cur_max_top_layer}, new_location_top_layer: {new_location_top_layer}")
    
    if new_location_top_layer+new_layer_len > board.shape[0]:
        raise (ValueError(f"Not enough space at destination location, number of new layer to move {new_layer_len} current top layer at destination {new_location_top_layer} max height {board.shape[0]}"))

    
    move_shapes_list = []

    for layer in range(start_range, end_range):
        shape = short_to_long[board[layer, 0, x1_, y1_]]
        color = board[layer, 1, x1_, y1_]
        color = short_to_long_color[color.lower()]
        move_shapes_list.append((shape, color, layer))

    source_list = []
    dest_list = []
    dest_layer = new_location_top_layer

    for shape, color, layer in move_shapes_list:

        try:
            put(board, shape, color, x2_+1, y2_+1)
            # Tracking shape from the source and destination locations for later clearing
            source_list.append((shape, color, layer))
            dest_list.append((shape, color, dest_layer))
            dest_layer += 1

        except Exception as e:
            # If placement fails, clear the objects at the destination location
            _clear_cell_shapes(board, x2_, y2_, dest_list[::-1])
            #raise e
            raise ValueError(f"Failed to move {shape} to ({x2_+1}, {y2_+1}): {str(e)}") from e
    # If all placements are successful, clear the objects at the source location
    _clear_cell_shapes(board, x1_, y1_, source_list)
    return board

def clear(board):
    board[:, :, :, :] = "0"
    return board


def remove(board, x, y, shape, color):
    x_ = (x - 1)
    y_ = (y - 1)
    if x_ >= board.shape[2] or y_ >= board.shape[3] or x_ < 0 or y_ < 0:
        raise (DimensionsMismatchError(f"Removal not possible - ({x},{y}) coordinates out of bounds"))

    top_layer = get_top_layer(board, x_, y_)
    if top_layer == 0:
        raise (ValueError(f"No shapes to remove at the specified location ({x},{y})"))
    
    if shape not in long_to_short:
        raise (ValueError(f"Unknown shape ({shape}) used in remove(). Removeshape currently supports only screw, washer, nut, bridge-h, and bridge-v shapes. If you want to remove objects, please use clear()."))
    
    if color.lower() not in long_to_short_color:
        raise (ValueError(f"Unknown color ({color}) used in remove(). Removeshape currently supports only red, green, blue, and yellow colors. If you want to remove objects, please use clear()."))    

    board_shape = board[top_layer - 1, 0, x_, y_]
    board_color = board[top_layer - 1, 1, x_, y_]

    if board_shape in ["R", "B"]:
        raise (ValueError(f"Non-anchor token found (perhaps a bridge was partially removed) at location ({x}, {y}). Please note that the bridge should be removed as a whole, and only the topmost shape can be removed at a time."))

    if board_shape == "0" or board_shape not in short_to_long:
        raise (ValueError(f"Unknown shape at location ({x}, {y}) ; board_shape: {board_shape}, board_color: {short_to_long_color[board_color]}"))

    color_formatted = long_to_short_color[color.lower()]

    if short_to_long[board_shape] != shape or board_color != color_formatted:
        raise (ValueError(f"Topmost shape({short_to_long[board_shape]}), color ({short_to_long_color[board_color]}) at location ({x}, {y}) does not match with the input data:({shape}, {color}). Please note that only the topmost shape can be removed at a time."))

    if shape == "bridge-h":
        top_layer_next_col = get_top_layer(board, x_, y_ + 1)
        if top_layer != top_layer_next_col:
            raise (ValueError(f"Bridge shape incomplete at location ({x}, {y})"))
        board[top_layer - 1, 0, x_, y_ + 1] = "0"
        board[top_layer - 1, 1, x_, y_ + 1] = "0"
    elif shape == "bridge-v":
        top_layer_next_row = get_top_layer(board, x_ + 1, y_)
        if top_layer != top_layer_next_row:
            raise (ValueError(f"Bridge shape incomplete at location ({x}, {y})"))
        board[top_layer - 1, 0, x_ + 1, y_] = "0"
        board[top_layer - 1, 1, x_ + 1, y_] = "0"

    board[top_layer - 1, 0, x_, y_] = "0"
    board[top_layer - 1, 1, x_, y_] = "0"


    return board