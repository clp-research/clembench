import os
from typing import Dict
import re
import ast
import base64
import traceback
import numpy as np
from utils.coco import (
#from coco import (
    init_board,
    plot_board,
    put,
    move,
    remove,
    clear,
    SameShapeStackingError,
    SameShapeAtAlternateLevels,
    NotOnTopOfScrewError,
    DepthMismatchError,
)
import logging

logger = logging.getLogger(__name__)

class PrepareASCIIRep:

    def __init__(self):
        self.occupied_cells = []


    def _list_occupied_cells_with_details(self, board):
        occupied_cells = {}

        if board is None:
            return occupied_cells

        for row in range(board.shape[2]):
            for col in range(board.shape[3]):
                cell_elements = []
                # check each layer for the current cell
                for layer in range(board.shape[0]):
                    # get shape and color
                    shape = board[layer, 0, row, col]
                    color = board[layer, 1, row, col]
                    # If the shape is not '0', then the cell is occupied
                    if shape != "0":
                        cell_elements.append((shape, color))

                if cell_elements:
                    occupied_cells[f"{row}:{col}"] = cell_elements

        return occupied_cells

    def _execute_code(self, gt_code: dict, board_size: dict) -> np.ndarray:
        """Execute the ground truth code and return the board state."""
        board = init_board(board_size["rows"], board_size["cols"])
        gt_exec_code = f"{gt_code['function']}\n{gt_code['usage']}"
        try:
            exec(gt_exec_code)
        except Exception as e:
            #print(f"Error executing code: {e}")
            logger.error(f"Error executing code: {e}")
            return None
        return board
    
    def undo(self, board: np.ndarray) -> np.ndarray:
        if not self.occupied_cells:
            logger.info(f"No occupied cells to undo.")
            return None, None
        move_locations = self.occupied_cells.pop()
        logger.info(f"Undoing last move, removed occupied cells: {move_locations}")
        for key in move_locations:
            logger.info(f"Clearing cell at {key}")
            for shape, color in move_locations[key]:
                row, col = map(int, key.split(":"))
                logger.info(f"Removing shape {shape} with color {color} at ({row}, {col})")
                remove(board, row, col, shape, color)
        return board
    
    def _save_skills(self, skill_name, skill_code):
        if skill_name not in self.avail_skills:
            self.avail_skills[skill_name] = []

        self.avail_skills[skill_name]
    
    #def _process_remove_object(self, board: np.ndarray, x, y, shape: str, color: str) -> np.ndarray:


    def removeshape(self, board: np.ndarray, x, y, shape: str, color: str) -> np.ndarray:
        if shape is None or color is None:
            logger.info(f"Shape or color is None, cannot remove shape.")
            return board
        if shape in ["bridge-h-right", "bridge-v-bottom"]:
            logger.info(f"Shape {shape} might have been removed with a different method.")
            return board
        if shape == "bridge-h-left":
            shape = "bridge-h"
        if shape == "bridge-v-top":
            shape = "bridge-v"
        logger.info(f"Removing shape {shape} with color {color} from the board.")
        return remove(board, x, y, shape, color)

    def set_occupied_cells(self, occupied_cells: Dict[str, list]) -> np.ndarray:
        if self.occupied_cells:
            use_cells = {}
            #compute the difference over the previous occupied cells
            prev_occupied_cells = self.occupied_cells[-1]
            for key in occupied_cells:
                if key in prev_occupied_cells:
                    prev_elements = prev_occupied_cells[key]
                    curr_elements = occupied_cells[key]
                    if len(curr_elements) > len(prev_elements):
                        logger.info(f"New elements added at cell {key}: {curr_elements[len(prev_elements):]}")
                        use_cells[key] = curr_elements[len(prev_elements):]
                else:
                    logger.info(f"New cell occupied at {key}: {occupied_cells[key]}")
                    use_cells[key] = occupied_cells[key]
        else:
            use_cells = occupied_cells

        logger.info(f"Setting occupied cells: {use_cells}")
        self.occupied_cells.append(use_cells)

    def execute_generated_response(self, response: str, board_size: dict, board: np.ndarray) -> np.ndarray:
        """Execute the generated response code and return the updated board state."""
        if board is None:
            logger.info(f"Board is None, initializing a new board with the size: {board_size}")
            board = init_board(board_size["rows"], board_size["cols"])

        code_stats = {"move": 0, "remove": 0, "clear": 0, "undo": 0}

        if "removeshape" in response:
            response = response.replace("removeshape", "self.removeshape")
            code_stats["remove"] += 1
        elif "undo" in response:
            response = response.replace("undo", "self.undo")
            code_stats["undo"] += 1
        else:
            if "move(" in response:
                code_stats["move"] += 1
            if "clear(" in response:
                code_stats["clear"] += 1

        logger.info(f"Executing response:\n{response}")
        error = None

        try:
            exec(response)
            logger.info(f"Response executed successfully.")
        except Exception as e:
            logger.error(f"Error executing response: {e}")
            error = str(e)
        return board, error, code_stats
    
    def execute_generated_response_skill(self, response: str, board_size: dict, board: np.ndarray, skillfunc_def) -> np.ndarray:
        """Execute the generated response code and return the updated board state."""
        if board is None:
            logger.info(f"Board is None, initializing a new board with the size: {board_size}")
            board = init_board(board_size["rows"], board_size["cols"])

        if "removeshape" in response:
            response = response.replace("removeshape", "self.removeshape")
        elif "undo" in response:
            response = response.replace("undo", "self.undo")


        out_code = f"{skillfunc_def}\n{response}"   
        logger.info(f"Executing response:\n{out_code}")

        """
        env = {
            "__builtins__": __builtins__,  # or restrict if you want sandboxing
            "np": np,
            "board": board,
            "board_size": board_size,
            # coco APIs
            "init_board": init_board,
            "put": put,
            "move": move,
            "clear": clear,
        }
        """        
        try:
            #exec(out_code, env, env)
            exec(out_code)
            logger.info(f"Response executed successfully.")
        except Exception as e:
            logger.error(f"Error executing response: {e}")
            logger.error("Error type: %s", type(e).__name__)
            logger.error("Error repr: %r", e)
            logger.error("Traceback:\n%s", traceback.format_exc())            
            return None, str(e), None       
        #board = env.get("board", board)
        return board, None, None


    
    def execute_generated_response_skill_working(self, response: str, board_size: dict, board: np.ndarray, skillfunc_def) -> np.ndarray:
        """Execute the generated response code and return the updated board state."""
        if board is None:
            logger.info(f"Board is None, initializing a new board with the size: {board_size}")
            board = init_board(board_size["rows"], board_size["cols"])

        clearfunc_code = """
def clear(board):
    board[:, :, :, :] = "0"
    return board
"""

        out_code = f"{clearfunc_code}\n{skillfunc_def}\n{response}"
        logger.info(f"Executing response:\n{out_code}")
        try:
            exec(out_code)
            logger.info(f"Response executed successfully.")
        except Exception as e:
            logger.error(f"Error executing response: {e}")
            return None, str(e), None
        return board, None, None

    def execute_optimized_response(self, board_size: dict, board: np.ndarray, optim_func, func_usage) -> np.ndarray:
        if board is None:
            logger.info(f"Board is None, initializing a new board with the size: {board_size}")
            board = init_board(board_size["rows"], board_size["cols"])
        #logger.info(f"Executing optimized function: {optim_func}")

        exec_code = f"{optim_func}\n{func_usage}"
        try:
            exec(exec_code)
            logger.info(f"Optimized function executed successfully.")
        except Exception as e:
            logger.error(f"Error executing optimized function: {e}")
            error = str(e)
            return None, error
        return board, None


    def _elaborate_shape_color(self, shape: str, color: str) -> str:
        """Elaborate the shape and color into a string representation."""
        shape = shape.upper()
        color = color.upper()
        if shape == "W":
            shape = "washer"
        if shape == "N":
            shape = "nut"
        if shape == "S":
            shape = "screw"
        if shape == "L":
            shape = "bridge-h-left"
        if shape == "R":
            shape = "bridge-h-right"
        if shape == "T":
            shape = "bridge-v-top"
        if  shape == "B":
            shape = "bridge-v-bottom"
        if color == "R":
            color = "red"
        if color == "B":
            color = "blue"
        if color == "G":
            color = "green"
        if color == "Y":
            color = "yellow"

        return f"{shape}:{color}", shape, color
        #return shape, color

    def _elaborate_shape_color_optim(self, shape: str, color: str) -> str:
        """Elaborate the shape and color into a string representation."""
        shape = shape.upper()
        color = color.upper()
        if shape == "W":
            shape = "washer"
        if shape == "N":
            shape = "nut"
        if shape == "S":
            shape = "screw"
        if shape == "L":
            shape = "bridge-h"
        if shape == "R":
            shape = "bridge-h"
        if shape == "T":
            shape = "bridge-v"
        if  shape == "B":
            shape = "bridge-v"
        if color == "R":
            color = "red"
        if color == "B":
            color = "blue"
        if color == "G":
            color = "green"
        if color == "Y":
            color = "yellow"

        return f"{shape}:{color}", shape, color
        #return shape, color        
    
    def _prepare_shape_color_dict(self, elements, use_optim=False) -> dict:
        shapes = []
        colors = []
        for shape, color in elements:
            if use_optim:
                _, shape_, color_ = self._elaborate_shape_color_optim(shape, color)
            else:
                _, shape_, color_ = self._elaborate_shape_color(shape, color)
            shapes.append(shape_)
            colors.append(color_)

        return {"shapes": shapes, "colors": colors}
    

    def get_occupied_cells(self, gt_code: dict, board_size: dict) -> Dict[str, list]:
        board = self._execute_code(gt_code, board_size)
        if board is None:
            return None

        occupied_cells = self._list_occupied_cells_with_details(board)
        return occupied_cells
    
    def _list_occupied_cells_with_repeats(self, combo_name: str, colors: list, repeat_locations: list):
        occupied_cells = {}
        for location in repeat_locations:
            row, col = location[0]-1, location[1]-1
            if f"{row}:{col}" in occupied_cells:
                #print(f"Multiple objects at location {row},{col} for combo {combo_name}")
                #input()
                logger.info(f"Multiple objects at location {row},{col} for combo {combo_name}")
            occupied_cells[f"{row}:{col}"] = [(combo_name, f"{colors}")]
        return occupied_cells 

    def get_layer_representation_rb_reuse(self, combo_name: str, colors: list, repeat_locations: list):
        # No stacking of the objects, so only one layer
        layer_rep = "Grid levels (bottom to top):\n"
        max_layers = 1

        layers_info = {}
        for layer in range(max_layers):
            layers_info[layer+1] = []     
            for loc in repeat_locations:
                row = loc[0]
                col = loc[1]
                shape_info = {"shapes": f"[{combo_name}]", "colors": f"{colors}"}
                use_key = f"row: {row}, col: {col}"
                layers_info[layer+1].append(f"{use_key}: {shape_info}")

        for layer in layers_info:
            layer_rep += f"Level {layer}:\n"
            for cell_info in layers_info[layer]:
                layer_rep += f"\t{cell_info}\n"

        return layer_rep            
    
    def get_ascii_representation_rb(self, board_size: dict, gt_code: dict,  combo_name: str, colors: list, repeat_locations: list) -> str:
        """Convert the ground truth code to an ASCII representation."""
        if board_size is None or "rows" not in board_size or "cols" not in board_size:
            logger.info(f"Board size is invalid, cannot generate ASCII representation.")
            return None, None, None, None
        if combo_name is None or combo_name == "":
            logger.info(f"Combo name is invalid, cannot generate ASCII representation.")
            return None, None, None, None
        
        if colors is None or not colors:
            logger.info(f"Colors is None, cannot generate ASCII representation.")
            return None, None, None, None
        
        if repeat_locations is None or not repeat_locations:
            logger.info(f"Repeat locations is None, cannot generate ASCII representation.")
            return None, None, None, None

        board = self._execute_code(gt_code, board_size)
        if board is None:
            return None, None, None, None
        
        occupied_cells = self._list_occupied_cells_with_details(board)

        occupied_cells_repeat = self._list_occupied_cells_with_repeats(combo_name, colors, repeat_locations)

        layer_rep = self.get_layer_representation_rb_reuse(combo_name, colors, repeat_locations)
        return layer_rep, board, occupied_cells, occupied_cells_repeat       

    def get_ascii_representation(self, gt_code: dict, board_size: dict) -> str:
        """Convert the ground truth code to an ASCII representation."""
        board = self._execute_code(gt_code, board_size)
        if board is None:
            return None

        occupied_cells = self._list_occupied_cells_with_details(board)
        #print(occupied_cells)
        logger.info(f"Occupied cells: {occupied_cells}")
        layer_rep = self.get_layer_representation(occupied_cells)
        return layer_rep, board
        ascii_representation = "[\n"
        num_rows = board_size["rows"]
        num_cols = board_size["cols"]
        """
        for row in range(num_rows):
            ascii_representation += "\t"
            for col in range(num_cols):
                cell_key = f"{row}:{col}"
                if cell_key in occupied_cells:
                    elements = occupied_cells[cell_key]
                    ascii_representation += f"({', '.join(self._elaborate_shape_color(shape, color) for shape, color in elements)}) "
                else:
                    ascii_representation += "▢ " 
            ascii_representation = ascii_representation.strip() + "\n"
        ascii_representation += "]\n"
        """
        for row in range(num_rows):
            for col in range(num_cols):
                cell_key = f"{row}:{col}"
                if cell_key in occupied_cells:
                    elements = occupied_cells[cell_key]
                    #print(elements)
                    #input()
                    logger.info(f"Elements are: {elements}")
                    #shape_color_str = (', '.join(self._elaborate_shape_color(shape, color) for shape, color in elements))
                    shape_info = self._prepare_shape_color_dict(elements)
                    cell_data = {f"row: {row+1}, col: {col+1}, value: ({shape_info})\n"}
                    ascii_representation += f"\t{cell_data}"

            ascii_representation = ascii_representation.strip() + "\n"
        ascii_representation += "]\n"
        #print(ascii_representation)
        logger.info(f"ascii_representation: {ascii_representation}")

        return ascii_representation, board


    def get_ascii_representation_from_combo_names(self, board: str, board_size: dict, reuse_skills: list, funcdef: str) -> str:
        """Generate an ASCII representation from the board state."""
        if board is None or board_size is None or not reuse_skills:
            logger.info(f"Board generation call is None, cannot generate ASCII representation.")
            error = "Invalid input for generating ASCII representation from combo names."
            return None, error, None, None
        logger.debug(f"Generating ASCII representation from the board generation call.")
        occupied_cells = self._list_occupied_cells_with_details(board)
        #print(occupied_cells)


        try:
            newfuncboard = init_board(board_size["rows"], board_size["cols"])
            for skill in reuse_skills:
                usage = f"{skill['name']}(board, {skill['colors']}, {skill['x']}, {skill['y']})"
                self.execute_generated_response_skill(usage, board_size, newfuncboard, funcdef)
            newoccupied_cells = self._list_occupied_cells_with_details(newfuncboard)
            logger.info(f"Generated occupied cells from reuse skills: {newoccupied_cells}")

            if newoccupied_cells != occupied_cells:
                # TODO: There should be a way to fix this -> check 61 episode
                logger.error(f"Mismatch in occupied cells from reuse skills and board generation call.")
                error = "Mismatch in occupied cells from reuse skills and board generation call."
                return None, error, None, None
            

            # TODO: Because only the first name, colors are taken, if there are issues in other locations, it is misleading to same values -> Fix this
            combo_name = reuse_skills[0]['name']
            combo_colors = reuse_skills[0]['colors']
            repeat_locations = [[skill['x'],skill['y']] for skill in reuse_skills]
            occupied_cells_repeat = self._list_occupied_cells_with_repeats(combo_name, combo_colors, repeat_locations)

            layer_rep = self.get_layer_representation_rb_reuse(combo_name, combo_colors, repeat_locations)
            return layer_rep, board, occupied_cells, occupied_cells_repeat 
        except Exception as e:
            logger.error(f"Error executing reuse skill: {e}")
            error = str(e)
            return None, error, None, None

    def get_layer_representation_diff_rb(self, gt_rep_locations, gen_rep_locations):
        """Generate a layer-wise ASCII representation from the occupied cells."""
        if not gt_rep_locations or not gen_rep_locations:
            logger.error(f"No occupied cells provided.{gt_rep_locations}, {gen_rep_locations}")
            return None
        logger.info(f"GT_Rep Locations: {gt_rep_locations}")
        logger.info(f"Gen_Rep Locations: {gen_rep_locations}")


        diff_rep = f"Level 1:\n"
        for location in gt_rep_locations:
            if location not in gen_rep_locations:
                diff_rep += f"\trow: {location[0]}, col: {location[1]}: Missing\n"
            else:
                diff_rep += f"\trow: {location[0]}, col: {location[1]}: Identical\n"

        for location in gen_rep_locations:
            if location not in gt_rep_locations:
                diff_rep += f"\trow: {location[0]}, col: {location[1]}: Extra\n"

        return diff_rep
    
    def _extract_row_col_shape_colors_from_gridrep(self, gridrep):
        pattern = re.compile(
            r"row:\s*(\d+),\s*col:\s*(\d+):\s*\{'shapes':\s*'([^']+)',\s*'colors':\s*\"([^\"]+)\"\}"
        )

        results = []

        for match in pattern.findall(gridrep):
            row, col, shapes, colors = match
            results.append({
                "row": int(row),
                "col": int(col),
                "shapes": shapes,
                "colors": colors
            })

        for r in results:
            r["colors"] = ast.literal_eval(r["colors"])


        logger.info(f"Extracted row, col, shapes, colors from grid representation:\n{results}")
        return results    
    
    def unusedcomparegrid(self, gtrep, genrep):
        if gtrep is None or genrep is None:
            error = f"One of the grid representations is None. GT: {gtrep}, Gen: {genrep}"
            logger.error(error)
            return None, error

        gtrep = self._extract_row_col_shape_colors_from_gridrep(gtrep)
        genrep = self._extract_row_col_shape_colors_from_gridrep(genrep)

        diff_rep = f"Level 1:\n"
        for index, glitem in enumerate(gtrep):
            if index < len(genrep):
                genitem = genrep[index]
                if glitem["row"] == genitem["row"] and glitem["col"] == genitem["col"]:
                    if glitem["shapes"] == genitem["shapes"] and glitem["colors"] == genitem["colors"]:
                        #logger.info(f"Match found for location row: {glitem['row']}, col: {glitem['col']}")
                        diff_rep += f"\trow: {glitem['row']}, col: {glitem['col']}: Identical\n"
                    else:
                        if glitem["shapes"] != genitem["shapes"]:
                            diff_rep += f"\trow: {glitem['row']}, col: {glitem['col']}: Shape Mismatch (Goal: {glitem['shapes']}, Player: {genitem['shapes']})\n"
                        elif glitem["colors"] != genitem["colors"]:
                            if type(glitem["colors"]) != type(genitem["colors"]):
                                if set(glitem["colors"]) != set(genitem["colors"]):
                                    diff_rep += f"\trow: {glitem['row']}, col: {glitem['col']}: Color Mismatch (Goal: {glitem['colors']}, Player: {genitem['colors']})\n"
                                else:
                                    diff_rep += f"\trow: {glitem['row']}, col: {glitem['col']}: Identical\n"

                            else:
                                diff_rep += f"\trow: {glitem['row']}, col: {glitem['col']}: Color Mismatch (Goal: {glitem['colors']}, Player: {genitem['colors']})\n"
                        #logger.info(f"Mismatch found for location row: {glitem['row']}, col: {glitem['col']}")
                else:
                    #logger.info(f"Location mismatch between goal and generated representation at index {index}")
                    diff_rep += f"\trow: {glitem['row']}, col: {glitem['col']}: Missing\n"
            else:
                #logger.info(f"No corresponding generated item for goal item at index {index}")
                diff_rep += f"\trow: {glitem['row']}, col: {glitem['col']}: Missing\n"

        for index, genitem in enumerate(genrep):
            if index >= len(gtrep):
                #logger.info(f"Extra generated item at index {index}")
                diff_rep += f"\trow: {genitem['row']}, col: {genitem['col']}: Extra\n"

        return diff_rep, None

    
    def comparegridrep(self, gtrep, genrep):
        if gtrep is None or genrep is None:
            error = f"One of the grid representations is None. GT: {gtrep}, Gen: {genrep}"
            logger.error(error)
            return None, error

        gtrep = self._extract_row_col_shape_colors_from_gridrep(gtrep)
        genrep = self._extract_row_col_shape_colors_from_gridrep(genrep)

        def _colors_equal(a, b):
            # Handle list/tuple comparison: convert both to lists for order-dependent comparison
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                return list(a) == list(b)
            return a == b

        gt_map = {(item["row"], item["col"]): item for item in gtrep}
        gen_map = {(item["row"], item["col"]): item for item in genrep}

        diff_rep = f"Level 1:\n"
        all_keys = sorted(set(gt_map.keys()).union(gen_map.keys()))
        for row, col in all_keys:
            gt_item = gt_map.get((row, col))
            gen_item = gen_map.get((row, col))
            if gt_item is None:
                diff_rep += f"\trow: {row}, col: {col}: Extra\n"
                continue
            if gen_item is None:
                diff_rep += f"\trow: {row}, col: {col}: Missing\n"
                continue

            if gt_item["shapes"] == gen_item["shapes"] and _colors_equal(gt_item["colors"], gen_item["colors"]):
                diff_rep += f"\trow: {row}, col: {col}: Identical\n"
            else:
                if gt_item["shapes"] != gen_item["shapes"]:
                    diff_rep += f"\trow: {row}, col: {col}: Shape Mismatch (Goal: {gt_item['shapes']}, Player: {gen_item['shapes']})\n"
                elif not _colors_equal(gt_item["colors"], gen_item["colors"]):
                    diff_rep += f"\trow: {row}, col: {col}: Color Mismatch (Goal: {gt_item['colors']}, Player: {gen_item['colors']})\n"

        return diff_rep, None

    def get_layer_representation_diff_rb_withcolorsshapes(self, gt_data, gen_data):
        """Generate a layer-wise ASCII representation from the occupied cells."""

        if "shapes" not in gt_data or "colors" not in gt_data or "locations" not in gt_data:
            logger.error(f"GT data is missing required keys: {gt_data}")
            return None
        if "shapes" not in gen_data or "colors" not in gen_data or "locations" not in gen_data:
            logger.error(f"Generated data is missing required keys: {gen_data}")
            return None
        
        if gt_data["shapes"] is None or gt_data["colors"] is None or gt_data["locations"] is None:
            logger.error(f"GT data has None values: {gt_data}")
            return None
        
        if gen_data["shapes"] is None or gen_data["colors"] is None or gen_data["locations"] is None:
            logger.error(f"Generated data has None values: {gen_data}")
            return None
        
        gt_rep_locations = gt_data["locations"]
        gen_rep_locations = gen_data["locations"]
        
        if not gt_rep_locations or not gen_rep_locations:
            logger.error(f"No occupied cells provided.{gt_rep_locations}, {gen_rep_locations}")
            return None
        logger.info(f"GT_Rep Locations: {gt_rep_locations}")
        logger.info(f"Gen_Rep Locations: {gen_rep_locations}")


        diff_rep = f"Level 1:\n"
        for loc_index, location in enumerate(gt_rep_locations):
            if location not in gen_rep_locations:
                diff_rep += f"\trow: {location[0]}, col: {location[1]}: Missing\n"
            else:
                #Compare shapes and colors
                if gen_data["shapes"][loc_index] != gt_data["shapes"]:
                    diff_rep += f"\trow: {location[0]}, col: {location[1]}: Shape Mismatch (Goal: {gt_data['shapes']}, Player: {gen_data['shapes'][loc_index]})\n"
                elif gen_data["colors"][loc_index] != gt_data["colors"]:
                    diff_rep += f"\trow: {location[0]}, col: {location[1]}: Color Mismatch (Goal: {gt_data['colors']}, Player: {gen_data['colors'][loc_index]})\n"
                else:
                    diff_rep += f"\trow: {location[0]}, col: {location[1]}: Identical\n"

        for location in gen_rep_locations:
            if location not in gt_rep_locations:
                diff_rep += f"\trow: {location[0]}, col: {location[1]}: Extra\n"

        return diff_rep


    def get_func_details(self, repeat_code, combo_name):        
        func_header = f"def {combo_name}(board, colors, x, y):\n\tcoordinates.append((x, y))\n\tcolorslist.append(colors)\n"
        coordinates = []
        colorslist = []
        board = init_board(8, 8)
        ns = {
            "coordinates": coordinates,
            "board": board,
            "colorslist": colorslist,
            }

        rcode_lines = repeat_code.strip().split('\n')
        updatedrcode = []
        for line in rcode_lines:
            if line.strip().startswith("clear" + "(") or line.strip().startswith("remove" + "(") or line.strip().startswith("move" + "("):
                continue
            else:
                updatedrcode.append(line)
        use_repeat_code = '\n'.join(updatedrcode)
        logger.info(f"Before cleaning repeat code:\n{repeat_code}")
        logger.info(f"After cleaning repeat code:\n{use_repeat_code}")


        outcode = func_header + use_repeat_code
        logger.info(f"Executing function code:\n{outcode}")
        exec(outcode, ns)
        if isinstance(coordinates, list) and len(coordinates) == 0:
            logger.info(f"No coordinates found for combo_name: {combo_name}, repeat_code:\n{repeat_code}")

        if isinstance(colorslist, list) and len(colorslist) == 0:
            logger.info(f"No colors found for combo_name: {combo_name}, repeat_code:\n{repeat_code}")

        func_name_list = [combo_name for _ in coordinates]
        func_colors_list = colorslist
        x_list = [coord[0] for coord in coordinates]
        y_list = [coord[1] for coord in coordinates]

        return func_name_list, func_colors_list, x_list, y_list   


    def saveboard(self, board, filename):
        """Save the board state to a file."""
        if board is None:
            logger.info(f"Board is None, cannot save to file.")
            return
        
        #/home/admin/Desktop/codebase/cocobots/testimageccbts_local/clembench/imageccbts/
        os.makedirs("instanceplots/", exist_ok=True)
        plot_board(board, f"instanceplots/{filename}")
        
        logger.info(f"Board saved to {filename}")
    

    def get_ascii_representation_forvalidation(self, gt_code: dict, board_size: dict) -> str:
        """Convert the ground truth code to an ASCII representation."""
        board = self._execute_code(gt_code, board_size)
        if board is None:
            return None

        occupied_cells = self._list_occupied_cells_with_details(board)
        return occupied_cells
    
    def get_ascii_representation_from_board_forvalidation(self, board: np.ndarray, board_size: dict) -> str:
        occupied_cells = self._list_occupied_cells_with_details(board)
        return occupied_cells    

    
    def get_ascii_representation_from_board_layers(self, board: np.ndarray, board_size: dict) -> str:
        """Generate an ASCII representation from the board state."""
        if board is None or board_size is None:
            logger.info(f"Board is None, cannot generate ASCII representation.")
            return None, None
        logger.debug(f"Generating ASCII representation from the board state.")
        occupied_cells = self._list_occupied_cells_with_details(board)
        logger.info(f"Occupied cells: {occupied_cells}")
        layer_rep = self.get_layer_representation(occupied_cells)
        return layer_rep, occupied_cells

        ascii_representation = "[\n"

        num_rows = board_size["rows"]
        num_cols = board_size["cols"]

        for row in range(num_rows):
            for col in range(num_cols):
                cell_key = f"{row}:{col}"
                if cell_key in occupied_cells:
                    elements = occupied_cells[cell_key]
                    #ascii_representation += f"({', '.join(self._elaborate_shape_color(shape, color) for shape, color in elements)}) "
                    #ascii_representation += f"{', '.join(self._elaborate_shape_color(shape, color) for shape, color in elements)} "

                    shape_info = self._prepare_shape_color_dict(elements)
                    cell_data = {f"row: {row+1}, col: {col+1}, value: ({shape_info})\n"}
                    ascii_representation += f"\t{cell_data}"

            ascii_representation = ascii_representation.strip() + "\n" #TODO: Do we need a comman for next line?

        ascii_representation += "]\n"

        return ascii_representation


    def get_ascii_representation_from_board(self, board: np.ndarray, board_size: dict) -> str:
        """Generate an ASCII representation from the board state."""
        if board is None or board_size is None:
            logger.info(f"Board is None, cannot generate ASCII representation.")
            return None
        logger.debug(f"Generating ASCII representation from the board state.")
        occupied_cells = self._list_occupied_cells_with_details(board)
        logger.info(f"Occupied cells: {occupied_cells}")
        ascii_representation = "[\n"

        num_rows = board_size["rows"]
        num_cols = board_size["cols"]
        for row in range(num_rows):
            ascii_representation += "\t["
            for col in range(num_cols):
                cell_key = f"{row}:{col}"
                if cell_key in occupied_cells:
                    elements = occupied_cells[cell_key]
                    #ascii_representation += f"({', '.join(self._elaborate_shape_color(shape, color) for shape, color in elements)}) "
                    #ascii_representation += f"{', '.join(self._elaborate_shape_color(shape, color) for shape, color in elements)} "
                    for shape, color in elements:
                        shape_color_str, *_ = self._elaborate_shape_color(shape, color)
                        ascii_representation += f"({shape_color_str})"
                else:
                    #ascii_representation += "▢ " # Using a square to represent an empty cell
                    #ascii_representation += "⬜ " # Using a square to represent an empty cell 
                    ascii_representation += "null, " # Using a square to represent an empty cell 
            ascii_representation = ascii_representation.strip() + "],\n"

        ascii_representation += "]\n"

        return ascii_representation


    def get_empty_board(self, board_size: dict) -> np.ndarray:
        """Generate an empty board state."""
        board = init_board(board_size["rows"], board_size["cols"])
        return board
    
    def get_image_gen_board(self, board: np.ndarray, filename: str, delete_after: bool = False) -> str:
        """Generate an image from the board state and return its base64 representation."""
        if board is None:
            logger.info(f"Board is None, cannot generate image.")
            return None

        self.saveboard(board, filename)
        with open(f"instanceplots/{filename}", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        if delete_after:
            os.remove(f"instanceplots/{filename}")
        
        logger.info(f"Generated base64 image from board.{encoded_string[:10]}")
        return encoded_string
    
    def get_image_base64_from_filepath(self, filepath: str) -> str:
        """Generate a base64 representation from an image file path."""
        if filepath is None or not os.path.exists(filepath):
            logger.info(f"Filepath is invalid or does not exist: {filepath}")
            return None

        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        logger.info(f"Generated base64 image from filepath.{encoded_string[:10]}")
        return encoded_string


    def get_empty_ascii_representation_old(self, board_size: dict) -> str:
        """Generate an empty ASCII representation for the board."""

        num_rows = board_size["rows"]
        num_cols = board_size["cols"]
        ascii_representation = "[\n"
        for row in range(num_rows):
            ascii_representation += "\t"
            for col in range(num_cols):
                ascii_representation += "▢ "  # Using a square to represent an empty cell
        ascii_representation = ascii_representation.strip() + "\n"
        ascii_representation += "]\n"
        return ascii_representation
    
    def get_empty_ascii_representation(self, board_size: dict) -> str:
        """Generate an empty ASCII representation for the board."""

        num_rows = board_size["rows"]
        num_cols = board_size["cols"]
        rows = [f"\t[{', '.join(['null'] * num_cols)}]" for _ in range(num_rows)]
        ascii_representation = "[\n" + ",\n".join(rows) + "\n]\n"
        return ascii_representation
    

    def get_layer_representation(self, occupied_cells, optim=False):
        """Generate a layer-wise ASCII representation from the occupied cells."""
        if not occupied_cells:
            #print(f"No occupied cells provided.")
            logger.info(f"No occupied cells provided.")
            return None
        

        layer_rep = "Grid levels (bottom to top):\n"

        max_layers = max(len(elements) for elements in occupied_cells.values())
        #logger.info(f"Max layers found: {max_layers}")        

        layers_info = {}        
        for layer in range(max_layers):
            layers_info[layer+1] = []     
            for key, value in occupied_cells.items():
                if layer >= len(value):
                    continue
                #print(f"{key}: {value[layer]}")
                shape_info = self._prepare_shape_color_dict([value[layer]], optim)
                row, col = map(int, key.split(":"))
                use_key = f"row: {row+1}, col: {col+1}"
                layers_info[layer+1].append(f"{use_key}: {shape_info}")
        
        for layer in layers_info:
            layer_rep += f"Level {layer}:\n"
            for cell_info in layers_info[layer]:
                layer_rep += f"\t{cell_info}\n"

        return layer_rep
    
    def get_layer_representation_for_optimization(self, inputdata, use_occupied_cells=False):
        """Generate a layer-wise ASCII representation from the occupied cells."""
        if inputdata is None:
            logger.error(f"Not a valid board {inputdata}")
            return None, None

        if use_occupied_cells:
            gen_occupied_cells = inputdata
        else:
            gen_occupied_cells  = self._list_occupied_cells_with_details(inputdata)
            logger.info(f"Generated occupied cells from board: {gen_occupied_cells}")
        if gen_occupied_cells is None:
            logger.error(f"No occupied cells provided.{gen_occupied_cells}")
            return None, None

        gen_levels = self.get_layer_representation(gen_occupied_cells, True)

        logger.info(f"GenBoard_Levels: {gen_levels}")
        return gen_levels, gen_occupied_cells         

    def get_layer_representation_diff_for_optimization(self, gt_occupied_cells, gen_occupied_cells):
        """Generate a layer-wise ASCII representation from the occupied cells."""
        if not gt_occupied_cells or not gen_occupied_cells:
            logger.error(f"No occupied cells provided.{gt_occupied_cells}, {gen_occupied_cells}")
            return None


        gt_levels = {}
        gt_max_layers = max(len(elements) for elements in gt_occupied_cells.values())
        for layer in range(gt_max_layers):
            gt_levels[layer+1] = {}
            for key, value in gt_occupied_cells.items():
                if layer >= len(value):
                    continue
                shape_info = self._prepare_shape_color_dict([value[layer]], True)
                row, col = map(int, key.split(":"))
                use_key = f"{row+1}:{col+1}"
                #print(use_key)
                gt_levels[layer+1][f"{use_key}"] = shape_info

        gen_levels = {}
        gen_max_layers = max(len(elements) for elements in gen_occupied_cells.values())
        for layer in range(gen_max_layers):
            gen_levels[layer+1] = {}
            for key, value in gen_occupied_cells.items():
                if layer >= len(value):
                    continue
                shape_info = self._prepare_shape_color_dict([value[layer]], True)
                row, col = map(int, key.split(":"))
                use_key = f"{row+1}:{col+1}"
                #print(use_key)
                gen_levels[layer+1][f"{use_key}"] = shape_info                

        logger.info(f"GT_Levels: {gt_levels}")
        logger.info(f"Gen_Levels: {gen_levels}")


        max_level = max(gt_max_layers, gen_max_layers)
        diff_rep = "Differences between target grid and player's grid (bottom to top):\n"
        for level in range(1, max_level+1):
            diff_rep += f"Level {level}:\n"
            gt_level = gt_levels.get(level, {})
            gen_level = gen_levels.get(level, {})
            all_keys = set(gt_level.keys()).union(set(gen_level.keys()))
            all_keys = sorted(all_keys, key=lambda x: (int(x.split(":")[0]), int(x.split(":")[1])))
            for key in all_keys:
                gt_value = gt_level.get(key)
                gen_value = gen_level.get(key)
                row, col = map(int, key.split(":"))                
                if gt_value != gen_value:
                    diff_rep += f"\trow: {row}, col: {col}:\n"
                    if gt_value and not gen_value:
                        diff_rep += f"\t\t- Missing: {gt_value}\n"
                    elif not gt_value and gen_value:
                        diff_rep += f"\t\t- Extra: {gen_value}\n"
                    else: #different values in GT and Generated
                        diff_rep += f"\t\t- Expected: {gt_value}\n"
                        diff_rep += f"\t\t- Current Status: {gen_value}\n"
                else:
                    diff_rep += f"\trow: {row}, col: {col}: Identical\n"

        return diff_rep


    def get_layer_representation_diff(self, gt_occupied_cells, gen_occupied_cells):
        """Generate a layer-wise ASCII representation from the occupied cells."""
        if not gt_occupied_cells or not gen_occupied_cells:
            logger.error(f"No occupied cells provided.{gt_occupied_cells}, {gen_occupied_cells}")
            return None


        gt_levels = {}
        gt_max_layers = max(len(elements) for elements in gt_occupied_cells.values())
        for layer in range(gt_max_layers):
            gt_levels[layer+1] = {}
            for key, value in gt_occupied_cells.items():
                if layer >= len(value):
                    continue
                shape_info = self._prepare_shape_color_dict([value[layer]])
                row, col = map(int, key.split(":"))
                use_key = f"{row+1}:{col+1}"
                #print(use_key)
                gt_levels[layer+1][f"{use_key}"] = shape_info

        gen_levels = {}
        gen_max_layers = max(len(elements) for elements in gen_occupied_cells.values())
        for layer in range(gen_max_layers):
            gen_levels[layer+1] = {}
            for key, value in gen_occupied_cells.items():
                if layer >= len(value):
                    continue
                shape_info = self._prepare_shape_color_dict([value[layer]])
                row, col = map(int, key.split(":"))
                use_key = f"{row+1}:{col+1}"
                #print(use_key)
                gen_levels[layer+1][f"{use_key}"] = shape_info                

        logger.info(f"GT_Levels: {gt_levels}")
        logger.info(f"Gen_Levels: {gen_levels}")


        max_level = max(gt_max_layers, gen_max_layers)
        diff_rep = "Differences between target grid and player's grid (bottom to top):\n"
        for level in range(1, max_level+1):
            diff_rep += f"Level {level}:\n"
            gt_level = gt_levels.get(level, {})
            gen_level = gen_levels.get(level, {})
            all_keys = set(gt_level.keys()).union(set(gen_level.keys()))
            all_keys = sorted(all_keys, key=lambda x: (int(x.split(":")[0]), int(x.split(":")[1])))
            for key in all_keys:
                gt_value = gt_level.get(key)
                gen_value = gen_level.get(key)
                row, col = map(int, key.split(":"))                
                if gt_value != gen_value:
                    diff_rep += f"\trow: {row}, col: {col}:\n"
                    if gt_value and not gen_value:
                        diff_rep += f"\t\t- Missing: {gt_value}\n"
                    elif not gt_value and gen_value:
                        diff_rep += f"\t\t- Extra: {gen_value}\n"
                    else: #different values in GT and Generated
                        diff_rep += f"\t\t- Expected: {gt_value}\n"
                        diff_rep += f"\t\t- Current Status: {gen_value}\n"
                else:
                    diff_rep += f"\trow: {row}, col: {col}: Identical\n"

        return diff_rep
    
    
    def encode_image_to_base64(self, filepath):
        """Generate a base64 encoded image from the board state."""        
        #/home/admin/Desktop/codebase/cocobots/testimageccbts_local/clembench/imageccbts/
        with open(f"instanceplots/{filepath}", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        logger.info(f"Generated base64 image from board.{encoded_string[:10]}")
        return f"data:image/png;base64,{encoded_string}"


    def getshapescolorsfromoppcells(self, occupied_cells):
        shapes_colors = {}
        skip_cells = {}        
        shapes = []
        colors = []        
        for key, elements in occupied_cells.items():
            row, col = map(int, key.split(":"))
            for index, element in enumerate(elements):
                shape, color = element
                if shape == "L":
                    next_key = f"{row}:{col+1}"
                    if next_key in occupied_cells:
                        next_elements = occupied_cells[next_key]
                        if index > len(next_elements)-1:
                            #print("Index is not matching for bridge-h-right")
                            #input()
                            logger.info(f"Index is not matching for bridge-h-right at {next_key} for shape L at {key}")
                        else:
                            next_shape, next_color = next_elements[index]
                            if next_shape == "R" and next_color == color:
                                if next_key not in skip_cells:
                                    skip_cells[next_key] = []

                                skip_cells[next_key].append(index)
                            else:
                                #print(f"Bridge-h-right not found or color mismatch: {next_shape}, {next_color}")
                                #input()
                                logger.info(f"Bridge-h-right not found or color mismatch at {next_key} for shape L at {key}. Found shape: {next_shape}, color: {next_color}")
                elif shape == "T":
                    next_key = f"{row+1}:{col}"
                    if next_key in occupied_cells:
                        next_elements = occupied_cells[next_key]
                        if index > len(next_elements)-1:
                            #print("Index is not matching for bridge-v-bottom")
                            #input()
                            logger.info(f"Index is not matching for bridge-v-bottom at {next_key} for shape T at {key}")
                        else:
                            next_shape, next_color = next_elements[index]
                            if next_shape == "B" and next_color == color:
                                if next_key not in skip_cells:
                                    skip_cells[next_key] = []

                                skip_cells[next_key].append(index)
                            else:
                                #print(f"Bridge-v-bottom not found or color mismatch: {next_shape}, {next_color}")
                                #input()   
                                logger.info(f"Bridge-v-bottom not found or color mismatch at {next_key} for shape T at {key}. Found shape: {next_shape}, color: {next_color}")

                elif shape in ["R", "B"]:
                    if key in skip_cells and index in skip_cells[key]:
                        #print(f"Skipping shape {shape} at {key} as it's part of a bridge already processed.")
                        continue
                    else:
                        #print(f"Shape {shape} at {key} is not part of a bridge or already processed.")
                        #print(skip_cells)
                        #input()
                        logger.info(f"Shape {shape} at {key} is not part of a bridge or already processed. Skip cells: {skip_cells}")
                shapes_dict, shape_str, color_str = self._elaborate_shape_color_optim(shape, color)                             
                #print(f"Processed shape-color: {shapes_dict}, {shape_str}, {color_str}")
                    
                #shape_str, color_str = self._elaborate_shape_color(shape, color)
                shapes.append(shape_str)
                colors.append(color_str)
        return shapes, colors




if __name__ == "__main__":
    prepare_ascii_rep = PrepareASCIIRep()
    gt_code = {
        "function": "def wn(board, colors, x, y):\n    shapes = ['bridge-v', 'nut']\n    for shape, color, dx, dy in zip(shapes, colors, [0, 0], [0, 0]):\n            put(board, shape, color, x + dx, y + dy)",
        "usage": "wn(board, ('green', 'blue'), 0, 7)"
    }
    board_size = {"rows": 8, "cols": 8}
    #ascii_rep, _ = prepare_ascii_rep.get_ascii_representation(gt_code, board_size)
    #occupied_cells = prepare_ascii_rep.get_ascii_representation_forvalidation(gt_code, board_size)
    #print(ascii_rep)
    #print(occupied_cells)
    #empty_rep = prepare_ascii_rep.get_empty_ascii_representation(board_size)
    #print(empty_rep)
    #board = init_board(8,8)
    #put(board, "bridge-v", "green", 0, 7)
    #put(board, "washer", "red", 2, 7)
    #put(board, "nut", "blue", 0, 7)
    #gencode_rep, gencode_occupied = prepare_ascii_rep.get_ascii_representation_from_board_sparse(board, board_size)
    #layer_rep = prepare_ascii_rep.get_layer_representation(gencode_rep)
    #print(layer_rep)
    #gencode_rep = prepare_ascii_rep.get_ascii_representation_from_board_forvalidation(board, board_size)
    #print(gencode_rep)
    #print(gencode_occupied)
    #gt_occupied_cells = {1: {'2:6': {'shapes': ['washer'], 'colors': ['red']}, '2:7': {'shapes': ['washer'], 'colors': ['green']}}, 2: {'2:6': {'shapes': ['bridge-h-left'], 'colors': ['yellow']}, '2:7': {'shapes': ['bridge-h-right'], 'colors': ['yellow']}}}
    #gencode_occupied = {1: {'2:6': {'shapes': ['washer'], 'colors': ['red']}, '2:7': {'shapes': ['washer'], 'colors': ['green']}}}
    gt_occupied_cells = {'1:6': [('W', 'r'), ('L', 'g')], '1:7': [('N', 'y'), ('R', 'g')]}#{'0:5': [('L', 'b')], '0:6': [('R', 'b'), ('W', 'y'), ('N', 'g')]}#{'0:7': [('N', 'y'), ('T', 'b')], '1:7': [('W', 'g'), ('B', 'b')]}
    gencode_occupied = {'1:6': [('W', 'r')]}#{'0:5': [('L', 'b')], '0:6': [('R', 'b')]}#{'0:7': [('N', 'y')]}
    #diff_grid = prepare_ascii_rep.get_layer_representation_diff(gt_occupied_cells, gencode_occupied)
    #diff_grid = prepare_ascii_rep.get_layer_representation_diff_for_optimization(gt_occupied_cells, gencode_occupied)
    #print(diff_grid)
    shapes, colors = prepare_ascii_rep.getshapescolorsfromoppcells(gencode_occupied)
    print(shapes, colors)
