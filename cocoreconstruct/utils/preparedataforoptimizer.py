import os
from typing import Dict
import base64
import numpy as np
import json
import shutil

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PrepareDataForOptimizer:

    def __init__(self):
        pass


    def run(self, base_dir):
        if base_dir is None or not os.path.exists(base_dir):
            raise RuntimeError(f"PrepareDataForOptimizer.run called with invalid base_dir: {base_dir}")
        data_for_optimizer = {}

        for filename in os.listdir(base_dir):
            if not filename.endswith(".json") or "combo_name_" not in filename:
                logger.debug(f"Skipping file: {filename}")
                continue
            version_number = filename.split("_v")[1].split(".json")[0]          
            if version_number.isdigit():
                version_number = int(version_number)
            else:
                logger.debug(f"Skipping file with invalid version number: {filename}")
                continue
            filepath = os.path.join(base_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Error reading JSON file {filepath}: {e}")
                print(f"Error reading JSON file {filepath}: {e}")
                continue

            logger.info(f"Processing file: {filename} with version: {version_number}")
            board_info = data.get("board_info", {})
            if not board_info:
                logger.debug(f"Skipping entry with missing boardinfo in file {filename}")
                continue
            board_type = board_info.get("board_type", "unknown")
            object_type = board_info.get("object_type", "unknown")
            variant = board_info.get("variant", "unknown")
            total_shapes = board_info.get("total_shapes", 0)
            combo_name = board_info.get("combo_name", "unknown")

            if any(x is None for x in [board_type, object_type, variant]):
                logger.debug(f"Skipping entry with incomplete boardinfo in file {filename}")
                continue

            if board_type not in data_for_optimizer:
                data_for_optimizer[board_type] = {}
            if object_type not in data_for_optimizer[board_type]:
                data_for_optimizer[board_type][object_type] = {}
            variant_mod = f"reconstruct-{variant}"
            #if variant_mod not in data_for_optimizer[board_type][object_type]:
            #    data_for_optimizer[board_type][object_type][variant_mod] = {}
            if total_shapes not in data_for_optimizer[board_type][object_type]:
                data_for_optimizer[board_type][object_type][total_shapes] = {}
            if combo_name not in data_for_optimizer[board_type][object_type][total_shapes]:
                data_for_optimizer[board_type][object_type][total_shapes][combo_name] = []

            entry = {
                "combo_name": combo_name,
                "shapes": board_info.get("shapes", None),
                "colors": board_info.get("colors", None),
                "x": board_info.get("x", None),
                "y": board_info.get("y", None),
                "orientations": board_info.get("orientations", None),
                "rows": board_info.get("size", {}).get("rows", None),
                "cols": board_info.get("size", {}).get("cols", None),
                "min_rows": board_info.get("min_rows", None),
                "min_cols": board_info.get("min_cols", None),
                "code": board_info.get("synthetic_code", None),
                "dialogues": {variant_mod:{"instructions":[{"<Programmer>": board_info.get("synthetic_instructions", None)}]}},
                "quadrant": board_info.get("quadrant", None),
                "seed_template": board_info.get("seed_template_name", None),
                "instructions": board_info.get("synthetic_instructions", None),
            }

            for key, value in data.items():
                if key == "board_info":
                    continue
                # Further processing if needed
                #if key == "gen_occupied_cells":
                #    logger.info(f"gen_occupied_cells {value}")
                #    print(f"gen_occupied_cells {value}")
                entry[key] = value
            data_for_optimizer[board_type][object_type][total_shapes][combo_name].append(entry)
        #Sort datdata_for_optimizer by total_shapes
        for board_type in data_for_optimizer:
            for object_type in data_for_optimizer[board_type]:
                sorted_total_shapes = dict(sorted(data_for_optimizer[board_type][object_type].items()))
                data_for_optimizer[board_type][object_type] = sorted_total_shapes
        return data_for_optimizer

    def save_data(self, data: Dict, output_filepath: str):
        try:
            with open(output_filepath, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Data for optimizer saved to {output_filepath}")
        except Exception as e:
            logger.error(f"Error saving data to {output_filepath}: {e}")

    def copy_data(self, sourcefilepath: str, destination_folder: str) -> None:
        #Copy the generated json to cocooptimizer/resources/data/en/ path
        shutil.copy(sourcefilepath, destination_folder)
        logger.info(f"Copied data from {sourcefilepath} to {destination_folder}")


    def main(self, base_dir: str, output_filepath: str):
        data = self.run(base_dir)
        self.save_data(data, output_filepath)
        self.copy_data(output_filepath, "/home/admin/Desktop/codebase/cocobots/testimageccbts_local/clemnew/clembench/cocooptimizer/resources/data/en")
        print("Data preparation complete.")

if __name__ == "__main__":
    base_dir = "reconstruct-data-pairs"
    output_filepath = "reconstructer_data.json"
    preparer = PrepareDataForOptimizer()
    preparer.main(base_dir, output_filepath)
