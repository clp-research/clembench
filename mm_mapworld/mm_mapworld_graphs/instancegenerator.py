from typing import Dict, List

from clemcore.clemgame import GameInstanceGenerator
import sys
import os

sys.path.append(os.path.abspath('../clembench/mm_mapworld'))
from mm_mapworld.mm_mapworld_maps import AbstractMap

import numpy as np
import os
import random
import json
import shutil
from mm_mapworld.config_languages import LANG_CONFIG

# set the name of the game in the script, as you named the directory
# this name will be used everywhere, including in the table of results

NUM_INSTANCES = 10
GRIDS = {"small": (4, 4), "medium": (4, 4), "large": (4, 4)}
SIZES = {"small": 4, "medium": 6, "large": 8}  # num_nodes
# SEED = 42
RANDOM_PATH = "random_test_images"
IMAGE_PATH = os.path.join("..", "mm_mapworld_main", "resources", "images")
DATASET_PATH = os.path.join("..", "mm_mapworld_main", "resources", "ade_20k_reduced", "ade_imgs")
MAPPING_PATH = os.path.join("..", "mm_mapworld_main", "resources", "ade_20k_reduced", "captions.json")
GAME_IMAGES_DIR = os.path.join("resources", "images")
MOVE_CONSTRUCTION = "GO: "
STOP_CONSTRUCTION = "DONE"
GRAPH_REGEX = "\"graph\":\s*(\{\s*\"nodes\"\s*:\s*\[.*\]\s*,\s*\"edges\"\s*:\s*\{.*\})\s*\}$"
RESPONSE_REGEX = "^\{[\s]*\"action\":\s*\"([^\{]*?)\"\s*,\s*\"description\":\s*\"([^\{]*?)\"[\s]*,\s*"
RESPONSE_REGEX += GRAPH_REGEX
DONE_REGEX = '^DONE$'
MOVE_REGEX = '^GO:\s*(north|east|west|south)$'


class MmMapWorldGraphsInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def get_path_to(self, directory):
        return os.path.join(self.game_path, directory)

    def prepare_images_dir(self):
        images_dir = self.get_path_to(GAME_IMAGES_DIR)
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
        os.makedirs(images_dir)

    def copy_image(self, node_img):
        src_dir = self.get_path_to(DATASET_PATH)
        src = os.path.join(src_dir, node_img)
        tgt_dir = self.get_path_to(GAME_IMAGES_DIR)
        file_name = os.path.basename(node_img)  # remove the subdirectory e.g. pantry
        tgt = os.path.join(tgt_dir, file_name)
        shutil.copy(src, tgt)
        # return relative path for dynamic lookup during gameplay
        return os.path.join(GAME_IMAGES_DIR, file_name)

    def assign_images(self, mapping: Dict, nodes: List):
        cats = mapping.keys()
        cats_inside = [cat for cat in cats if 'outdoor' not in cat]
        chosen_cats = np.random.choice(cats_inside, size=len(nodes))
        imgs = {}
        cat_mapping = {}
        for i in range(len(nodes)):
            if nodes[i] in cat_mapping:
                cat_mapping[nodes[i]].append(chosen_cats[i])
            else:
                cat_mapping[nodes[i]] = [chosen_cats[i]]

            node_img = np.random.choice(cat_mapping[nodes[i]])
            after_copy_path = self.copy_image(node_img)
            imgs[nodes[i]] = after_copy_path
        return imgs, cat_mapping

    def create_instances(self, mapping: Dict, grid_size: int, graph_size: int, num_instances=NUM_INSTANCES):
        instances = []
        for i in range(num_instances):
            map = AbstractMap(*grid_size, graph_size)
            nodes = [str(n) for n in map.G]
            edges = list(map.G.edges())
            rev_edges = [(edge[1], edge[0]) for edge in edges]
            edges.extend(rev_edges)
            img_ref, cat_ref = self.assign_images(mapping, nodes)
            instances.append({
                'nodes': nodes,
                'edges': [str(e) for e in edges],
                'imgs': img_ref,
                'cats': cat_ref,
                'start': random.choice(nodes),
                'use_loop_warning': True,
                'use_turn_limit_warning': True
            })
        return instances

    def instance_from_config(self, config, mapping, prompts):
        instances = self.create_instances(mapping,
                                          grid_size=GRIDS[config.get('size', 'large')],
                                          graph_size=SIZES[config.get('size', 'large')],
                                          num_instances=config.get('num_instances', NUM_INSTANCES)
                                          )
        for i in range(len(instances)):
            if config.get('one_shot', 0):
                instances[i]['initial_prompt'] = prompts['initial_one_shot']
            else:
                instances[i]['initial_prompt'] = prompts['initial']
            instances[i]['success_response'] = prompts['later_success']
            instances[i]['invalid_response'] = prompts['later_invalid']
            instances[i]['reprompt'] = config.get('reprompt', False)
            instances[i]['use_images'] = config.get('use_images', True)
            instances[i]["reprompt_format"] = prompts["reprompt_format"]
            instances[i]["limit_warning"] = prompts["limit_warning"]
            instances[i]["loop_warning"] = prompts["loop_warning"]
        return instances

    def on_generate(self, seed: int, **kwargs):
        language = kwargs.get("language") or kwargs.get("lang", "en")
        lang = LANG_CONFIG[language]
        lang_dir = lang.get("prompt_dir", "")

        prompts = {
            'initial': self.load_template(os.path.join("resources", "initial_prompts", lang_dir, "prompt.template")),
            'initial_one_shot': self.load_template(
                os.path.join("resources", "initial_prompts", lang_dir, "prompt_one_shot.template")),
            'later_success': self.load_template(os.path.join("resources", "later_prompts", lang_dir, "successful_move.template")),
            'later_invalid': self.load_template(os.path.join("resources", "later_prompts", lang_dir, "invalid_move.template")),
            'reprompt_format': self.load_template(os.path.join("resources", "reprompts", lang_dir, "invalid_format.template")),
            'limit_warning': self.load_template(os.path.join("resources", "later_prompts", lang_dir, "turn_limit.template")),
            'loop_warning': self.load_template(os.path.join("resources", "later_prompts", lang_dir, "loop.template")),
        }
        experiments = {
            'small': {"size": "small", "reprompt": False, "one_shot": True},
            'medium': {"size": "medium", "reprompt": False, "one_shot": True},
            'large': {"size": "large", "reprompt": False, "one_shot": True}
        }

        self.prepare_images_dir()

        mapping_path = self.get_path_to(MAPPING_PATH)
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        for experiment_name, experiment_config in experiments.items():
            experiment = self.add_experiment(experiment_name)
            game_id = 0
            generated_instances = self.instance_from_config(experiment_config, mapping, prompts)
            for inst in generated_instances:
                instance = self.add_game_instance(experiment, game_id)
                for key, value in inst.items():
                    instance[key] = value
                instance["move_construction"] = f'{lang["MOVE"]}: '
                instance["stop_construction"] = lang["DONE"]
                instance["response_regex"] = RESPONSE_REGEX
                instance["done_regex"] = f'^{lang["DONE"]}$'
                instance["move_regex"] = (rf'^{lang["MOVE"]}:\s*(' + "|".join(lang["DIRECTIONS"]) + r')$')
                instance["directions"] = lang["DIRECTIONS"]
                instance["dir_to_delta"] = lang["DIR_TO_DELTA"]
                game_id += 1


if __name__ == '__main__':
    # always call this, which will actually generate and save the JSON file
    import json

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    with open(os.path.join(ROOT, "SUPPORTED_LANGUAGES.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    GAME_NAME = "mm_mapworld"
    supported_languages = [lang for lang, data in config["languages"].items() if GAME_NAME in data["games"]]
    if not supported_languages:
        print(f"No languages configured for game '{GAME_NAME}'")
    for lang in supported_languages:
        print(f"Generating instances for language '{lang}'")
        MmMapWorldGraphsInstanceGenerator().generate(seed=42, lang=lang, filename=f"instances_{lang}.json")
