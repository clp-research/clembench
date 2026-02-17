"""Template script for instance generation.

usage:
python3 instancegenerator.py
Creates instance.json file in ./in

"""
from math import exp
import os
import logging
import json
import random
from copy import deepcopy
from shutil import move
from string import Template

from matplotlib.pyplot import grid
from numpy import empty, object_
from pandas import value_counts
from regex import T
from clemcore.clemgame import GameInstanceGenerator
from resources.game_state.utils import EMPTY_SYMBOL, TEXT_BASED, IMAGE_BASED, place_objects, GameObject
from resources.game_state.game_state import GridState, SemanticGridState

logger = logging.getLogger(__name__)

# Seed for reproducibility
SEED = 73128361
ICON_METADATA_PATH = "resources/icons/metadata.json"

class CleanUpInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed, config, n_instances, language, modality, **kwargs):
        logger.info(f"Generating a total of {n_instances * len(config['experiments'])} instances for language '{language}' and modality '{modality}'")
        self.modality = modality
        self.objects = config['objects']
        self.language = language
        self.initial_prompt = self.load_config_json('resources/initial_prompt.json')
        self.start_messages = self.load_config_json(f'resources/start_messages.json')
        self.commands = self.load_config_json('resources/commands.json')
        move_messages = self.load_config_json('resources/move_messages.json')
        parse_errors = self.load_config_json('resources/parse_errors.json')
        intermittent_prompts = self.load_config_json('resources/intermittent_prompts.json')
        intermittent_prompts['invalid_response'] = Template(intermittent_prompts['invalid_response']).safe_substitute(
            say=self.commands['say'],
            move=self.commands['move'],
            empty_symbol=EMPTY_SYMBOL
        )
        template_instance = {
            "intermittent_prompts": intermittent_prompts,
            "say_pattern": self.commands['say_pattern'],
            "move_pattern": self.commands['move_pattern'],
            "parse_errors": parse_errors,
            "move_messages": move_messages,
            "terminate_question": self.commands['terminate_question'],
            "terminate_answer": self.commands['terminate_answer']
        }

        restricted_patterns = self.load_config_json('resources/restricted_patterns.json')
        if restricted_patterns:
            template_instance["restricted_patterns"] = restricted_patterns

        for experiment_conf in config['experiments']:
            self.current_experiment_conf = deepcopy(experiment_conf)
            sampled_backgrounds = []
            if self.modality in TEXT_BASED:
                # For each experiment and object count, sample n_instances backgrounds
                # This way we ensure that easy_3obj/instance_0000 has the same background
                # as easy_5obj/instance_0000 and so on.
                for _ in range(n_instances):
                    background, background_stats = self.sample_background()
                    sampled_backgrounds.append((background, background_stats))
            else:
                # For image modality, we use the same background for all instances
                background, background_stats = self.sample_background()
                sampled_backgrounds = [(background, background_stats)] * n_instances
            for object_count in config['objects']:
                if self.modality in IMAGE_BASED:
                    self.current_experiment_conf = {key: object_count if val == 'OBJECT_COUNT' else val for key, val in self.current_experiment_conf.items()}
                experiment_name = f"{self.current_experiment_conf['name']}_{object_count}obj_{language}"
                experiment = self.add_experiment(experiment_name)
                # max_penalties: 2 'free' penalties for each player for format errors etc.
                #                + config['penalty_factor'] * object_count
                max_penalties = config['penalty_factor'] * int(object_count) + 2
                max_rounds = int(object_count) * 4
                for instance_id in range(n_instances):
                    game_instance = self.add_game_instance(experiment, instance_id)
                    game_instance['modality'] = modality
                    game_instance['language'] = language
                    game_instance['max_penalties'] = max_penalties
                    game_instance['max_rounds'] = max_rounds
                    # self.background, background_stats = self.sample_background()
                    self.background, background_stats = sampled_backgrounds[instance_id]
                    if self.modality in TEXT_BASED:
                        if self.modality != 'semantic_text':
                            game_instance['empty_cells'] = background_stats['empty']
                            game_instance['total_cells'] = background_stats['total']
                        game_instance['empty_symbol'] = EMPTY_SYMBOL
                    game_instance['background'] = self.background
                    objects_1 = self.get_objects(object_count)
                    objects_2 = deepcopy(objects_1)
                    for key in template_instance:
                        game_instance[key] = deepcopy(template_instance[key])
                    game_instance['intermittent_prompts']['penalty_counter'] = Template(game_instance['intermittent_prompts']['penalty_counter']).safe_substitute(max_penalties=max_penalties)
                    game_instance['intermittent_prompts']['round_counter'] = Template(game_instance['intermittent_prompts']['round_counter']).safe_substitute(max_rounds=max_rounds)
                    game_instance['objects_1'] = place_objects(self.modality, objects_1, game_instance['background'])
                    game_instance['objects_2'] = place_objects(self.modality, objects_2, game_instance['background'])
                    
                    object_string = "'" + "', '".join([obj['id'] for obj in objects_1]) + "'"
                    grid_1 = None
                    grid_2 = None
                    if self.modality in TEXT_BASED:
                        cls = SemanticGridState if self.modality == 'semantic_text' else GridState
                        grid_1 = str(cls(background=self.background, objects=objects_1))
                        grid_2 = str(cls(background=self.background, objects=objects_2))
                        # gridstate_cls = SemanticGridState if self.modality == 'semantic_text' else GridState
                        # object_string = "'" + "', '".join([obj['id'] for obj in objects_1]) + "'"
                        # grid_1 = str(SemanticGridState(background=self.background, objects=objects_1))
                        # grid_2 = str(SemanticGridState(background=self.background, objects=objects_2))
                    
                    p1_initial_prompt = self.prepare_initial_prompt(grid=grid_1, max_penalties=max_penalties, max_rounds=max_rounds, object_string=object_string)
                    if modality in IMAGE_BASED:
                        p2_initial_prompt = p1_initial_prompt
                    else:
                        p2_initial_prompt = self.prepare_initial_prompt(grid=grid_2, max_penalties=max_penalties, max_rounds=max_rounds, object_string=object_string)
                    game_instance['p1_initial_prompt'] = p1_initial_prompt + self.start_messages['p1_start']
                    game_instance['p2_initial_prompt'] = p2_initial_prompt + self.start_messages['p2_start']

    def load_config_json(self, file_path: str) -> dict:
        """
        Load a JSON file from the game directory.
        If the JSON file contains entries for different languages, load the specified language.
        Then, for all keys that differ on modality, load the specified modality.
        Modalities can also be combined with a comma, like 'text,hybrid'
        """
        data = super().load_json(file_path)
        if self.language in data:
            data = data[self.language]
        for key, value in data.items():
            if self.modality in key.split(','):
                data = value
                break
        else:
            for key in data:
                # If there are different entries for different modalities, take the one for the current modality
                # Otherwise, keep the entry as it is
                if isinstance(data[key], dict):
                    for subkey, value in data[key].items():
                        if self.modality in subkey.split(','):
                            data[key] = value
                            break
        return data
    
    def sample_background(self):
        """
        Samples a background (text grid) from the provided dictionary.
        Might implement sampling background images in the future.
        """
        if modality in TEXT_BASED:
            if modality == 'semantic_text': 
                grid_data = self.load_config_json('resources/backgrounds/semantic_grids.json')
                background_stats = None
            else: 
                grid_data = self.load_config_json('resources/backgrounds/grids.json')
                info = grid_data['info'][self.current_experiment_conf['grid_config']]
                empty_cells = info['empty_cells']
                total_cells = info['total_cells']
                background_stats = {'empty': empty_cells, 'total': total_cells}

            backgrounds = grid_data[self.current_experiment_conf['grid_config']]
            background = random.choice(list(backgrounds.values()))
        else:
            background_stats = None
            background = self.game_path + '/resources/backgrounds/kitchen.png'
        return background, background_stats
    
    def get_objects(self, object_count):
        objects = []
        if self.modality in TEXT_BASED:
            for letter in self.objects[object_count]:
                object = GameObject(id=letter, coord=(None, None))
                objects.append(object)
        else:
            object_categories = int(self.current_experiment_conf['object_categories'])
            colored = self.current_experiment_conf['colored']
            objects_per_color = int(self.current_experiment_conf['objects_per_color'])

            metadata = self.load_json(ICON_METADATA_PATH)
            if colored:  # sampling colored icons
                category_sample_base = [key for key in metadata.keys() if set(metadata[key].keys()) != set(['black'])]
            else:  # sampling black icons
                category_sample_base = [key for key in metadata.keys() if 'black' in metadata[key]]

            sampled_categories = random.sample(category_sample_base, k=int(object_categories))
            for category in sampled_categories:
                if colored:
                    color = random.choice(list(set(metadata[category].keys()) - set(['black'])))
                else:
                    color = 'black'
                for icon in random.sample(metadata[category][color], k=objects_per_color):
                    objects.append(icon)
        return objects

    def prepare_initial_prompt(self, max_penalties, max_rounds, grid=None, object_string=None) -> str:
        initial_prompt = ""
        for _, value in self.initial_prompt.items():
            initial_prompt += value + "\n"
        initial_prompt = Template(initial_prompt).safe_substitute(
            grid=grid,
            objects=object_string,
            max_rounds=max_rounds,
            max_penalties=max_penalties,
            say=self.commands['say'],
            move=self.commands['move'],
            end_1=self.commands['end_1'],
            end_2=self.commands['end_2'],
            empty_symbol=EMPTY_SYMBOL
        )
        return initial_prompt

    
if __name__ == '__main__':
    import json

    instance_generator = CleanUpInstanceGenerator()
    experiments = instance_generator.load_json('resources/experiments.json')
    # experiments = json.load(open('resources/experiments.json', 'r', encoding='utf-8'))
    n_instances = experiments.get('n_instances', 2)
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    with open(os.path.join(ROOT, "SUPPORTED_LANGUAGES.json"), "r", encoding="utf-8") as f:
        lang_config = json.load(f)
    GAME_NAME = "clean_up"
    supported_languages = [lang for lang, data in lang_config["languages"].items() if GAME_NAME in data["games"]]

    if not supported_languages:
        print(f"No languages configured for game '{GAME_NAME}'")

    for language in supported_languages:
        for modalities, config in experiments['modalities'].items():
            for modality in modalities.split(','):
                print(f"Generating instances for language '{language}' and modality '{modality}'")
                file_name = config.get('instances', 'instances')
                file_name = f"{file_name}_{language}.json"
                # config = experiments['modalities'][modality]
                CleanUpInstanceGenerator().generate(filename=file_name, language=language, modality=modality, config=config, n_instances=n_instances, seed=SEED)
