"""Template script for instance generation.

usage:
python3 instancegenerator.py
Creates instance.json file in ./in

"""
import json
import os
import random
import logging
import numpy as np
#import requests
#import spacy
#import argparse

from clemcore.clemgame import GameInstanceGenerator
from utils.calculate_benchmarking import compute_instance_benchmarks 
from utils.pseudo_poly_knapsack import dp_pseudo_poly_knapsack

logger = logging.getLogger(__name__)


GAME_NAME = 'air_balloon_survival'
N_INSTANCES = 5
SEED = 1

ONLY_INDIVIDUAL = False # variable indicates if only individual player scores are needed. (Allows for games with more items)
LANGUAGE = 'en' # must be 'de' or 'en'
ITEMS_LIST = f'resources/{LANGUAGE}/items.json'  # Path to a text file containing items for the game
N_ITEMS = 15 # Number of items to select for each game instance
CATEGORY = "generated_words"  # must be 'generated_words' or 'common_nouns'
WEIGHT_RANGE = (1,1001) # range of possible weights underlying uniform distribution for weight sampling
LIMIT_ALPHA = 0.5 # constant which determines the maximum weight limit from real unit interval. high means many sets admissible, low means few

# next determines the type of correlation between weights and player 1 utility scale
PLAYER_1_PREFERENCES = 'uncorrelated' # must be 'uncorrelated', 'weakly-correlated', 'strongly-correlated', 'inversely-correlated' or 'subset-sum'
PLAYER_2_PREFERENCES  = None # must be None or one of those listed for PLAYER_1. If None defaults to the same as PLAYER_2_MODE

# next determines the correlation between two player's scales (note that despite independent setting, the same deterministic utility scale computations for both players will result in equal scales e.g. strong correlation
NEGOTIATION_MODE = 'independent' # possible values: 'independent', 'inverse', 'equal'

STRATEGIC_REASONING = True
REQUIRE_ARGUMENT = True # This can be set to False, but it results in limited negotiation which is undesirable as a whole
EMPTY_TURN_LIMIT = 15
PATIENCE = 2

# Seed for reproducibility
random.seed(SEED)

class AirBalloonInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed, **kwargs):
        language = kwargs["lang"]
        lang_resources = self.load_language_resources(language)

        regex_patterns = lang_resources["regex_patterns"]
        items = lang_resources["items"][:N_ITEMS]
        
        prompt_1 = lang_resources["prompt_player_1"]
        prompt_2 = lang_resources["prompt_player_2"]
        sr_tag = lang_resources['sr_tag']
        arg_tag = lang_resources['arg_tag']
        prop_tag = lang_resources['prop_tag']
        agree_tag = lang_resources['agree_tag']

        require_argument_prompt = lang_resources["require_argument"]

        #Name the experiments within the instance file
        experiment_name = f"{GAME_NAME}_{language}_{CATEGORY}_{NEGOTIATION_MODE}_{PLAYER_1_PREFERENCES}_{PLAYER_2_PREFERENCES}"
        experiment = self.add_experiment(experiment_name)

        for game_id in range(N_INSTANCES):
            instance = self.add_game_instance(experiment, game_id)

            item_weights = self.generate_weights(items, w_range=WEIGHT_RANGE) #update variable
            
            p1_prefs, p2_prefs = self.generate_preferences(item_weights,
                                                           player_1_mode=PLAYER_1_PREFERENCES,
                                                           player_2_mode=PLAYER_2_PREFERENCES,
                                                           setting=NEGOTIATION_MODE) #update variable

            instance['negotiation_mode'] = NEGOTIATION_MODE
            instance['item_weights'] = item_weights
            instance['player1_preferences'] = p1_prefs
            instance['player2_preferences'] = p2_prefs
            instance['max_weight'] = int(LIMIT_ALPHA * sum(item_weights.values()))

            if ONLY_INDIVIDUAL:
                instance['only_individual'] = True
                instance['max_u1'], best_deal_player_1 = dp_pseudo_poly_knapsack(item_weights, p1_prefs, instance['max_weight'])
                instance['max_u2'], best_deal_player_2 = dp_pseudo_poly_knapsack(item_weights, p2_prefs, instance['max_weight'])
                instance['max_total_utility'] = instance['max_u1'] + instance['max_u2']
                instance['max_harmonic'] = 1 
                instance['pareto_optima_count'] = 0
                instance['pareto_optima_sets'] = []
                instance['best_deal_player_1'] = list(best_deal_player_1)
                instance['best_deal_player_2'] = list(best_deal_player_2)
                # this is just used for the mock responses
                mock_set = list(best_deal_player_1)
            
            else:
                # calculate and log episode's benchmarking values
                benchmarks = compute_instance_benchmarks(items, item_weights,
                                                        p1_prefs, p2_prefs, instance['max_weight'])
                instance['only_individual'] = False
                instance['max_u1'] = benchmarks['max_u1']
                instance['max_u2'] = benchmarks['max_u2']
                instance['max_total_utility'] = benchmarks['max_total_utility']
                instance['max_harmonic'] = benchmarks['max_harmonic']
                instance['pareto_optima_count'] = benchmarks['pareto_optima_count']
                instance['pareto_optima_sets'] = benchmarks['pareto_optima_sets']
                # this is used for the mock instances
                mock_set = benchmarks['best_deal']
            


            # store empty turn limit and patience in the instance file
            instance["empty_turn_limit"] = EMPTY_TURN_LIMIT
            instance["patience"] = PATIENCE

            instance["strategic_reasoning"] = STRATEGIC_REASONING
            instance["require_argument"] = REQUIRE_ARGUMENT

            mock_response = ''
            if instance['strategic_reasoning']:
                mock_response += sr_tag
                mock_response += "{'blablabla'}\n"
            if instance['require_argument']:
                mock_response += arg_tag + "{'blablabla'}\n"
            instance['mock_response_p1'] = mock_response + prop_tag + str({str(item) for item in mock_set})
            instance['mock_response_p2'] = mock_response + agree_tag + str({str(item) for item in mock_set})


            for prompt in lang_resources["game_error_prompts"]:
                instance[prompt] = lang_resources["game_error_prompts"][prompt]
            for prompt in lang_resources["parse_error_prompts"]:
                instance[prompt] = lang_resources["parse_error_prompts"][prompt]

            strategic_reasoning_format = lang_resources["strategic_reasoning"]["format"]
            strategic_reasoning_rule = lang_resources["strategic_reasoning"]["rule"]

            instance['player1_initial_prompt'] = self.create_prompt(prompt_1,
                                                                    item_weights,
                                                                    p1_prefs,
                                                                    instance['max_weight'],
                                                                    strategic_reasoning_format,
                                                                    strategic_reasoning_rule,
                                                                    require_argument_prompt
                                                                    )
            instance['player2_initial_prompt'] = self.create_prompt(prompt_2,
                                                                    item_weights,
                                                                    p2_prefs,
                                                                    instance['max_weight'],
                                                                    strategic_reasoning_format,
                                                                    strategic_reasoning_rule,
                                                                    require_argument_prompt
                                                                    )
            
            instance.update(regex_patterns)

    def load_language_resources(self, language: str) -> dict:
        base_path = os.path.join(os.path.dirname(__file__), "resources", language)

        # Load regex patterns
        with open(os.path.join(base_path, "regex_patterns.json"), encoding="utf-8") as f:
            regex_patterns = json.load(f)

        # Load items dictionary and extract from given category
        with open(os.path.join(base_path, "items.json"), encoding="utf-8") as f:
            item_dict = json.load(f)
        items = item_dict.get(CATEGORY, [])
        random.shuffle(items) # Shuffle items to ensure randomness

        # Load initial prompts (as raw text or templates)
        with open(os.path.join(base_path, "prompt_player_1.template"), encoding="utf-8") as f:
            prompt_1 = f.read()
        with open(os.path.join(base_path, "prompt_player_2.template"), encoding="utf-8") as f:
            prompt_2 = f.read()

        # load error prompts
        with open(os.path.join(base_path, "game_error_prompts.json"), encoding="utf-8") as f:
            game_error_prompts = json.load(f)
        with open(os.path.join(base_path, "parse_error_prompts.json"), encoding="utf-8") as f:
            parse_error_prompts = json.load(f)

        # these are needed to create mock responses for a given language
        with open(os.path.join(base_path, "tags.json"), encoding="utf-8") as f:
            sr_arg_tag_dict = json.load(f)
            sr_tag = sr_arg_tag_dict['sr']
            arg_tag = sr_arg_tag_dict['arg']
            prop_tag = sr_arg_tag_dict['prop']
            agree_tag = sr_arg_tag_dict['agree']

        # load strategic reasoning format and rule
        if STRATEGIC_REASONING:
            with open(os.path.join(base_path, "strategic_reasoning.json"), encoding="utf-8") as f:
                strategic_reasoning_dict = json.load(f)
        else:
            strategic_reasoning_dict = {"rule": "", "format": ""}

        if REQUIRE_ARGUMENT:
            with open(os.path.join(base_path, "require_argument.txt"), encoding="utf-8") as f:
                require_argument = f.read()
        else:
            require_argument = ""


        return {
            "regex_patterns": regex_patterns,
            "items": items,
            "prompt_player_1": prompt_1,
            "prompt_player_2": prompt_2,
            "parse_error_prompts": parse_error_prompts,
            "game_error_prompts": game_error_prompts,
            "strategic_reasoning": strategic_reasoning_dict,
            "require_argument": require_argument,
            "sr_tag": sr_tag,
            "arg_tag": arg_tag,
            "prop_tag": prop_tag,
            "agree_tag": agree_tag
            }
    
    def create_prompt(self, prompt: str, weights: dict[str, int],
                      prefs: dict[str, int], limit: int,
                      strategic_reasoning_format: str, strategic_reasoning_rule: str,
                      require_argument_str: str) -> str:
        
        text = prompt.replace("$LIMIT$", str(limit))
        text = text.replace("$ITEM_WEIGHTS$", json.dumps(weights))
        text = text.replace("$UTILITY_SCALE_PLAYER$", json.dumps(prefs))
        text = text.replace("$STRATEGIC_REASONING_FORMAT$", strategic_reasoning_format if STRATEGIC_REASONING else '')
        text = text.replace("$STRATEGIC_REASONING_RULE$", strategic_reasoning_rule if STRATEGIC_REASONING else '')
        text = text.replace("$REQUIRE_ARGUMENT$", require_argument_str if REQUIRE_ARGUMENT else '')

        return text
    

    def generate_weights(self, items: list[str], w_range: tuple = WEIGHT_RANGE) -> dict[str, int]:

        return {item: random.randint(w_range[0], w_range[1]) for item in items}


    def generate_preferences(self,
                             weights: dict,
                             player_1_mode: str = 'uncorrelated',
                             player_2_mode: str | None = None,
                             setting: str = 'independent') -> tuple[dict[str, int], dict[str, int]]:
        """
        Generate player-specific utility scales for items.
        
        Args:
            item_weights: dict mapping item names to weights
            player_1_mode: type of correlation for player 1 ('uncorrelated', 'weakly', 'strongly', 'subset', 'inverse')
            player_2_mode: same for player 2. Only useful when used with setting==independent (defaults to same mode as player 1)
            setting: 'equal' -> same scales, 'inverse' -> player 2 has inverse utilities, 'random' -> independent utility scales

        Returns:
            (player_1_utilities, player_2_utilities)
        """

        items = list(weights.keys())
        if player_2_mode is None:
            player_2_mode = player_1_mode

        R = max(weights.values())
        r = R // 10
        q = R // 10
        K = max(weights.values()) + 1
        
        rng1 = random.Random(SEED)
        rng2 = random.Random(SEED + 1)

        def gen_player_scale(mode: str, rng):
            scale = {}
            for item in items:
                w = weights[item]
                if mode == 'uncorrelated':
                    u = rng.randint(1, R)
                elif mode == 'weakly-correlated':
                    u = w + rng.randint(-r, r)
                    u = max(1, u)
                elif mode == 'strongly-correlated':
                    u = w + q
                elif mode == 'subset-sum':
                    u = w
                elif mode == 'inversely-correlated':
                    u = K - w
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                scale[item] = u
            return scale

        # same utility scale
        if setting == 'equal':
            player_1_scale = gen_player_scale(player_1_mode, rng1)
            player_2_scale = player_1_scale.copy()
        # inverse ranking
        elif setting == 'inverse':
            player_1_scale = gen_player_scale(player_1_mode, rng1)
            # Sort player 1 items by utility (ascending)
            sorted_items_by_p1 = sorted(player_1_scale.items(), key=lambda x: x[1])
            # Get the actual utility values sorted ascending
            values_sorted_by_p1 = [v for _, v in sorted_items_by_p1]
            # Get the items sorted descending (highest P1 first)
            items_in_inverse_order = [item for item, _ in sorted_items_by_p1[::-1]]
            # Assign: highest P1 gets lowest utility from values
            player_2_scale = {item: values_sorted_by_p1[i] for i, item in enumerate(items_in_inverse_order)}

        # player 1 and player two get independently created utility scales
        elif setting == 'independent':
            player_1_scale = gen_player_scale(player_1_mode, rng1)
            player_2_scale = gen_player_scale(player_2_mode, rng2)
        else:
            raise ValueError(f"Unknown setting: {setting}")
        return player_1_scale, player_2_scale

if __name__ == '__main__':
    import json

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    with open(os.path.join(ROOT, "SUPPORTED_LANGUAGES.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    GAME_NAME = "air_balloon_survival"
    LANGUAGES = [lang for lang, data in config["languages"].items() if GAME_NAME in data["games"]]
    if not LANGUAGES:
        print(f"No languages configured for game '{GAME_NAME}'")
    for lang in LANGUAGES:
        print(f"Generating instances for language: {lang}")
        AirBalloonInstanceGenerator().generate(filename=f"instances_{lang}.json", seed=SEED, lang=lang)
