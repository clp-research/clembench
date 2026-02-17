'''
The script generates game instances for the Deal or No Deal (DoND) game. It selects
a set of item types and generates a the item counts and value function for each player.
It will generate instances for all three game modes, and for all three supported
languages. Creates `instance_${mode}_${lang}.json` files in `./in`.
'''

import os
import random
import logging
import argparse

from clemcore.clemgame import GameInstanceGenerator

# Different experiment configurations.
modes = ['coop', 'semi', 'comp']
languages = ['en', 'de', 'it', 'hu']

# Number of instances per mode and language combination.
n_instances = 50
n_messages = 5          # Number of messages are allowed per player.
n_item_types = (3, 5)   # The number of different item types.
n_items = (5, 8)        # The number of different items.
n_points = 10           # Number of value points per player.

logger = logging.getLogger(__name__)


class DealOrNoDealGameInstanceGenerator(GameInstanceGenerator):
    '''
    Instance generator for the Deal or No Deal game. One instance of the generator
    creates instances for one mode and language combination.
    '''

    def __init__(self, mode: str, language: str):
        super().__init__(os.path.dirname(__file__))
        self.mode = mode
        self.language = language

    def on_generate(self, seed=None, **kwargs):
        # Load the list of possible item words.
        item_words = self.load_json(
            f'resources/{self.language}/possible_items.json'
        )
        item_words_plural = self.load_json(
            f'resources/{self.language}/possible_items_plural.json'
        )
        assert isinstance(item_words, list)
        assert isinstance(item_words_plural, list)
        # Setup experiment level information.
        experiment = self.add_experiment(f'{self.mode}_{self.language}')
        experiment['max_turns'] = n_messages
        experiment['mode'] = self.mode
        experiment['language'] = self.language
        # Just load the template. Will be filled in by the game master.
        experiment['initial_prompt'] = self.load_template(
            f'resources/{self.language}/initial_{self.mode}'
        )
        experiment['proposal_early'] = self.load_template(
            f'resources/{self.language}/proposal_early'
        )
        experiment['proposal_timeout'] = self.load_template(
            f'resources/{self.language}/proposal_timeout'
        )
        # Create the instances.
        target_id = 0
        while target_id < n_instances:
            while True:
                # Sample a new set of item types.
                num_types = random.randint(n_item_types[0], n_item_types[1])
                item_types_idx = random.sample(
                    range(len(item_words)), k=num_types
                )
                item_types = [item_words[i] for i in item_types_idx]
                item_types_plural = [
                    item_words_plural[i] for i in item_types_idx
                ]
                # Distribute item counts.
                num_items = random.randint(n_items[0], n_items[1])
                item_counts = [1] * num_types
                while sum(item_counts) < num_items:
                    item_counts[random.randrange(len(item_counts))] += 1
                # Create value functions.
                values_a = [
                    random.randint(0, n_points) for _ in range(num_types)
                ]
                values_b = [
                    random.randint(0, n_points) for _ in range(num_types)
                ]
                if self.is_valid_instance(item_counts, values_a, values_b):
                    break
            # Add the new game instance.
            game_instance = self.add_game_instance(experiment, target_id)
            game_instance['item_types'] = item_types
            game_instance['item_types_plural'] = item_types_plural
            game_instance['item_counts'] = item_counts
            game_instance['player_a_values'] = values_a
            game_instance['player_b_values'] = values_b
            target_id += 1

    def is_valid_instance(self, item_counts: list[int], values_a: list[int], values_b: list[int]) -> bool:
        '''
        Check whether the two value functions satisfy the following conditions:
        * Each player can get a maximum of `n_points` points.
        * Each item is valued by at least one of the players.
        * At least one item is values by both players.
        '''
        if sum(count * value for count, value in zip(item_counts, values_a)) != n_points:
            return False  # Values for player A don't add up to `n_points`.
        if sum(count * value for count, value in zip(item_counts, values_b)) != n_points:
            return False  # Values for player B don't add up to `n_points`.
        if any(val_a == 0 and val_b == 0 for val_a, val_b in zip(values_a, values_b)):
            return False  # There is some items not valued by any player.
        if all(val_a == 0 or val_b == 0 for val_a, val_b in zip(values_a, values_b)):
            return False  # There is no item valued by both players.
        return True


if __name__ == '__main__':
    import json

    parser = argparse.ArgumentParser(
        description='Generate Deal or No Deal game instances.')
    _ = parser.parse_args()

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    with open(os.path.join(ROOT, "SUPPORTED_LANGUAGES.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    GAME_NAME = "dond"
    supported_languages = [lang for lang, data in config["languages"].items() if GAME_NAME in data["games"]]

    if not supported_languages:
        print(f"No languages configured for game '{GAME_NAME}'")
    for lang in supported_languages:
        random.seed(3141592)
        for mode in modes:
            print(f"Generating {mode} instances for language '{lang}'")
            DealOrNoDealGameInstanceGenerator(mode=mode, language=lang).generate(filename=f'instances_{mode}_{lang}.json')
