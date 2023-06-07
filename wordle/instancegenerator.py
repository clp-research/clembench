"""
Randomly generate templates for the private/shared game.

Creates files in ./instances and ./requests
"""
from tqdm import tqdm

import re
import random

from clemgame.clemgame import GameInstanceGenerator
from games.wordle.utils.instance_utils import InstanceUtils

GAME_NAME = "wordle"


class WordleGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self, game_name):
        super().__init__(game_name)
        self.game_name = game_name

    def on_generate(self):
        self.experiment_config = self.load_json("resources/config.json")
        self.instance_utils = InstanceUtils(self.experiment_config, self.game_name)

        target_words_test_dict = self.instance_utils.select_target_words()

        word_difficulty = list(target_words_test_dict.keys())

        for difficulty in word_difficulty:
            experiment_name = f'{difficulty}_words_{self.experiment_config["name"]}'
            experiment = self.add_experiment(experiment_name)
            self.instance_utils.update_experiment_dict(experiment)

            for index, word in enumerate(target_words_test_dict[difficulty]):
                game_instance = self.add_game_instance(experiment, index + 1)
                self.instance_utils.update_game_instance_dict(
                    game_instance, word, difficulty
                )


if __name__ == "__main__":
    WordleGameInstanceGenerator(GAME_NAME).generate()
