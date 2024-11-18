"""
Generate instances for the game.

Creates files in ./in
"""
from tqdm import tqdm
import os
import logging
from clemcore.clemgame import GameInstanceGenerator

logger = logging.getLogger(__name__)

LANGUAGE = "en"


class HelloGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self):
        # Create an experiment (here for english greetings)
        experiment = self.add_experiment(f"greet_{LANGUAGE}")
        experiment["language"] = LANGUAGE  # experiment parameters

        # Load a list of prepared names to choose from
        names = self.load_file("resources/names", file_ending=".txt").split("\n")

        # Load the prepared initial prompt
        prompt = self.load_template("resources/initial_prompts/prompt")

        # We create one game for each name
        for game_id in tqdm(range(len(names))):
            target_name = names[game_id]

            # Replace the name in the templated initial prompt
            instance_prompt = prompt.replace("$NAME$", target_name)

            # Create a game instance
            game_instance = self.add_game_instance(experiment, game_id)
            game_instance["prompt"] = instance_prompt  # game parameters
            game_instance["target_name"] = target_name  # game parameters


if __name__ == '__main__':
    # The resulting instances.json is automatically saved to the "in" directory of the game folder
    HelloGameInstanceGenerator().generate()
