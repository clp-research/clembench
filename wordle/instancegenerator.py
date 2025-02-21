"""Generate instances for the wordle game.
All three variants' instances will be created as different experiments, with both difficulty levels each.
Creates files in ./instances
"""
import logging
import os

from clemcore.clemgame import GameInstanceGenerator

from wordle.utils.instance_utils import InstanceUtils

LANGUAGE = "en"

logger = logging.getLogger(__name__)
GAME_NAME = "wordle"
# SEED = "17, 42"  # seed for old/v1.6 instances
SEED = "28"

class WordleGameInstanceGenerator(GameInstanceGenerator):
    """Generate instances for wordle."""
    def __init__(self, game_name):
        super().__init__(os.path.dirname(os.path.abspath(__file__)))
        self.game_name = game_name

    def load_instances(self):
        return self.load_json("in/instances")
    
    def _setresponseformatkeywords(self, language):
        """Set the response keyword language to be used.
        To support a new language, update the file resources/langconfig.json.
        Do not change the key names.
        Only update their values with the text in the new language.
        Args:
            language: The two-letter language code for the language to load. Must be matched by a language key in
                langconfig.json.
        """
        # The relative path is used to force looking for the file in the wordle directory
        lang_keywords = self.load_json("resources/langconfig")
        self.lang_keywords = lang_keywords[language]    

    def generate(self, filename: str ="instances", **kwargs):
        """Generate the game benchmark and store the instances JSON files.
        Modified from GameInstanceGenerator to create three instance JSON files for the different wordle variants:
        Instance files are stored by the on_generate method instead of this method.
        Args:
            filename: The base name of the instances JSON files to be stored in the 'in' subdirectory. This name will be
                suffixed with the game variants' names. Defaults to 'instances'.
            kwargs: Keyword arguments (or dict) to pass to the on_generate method.
        """
        # load the variants configuration file:
        self.experiment_config = self.load_json("resources/config.json")
        # load response keywords for the specified language:
        self._setresponseformatkeywords(LANGUAGE)

        for variant, variant_config in self.experiment_config.items():
            print(f"Creating instance JSON for wordle variant '{variant}' with config: {variant_config}")
            # generate variant instances and store them:
            self.on_generate(filename, variant, variant_config)
            # reset the instance attribute to assure separated variant instance files:
            self.instances = dict(experiments=list())

    def on_generate(self, filename: str, variant: str, variant_config: dict):
        """Generate instances for a wordle variant and store them in a separate JSON file.
        Args:
            filename: The base name of the instances JSON files to be stored in the 'in' subdirectory. This name will be
                suffixed with the game variants' names.
            variant: The variant name as string. Must start with 'wordle', followed by the variant suffix.
            variant_config: The variant configuration dict. Must contain 'use_clue' and 'use_critic' bool keys, and the
                experiment identifier 'name' string key.
        """
        self.instance_utils = InstanceUtils(
            os.path.dirname(os.path.abspath(__file__)),
            variant_config,
            variant,
            LANGUAGE)

        target_words_test_dict = self.instance_utils.select_target_words(SEED)

        word_difficulty = list(target_words_test_dict.keys())

        for difficulty in word_difficulty:
            experiment_name = f'{difficulty}_words_{variant_config["name"]}'
            experiment = self.add_experiment(experiment_name)
            self.instance_utils.update_experiment_dict(experiment, self.lang_keywords)

            for index, word in enumerate(target_words_test_dict[difficulty]):
                game_instance = self.add_game_instance(experiment, index + 1)
                self.instance_utils.update_game_instance_dict(
                    game_instance, word, difficulty
                )
        # get and add variant suffix to the file name:
        variant_suffix = variant.split("wordle")[1]
        # print("variant_suffix:", variant_suffix)
        variant_filename = f"{filename}{variant_suffix}.json"
        # store the variant instances file:
        self.store_file(self.instances, variant_filename, sub_dir="in")


if __name__ == "__main__":
    WordleGameInstanceGenerator(GAME_NAME).generate()                