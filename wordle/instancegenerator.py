import os
from clemcore.clemgame import GameInstanceGenerator
from wordle.utils.instance_utils import InstanceUtils


# SEED = "17, 42"  # seed for old/v1.6 instances
# SEED = "28"

class WordleGameInstanceGenerator(GameInstanceGenerator):
    """Generate instances for wordle."""

    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed: int, **kwargs):
        # The variant name as string. Must start with 'wordle', followed by the variant suffix.
        variant = kwargs["variant"]
        variant_config = self.load_json("resources/config.json")[variant]
        # The two-letter language code for the language to load. Must be matched by a language key in langconfig.json.
        lang = kwargs["lang"]
        lang_keywords = self.load_json("resources/langconfig")[lang]
        # Loaded response keywords for the specified language.
        # To support a new language, update the file resources/langconfig.json.
        # Do not change the key names. Only update their values with the text in the new language.

        print(f"Creating instance JSON for wordle variant '{variant}' with config: {variant_config}")
        instance_utils = InstanceUtils(self.game_path, variant_config, variant, lang)
        target_words_test_dict = instance_utils.select_target_words(seed)
        word_difficulty = list(target_words_test_dict.keys())

        for difficulty in word_difficulty:
            experiment_name = f'{difficulty}_words_{variant_config["name"]}'
            experiment = self.add_experiment(experiment_name)
            instance_utils.update_experiment_dict(experiment, lang_keywords)

            for index, word in enumerate(target_words_test_dict[difficulty]):
                game_instance = self.add_game_instance(experiment, index + 1)
                instance_utils.update_game_instance_dict(game_instance, word, difficulty)


if __name__ == "__main__":
    for variant in ["wordle", "wordle_withclue", "wordle_withcritic"]:
        file_name = "instances.json"
        variant_suffix = variant.split("_")
        if len(variant_suffix) > 1:
            file_name = f"instances_{variant_suffix[-1]}.json"
        print(f"Generate {file_name} for {variant}")
        WordleGameInstanceGenerator().generate(filename=file_name, seed=28, variant=variant, lang="en")
