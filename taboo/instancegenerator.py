"""The script generates game instances for the Taboo game. It selects target words and generates a list of related words.
The script uses either ConceptNet or the OpenAI API to retrieve or generate these related words.

usage:
python3 instancegenerator.py
Creates instance.json file in ./in

"""
import os
import random
import logging
import argparse

from clemcore.clemgame import GameInstanceGenerator

N_INSTANCES = 20  # how many different target words
N_GUESSES = 3  # how many tries the guesser will have
N_RELATED_WORDS = 3
VERSION = "v2.0"

HIGH_LIST = "resources/{}/high_frequency_taboo_words.json"
LOW_LIST  = "resources/{}/low_frequency_taboo_words.json"
PROMPT_PATH = "resources/initial_prompts/{}/{}"

logger = logging.getLogger(__name__)


# Seed for reproducibility
# random.seed(87326423)  # v1 seed
# random.seed(73128361)  # v2.0 seed

class TabooGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed: int, **kwargs):
        random.seed(seed)
        # prepare related word generation
        lang = kwargs["lang"]
        #mode = kwargs["mode"]
        #assert mode == "manual", "Only support manual related word selection for now"

        high_words = self.load_json(HIGH_LIST.format(lang))
        low_words = self.load_json(LOW_LIST.format(lang))

        for frequency, entries in [("high", high_words), ("low", low_words)]:
            print("\nSampling from freq:", frequency)

            entries = list(entries)
            random.shuffle(entries)

            experiment = self.add_experiment(f"{frequency}_{lang}")
            experiment["max_turns"] = N_GUESSES
            experiment["describer_initial_prompt"] = self.load_template(PROMPT_PATH.format(lang, "initial_describer.template"))
            experiment["guesser_initial_prompt"] = self.load_template(PROMPT_PATH.format(lang, "initial_guesser.template"))

            target_id = 0
            for entry in entries:
                if target_id >= N_INSTANCES:
                    break

                target = entry.get("target_word")
                related_words = entry.get("related_word", [])
                target_stem = entry.get("target_word_stem")
                related_stems = entry.get("related_word_stem", [])

                if not target or len(target) < 3:
                    continue

                if not isinstance(related_words, list) or len(related_words) < 1:
                    print(f"Skipping '{target}' (no related words)")
                    continue

                game_instance = self.add_game_instance(experiment, target_id)
                game_instance["target_word"] = target
                game_instance["related_word"] = related_words
                if target_stem:
                    game_instance["target_word_stem"] = target_stem

                if related_stems:
                    game_instance["related_word_stem"] = related_stems

                target_id += 1

            if target_id < N_INSTANCES:
                print(
                    f"Warning: only generated {target_id}/{N_INSTANCES} instances for {frequency}"
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Taboo game instances.")
    parser.add_argument("-l", "--lang", default="hu", help="Language code (e.g. en, de, hu)")

    args = parser.parse_args()
    TabooGameInstanceGenerator().generate(seed=4723848, lang=args.lang, filename=f"instances_{args.lang}.json")
