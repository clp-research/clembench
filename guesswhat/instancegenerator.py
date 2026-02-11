"""
Generate instances for the Guess What game.

"""

import os
import sys

print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

import random
import json
from tqdm import tqdm
from clemcore.clemgame import GameInstanceGenerator
from resources.lang_config import LANG_CONFIG

# number of words per episode
num_words = 8

# number of instances per experiment
N_INSTANCES = 10


class GuessWhatGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self, lang="en"):
        super().__init__(os.path.dirname(__file__))
        self.lang = lang
        self.lang_cfg = LANG_CONFIG[lang]["guesswhat"]
        base_path = os.path.join(os.path.dirname(__file__), "resources", lang)

        category_file_path = os.path.join(base_path, "categories.json")
        if not os.path.exists(category_file_path):
            raise FileNotFoundError(f"Missing categories.json for language '{lang}' at {category_file_path}")

        with open(category_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "Categories" in data:
            self.categories = self.categories = {c["Category"]: sum((sub["Members"] for sub in c["Subcategories"]), []) for c in data["Categories"]}

        elif "words" in data:
            self.categories = data["words"]

        else:
            raise ValueError(f"Unknown categories.json format in {category_file_path}. Top-level keys: {list(data.keys())}")

    def on_generate(self, seed: int, **kwargs):
        random.seed(seed)
        output_instances = {
            "experiments": []
        }
        self.generate_mix_set(output_instances)

    def generate_mix_set(self, output_instances):
        output_instance_details = {"Level_1": [], "Level_2": [], "Level_3": []}


        for level in [1, 2, 3]:
            experiment_name = f"Level_{level}"
            experiment = self.add_experiment(experiment_name)
            cfg = self.lang_cfg

            max_turns = num_words
            experiment["max_turns"] = max_turns
            experiment["question_tag"] = cfg["QUESTION"]
            experiment["answer_tag"] = cfg["ANSWER"]
            experiment["guess_tag"] = cfg["GUESS"]
            experiment["answer_variations"] = cfg["ANSWER_VARIATIONS"]
            patterns = cfg["PATTERNS"]
            experiment["letter_based_pattern"] = patterns["LETTER"]
            experiment["direct_guess_pattern"] = patterns["DIRECT"]
            experiment["length_question_pattern"] = patterns["LENGTH"]
            experiment["syllable_question_pattern"] = patterns["SYLLABLE"]
            experiment["pos_question_pattern"] = patterns["POS"]
            answerer_prompt = self.load_template(f"resources/{self.lang}/initial_prompts/answerer_prompt")
            guesser_prompt = self.load_template(f"resources/{self.lang}/initial_prompts/guesser_prompt")

            experiment["answerer_initial_prompt"] = answerer_prompt
            experiment["guesser_initial_prompt"] = guesser_prompt

            #used_words = set()
            game_instances = []
            for game_id in tqdm(range(N_INSTANCES)):
                used_words = set()
                instance, instance_details = self.generate_instance(level, used_words)
                if instance:
                    game_instance = self.add_game_instance(experiment, game_id)
                    game_instance["target_word"] = instance["target"]
                    game_instance["candidate_list"] = instance["items"]
                    game_instances.append(game_instance)

                    output_instance_details[experiment_name].append(instance_details)

            experiment["game_instances"] = game_instances
            output_instances["experiments"].append(experiment)

    def generate_instance(self, level, used_words):
        instance = {"items": [], "target": ""}
        instance_details = {"items": [], "target": ""}
        used_categories = set()
        categories = list(self.categories.keys())
        instance = {"items": [], "target": ""}

        required_words = num_words

        if level == 1:
            # 8 categories x 1 word
            chosen_categories = random.sample(categories, num_words)
            for cat in chosen_categories:
                word = random.choice([w for w in self.categories[cat] if w not in used_words] or self.categories[cat])
                used_words.add(word)
                instance["items"].append(word)

        elif level == 2:
            # 4 categories x 2 words
            chosen_categories = random.sample(categories, 4)
            for cat in chosen_categories:
                words = random.sample([w for w in self.categories[cat] if w not in used_words] or self.categories[cat],2)
                for word in words:
                    used_words.add(word)
                    instance["items"].append(word)

        elif level == 3:
            # 1 category x 8 words
            cat = random.choice(categories)
            words = random.sample([w for w in self.categories[cat] if w not in used_words] or self.categories[cat], num_words)
            for word in words:
                used_words.add(word)
                instance["items"].append(word)

        else:
            return None, None

        if len(instance["items"]) == num_words:
            instance["target"] = random.choice(instance["items"])
            instance_details["items"] = instance["items"]
            instance_details["target"] = instance["target"]
            return instance, instance_details

        return None, None
    """
        def find_valid_categories(level):
            if level == 1:
                return [
                    c for c in self.categories if c["Category"] not in used_categories
                    and len([sub for sub in c["Subcategories"] if len(sub["Members"]) >= 1]) >= 2
                ]
            elif level == 2:
                return [
                    c for c in self.categories if c["Category"] not in used_categories
                    and len([sub for sub in c["Subcategories"] if len(sub["Members"]) >= 2]) >= 2
                ]
            elif level == 3:
                return [
                    c for c in self.categories if c["Category"] not in used_categories
                    and len([sub for sub in c["Subcategories"] if len(sub["Members"]) >= 2]) >= 4
                ]
            return []

        def find_valid_subcategories(category, level):
            if level == 1:
                return [sub for sub in category["Subcategories"] if len(sub["Members"]) >= 1]
            elif level == 2:
                return [sub for sub in category["Subcategories"] if len(sub["Members"]) >= 2]
            elif level == 3:
                return [sub for sub in category["Subcategories"] if len(sub["Members"]) >= 2]
            return []

        # Retry limit to prevent infinite loops
        for _ in range(100):
            instance["items"].clear()
            instance_details["items"].clear()

            if level == 1:
                valid_categories = find_valid_categories(level)
                if len(valid_categories) < 4:
                    print("Warning: Not enough valid categories for Level 1. Using available categories.")
                    valid_categories = [c for c in self.categories if c["Category"] not in used_categories]

                selected_categories = random.sample(valid_categories, min(4, len(valid_categories)))
                used_categories.update(cat["Category"] for cat in selected_categories)

                for category in selected_categories:
                    subcategories = find_valid_subcategories(category, level)
                    if len(subcategories) < 2:
                        subcategories = [sub for sub in category["Subcategories"]]
                    selected_subcategories = random.sample(subcategories, min(2, len(subcategories)))
                    for sub in selected_subcategories:
                        available_words = [w for w in sub["Members"] if w not in used_words]
                        if len(available_words) < 1:
                            continue
                        word = random.choice(available_words)
                        used_words.add(word)
                        instance["items"].append(word)
                        instance_details["items"].append({
                            "word": word,
                            "category": category["Category"],
                            "feature": sub["Subcategory"]
                        })

            elif level == 2:
                valid_categories = find_valid_categories(level)
                if len(valid_categories) < 2:
                    valid_categories = [c for c in self.categories if c["Category"] not in used_categories]

                selected_categories = random.sample(valid_categories, min(2, len(valid_categories)))
                used_categories.update(cat["Category"] for cat in selected_categories)

                for category in selected_categories:
                    subcategories = find_valid_subcategories(category, level)
                    if len(subcategories) < 2:
                        subcategories = [sub for sub in category["Subcategories"]]
                    selected_subcategories = random.sample(subcategories, min(2, len(subcategories)))
                    for sub in selected_subcategories:
                        available_words = [w for w in sub["Members"] if w not in used_words]
                        if len(available_words) < 2:
                            continue
                        words = random.sample(available_words, 2)
                        for word in words:
                            used_words.add(word)
                            instance["items"].append(word)
                            instance_details["items"].append({
                                "word": word,
                                "category": category["Category"],
                                "feature": sub["Subcategory"]
                            })

            elif level == 3:
                valid_categories = find_valid_categories(level)
                if len(valid_categories) < 1:
                    valid_categories = [c for c in self.categories if c["Category"] not in used_categories]

                selected_category = random.choice(valid_categories)
                used_categories.add(selected_category["Category"])

                subcategories = find_valid_subcategories(selected_category, level)
                if len(subcategories) < 4:
                    subcategories = [sub for sub in selected_category["Subcategories"]]
                selected_subcategories = random.sample(subcategories, min(4, len(subcategories)))

                for sub in selected_subcategories:
                    available_words = [w for w in sub["Members"] if w not in used_words]
                    if len(available_words) < 2:
                        continue
                    words = random.sample(available_words, 2)
                    for word in words:
                        used_words.add(word)
                        instance["items"].append(word)
                        instance_details["items"].append({
                            "word": word,
                            "category": selected_category["Category"],
                            "feature": sub["Subcategory"]
                        })

            if len(instance["items"]) >= required_words:
                instance["items"] = instance["items"][:required_words]
                instance_details["items"] = instance_details["items"][:required_words]
                instance["target"] = random.choice(instance["items"])
                instance_details["target"] = instance["target"]
                return instance, instance_details

        print("Error: Could not generate a valid instance after several attempts.")
        return None, None
    """


if __name__ == '__main__':
    for lang in LANG_CONFIG.keys():
        GuessWhatGameInstanceGenerator(lang=lang).generate(seed=42, filename=f"instances_{lang}.json")
