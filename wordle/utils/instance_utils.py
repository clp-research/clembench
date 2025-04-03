import random
import requests
import os

import zipfile

from clemcore.clemgame import GameResourceLocator


class InstanceUtils(GameResourceLocator):
    def __init__(self, game_path, experiment_config, game_name, language):
        super().__init__(path=game_path)
        self.experiment_config = experiment_config
        self.game_name = game_name
        self.language = language
        self.common_config = self.load_json("resources/common_config")
        self.langconfig = self.load_json("resources/langconfig")[self.language]

    def read_inital_prompt(self, use_clue, use_critic):
        guesser_prompt = ""
        guesser_critic_prompt = ""

        # Hardcoded the game_name argument in the load_template function to wordle
        # During the instance creation of wordle_withclue, wordle_withcritic, game_name would be different and that leads to file not found error

        if use_critic:
            guesser_prompt = self.load_template(f"resources/initial_prompts/{self.language}/guesser_withcritic_prompt")
            guesser_critic_prompt = self.load_template(f"resources/initial_prompts/{self.language}/critic_prompt")
        else:
            if use_clue:
                guesser_prompt = self.load_template(
                    f"resources/initial_prompts/{self.language}/guesser_withclue_prompt")
            else:
                guesser_prompt = self.load_template(f"resources/initial_prompts/{self.language}/guesser_prompt")

        return guesser_prompt, guesser_critic_prompt

    def download_nytcrosswords(self):
        # Requires kaggle authentication for successfully downloading the file; see README.md
        kaggle_credentials = self.load_json("wordle_keys")['kaggle']
        os.environ['KAGGLE_USERNAME'] = kaggle_credentials['username']
        os.environ['KAGGLE_KEY'] = kaggle_credentials['key']

        if os.environ['KAGGLE_USERNAME'] == "<your-kaggle-user-name>" or os.environ['KAGGLE_KEY'] == "<kaggle-api-key>":
            print("Please provide your kaggle credentials in the instance_utils.py file\n")
            return

        print("Downloading nytcrosswords...")

        fp = f"resources/target_words/{self.language}/"

        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("darinhawley/new-york-times-crossword-clues-answers-19932021", path=fp)

        # Unzip the file
        with zipfile.ZipFile(fp + "/new-york-times-crossword-clues-answers-19932021.zip", "r") as zip_ref:
            zip_ref.extractall(fp)
        print("Stored the nytc crosswords clues file", fp)

    def download_allowed_words(self):
        print("Downloading wordle recognized words for EN Language...")
        url = self.langconfig["official_recognized_words_file_url"]
        r = requests.get(url, allow_redirects=True)
        fp = f"resources/target_words/{self.language}/"
        self.store_file(r.content.decode("utf-8"), "official_recognized_words.txt", fp)
        print("Stored the wordle recognized words file", fp)

    def read_file_contents(self, filename, file_ext="txt"):
        if file_ext == "csv":
            words_dict = {}
            try:
                words_list = self.load_csv(f"resources/{filename}")
            except FileNotFoundError:
                # File not available, downloading
                if filename.endswith("nytcrosswords.csv"):
                    self.download_nytcrosswords()
                    words_list = self.load_csv(f"resources/{filename}")
                else:
                    print(f"Word clues file {filename} not found, check and download the relevant files")
                    return []

            if "nytcrosswords.csv" in filename:
                for word in words_list:
                    words_dict[word[1].lower().strip()] = word[2].lower().strip()
                return words_dict
            else:
                return words_list

        elif file_ext == "txt":
            try:
                words = self.load_file(f"resources/{filename}")
            except FileNotFoundError:
                if filename.endswith("official_recognized_words.txt"):
                    self.download_allowed_words()
                    words = self.load_file(f"resources/{filename}")
                else:
                    print(f"File {filename} not found")
                    return []

            words = words.strip()
            if words:
                words_list = words.split("\n")
                words_list = [word.lower().strip() for word in words_list]
            else:
                words_list = []
            return words_list

    def categorize_target_words(self, unigram_freq_sorted_dict, clue_words_dict):
        easy_words = unigram_freq_sorted_dict[: int(len(unigram_freq_sorted_dict) / 3)]
        medium_words = unigram_freq_sorted_dict[
                       int(len(unigram_freq_sorted_dict) / 3): int(
                           2 * len(unigram_freq_sorted_dict) / 3
                       )
                       ]
        hard_words = unigram_freq_sorted_dict[
                     int(2 * len(unigram_freq_sorted_dict) / 3):
                     ]

        easy_words_list = [word[0] for word in easy_words]
        medium_words_list = [word[0] for word in medium_words]
        hard_words_list = [word[0] for word in hard_words]

        clue_words_keys = set(list(clue_words_dict.keys()))
        easy_words_list = set(easy_words_list).intersection(set(clue_words_keys))
        easy_words_list = list(easy_words_list)

        medium_words_list = set(medium_words_list).intersection(set(clue_words_keys))
        medium_words_list = list(medium_words_list)

        hard_words_list = set(hard_words_list).intersection(set(clue_words_keys))
        hard_words_list = list(hard_words_list)

        return easy_words_list, medium_words_list, hard_words_list

    def get_target_word_freq(self, word_list, freq_dict):
        data = {}
        for word in word_list:
            if word not in freq_dict:
                continue
            data[word] = freq_dict[word]
        return data

    def read_word_lists(self):
        official_words = []
        # officially recognized wordle words are downloaded from
        # https://github.com/3b1b/videos/blob/master/_2022/wordle/data/allowed_words.txt
        official_words = self.read_file_contents(f"target_words/{self.language}/official_recognized_words.txt")

        # wordle target words are downloaded from
        # https://github.com/3b1b/videos/blob/master/_2022/wordle/data/possible_words.txt
        # file_name = self.common_config["target_words_file_name"]
        # target_words = self.read_file_contents(file_name)

        # Uni-gram frequency data is downloaded from
        # https://www.kaggle.com/datasets/rtatman/english-word-frequency
        # file_name = self.common_config["unigram_freq_file_name"]
        # unigram_freq_dict = self.read_file_contents(file_name, file_ext="csv")

        # target_freq_dict = self.get_target_word_freq(target_words, unigram_freq_dict)
        # unigram_freq_sorted_dict = sorted(target_freq_dict.items(), key=lambda x: x[1])

        # Crosswords Clues are downloaded from
        # https://www.kaggle.com/datasets/darinhawley/new-york-times-crossword-clues-answers-19932021

        # Crosswords Clues are: List of lists and each sublist has: [date, word, clue]
        # We are only interested in the word and clue
        # Data cleanup happens inside the read_file_contents function
        word_clues_dict = {}
        word_clues_dict = self.read_file_contents(f"target_words/{self.language}/nytcrosswords.csv", file_ext="csv")

        # Currently the categorized words are read directly from the files
        #   without doing the categorization during instance generation
        # Check the file dump_categorized_words.py to see how the
        #   categorization is done
        # easy_words_list, medium_words_list, hard_words_list = self._categorize_target_words(unigram_freq_sorted_dict, word_clues_dict)
        easy_words_list = self.read_file_contents(f"target_words/{self.language}/easy_words.txt")
        medium_words_list = self.read_file_contents(f"target_words/{self.language}/medium_words.txt")
        hard_words_list = self.read_file_contents(f"target_words/{self.language}/hard_words.txt")

        if not official_words or not word_clues_dict:
            print("Error in reading the word lists, check and download the relevant files")
            return "DATA_NOT_AVAILABLE"

        self.official_words = official_words
        # self.target_words = target_words
        self.word_clues_dict = word_clues_dict
        self.easy_words_list = easy_words_list
        self.medium_words_list = medium_words_list
        self.hard_words_list = hard_words_list

    def select_target_words(self, use_seed):
        # use_seed = self.common_config["seed_to_select_target_word"]
        number_of_target_words = self.common_config["number_of_target_words"]

        target_words_test_dict = {}

        data_to_read = self.read_word_lists()
        if data_to_read == "DATA_NOT_AVAILABLE":
            return target_words_test_dict

        if "high_frequency" in self.common_config["supported_word_difficulty"]:
            if self.easy_words_list:
                random.seed(use_seed)
                target_words_test_dict["high_frequency"] = random.sample(
                    self.easy_words_list, k=number_of_target_words["high_frequency"]
                )

        if "medium_frequency" in self.common_config["supported_word_difficulty"]:
            if self.medium_words_list:
                random.seed(use_seed)
                target_words_test_dict["medium_frequency"] = random.sample(
                    self.medium_words_list, k=number_of_target_words["medium_frequency"]
                )

        if "low_frequency" in self.common_config["supported_word_difficulty"]:
            if self.hard_words_list:
                random.seed(use_seed)
                target_words_test_dict["low_frequency"] = random.sample(
                    self.hard_words_list, k=number_of_target_words["low_frequency"]
                )

        return target_words_test_dict

    def update_experiment_dict(self, experiment, lang_keywords):
        experiment["common_config"] = self.common_config
        experiment["common_config"]["max_word_length"] = lang_keywords["max_word_length"]
        experiment["use_clue"] = self.experiment_config["use_clue"]
        experiment["use_critic"] = self.experiment_config["use_critic"]

        (
            guesser_prompt,
            guesser_critic_prompt,
        ) = self.read_inital_prompt(experiment["use_clue"], experiment["use_critic"])

        experiment["guesser_prompt"] = guesser_prompt
        experiment["guesser_critic_prompt"] = guesser_critic_prompt
        experiment["lang_keywords"] = lang_keywords
        experiment["lang_keywords"]["official_words_list"] = self.official_words

    def update_game_instance_dict(self, game_instance, word, difficulty):
        game_instance["target_word"] = word
        game_instance["target_word_clue"] = self.word_clues_dict[word]
        game_instance["target_word_difficulty"] = difficulty
