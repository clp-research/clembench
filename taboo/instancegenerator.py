"""The script generates game instances for the Taboo game. It selects target words and generates a list of related words.
The script uses either ConceptNet or the OpenAI API to retrieve or generate these related words.

usage:
python3 instancegenerator.py
Creates instance.json file in ./in

"""
import json
import os
import random
import logging
import openai
import requests
import spacy
import argparse

import nltk

from clemcore.clemgame import GameInstanceGenerator

from utils.select_taboo_words import is_function_word

N_INSTANCES = 20  # how many different target words
N_GUESSES = 3  # how many tries the guesser will have
N_RELATED_WORDS = 3
LANGUAGE = "en"
VERSION = "v2.0"

WORD_LISTS = f"resources/target_words/{LANGUAGE}/taboo_word_lists.json"

logger = logging.getLogger(__name__)

# Seed for reproducibility
# random.seed(87326423)  # v1 seed
random.seed(73128361)  # v2.0 seed

# Set up OpenAI API key if using openai
OPENAI_API_KEY = ""  # Insert your OpenAI API key


class TabooGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(os.path.dirname(__file__))
        self.n = N_RELATED_WORDS
        self.language = LANGUAGE
        # Variable for storing spaCy model (only loaded if conceptnet is used)
        self.tagger = None
        # Using nltk Snowball stemmer:
        self.stemmer = nltk.stem.SnowballStemmer('english')

    def on_generate(self, mode):
        # prepare related word generation
        if mode == "conceptnet":
            # this is currently the default
            self.filename = f"instances_{VERSION}_{LANGUAGE}.json"
            # Load spaCy model for filtering nouns
            self.tagger = spacy.load("en_core_web_sm")
            generator_function = self.get_related_words_from_conceptnet
        else:
            self.filename = f"instances_{VERSION}_{LANGUAGE}_{mode}.json"
            if mode == "openai":
                openai.api_key = OPENAI_API_KEY
                generator_function = self.generate_related_words_from_openai

        taboo_words = self.load_json(file_name=WORD_LISTS)

        for frequency in ["high", "medium", "low"]:
            print("\nSampling from freq:", frequency)

            experiment = self.add_experiment(f"{frequency}_{LANGUAGE}")
            experiment["max_turns"] = N_GUESSES
            experiment["describer_initial_prompt"] = self.load_template("resources/initial_prompts/initial_describer")
            experiment["guesser_initial_prompt"] = self.load_template("resources/initial_prompts/initial_guesser")

            target_id = 0
            while target_id < N_INSTANCES:
                if not taboo_words[frequency]:
                    print(
                        f"No more words available to sample. Only found the requested {N_RELATED_WORDS} related words for {target_id} target words.")
                    break
                # Sample a new target word
                target = random.choice(taboo_words[frequency])
                taboo_words[frequency].remove(target)
                # only use words of length 3 or greater:
                if len(target) < 3:
                    continue

                # Generate related words
                print(f"Retrieving related words for '{target}'")
                related_words = []
                if mode == "conceptnet":
                    related_words = self.get_related_words_from_conceptnet(target)
                elif mode == "openai":
                    related_words = self.generate_related_words_from_openai(target)

                if len(related_words) < N_RELATED_WORDS and not mode == "manual":
                    print(f"Skipping '{target}' due to lack of related words.")
                    continue  # Skip this word and try another

                else:
                    # stem words:
                    target_word_stem = self.stemmer.stem(target)
                    related_word_stem = [self.stemmer.stem(related_word) for related_word in related_words]
                    # The nltk SnowballStemmer is not reliable - manual inspection and correction still needed!
                    # Add a valid game instance
                    game_instance = self.add_game_instance(experiment, target_id)
                    game_instance["target_word"] = target
                    game_instance["related_word"] = related_words
                    game_instance["target_word_stem"] = target_word_stem
                    game_instance["related_word_stem"] = related_word_stem
                    target_id += 1

    def get_related_words_from_conceptnet(self, word, filter_nouns=False):
        """
        Fetch related words from ConceptNet and filter for nouns.
        """
        try:
            url = f"http://api.conceptnet.io/c/{self.language}/{word}/"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            # this could have safety checks, like checking for the word being slang

            related_words = set()
            edges = data.get("edges", [])
            for edge in edges:
                if edge.get("end", {}).get("language", "") == LANGUAGE:  # only use same language
                    related_term = edge.get("end", {}).get("label", "")
                    # make sure word is different from target, is only one word, and not a function word
                    if related_term.lower() != word.lower() \
                            and " " not in related_term \
                            and not is_function_word(related_term):
                        if filter_nouns:
                            if self.is_noun(related_term):
                                related_words.add(related_term)
                        else:
                            related_words.add(related_term)
                    if len(related_words) >= self.n:
                        break
            return list(related_words)

        except Exception as e:
            print(f"Error fetching related words for '{word}': {e}")
            return []

    def is_noun(self, word):
        """
        Checks if a word is a noun
        :param word: the word to be checked
        :return: True if word is a noun according to spacy, False otherwise
        """
        doc = self.tagger(word)  # Analyse word with spaCy
        for token in doc:
            if token.pos_ == "NOUN":
                return True
        return False

    def generate_related_words_from_openai(self, target_word):
        """
        Generates related words for a target word using the OpenAI API (new interface).
        :param target_word: The target word.
        :return: List of related words.
        """
        try:
            # Prompt for Chat model
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Give me {N_RELATED_WORDS} words that are related to '{target_word}'."}
            ]

            # Request to the ChatCompletion-API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # or "gpt-4.0-turbo" if available
                messages=messages,
                max_tokens=50,
                temperature=0.7
            )

            raw_response = response['choices'][0]['message']['content'].strip()

            # Standardize the response
            if "\n" in raw_response:  # Check for newline-separated list
                related_words = [line.split(".")[-1].strip() for line in raw_response.split("\n") if line.strip()]
            else:  # Assume comma-separated list
                related_words = raw_response.split(", ")

            return related_words[:self.n]  # limit the number of related words
        except Exception as e:
            logger.error(f"Error generating related words for '{target_word}': {e}")
            return []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Taboo game instances.")
    parser.add_argument("-m", "--mode", choices=["manual", "conceptnet", "openai"], default="conceptnet",
                        help="Choose whether to use ConceptNet or OpenAI.")
    args = parser.parse_args()
    TabooGameInstanceGenerator().generate(mode=args.mode)
