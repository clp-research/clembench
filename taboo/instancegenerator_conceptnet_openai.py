"""The script generates game instances for the Taboo game. It selects target words and generates a list of related words. 
The script uses either ConceptNet or the OpenAI API to retrieve or generate these related words."""

import os
import random
import logging
import openai
from tqdm import tqdm
import json
import requests
import spacy
import argparse

# Parameters
N_INSTANCES = 20  # Number of target words; 0 means "all"
N_GUESSES = 3  # Maximal number of trials
N_RELATED_WORDS = 3  # Number of related words for each target word
LANGUAGE = "en"
VERSION = "v1.5"
#OPENAI_API_KEY =   # Replace with your OpenAI API key

# Seed for reproducibility
random.seed(42)

# Set up OpenAI API key
#openai.api_key = OPENAI_API_KEY

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def setup_logging(log_file):
    """
    Set up logging to a specified log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def filter_nouns(words):
    """
    Filter words to include only nouns.
    """
    nouns = []
    for word in words:
        doc = nlp(word)
        for token in doc:
            if token.pos_ == "NOUN":
                nouns.append(token.text)
    return nouns


def get_related_words_from_conceptnet(word, n=N_RELATED_WORDS, language=LANGUAGE):
    """
    Fetch related words from ConceptNet and filter for nouns.
    """
    try:
        url = f"http://api.conceptnet.io/c/{language}/{word}/"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        related_words = []
        for edge in data.get("edges", []):
            related_term = edge.get("end", {}).get("label", "")
            if related_term.lower() != word.lower():
                related_words.append(related_term)

        # Filter nouns
        related_nouns = filter_nouns(related_words)
        return related_nouns[:n]
    except Exception as e:
        logger.error(f"Error fetching related words for '{word}': {e}")
        return []


def generate_related_words_from_openai(target_word, n=3):
    """
    Generate related words using OpenAI's API.
    """
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Give me {n} words that are related to '{target_word}'."},
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=50,
            temperature=0.7,
        )

        raw_response = response['choices'][0]['message']['content'].strip()
        logger.debug(f"Raw OpenAI response for '{target_word}': {raw_response}")

        if "\n" in raw_response:
            related_words = [line.split(".")[-1].strip() for line in raw_response.split("\n") if line.strip()]
        else:
            related_words = raw_response.split(", ")

        logger.info(f"Extracted related words for '{target_word}': {related_words}")
        return related_words[:n]
    except Exception as e:
        logger.error(f"Error generating related words for '{target_word}': {e}")
        return []


class TabooGameInstanceGenerator:
    def __init__(self, mode):
        self.base_dir = os.path.dirname(__file__)
        self.used_words = set()
        self.mode = mode

    def load_file(self, file_name, file_ending):
        path = os.path.join(self.base_dir, file_name + file_ending)
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()

    def add_experiment(self, name):
        return {
            "name": name,
            "game_instances": [],
            "max_turns": N_GUESSES,
            "describer_initial_prompt": self.load_file("resources/initial_prompts/initial_describer", ".template"),
            "guesser_initial_prompt": self.load_file("resources/initial_prompts/initial_guesser", ".template"),
        }

    def add_game_instance(self, experiment, game_id):
        game_instance = {"game_id": game_id}
        experiment["game_instances"].append(game_instance)
        return game_instance

    def generate(self):
        if self.mode == "conceptnet":
            log_file = "taboo_game_conceptnet.log"
            output_file = os.path.join(self.base_dir, "in", "instances_conceptnet.json")
            generator_func = get_related_words_from_conceptnet
        elif self.mode == "openai":
            log_file = "taboo_game_openai.log"
            output_file = os.path.join(self.base_dir, "in", "instances_openai.json")
            generator_func = generate_related_words_from_openai
        else:
            raise ValueError("Invalid mode selected.")

        global logger
        logger = setup_logging(log_file)

        experiments = []
        for frequency in ["medium"]:
            logger.info(f"Processing '{frequency}' frequency...")
            file_path = f"resources/target_words/{LANGUAGE}/{frequency}_freq_100_{VERSION}"
            target_words = self.load_file(file_path, ".txt").split('\n')

            if N_INSTANCES > 0:
                assert len(target_words) >= N_INSTANCES, \
                    f'There are fewer target words ({len(target_words)}) than needed ({N_INSTANCES}).'

            available_words = set(target_words)
            self.used_words = set()
            experiment = self.add_experiment(f"{frequency}_{LANGUAGE}")

            game_id = 0
            while game_id < N_INSTANCES:
                remaining_words = available_words - self.used_words
                if not remaining_words:
                    logger.warning("No more words available to sample.")
                    break

                target = random.choice(list(remaining_words))
                self.used_words.add(target)

                related_words = generator_func(target, n=N_RELATED_WORDS)
                if not related_words:
                    logger.info(f"Skipping '{target}' due to lack of related words.")
                    continue

                game_instance = self.add_game_instance(experiment, game_id)
                game_instance["target_word"] = target
                game_instance["related_word"] = related_words
                logger.info(f"Added game instance for '{target}' with related words: {related_words}")
                game_id += 1

            experiments.append(experiment)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"experiments": experiments}, f, indent=2, ensure_ascii=False)
        logger.info(f"Instances successfully generated and saved in '{output_file}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Taboo game instances.")
    parser.add_argument("--mode", choices=["conceptnet", "openai"], required=True,
                        help="Choose whether to use ConceptNet or OpenAI.")
    args = parser.parse_args()

    generator = TabooGameInstanceGenerator(mode=args.mode)
    generator.generate()
