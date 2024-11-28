import os
import random
import logging
from tqdm import tqdm
import json
import requests
import spacy  # Für Wortartenanalyse

# Lade das spaCy-Modell
nlp = spacy.load("en_core_web_sm")  # Englischsprachiges Modell

# Parameters
N_INSTANCES = 20  # Number of target words; 0 means "all"
N_GUESSES = 3  # Maximal number of trials
N_RELATED_WORDS = 3  # Number of related words for each target word
LANGUAGE = "en"
VERSION = "v1.5"

# Setup Logging to a file
LOG_FILE = "taboo_game_conceptnet.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Seed for reproducibility
random.seed(42)

def filter_nouns(words):
    """
    Filtert Wörter, sodass nur Nomen zurückgegeben werden.
    :param words: Liste von Wörtern
    :return: Liste von Nomen
    """
    nouns = []
    for word in words:
        doc = nlp(word)  # Analysiere das Wort mit spaCy
        for token in doc:
            if token.pos_ == "NOUN":  # Überprüfe, ob das Wort ein Nomen ist
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

        # Filter nur Nomen
        related_nouns = filter_nouns(related_words)
        return related_nouns[:n]  # Begrenze die Anzahl auf 'n'
    except Exception as e:
        logger.error(f"Error fetching related words for '{word}': {e}")
        return []

class TabooGameInstanceGenerator:
    """
    Generates game instances for the Taboo game and saves them to a file.
    """
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.used_words = set()  # To track already used target words

    def load_file(self, file_name, file_ending):
        """
        Loads a file from the resources folder.
        """
        path = os.path.join(self.base_dir, file_name + file_ending)
        try:
            with open(path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise

    def add_experiment(self, name):
        """
        Adds an experiment.
        """
        return {
            "name": name,
            "game_instances": [],
            "max_turns": N_GUESSES,
            "describer_initial_prompt": self.load_file("resources/initial_prompts/initial_describer", ".template"),
            "guesser_initial_prompt": self.load_file("resources/initial_prompts/initial_guesser", ".template")
        }

    def add_game_instance(self, experiment, game_id):
        """
        Adds a game instance to an experiment.
        """
        game_instance = {"game_id": game_id}
        experiment["game_instances"].append(game_instance)
        return game_instance

    def generate(self):
        """
        Generates the game instances and saves them to a file.
        """
        output_file = os.path.join(self.base_dir, "in", "instances_conceptnet_filtered_nouns.json")
        experiments = []

        for frequency in ["medium"]:  # You can adjust to process other frequencies
            logger.info(f"Processing '{frequency}' frequency...")
            # Load the target words
            file_path = f"resources/target_words/{LANGUAGE}/{frequency}_freq_100_{VERSION}"
            target_words = self.load_file(file_path, ".txt").split('\n')
            if N_INSTANCES > 0:
                assert len(target_words) >= N_INSTANCES, \
                    f'There are fewer target words ({len(target_words)}) than needed ({N_INSTANCES}).'

            available_words = set(target_words)  # Use a set for efficient lookup
            self.used_words = set()  # Reset the used words for this frequency
            experiment = self.add_experiment(f"{frequency}_{LANGUAGE}")

            game_id = 0
            while game_id < N_INSTANCES:
                # Sample a new target word that has not been used yet
                remaining_words = available_words - self.used_words
                if not remaining_words:
                    logger.warning("No more words available to sample.")
                    break

                target = random.choice(list(remaining_words))
                self.used_words.add(target)

                # Generate related words and filter for nouns
                related_words = get_related_words_from_conceptnet(target, n=N_RELATED_WORDS)
                if not related_words or all(word == "No related word" for word in related_words):
                    logger.info(f"Skipping '{target}' due to lack of related nouns.")
                    continue  # Skip this word and try another

                # Add a valid game instance
                game_instance = self.add_game_instance(experiment, game_id)
                game_instance["target_word"] = target
                game_instance["related_word"] = related_words

                logger.info(f"Added game instance for target word '{target}' with related nouns: {related_words}")
                game_id += 1  # Increment only for valid instances

            experiments.append(experiment)

        # Save JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"experiments": experiments}, f, indent=2, ensure_ascii=False)
        logger.info(f"Instances successfully generated and saved in '{output_file}'.")

if __name__ == '__main__':
    TabooGameInstanceGenerator().generate()
