import os
import random
import logging
import openai
from tqdm import tqdm
import json

# Parameters
N_INSTANCES = 20  # number of the target words; 0 means "all"
N_GUESSES = 3  # Maximal numbers of trials
N_RELATED_WORDS = 3  # numbers of related words for each target word
LANGUAGE = "en"
VERSION = "v1.5"
OPENAI_API_KEY = "" # Replace with your OpenAI API key

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("taboo_game_openai.log"),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# Seed for reproducibility
random.seed(42)

# set up OpenAI API-key 
openai.api_key = OPENAI_API_KEY

def generate_related_words_from_openai(target_word, n=3):
    """
    Generates related words for a target word using the OpenAI API (new interface).
    :param target_word: The target word.
    :param n: Number of related words.
    :return: List of related words.
    """
    try:
        # Prompt for Chat model
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Give me {n} words that are related to '{target_word}'."}
        ]

        # Request to the ChatCompletion-API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4.0-turbo" if available
            messages=messages,
            max_tokens=50,
            temperature=0.7
        )

        # Extract the response of the model
        raw_response = response['choices'][0]['message']['content'].strip()

        # Standardize the response
        if "\n" in raw_response:  # Check for newline-separated list
            related_words = [line.split(".")[-1].strip() for line in raw_response.split("\n") if line.strip()]
        else:  # Assume comma-separated list
            related_words = raw_response.split(", ")

        return related_words[:n]  # limit the number of related words
    except Exception as e:
        logger.error(f"Error generating related words for '{target_word}': {e}")
        return []

class TabooGameInstanceGenerator:
    """
    Generates game instances for the Taboo game and saves them to a file.
    """
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)

    def load_file(self, file_name, file_ending):
        """
        Loads a file from the resources folder.
        """
        path = os.path.join(self.base_dir, file_name + file_ending)
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()

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
        output_file = os.path.join(self.base_dir, "in", "instances_openai.json")
        experiments = []

        for frequency in ["high", "medium", "low"]:
            logger.info(f"Processing frequency: {frequency}")

            # Load the target words
            file_path = f"resources/target_words/{LANGUAGE}/{frequency}_freq_100_{VERSION}"
            target_words = self.load_file(file_path, ".txt").split('\n')
            if N_INSTANCES > 0:
                assert len(target_words) >= N_INSTANCES, \
                    f'There are fewer target words ({len(target_words)}) than needed ({N_INSTANCES}).'
            
            sampled_words = set()
            valid_instances = 0
            experiment = self.add_experiment(f"{frequency}_{LANGUAGE}")

            while valid_instances < N_INSTANCES:
                target = random.choice(target_words)

                # Skip already-sampled words
                if target in sampled_words:
                    continue
                
                sampled_words.add(target)
                related_words = generate_related_words_from_openai(target, n=N_RELATED_WORDS)

                # Skip words without valid related words
                if not related_words:
                    logger.warning(f"Skipping target word '{target}' due to no related words found.")
                    continue
                
                # Add valid instance
                game_instance = self.add_game_instance(experiment, valid_instances)
                game_instance["target_word"] = target
                game_instance["related_word"] = related_words
                valid_instances += 1

            experiments.append(experiment)

        # Save JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"experiments": experiments}, f, indent=2, ensure_ascii=False)
        logger.info(f"Instances successfully generated and saved in '{output_file}'.")

if __name__ == '__main__':
    TabooGameInstanceGenerator().generate()
