"""
Utilities for handling WinoDict-generated new-words.
"""

from typing import Dict, List, Optional, Sequence, Set

import nltk
from nltk import corpus
from nltk import lm

from adventuregame.resources.new_word_generation.wino_dict.create_new_words import NGramGenerator, generate_ngram_examples, read_morph_rules
from adventuregame.resources.new_word_generation.wino_dict.create_new_words import add_morphology_to_examples


def read_new_words_file(file_path:str) -> Dict:
    """Read a winodict-generated new words file and parse it into a dict.
    Args:
        file_path: Path to the .tsv file containing the generated new words.
    Returns:
        A dict containing the generated new words, with POS dict for each word.
    """
    with open(file_path, 'r', encoding='utf-8') as new_words_file:
        new_words_raw = new_words_file.read().split("\n")
    new_words_dict = dict()
    # read lines, one new word per line:
    for new_word_line in new_words_raw[:-1]:
        # split by tabs:
        tab_split_line = new_word_line.split("\t")
        # parse POS list string:
        pos_dict = dict()
        pos_split_1 = tab_split_line[2].split(",")
        for pos_split in pos_split_1:
            pos_split_2 = pos_split.split(":")
            pos_dict[pos_split_2[0]] = pos_split_2[1]
        # add to overall dict:
        new_words_dict[tab_split_line[0]] = {'wino_dict_value': tab_split_line[1], 'pos': pos_dict}

    return new_words_dict


def generate_winodict_words(num_characters: int = 3, min_length: int = 5, max_length: int = 12, num_buckets: int = 5,
                            min_score: float = -30.0, num_iterations: int = 3000, seed: int = 42,
                            morph_rule_path: str = "wino_dict/morph_rules.txt") -> List:
    """Generate new words using winodict functions.
    Based on https://github.com/google-research/language/tree/master/language/wino_dict/create_new_words.py
    Args:
        num_characters: Number of characters in the probabilistic model.
        min_length: Minimum length to use for word generation.
        max_length: Max length to use for word generation.
        num_buckets: Number of score buckets to use, sorting words from more to less probable. Max number of examples
            per bucket will be the number of iterations divided by the number of buckets.
        min_score: Minimum winodict score of generated words.
        num_iterations: Approximate number of examples to generate. Depending on bucketing and filters, could end up
            being less.
        seed: Random seed. If unchanged, output will be deterministic.
        morph_rule_path: Path to the set of morphology rules.
    Returns:
        A list of winodict new word objects.
    """
    print("Winodict word generation: Started.")
    nltk.download("words")
    generator = NGramGenerator(num_characters, set(corpus.words.words()))
    print("Winodict word generation: Finished training model with nltk vocab.")
    examples = generate_ngram_examples(generator, min_length, max_length, num_buckets, min_score, num_iterations, seed)
    print(f"Winodict word generation: Generated {len(examples)} initial examples.")
    print(f"Winodict word generation: Reading morphology from {morph_rule_path}")
    morph_rules = read_morph_rules(morph_rule_path)
    morphed_examples = add_morphology_to_examples(examples, morph_rules)
    print(f"Winodict word generation: Successfully created {len(morphed_examples)} final examples")

    return morphed_examples


def get_winodict_words(seed: int = 42) -> Dict:
    """Generate new words using winodict functions and convert into dict format."""
    # generate new words list using winodict functions:
    generated_new_words = generate_winodict_words(seed=seed)
    # convert each new word list to dict:
    new_words_dict = dict()
    for new_word in generated_new_words:
        new_word_str = new_word.scored_ngram.word
        new_words_dict[new_word_str] = dict()
        new_words_dict[new_word_str]['wino_dict_score'] = new_word.scored_ngram.score
        new_words_dict[new_word_str]['pos'] = new_word.morphology
    return new_words_dict

if __name__ == "__main__":
    new_words_dict = get_winodict_words()
    print(new_words_dict)