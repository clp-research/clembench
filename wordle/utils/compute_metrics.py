import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def turns_closeness(guesser_feedbacks: List[str]):
    """
    Assuming records contain turns_data in the below format
    [['creek', 'c<red> r<red> e<red> e<red> k<green>'], ['sneak', 's<green> n<yellow> e<red> a<red> k<green>']
    """

    score_list = []

    for feedback in guesser_feedbacks:
        # Add a score of 5 for letters in green
        # Add a score of 1 for letters in yellow
        # Add a score of 0 for letters in red
        score = 0
        for letter in feedback.split(" "):
            if "green" in letter:
                score += 5
            elif "yellow" in letter:
                score += 3
        score_list.append(score)

    return score_list


def turns_strategy(guesser_feedbacks: List[str], is_aborted: bool):
    """
    Assuming records contain turns_data in the below format
    [['creek', 'c<red> r<red> e<red> e<red> k<green>'], ['sneak', 's<green> n<yellow> e<red> a<red> k<green>']
    """
    if len(guesser_feedbacks) == 1:
        if is_aborted:
            return [0]
        return [100]  # Looks like the game was won in first guess!
    # For the first turn, there is no comparison possible, hence adding the strategy score as 0
    score_list = [0]
    for guesses in zip(guesser_feedbacks, guesser_feedbacks[1:]):
        guess1, guess2 = guesses
        guess1_dict, _ = extract_words_by_color_code(guess1)
        _, guess2_list = extract_words_by_color_code(guess2)
        guess1_not_use = []
        guess1_use = []
        guess1_change = []

        if "red" in guess1_dict:
            guess1_not_use = guess1_dict["red"]
        if "green" in guess1_dict:
            guess1_use = guess1_dict["green"]
        if "yellow" in guess1_dict:
            guess1_change = guess1_dict["yellow"]
        score = 0

        result = len(set(guess1_not_use) & set(guess2_list))
        if result:
            # Decrease score by 20 for each non-used letter present in
            # next guess
            score -= result * 20

        # TODO: Do I need to penalize position change?
        result = len(set(guess1_use) & set(guess2_list))
        if result:
            # Increase score by 20 for each green letter present in next
            # guess
            score += result * 20

        # TODO: Do I need to penalize position non-change?
        result = len(set(guess1_change) & set(guess2_list))
        if result:
            # Increase score by 10 for each yellow letter present in
            # next guess
            score += result * 10
        score_list.append(score)
    return score_list


def extract_words_by_color_code(guess_word):
    color_lable_dict = {}
    letters_list = []

    for letter_code in guess_word.split(" "):
        matches = re.findall(r"\b(\w+)\b\s*\<(.+?)\>", letter_code)
        for match in matches:
            letter = match[0].strip()
            letters_list.append(letter)
            color_code = match[1].strip()
            if color_code not in color_lable_dict:
                color_lable_dict[color_code] = []
            color_lable_dict[color_code].append(letter)
    return color_lable_dict, letters_list
