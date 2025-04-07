import os.path
from typing import Dict, Tuple, List, Union
import logging
import numpy as np

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer, GameRecorder
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_SUCCESS, BENCH_SCORE
from clemcore.utils import file_utils, string_utils

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

nltk.download('stopwords', quiet=True)
EN_STOPWORDS = stopwords.words('english')

EN_STEMMER = SnowballStemmer("english")

logger = logging.getLogger(__name__)


class WordGuesser(Player):

    def __init__(self, model: Model):
        super().__init__(model)
        self._custom_responses = ["Apple", "Banana", "Cherry"]

    def _custom_response(self, messages):
        word = self._custom_responses.pop(0)
        return f'GUESS: {word}'


class WordDescriber(Player):

    def __init__(self, model: Model):
        super().__init__(model)
        self._custom_responses = ["(1) My first clue is ...", "(2) My second clue is ...", "(3) My third clue is ..."]

    def _custom_response(self, messages):
        clue = self._custom_responses.pop(0)
        return f"CLUE: {clue}"


def check_clue(clue: str, target_word: str, related_words: List[str],
               stemmer=EN_STEMMER, return_clue=False) -> Union[Tuple[str, List[Dict]], List[Dict]]:
    clue = clue.replace("CLUE:", "")
    clue = clue.strip()
    clue = clue.lower()
    clue = string_utils.remove_punctuation(clue)
    clue_words = clue.split(" ")
    clue_words = [clue_word for clue_word in clue_words if clue_word not in EN_STOPWORDS]
    clue_word_stems = [stemmer.stem(clue_word) for clue_word in clue_words]
    errors = []
    target_word_stem = stemmer.stem(target_word)
    related_word_stems = [stemmer.stem(related_word) for related_word in related_words]

    for clue_word, clue_word_stem in zip(clue_words, clue_word_stems):
        if target_word_stem == clue_word_stem:
            errors.append({
                "message": f"Target word '{target_word}' (stem={target_word_stem}) "
                           f"is similar to clue word '{clue_word}' (stem={clue_word_stem})",
                "type": 0
            })
        for related_word, related_word_stem in zip(related_words, related_word_stems):
            if related_word_stem == clue_word_stem:
                errors.append({
                    "message": f"Related word '{related_word}' (stem={related_word_stem}) "
                               f"is similar to clue word '{clue_word}' (stem={clue_word_stem})",
                    "type": 1
                })
    if return_clue:
        return clue, errors
    return errors


class Taboo(DialogueGameMaster):
    """
    This class implements a taboo game in which player A (the WordDescriber) is describing a
    target word that player B (the WordGuesser) needs to guess. Player A cannot use the target
    word or related words in their explanation. Morphology is checked in check_clue().
    """

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)
        self.max_rounds: int = experiment["max_turns"]

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance

        self.target_word = game_instance["target_word"]
        self.related_words = game_instance["related_word"]

        describer_initial_prompt = self.experiment["describer_initial_prompt"]
        describer_initial_prompt = describer_initial_prompt.replace("$TARGET_WORD$", self.target_word)
        rel_words = f"- {self.related_words[0]}\n- {self.related_words[1]}\n- {self.related_words[2]}"
        describer_initial_prompt = describer_initial_prompt.replace("$REL_WORD$", rel_words)
        describer_initial_prompt = describer_initial_prompt.replace("$N$", str(self.max_rounds))

        guesser_initial_prompt = self.experiment["guesser_initial_prompt"]
        guesser_initial_prompt = guesser_initial_prompt.replace("$N$", str(self.max_rounds))

        self.describer = WordDescriber(self.player_models[0])
        self.guesser = WordGuesser(self.player_models[1])

        self.add_player(self.describer, initial_context=describer_initial_prompt)
        self.add_player(self.guesser, initial_prompt=guesser_initial_prompt)

        self.invalid_response = False
        self.clue_error = None
        self.guess_word = None

    def _does_game_proceed(self):
        """Proceed as long as the word hasn't been guessed and the maximum length isn't reached.
        """
        if self.is_terminal():
            if self.is_aborted():
                self.log_to_self("invalid format", "abort game")
            if self.is_clue_error():  # stop game if clue is wrong (for now)
                self.log_to_self("invalid clue", self.clue_error["message"])
            if self.is_turn_limit_reached():
                self.log_to_self("max rounds reached", str(self.max_rounds))
            if self.is_success():
                self.log_to_self("correct guess", "end game")
            return False
        return True

    def is_terminal(self):
        if self.is_aborted():
            return True
        if self.is_failure():
            return True
        if self.is_success():
            return True
        return False

    def is_aborted(self):
        return self.invalid_response

    def is_failure(self):
        if self.is_clue_error():
            return True
        if self.is_turn_limit_reached():
            return True
        return False

    def is_clue_error(self):
        return self.clue_error is not None

    def is_turn_limit_reached(self):
        return self.current_round >= self.max_rounds

    def is_success(self):
        return self.guess_word == self.target_word

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        if player == self.guesser:
            # validate response format
            if not utterance.startswith("GUESS:"):
                self.invalid_response = True
                return False
            self.log_to_self("valid response", "continue")
            # extract guess word
            guess_word = utterance.replace("GUESS:", "")
            guess_word = guess_word.strip()
            guess_word = guess_word.lower()
            guess_word = string_utils.remove_punctuation(guess_word)
            self.guess_word = guess_word.lower()
            self.log_to_self("valid guess", self.guess_word)
        if player == self.describer:
            # validate response format
            if not utterance.startswith("CLUE:"):
                self.invalid_response = True
                return False
            self.log_to_self("valid response", "continue")
            # validate clue
            clue, errors = check_clue(utterance, self.target_word, self.related_words, return_clue=True)
            if errors:
                error = errors[0]  # highlight single error
                self.clue_error = error
                return False
            self.log_to_self("valid clue", clue)
        return True

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        if player == self.describer:
            self.set_context_for(self.guesser, parsed_response)
        if player == self.guesser:
            self.set_context_for(self.describer, parsed_response)

    def compute_response_score(self, response, context):
        return 1 if self.is_success() else 0

    def compute_episode_score(self):
        if self.is_success():
            return 100 / (self.current_round + 1)  # zero-based
        return 0


class TabooScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        """ Episode level scores"""
        turn_scores = []
        prev_guess = None
        prev_guess_counter = 0
        prev_clue = None
        prev_clue_counter = 0
        invalid_response = False  # Note: This only takes into consideration that both players were compliant or not
        guesser_won = False
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            turn_score = {"guess": None, "clue": None, "request_count": 1}

            for event in turn:
                action = event["action"]
                if action["type"] == "invalid format":
                    invalid_response = True
                if action["type"] == "guess":
                    turn_score["guess"] = action["content"]
                if action["type"] == "clue":
                    turn_score["clue"] = action["content"]
                if action["type"] == "correct guess":
                    guesser_won = True

            if invalid_response:
                turn_score["violated_request_count"] = 1
                turn_score["parsed_request_count"] = 0
            else:
                turn_score["violated_request_count"] = 0
                turn_score["parsed_request_count"] = 1

            if turn_score["guess"] is not None and turn_score["guess"] == prev_guess:  # might be None, if clue is wrong
                prev_guess_counter += 1
            if turn_score["clue"] is not None and turn_score["clue"] == prev_clue:
                prev_clue_counter += 1
            self.log_turn_score(turn_idx, 'Accuracy', 1 if guesser_won else 0)
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_VIOLATED, turn_score["violated_request_count"])
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_PARSED, turn_score["parsed_request_count"])
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT, turn_score["request_count"])
            prev_guess = turn_score["guess"]
            prev_clue = turn_score["clue"]
            turn_scores.append(turn_score)

        violated_request_count = sum([turn["violated_request_count"] for turn in turn_scores])
        self.log_episode_score(METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)

        parsed_request_count = sum([turn["parsed_request_count"] for turn in turn_scores])
        self.log_episode_score(METRIC_REQUEST_COUNT_PARSED, parsed_request_count)

        request_count = sum([turn["request_count"] for turn in turn_scores])
        self.log_episode_score(METRIC_REQUEST_COUNT, request_count)

        self.log_episode_score(METRIC_REQUEST_SUCCESS, parsed_request_count / request_count)
        # checking the last guess (could be None) is ok,
        # b.c. the game ends only successfully, when there is a correct guess

        # Common metrics
        if invalid_response:  # whether a violation of the game rules happened (response not parsable)
            self.log_episode_score(METRIC_ABORTED, 1)
            self.log_episode_score(METRIC_SUCCESS, 0)
            self.log_episode_score(METRIC_LOSE, 0)
            # Game-specific metrics
            self.log_episode_score(BENCH_SCORE, np.nan)  # metric not applicable
        else:
            self.log_episode_score(METRIC_ABORTED, 0)
            if guesser_won:
                self.log_episode_score(METRIC_SUCCESS, 1)
                self.log_episode_score(METRIC_LOSE, 0)
                self.log_episode_score(BENCH_SCORE, 100 / len(turn_scores))  # how early the guesser found the word
            else:
                self.log_episode_score(METRIC_SUCCESS, 0)
                self.log_episode_score(METRIC_LOSE, 1)
                self.log_episode_score(BENCH_SCORE, 0)  # word not found

        # Game-specific metrics
        # How often the Guesser repeated a guess
        self.log_episode_score('Repetition-Guesser', prev_guess_counter)
        # How often the Describer repeated itself
        self.log_episode_score('Repetition-Describer', prev_clue_counter)
        # this might require a side-loop between describer and GM (game should not continue with Guesser)
        # self.log_episode_score('Rule-following', ...)


class TabooGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)
        # TODO: experiment could also be set through GameSpec

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return Taboo(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return TabooScorer(self.game_name, experiment, game_instance)


def main():
    # select one experiment and instance
    game_path = os.path.dirname(os.path.abspath(__file__))
    experiments = file_utils.load_json("in/instances.json", game_path)
    experiment_1 = experiments["experiments"][0]
    game_1 = experiment_1["game_instances"][0]
    master = Taboo("taboo", experiment_1, ["mock", "mock"])
    master.setup(**game_1)
    master.play()


if __name__ == '__main__':
    main()
