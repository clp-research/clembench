import os.path
from typing import Dict, Tuple, List, Union
import logging
import numpy as np
import re

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer, GameRecorder
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_SUCCESS, BENCH_SCORE
from clemcore.utils import file_utils, string_utils

from utils.guessvalidator import GuessValidator
from utils.compute_metrics import ComputeMetrics

logger = logging.getLogger(__name__)


class ResponseError(Exception):
    """
    General error class for problems with the player response.

    Developers can introduce more specific error types by subclassing this error.
    Alternatively, the 'reason' attribute can be used to define more granular error types.
    """

    def __init__(self, reason: str = None, response: str = None):
        """
        :param reason: (optional) a brief description of the cause
        :param response: (optional) the player's response
        """
        super().__init__(reason)
        self.reason = reason
        self.response = response

    def __str__(self):
        return f"{self.__class__.__name__}: {self.reason}"


class GameError(ResponseError):
    """Raised when a verbal action of a player causes problems for advancing the game."""
    pass


class RuleViolationError(GameError):
    """Raised when a verbal action of a player violates the specified game rules.

    For example:
        - taboo: mentioning the target word as the clue giver
        - wordle: guessing words that are not exactly 5 letters long
    """
    pass


class ProtocolError(ResponseError):
    """Raised when a message does not follow the communication protocol expected by the game master."""
    pass


class ParseError(ProtocolError):
    """
    This error is supposed to be raised when player messages cannot be parsed or understood by the game master e.g.
    because the response does not start with a specified prefix.

    For example:
        - taboo: clue giver messages should start with 'CLUE:'
        - wordle: guesser messages should start with 'GUESS:'
    """
    pass


class ResponseFormatter:

    def __init__(self, words):
        self.words = words

    def to_guesser_response(self, explanation, guess):
        return (f"{self.words['explanation_lang']} {explanation}\n"
                f"{self.words['guess_lang']} {guess}\n")

    def to_critic_response(self, agreement, explanation):
        """ The format of an actual response of the critic """
        return (f"{self.words['explanation_lang']} {explanation}\n"
                f"{self.words['agreement_lang']} {agreement}\n")

    def to_gm_response_for_guesser(self, feedback):
        return (f"{self.words['guess_feedback_lang']} {feedback}\n\n"
                f"{self.words['error_prompt_text']['INVALID_FORMAT']}\n"  # Provide your response only in this format.
                f"{self.words['explanation_lang']} {self.words['explanataion_details_lang']}\n"
                f"{self.words['guess_lang']} {self.words['guess_word_lang']}\n"
                )

    def to_gm_response_for_critic(self, clue, explanation, guess):
        """ The format of a message send by the GM to the critic"""
        return (f"{self.words['clue_lang']} {clue}\n"
                f"{self.words['explanation_lang']} {explanation}\n"
                f"{self.words['guess_lang']} {guess}\n\n"
                f"{self.words['error_prompt_text']['INVALID_FORMAT']}\n"  # Provide your response only in this format.
                f"{self.words['explanation_lang']} {self.words['explanataion_details_lang']}\n"
                f"{self.words['agreement_lang']} {self.words['agreement_word_lang']}\n"
                )

    def to_gm_turn_stats(self, stats):
        return '\n'.join(f'{key} = {value}' for key, value in stats.items())


class WordGuesser(Player):
    def __init__(self, model: Model, formatter: ResponseFormatter):
        super().__init__(model)
        self.formatter = formatter
        self._custom_responses = ["apple", "beach", "crane", "after", "those"]

    def _terminal_response(self, context: Dict) -> str:
        guess = input("Enter your guess: ")
        return self.formatter.to_guesser_response("human guesser", guess)

    def _custom_response(self, messages):
        guess = self._custom_responses.pop(0)
        return self.formatter.to_guesser_response("custom guesser", guess)


class WordCritic(Player):
    def __init__(self, model: Model, formatter: ResponseFormatter):
        super().__init__(model)
        self.formatter = formatter
        self._custom_responses = ["yes", "no", "no", "yes", "no"]

    def _terminal_response(self, context: Dict) -> str:
        feedback = input("Do you agree with the guess? (yes/no) ")
        return self.formatter.to_guesser_response("human feedback", feedback)

    def _custom_response(self, messages):
        feedback = self._custom_responses.pop(0)
        return self.formatter.to_guesser_response("custom critic", feedback)


def parse_response(player: Player, response: str, lang_keywords: Dict) -> Tuple[str, str]:
    """Parse guesser response and extract guess and explanation"""
    if not response or not response.startswith(lang_keywords["explanation_lang"]):
        error = ParseError("INVALID_START_WORD")
        error.reason = f"The response should always start with the keyword '{lang_keywords['explanation_lang']}'"
        raise error

    response = response.strip()
    lines = response.split("\n")
    if len(lines) > 2:
        error = ParseError("UNKNOWN_TAGS")
        error.reason = (f"The response should contain only the '{lang_keywords['guess_lang']}' and "
                        f"'{lang_keywords['explanation_lang']}' keywords and associated information.")
        raise error

    # Extract explanation and guess
    explanation_pattern = re.compile(rf"{lang_keywords['explanation_lang']}([^\n]*)", re.IGNORECASE)

    content_prefix = lang_keywords['guess_lang']
    if player == WordCritic:
        content_prefix = lang_keywords['agreement_lang']
    content_pattern = re.compile(rf"{content_prefix}([^\n]*)", re.IGNORECASE)

    explanation_match = explanation_pattern.search(response)
    content_match = content_pattern.findall(response)

    if len(content_match) != 1:
        error = ParseError("MORE_THAN_ONE_GUESS")
        error.reason = f"The response should contain the '{content_prefix}' keyword only once."
        raise error

    content = content_match[0].strip().lower()
    explanation = explanation_match.group(1).strip() if explanation_match else ""

    return content, explanation


def validate_guess(guess: str, lang_keywords: Dict):
    """Validate guess format and content"""
    if not guess.isalpha() or " " in guess:
        error = RuleViolationError("INVALID_FORMAT")
        error.reason = "The guess should be a single word and should only contain letters."
        raise error

    if len(guess) != lang_keywords["max_word_length"]:
        error = RuleViolationError("INVALID_WORD_LENGTH")
        error.reason = f"The length of the guessed word is not {lang_keywords['max_word_length']}."
        raise error

    if guess not in lang_keywords["official_words_list"]:
        error = RuleViolationError("NOT_VALID_WORD_FOR_GAME")
        error.reason = f"The guessed word is not a valid word for this game."
        raise error


def validate_agreement(agreement: str, lang_keywords: Dict):
    """Validate critic agreement"""
    if not agreement.isalpha() or " " in agreement:
        error = RuleViolationError("INVALID_FORMAT")
        error.reason = "The agreement should be a single word and should only contain letters."
        raise error

    if agreement not in lang_keywords["agreement_match_keywords_lang"]:
        error = RuleViolationError("NOT_VALID_CRITIC_WORD")
        error.reason = (f"The agreement should be one of the following: "
                        f"{lang_keywords['agreement_match_keywords_lang']}")
        raise error


class Wordle(DialogueGameMaster):
    """Basic Wordle game without clue or critic"""

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)
        self.max_rounds: int = experiment["common_config"]["n_turns"]
        self.lang_keywords = experiment["lang_keywords"]
        self.formatter = ResponseFormatter(self.lang_keywords)

    def _on_setup(self, **game_instance):
        self.target_word = game_instance["target_word"].strip().lower()
        self.invalid_response = False
        self.current_guess = None
        self.current_explanation = None
        self.guess_feedback = None
        self.guess_validator = GuessValidator(self.target_word)
        self._add_players()

    def _add_players(self):
        self.guesser = WordGuesser(self.player_models[0], self.formatter)
        self.add_player(self.guesser, initial_context=self.experiment["guesser_prompt"])

    def _does_game_proceed(self):
        if self.is_terminal():
            if self.is_aborted():
                self.log_to_self("invalid format", "game_result = ABORT")
            elif self.is_turn_limit_reached():
                self.log_to_self("max rounds reached", "game_result = LOSS")
            elif self.is_success():
                self.log_to_self("correct guess", "game_result = WIN")
            return False
        return True

    def is_terminal(self):
        return self.is_aborted() or self.is_failure() or self.is_success()

    def is_aborted(self):
        return self.invalid_response

    def is_failure(self):
        return self.is_turn_limit_reached() and not self.is_success()

    def is_turn_limit_reached(self):
        return self.current_round >= self.max_rounds

    def is_success(self):
        return self.current_guess == self.target_word

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        try:
            # Parse response of the only player: the guesser
            guess, explanation = parse_response(player, utterance, self.lang_keywords)
            self.current_guess = guess
            self.current_explanation = explanation
            # Validate guess
            validate_guess(guess, self.lang_keywords)
            return True
        except (ParseError, RuleViolationError) as e:
            # todo prepare re-prompting for invalid response somewhere; if max re-prompts then abort
            self.invalid_response = True
            self.log_to_self("metadata", e.reason)
            return False

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        # Provide feedback to guesser for next round
        self.guess_feedback = self.guess_validator.validate(self.current_guess)
        content = self.formatter.to_gm_response_for_guesser(self.guess_feedback)
        self.set_context_for(self.guesser, content)
        self.log_to_self("metadata", self.formatter.to_gm_turn_stats(self.get_turn_stats()))

    def get_turn_stats(self):
        return {
            "attempts": self.current_round,
            "target_word": self.target_word,
            "guess": self.current_guess,
            "guess_feedback": self.guess_feedback
        }

    def compute_response_score(self, response, context):
        return 1 if self.is_success() else 0

    def compute_episode_score(self):
        if self.is_success():
            return 100 / self.current_round
        return 0


class WordleWithClue(Wordle):
    """Wordle game with target word clue"""

    def _on_setup(self, **game_instance):
        super()._on_setup(**game_instance)
        # Set clue as initial context; will be appended to initial_prompt on the Player's first turn
        self.target_word_clue = game_instance["target_word_clue"].strip()
        self.set_context_for(self.guesser, f"{self.lang_keywords['clue_lang']} {self.target_word_clue}")

    def _add_players(self):
        self.guesser = WordGuesser(self.player_models[0], self.formatter)
        self.add_player(self.guesser, initial_prompt=self.experiment["guesser_prompt"])

    def get_turn_stats(self):
        return {
            "attempts": self.current_round,
            "target_word": self.target_word,
            "target_word_clue": self.target_word_clue,
            "guess": self.current_guess,
            "guess_feedback": self.guess_feedback
        }


class WordleWithCritic(WordleWithClue):
    """
    Wordle game with clue and critic player.

    In this variant a critic provides intermediate feedback to the guesser.

    Hence, the rounds have 3 turns: [guesser, critic, guesser].

    Only then the color feedback is given and it's the guessers turn again.
    """

    def _on_setup(self, **game_instance):
        super()._on_setup(**game_instance)
        self.target_word_difficulty = game_instance["target_word_difficulty"].strip()
        self.awaiting_critic = True  # whether the critic has already been consulted
        self.commit_guess = False  # guesser has an initial and a final guess (to commit == end the round)
        self.current_agreement = None
        self.current_agreement_explanation = None

    def _add_players(self):
        guesser_model = self.player_models[0]
        self.guesser = WordGuesser(guesser_model, self.formatter)
        self.add_player(self.guesser, initial_prompt=self.experiment["guesser_prompt"])

        critic_model = self.player_models[1] if len(self.player_models) > 1 else guesser_model
        self.critic = WordCritic(critic_model, self.formatter)
        self.add_player(self.critic, initial_prompt=self.experiment["guesser_critic_prompt"])

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        if player == self.guesser:
            return super()._validate_player_response()
        if player == self.critic:
            try:
                agreement, explanation = parse_response(player, utterance, self.lang_keywords)
                self.current_agreement = agreement
                self.current_agreement_explanation = explanation
                validate_agreement(agreement, self.lang_keywords)
                return True
            except (ParseError, RuleViolationError) as e:
                # todo prepare re-prompting for invalid response somewhere; if max re-prompts then abort
                self.invalid_response = True
                self.log_to_self("metadata", e.reason)
                return False

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        # Only provide game feedback after critic interaction is complete
        if player == self.guesser:
            if self.awaiting_critic:  # little state machine
                content = self.formatter.to_gm_response_for_critic(self.target_word_clue,
                                                                   self.current_explanation,
                                                                   self.current_guess)
                self.set_context_for(self.critic, content)
            else:  # another turn with the gm
                self.commit_guess = True
                super()._on_valid_player_response(player, parsed_response)
        if player == self.critic:
            content = self.formatter.to_critic_response(self.current_agreement, self.current_agreement_explanation)
            self.set_context_for(self.guesser, content)
            self.awaiting_critic = False

    def _start_next_round(self):
        return self.commit_guess

    def _on_before_round(self):
        self.awaiting_critic = True
        self.commit_guess = False

    def _should_pass_turn(self):
        return self.awaiting_critic

    def _does_game_proceed(self):
        # Proceed if waiting for critic response (skip termination checks)
        if self.awaiting_critic and not self.invalid_response:
            return True
        return super()._does_game_proceed()


class WordleScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.cm = ComputeMetrics()

    def compute_scores(self, episode_interactions: Dict) -> None:
        """Compute episode-level and turn-level scores"""

        # Extract game results from interactions
        aborted = False
        success = False
        total_turns = 0
        guesses = []

        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            total_turns = turn_idx + 1
            turn_success = False
            turn_aborted = False

            for event in turn:
                action = event["action"]
                if action["type"] == "invalid format" or action["type"] == "parse error":
                    turn_aborted = True
                    aborted = True
                elif action["type"] == "correct guess":
                    turn_success = True
                    success = True
                elif action["type"] == "valid guess" or action["type"] == "final guess":
                    # Extract guess from content
                    content = action["content"]
                    if " -> " in content:
                        guess_part = content.split(" -> ")[0]
                        guesses.append(guess_part)

            # Log turn-level scores
            self.log_turn_score(turn_idx, 'Accuracy', 1 if turn_success else 0)
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT, 1)
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_PARSED, 0 if turn_aborted else 1)
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_VIOLATED, 1 if turn_aborted else 0)

        # Compute episode-level metrics
        if aborted:
            self.log_episode_score(METRIC_ABORTED, 1)
            self.log_episode_score(METRIC_SUCCESS, 0)
            self.log_episode_score(METRIC_LOSE, 0)
            self.log_episode_score(BENCH_SCORE, np.nan)
        else:
            self.log_episode_score(METRIC_ABORTED, 0)
            if success:
                self.log_episode_score(METRIC_SUCCESS, 1)
                self.log_episode_score(METRIC_LOSE, 0)
                self.log_episode_score(BENCH_SCORE, 100 / total_turns)
            else:
                self.log_episode_score(METRIC_SUCCESS, 0)
                self.log_episode_score(METRIC_LOSE, 1)
                self.log_episode_score(BENCH_SCORE, 0)

        # Request count metrics
        self.log_episode_score(METRIC_REQUEST_COUNT, total_turns)
        parsed_count = total_turns if not aborted else total_turns - 1
        self.log_episode_score(METRIC_REQUEST_COUNT_PARSED, parsed_count)
        violated_count = 1 if aborted else 0
        self.log_episode_score(METRIC_REQUEST_COUNT_VIOLATED, violated_count)

        if total_turns > 0:
            self.log_episode_score(METRIC_REQUEST_SUCCESS, parsed_count / total_turns)
        else:
            self.log_episode_score(METRIC_REQUEST_SUCCESS, 0)

        # Game-specific metrics
        if guesses and not aborted:
            # Check for repeated guesses
            unique_guesses = len(set(guesses))
            total_guesses = len(guesses)
            repetitions = total_guesses - unique_guesses
            self.log_episode_score('Repetition-Guesser', repetitions)
        else:
            self.log_episode_score('Repetition-Guesser', 0)


class WordleGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        # Determine which variant to use based on experiment configuration
        use_clue = experiment.get("use_clue", False)
        use_critic = experiment.get("use_critic", False)

        if use_critic:
            return WordleWithCritic(self.game_name, self.game_path, experiment, player_models)
        elif use_clue:
            return WordleWithClue(self.game_name, self.game_path, experiment, player_models)
        else:
            return Wordle(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return WordleScorer(self.game_name, experiment, game_instance)
