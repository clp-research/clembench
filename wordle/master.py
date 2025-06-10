import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import logging
import numpy as np
import re

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, BENCH_SCORE

from utils.guessvalidator import GuessValidator
from utils.compute_metrics import turns_closeness, turns_strategy

logger = logging.getLogger(__name__)


class ResponseError(Exception):
    """
    General error class for problems with the player response.

    Developers can introduce more specific error types by subclassing this error.
    Alternatively, the 'reason' attribute can be used to define more granular error types.
    """

    def __init__(self, reason: Optional[str] = None, response: Optional[str] = None, key: Optional[str] = None):
        """
        :param reason: (optional) a brief description of the cause
        :param response: (optional) the player's response
        :param key: (optional) a key word
        """
        super().__init__(reason)
        self.reason = reason
        self.response = response
        self.key = key

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


class UnknownFiveLetterWordError(RuleViolationError):
    """Raised when the word is 5-letters but not part of the game's vocabulary"""
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

    # noinspection PyMethodMayBeStatic
    def to_gm_turn_stats(self, stats: Dict):
        return '\n'.join(f'{key} = {value}' for key, value in stats.items())

    def to_gm_reprompt_for_guesser(self, error: ResponseError):
        return (f"{self.words['error_prompt_text'][error.key]} "  # only white space separated
                f"{self.words['error_prompt_text']['RETRY']}\n\n"  # Please try again.
                f"{self.words['error_prompt_text']['INVALID_FORMAT']}\n"  # Provide your response only in this format.
                f"{self.words['explanation_lang']} {self.words['explanataion_details_lang']}\n"
                f"{self.words['guess_lang']} {self.words['guess_word_lang']}"
                )

    def to_gm_response_for_guesser(self, feedback: str):
        return (f"{self.words['guess_feedback_lang']} {feedback}\n\n"
                f"{self.words['error_prompt_text']['INVALID_FORMAT']}\n"  # Provide your response only in this format.
                f"{self.words['explanation_lang']} {self.words['explanataion_details_lang']}\n"
                f"{self.words['guess_lang']} {self.words['guess_word_lang']}"
                )

    def to_gm_response_for_guesser_with_critic(self, clue: str, explanation: str, agreement: str):
        """ The format of a message send by the GM to the guesser with critic"""
        # todo: having more than explanation, agreement (see to_critic_response()) in the response reduces performance
        return (f"{self.words['clue_lang']} {clue}\n"
                f"{self.words['guess_agreement_lang']} {agreement}\n"
                f"{self.words['agreement_explanation_lang']} {explanation}\n\n"
                f"{self.words['error_prompt_text']['INVALID_FORMAT']}\n"  # Provide your response only in this format.
                f"{self.words['explanation_lang']} {self.words['explanataion_details_lang']}\n"
                f"{self.words['guess_lang']} {self.words['guess_word_lang']}"
                )

    def to_gm_response_for_critic(self, clue: str, explanation: str, guess: str, is_initial_context: bool):
        """ The format of a message send by the GM to the critic"""
        if is_initial_context:  # todo consider to remove this case as w/o format hint reduces pairing performance
            # On critic's first turn we leave away the format information because already described in initial prompt
            return (f"{self.words['clue_lang']} {clue}\n"
                    f"{self.words['explanation_lang']} {explanation}\n"
                    f"{self.words['guess_lang']} {guess}"
                    )
        return (f"{self.words['clue_lang']} {clue}\n"
                f"{self.words['explanation_lang']} {explanation}\n"
                f"{self.words['guess_lang']} {guess}\n\n"
                f"{self.words['error_prompt_text']['INVALID_FORMAT']}\n"  # Provide your response only in this format.
                f"{self.words['explanation_lang']} {self.words['explanataion_details_lang']}\n"
                f"{self.words['agreement_lang']} {self.words['agreement_word_lang']}"
                )


class WordGuesser(Player):
    def __init__(self, model: Model, words: Dict, target_word: str):
        super().__init__(model)
        self.target_word = target_word
        self.words = words
        self._custom_responses = ["apple", "beach", "crane",
                                  "pathy",  # throw in an invalid word
                                  "after", "those", "horse"]

    def to_guesser_response(self, explanation: str, guess: str):
        """ Only for custom response behavior (mock); documents the expected response format """
        return (f"{self.words['explanation_lang']} {explanation}\n"
                f"{self.words['guess_lang']} {guess}")

    def _terminal_response(self, context: Dict) -> str:
        guess = input("Enter your guess: ")
        return self.to_guesser_response("human guesser", guess)

    def _custom_response(self, messages):  # for playing with_critic we need doulbe the amoutn of responses
        guess = self._custom_responses.pop(0)
        if random.randint(0, 100) < 10:  # let the player occasionally win
            guess = self.target_word
        if random.randint(0, 100) > 90:  # let the player occasionally abort (not 5-letter word)
            guess = "scrumbled eggs"
        return self.to_guesser_response("custom guesser", guess)


class WordCritic(Player):
    def __init__(self, model: Model, words: Dict):
        super().__init__(model)
        self.words = words
        self._custom_responses = ["yes", "no", "no", "yes", "no", "no"]

    def to_critic_response(self, explanation: str, agreement: str):
        """ Only for custom response behavior (mock); documents the expected response format """
        return (f"{self.words['explanation_lang']} {explanation}\n"
                f"{self.words['agreement_lang']} {agreement}")

    def _terminal_response(self, context: Dict) -> str:
        feedback = input("Do you agree with the guess? (yes/no) ")
        return self.to_critic_response("human feedback", feedback)

    def _custom_response(self, messages):
        feedback = self._custom_responses.pop(0)
        return self.to_critic_response("custom critic", feedback)


class ReflectingWordGuesser(WordGuesser):
    """ When playing with a critic, the word guesser has two turns:
        1. Turn: The guesser provides on initial guess which is given to the critic
        2. Turn: The guesser reflects on the feedback given by the critic and (potentially) adjusts the initial guess
    """

    def __init__(self, model: Model, words: Dict, target_word: str):
        super().__init__(model, words, target_word)
        # self._custom_responses = ["yes", "no", "no", "yes", "no", "no"] -- from the critic
        self._custom_responses = ["apple", "apple",
                                  "beach", "crane",
                                  "those", "horse",
                                  "after", "after",
                                  "worse", "morse",
                                  "quiet", "fight", ]


def parse_response(player: Player, response: str, words: Dict) -> Tuple[str, str]:
    """Parse guesser response and extract guess and explanation"""
    if not response or not response.startswith(words["explanation_lang"]):
        raise ParseError(f"The response should always start with the keyword '{words['explanation_lang']}'",
                         key="INVALID_START_WORD")

    response = response.strip()
    lines = response.split("\n")
    if len(lines) > 2:
        raise ParseError(f"The response should contain only the '{words['guess_lang']}' and "
                         f"'{words['explanation_lang']}' keywords and associated information.",
                         key="UNKNOWN_TAGS")

    # Extract explanation and guess
    explanation_pattern = re.compile(rf"{words['explanation_lang']}([^\n]*)", re.IGNORECASE)

    content_prefix = words['guess_lang']
    if isinstance(player, WordCritic):
        content_prefix = words['agreement_lang']
    content_pattern = re.compile(rf"{content_prefix}([^\n]*)", re.IGNORECASE)

    explanation_match = explanation_pattern.search(response)
    content_match = content_pattern.findall(response)

    if len(content_match) != 1:
        raise ParseError(f"The response should contain the '{content_prefix}' keyword exactly once.",
                         key="MORE_THAN_ONE_GUESS")

    content = content_match[0].strip().lower()
    explanation = explanation_match.group(1).strip() if explanation_match else ""

    return content, explanation


def validate_guess(guess: str, words: Dict):
    """Validate guess format and content"""
    if not guess.isalpha() or " " in guess:
        raise RuleViolationError("The guess should be a single word and should only contain letters.",
                                 key="INVALID_FORMAT")

    if len(guess) != words["max_word_length"]:
        raise RuleViolationError(f"The length of the guessed word is not {words['max_word_length']}.",
                                 key="INVALID_WORD_LENGTH")

    if guess not in words["official_words_list"]:
        raise UnknownFiveLetterWordError(f"The guessed word is not a valid word for this game.",
                                         key="NOT_VALID_WORD_FOR_GAME")


def validate_agreement(agreement: str, words: Dict):
    """Validate critic agreement"""
    if not agreement.isalpha() or " " in agreement:
        raise RuleViolationError("The agreement should be a single word and should only contain letters.",
                                 key="INVALID_FORMAT")

    if agreement not in words["agreement_match_keywords_lang"]:
        raise RuleViolationError(f"The agreement should be one of the following: "
                                 f"{words['agreement_match_keywords_lang']}",
                                 key="NOT_VALID_CRITIC_WORD")


@dataclass
class WordleGameState:
    # Wordle
    target_word: str
    words: Dict[str, str]
    max_rounds: int
    max_retry_per_error: int
    guesser_initial_prompt: str
    success: bool = False
    failure: bool = False
    aborted: bool = False
    valid_response: bool = False
    reprompt_attempts: int = 0
    error: Optional[ResponseError] = None
    current_guess: str = None
    current_explanation: str = None
    guess_feedback: str = None
    # WordleWithClue
    guesser_initial_clue: Optional[str] = None
    # WordleWithCritic
    critic_initial_prompt: Optional[str] = None
    awaiting_critic: Optional[bool] = None
    commit_guess: Optional[bool] = None
    current_agreement: Optional[str] = None
    current_agreement_explanation: Optional[str] = None


# interaction keys to log structured data for scoring or logging
GUESSER_GUESSES = "Guesser Guesses"
GUESSER_EXPLANATIONS = "Guesser Explanations"
GUESSER_FEEDBACKS = "Guesser Feedbacks"


class Wordle(DialogueGameMaster):
    """Basic Wordle game without clue or critic"""

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)
        # game specific logging
        self.request_counts: int = 0
        self.parsed_request_counts: int = 0
        self.violated_request_counts: int = 0
        self.guesser_guesses: List[str] = []
        self.guesser_explanations: List[str] = []
        self.guesser_feedbacks: List[str] = []

    def _on_setup(self, **game_instance):
        self.state = WordleGameState(
            target_word=game_instance["target_word"].strip().lower(),
            words=self.experiment["lang_keywords"],
            max_rounds=self.experiment["common_config"]["n_turns"],
            # NOT_VALID_WORD_FOR_GAME is the only entry in the dict; we only handle this case in the game for now
            max_retry_per_error=self.experiment["common_config"]["max_retry_per_error"]["NOT_VALID_WORD_FOR_GAME"],
            guesser_initial_prompt=self.experiment["guesser_prompt"]
        )
        self.guess_validator = GuessValidator(self.state.target_word)
        self.formatter = ResponseFormatter(self.state.words)
        self._add_players()

    def _add_players(self):
        self.guesser = WordGuesser(self.player_models[0], self.state.words, self.state.target_word)
        self.add_player(self.guesser, initial_context=self.state.guesser_initial_prompt)

    def _does_game_proceed(self):
        return not (self.state.success or self.state.failure or self.state.aborted)

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        self.request_counts += 1
        try:
            # Parse response of the only player: the guesser
            guess, explanation = parse_response(player, utterance, self.state.words)
            self.state.current_guess = guess
            self.state.current_explanation = explanation
            # Validate guess
            validate_guess(guess, self.state.words)
            self.parsed_request_counts += 1
            # Reset re-prompting states
            self.state.valid_response = True
            self.state.reprompt_attempts = 0
            self.state.error = None
            return True
        except (ParseError, RuleViolationError) as e:
            if isinstance(e, UnknownFiveLetterWordError):
                self.parsed_request_counts += 1  # in this case still count toward parsed requests, but re-prompt
            else:
                self.violated_request_counts += 1
            self.state.valid_response = False
            self.state.error = e
            self.log_to_self("metadata", f"Error: {e.reason}")
            return False

    def _should_pass_turn(self):
        if not self.state.valid_response:
            if isinstance(self.state.error, UnknownFiveLetterWordError):
                # perform re-prompting up to N times
                self.state.reprompt_attempts += 1
                if self.state.reprompt_attempts > self.state.max_retry_per_error:
                    self.log_to_self("invalid format", "game_result = ABORT")
                    self.state.aborted = True
                else:  # adjust re-prompt text
                    self.set_context_for(self.guesser, self.formatter.to_gm_reprompt_for_guesser(self.state.error))
            else:
                self.log_to_self("invalid format", "game_result = ABORT")
                self.state.aborted = True
            return False
        return True

    def _start_next_round(self) -> bool:
        return self.state.valid_response

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        self.state.guess_feedback = self.guess_validator.validate(self.state.current_guess)
        self.log_to_self("metadata", self.formatter.to_gm_turn_stats(self.get_turn_stats()))
        self.guesser_feedbacks.append(self.state.guess_feedback)
        self.guesser_guesses.append(self.state.current_guess)
        self.guesser_explanations.append(self.state.current_explanation)
        # Check terminal conditions
        if self.state.target_word == self.state.current_guess:
            self.log_to_self("correct guess", "game_result = WIN")
            self.state.success = True
        elif self.current_round + 1 >= self.state.max_rounds:  # zero-based rounds
            self.log_to_self("max rounds played", "game_result = LOSS")
            self.state.failure = True
        else:  # Provide word validation feedback to guesser for next round
            content = self.formatter.to_gm_response_for_guesser(self.state.guess_feedback)
            self.set_context_for(self.guesser, content)

    def get_turn_stats(self):
        return {
            "attempts": self.current_round + 1,
            "target_word": self.state.target_word,
            "guess": self.state.current_guess,
            "guess_feedback": self.state.guess_feedback
        }

    def compute_response_score(self, response, context):
        return 1 if self.state.success else 0

    def compute_episode_score(self):
        if self.state.success:
            return 100 / self.current_round
        return 0

    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, int(self.state.aborted))
        self.log_key(METRIC_LOSE, int(self.state.failure))
        self.log_key(METRIC_SUCCESS, int(self.state.success))

        self.log_key(METRIC_REQUEST_COUNT, self.request_counts)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.parsed_request_counts)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_counts)

        self.log_key(GUESSER_GUESSES, self.guesser_guesses)
        self.log_key(GUESSER_FEEDBACKS, self.guesser_feedbacks)
        self.log_key(GUESSER_EXPLANATIONS, self.guesser_explanations)


class WordleWithClue(Wordle):
    """Wordle game with target word clue"""

    def _on_setup(self, **game_instance):
        super()._on_setup(**game_instance)  # this calls _add_players()
        # Set clue as initial context; will be appended to initial_prompt on the Player's first turn
        self.state.guesser_initial_clue = game_instance["target_word_clue"].strip()

    def _add_players(self):
        self.guesser = WordGuesser(self.player_models[0], self.state.words, self.state.target_word)
        self.add_player(self.guesser, initial_prompt=self.state.guesser_initial_prompt)

    def _on_before_game(self):
        content = f"{self.state.words['clue_lang']} {self.state.guesser_initial_clue}"
        self.set_context_for(self.guesser, content)

    def get_turn_stats(self):
        return {
            "attempts": self.current_round + 1,
            "target_word": self.state.target_word,
            "target_word_clue": self.state.guesser_initial_clue,
            "guess": self.state.current_guess,
            "guess_feedback": self.state.guess_feedback
        }


GUESSER_GUESSES_COMMITTED = "Guesser Guesses Committed"
CRITIC_JUDGEMENTS = "Critic Judgements"


class WordleWithCritic(WordleWithClue):
    """
    Wordle game with clue and critic player.

    In this variant a critic provides intermediate feedback to the guesser.

    Hence, the rounds have 3 turns: [guesser, critic, guesser].

    Only then the color feedback is given and it's the guessers turn again.
    """

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)
        self.critics_judgements: List[str] = []

    def _on_setup(self, **game_instance):
        super()._on_setup(**game_instance)  # this calls _add_players()
        self.state.critic_initial_prompt = self.experiment["guesser_critic_prompt"]
        self.state.awaiting_critic = True  # whether the critic has already been consulted
        self.state.commit_guess = False  # guesser has an initial and a final guess (to commit == end the round)

    def _add_players(self):
        guesser_model = self.player_models[0]
        self.guesser = ReflectingWordGuesser(guesser_model, self.state.words, self.state.target_word)
        self.add_player(self.guesser, initial_prompt=self.state.guesser_initial_prompt)

        critic_model = self.player_models[1] if len(self.player_models) > 1 else guesser_model
        self.critic = WordCritic(critic_model, self.state.words)
        # set initial prompt from self.experiment because self.state.critic_initial_prompt is not yet set
        self.add_player(self.critic, initial_prompt=self.experiment["guesser_critic_prompt"])

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        if player == self.guesser:
            return super()._validate_player_response(player, utterance)
        if player == self.critic:
            self.request_counts += 1
            try:
                agreement, explanation = parse_response(player, utterance, self.state.words)
                self.state.current_agreement = agreement
                self.state.current_agreement_explanation = explanation
                validate_agreement(agreement, self.state.words)
                self.parsed_request_counts += 1
                return True
            except (ParseError, RuleViolationError) as e:
                # Immediately abort when critic fails to produce a valid response
                self.violated_request_counts += 1
                self.state.aborted = True
                self.log_to_self("metadata", e.reason)
                self.log_to_self("invalid format", "game_result = ABORT")
                return False

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        # Only provide game feedback after critic interaction is complete
        if player == self.guesser:
            if self.state.awaiting_critic:  # little state machine
                self.guesser_guesses.append(self.state.current_guess)
                content = self.formatter.to_gm_response_for_critic(self.state.guesser_initial_clue,
                                                                   self.state.current_explanation,
                                                                   self.state.current_guess,
                                                                   is_initial_context=self.current_round == 0)
                self.set_context_for(self.critic, content)
            else:  # another turn with the gm
                self.state.commit_guess = True
                super()._on_valid_player_response(player, parsed_response)
        if player == self.critic:
            self.critics_judgements.append(self.state.current_agreement)
            content = self.formatter.to_gm_response_for_guesser_with_critic(self.state.guesser_initial_clue,
                                                                            self.state.current_agreement_explanation,
                                                                            self.state.current_agreement)
            self.set_context_for(self.guesser, content)
            self.state.awaiting_critic = False

    def _start_next_round(self):
        return self.state.commit_guess  # this requires self.valid_response = True

    def _on_before_round(self):
        self.state.awaiting_critic = True
        self.state.commit_guess = False

    def _should_pass_turn(self):
        if self.current_player == self.critic:  # critic always passes turn
            return True
        if not super()._should_pass_turn():  # possible re-prompting of guesser
            return False
        return self.state.awaiting_critic  # pass turn only to get critic response

    def _does_game_proceed(self):
        # Proceed if waiting for critic response (skip success/failure conditions)
        # However the game might be aborted, when the guesser or critic fails to produce a valid response
        if self.state.awaiting_critic and not self.state.aborted:
            return True
        return super()._does_game_proceed()

    def _on_after_game(self):
        super()._on_after_game()
        # Note: For wordle with critic guesses has twice as many entries [initial_1, committed_1, ...]
        guesses_committed = [guess for guess in self.guesser_guesses[1::2]]
        guesses = [guess for guess in self.guesser_guesses[::2]]
        self.log_key(GUESSER_GUESSES, guesses)
        self.log_key(GUESSER_GUESSES_COMMITTED, guesses_committed)
        self.log_key(CRITIC_JUDGEMENTS, self.critics_judgements)


SPEED_SCORES = {
    1: 100,
    2: 100,
    3: 100,
    4: 50,
    5: 30,
    6: 20
}
GUESS_REPETITIONS = "Guess Repetitions"
CLOSENESS_SCORE = "Closeness Score"  # turn metric
STRATEGY_SCORE = "Strategy Score"  # turn metric


class WordleScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def score_turns(self, episode_interactions: Dict) -> None:
        guesser_feedbacks = episode_interactions[GUESSER_FEEDBACKS]

        if not guesser_feedbacks:
            self.log_turn_score(0, CLOSENESS_SCORE, np.nan)
            self.log_turn_score(0, STRATEGY_SCORE, np.nan)
            return

        closeness_scores = turns_closeness(guesser_feedbacks)
        for idx, score in enumerate(closeness_scores):
            self.log_turn_score(idx + 1, CLOSENESS_SCORE, score)

        strategy_scores = turns_strategy(guesser_feedbacks, is_aborted=episode_interactions[METRIC_ABORTED])
        for idx, score in enumerate(strategy_scores):
            self.log_turn_score(idx + 1, STRATEGY_SCORE, score)

    def compute_speed(self, episode_interactions):
        """
        Rank is computed based on the number of turns taken to guess the word.
        The lesser the number of turns, the higher the speed
        """
        num_rounds: int = len(episode_interactions["turns"])
        if self.game_name == "wordle":
            return SPEED_SCORES[num_rounds]
        return round(100 / num_rounds, 2)

    def compute_guess_repetition(self, episode_interactions):
        guesses = episode_interactions[GUESSER_GUESSES]
        return len(guesses) - len(set(guesses))

    def log_main_score(self, episode_interactions: Dict):
        if episode_interactions[METRIC_ABORTED]:
            self.log_episode_score(BENCH_SCORE, np.nan)
            self.log_episode_score(GUESS_REPETITIONS, np.nan)
        elif episode_interactions[METRIC_LOSE]:
            self.log_episode_score(BENCH_SCORE, 0)
            self.log_episode_score(GUESS_REPETITIONS, self.compute_guess_repetition(episode_interactions))
        elif episode_interactions[METRIC_SUCCESS]:
            self.log_episode_score(BENCH_SCORE, self.compute_speed(episode_interactions))
            self.log_episode_score(GUESS_REPETITIONS, self.compute_guess_repetition(episode_interactions))
        else:
            raise RuntimeError("Cannot compute BENCH_SCORE because neither aborted, lose nor success is set.")


REPETITION_ON_AGREEMENT = "Repetition-Guesser-On-Critic-Agreement"
ADJUSTMENT_ON_AGREEMENT = "Non-Repetition-Guesser-On-Critic-Agreement"
REPETITION_ON_DISAGREEMENT = "Repetition-Guesser-On-Critic-Disagreement"
ADJUSTMENT_ON_DISAGREEMENT = "Non-Repetition-Guesser-On-Critic-Disagreement"
CHANGE_OF_OPINION = "Change-Of-Opinion"  # turn metric


class WordleWithCriticScorer(WordleScorer):

    def change_of_opinion(self, guesses, guesses_committed, critic_feedbacks):
        """
        Change of opinion is computed based on the number of times the opinion is changed after the critic's opinion
        """
        total_yes = 0
        total_no = 0

        use_same_guess_yes = 0
        use_diff_guess_yes = 0
        use_same_guess_no = 0
        use_diff_guess_no = 0
        overall_change = []

        # Note: zip will truncate to the shortest list e.g. if a critic feedback is missing
        for guess, guess_mod, critic_agreement in zip(guesses, guesses_committed, critic_feedbacks):
            if guess != guess_mod:
                overall_change.append(1)
                if critic_agreement == "yes":
                    total_yes += 1
                    use_diff_guess_yes += 1
                else:
                    total_no += 1
                    use_diff_guess_no += 1
            else:
                overall_change.append(0)
                if critic_agreement == "yes":
                    total_yes += 1
                    use_same_guess_yes += 1
                else:
                    total_no += 1
                    use_same_guess_no += 1

        return {"total_yes": total_yes, "total_no": total_no,
                "use_same_guess_yes": use_same_guess_yes,
                "use_diff_guess_yes": use_diff_guess_yes,
                "use_same_guess_no": use_same_guess_no,
                "use_diff_guess_no": use_diff_guess_no,
                "overall_change": overall_change}

    def compute_guess_repetition(self, episode_interactions):
        guesses = episode_interactions[GUESSER_GUESSES_COMMITTED]
        return len(guesses) - len(set(guesses))

    def log_main_score(self, episode_interactions: Dict):
        super().log_main_score(episode_interactions)

        guesses = episode_interactions[GUESSER_GUESSES]
        guesses_committed = episode_interactions[GUESSER_GUESSES_COMMITTED]
        critic_judgements = episode_interactions[CRITIC_JUDGEMENTS]
        results = self.change_of_opinion(guesses, guesses_committed, critic_judgements)

        repetition_agreement = np.nan
        repetition_disagreement = np.nan
        adjustment_agreement = np.nan
        adjustment_disagreement = np.nan
        if results["overall_change"]:
            for idx, change in enumerate(results["overall_change"]):
                self.log_turn_score(idx + 1, CHANGE_OF_OPINION, change)

            total_agreements = results["total_yes"]
            if total_agreements > 0:
                repetition_agreement = round(results["use_same_guess_yes"] / total_agreements, 2)
                adjustment_agreement = round(results["use_diff_guess_yes"] / total_agreements, 2)
            else:
                repetition_agreement = 0
                adjustment_agreement = 0
            total_disagreements = results["total_no"]
            if total_disagreements > 0:
                repetition_disagreement = round(results["use_same_guess_no"] / total_disagreements, 2)
                adjustment_disagreement = round(results["use_diff_guess_no"] / total_disagreements, 2)
            else:
                repetition_disagreement = 0
                adjustment_disagreement = 0
        self.log_episode_score(REPETITION_ON_AGREEMENT, repetition_agreement)
        self.log_episode_score(ADJUSTMENT_ON_AGREEMENT, adjustment_agreement)
        self.log_episode_score(REPETITION_ON_DISAGREEMENT, repetition_disagreement)
        self.log_episode_score(ADJUSTMENT_ON_DISAGREEMENT, adjustment_disagreement)


class WordleGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        if self.game_name == "wordle_withcritic":
            return WordleWithCritic(self.game_name, self.game_path, experiment, player_models)
        elif self.game_name == "wordle_withclue":
            return WordleWithClue(self.game_name, self.game_path, experiment, player_models)
        else:
            return Wordle(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        if self.game_name == "wordle_withcritic":
            return WordleWithCriticScorer(self.game_name, experiment, game_instance)
        return WordleScorer(self.game_name, experiment, game_instance)
