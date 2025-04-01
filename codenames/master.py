from typing import Dict, List, Tuple, Union
from string import Template
import random, copy
import logging
import os

from clemcore import backends
from clemcore.clemgame import GameScorer, GameBenchmark, Player, DialogueGameMaster, GameSpec
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED

from constants import *
from validation_errors import *
from players import ClueGiver, Guesser
from board import CodenamesBoard
from scorer import CodenamesScorer

logger = logging.getLogger(__name__)


class CodenamesGame(DialogueGameMaster):
    """This class implements a codenames game in which player A
    is giving a clue for a set of target words on a board, 
    which player B has to guess from the given clue.
    """

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[backends.Model]):
        super().__init__(game_name, game_path, experiment, player_models)
        self.experiment = experiment
        self.opponent_difficulty: bool = experiment[OPPONENT_DIFFICULTY]

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance
        self.board: CodenamesBoard = CodenamesBoard(game_instance[ASSIGNMENTS][TEAM],
                                                    game_instance[ASSIGNMENTS][OPPONENT],
                                                    game_instance[ASSIGNMENTS][INNOCENT],
                                                    game_instance[ASSIGNMENTS][ASSASSIN],
                                                    game_instance[BOARD],
                                                    self.experiment["flags"])

        self.aborted: bool = False
        self.lost: bool = False
        self.assassin_won: bool = False
        self.invalid_response: bool = False
        self.request_count = 0
        self.parsed_request_count = 0
        self.violated_request_count = 0

        self.cluegiver: ClueGiver = ClueGiver(self.player_models[0], self.experiment["flags"])
        self.guesser: Guesser = Guesser(self.player_models[1], self.experiment["flags"])
        self.add_player(self.cluegiver)
        self.add_player(self.guesser)

    def _was_target(self, word: str):
        return word in self.cluegiver.targets

    def _get_cluegiver_prompt(self, initial=False) -> str:
        folder = "initial_prompts" if initial else "intermittent_prompts"
        path = f"{GAME_PATH}/resources/{folder}/prompt_cluegiver"
        prompt_cluegiver = self.load_template(path)

        team_words = ", ".join(self.board.get_hidden_words(TEAM))
        opponent_words = ", ".join(self.board.get_hidden_words(OPPONENT))
        innocent_words = ", ".join(self.board.get_hidden_words(INNOCENT))
        assassin_words = ", ".join(self.board.get_hidden_words(ASSASSIN))

        instance_prompt_cluegiver = Template(prompt_cluegiver).substitute(team_words=team_words,
                                                                          opponent_words=opponent_words,
                                                                          innocent_words=innocent_words,
                                                                          assassin_words=assassin_words)
        return instance_prompt_cluegiver

    def _get_guesser_initial_prompt(self):
        return self._get_guesser_prompt("initial_prompts")

    def _get_guesser_intermittent_prompt(self):
        return self._get_guesser_prompt("intermittent_prompts")

    def _get_guesser_prompt(self, folder) -> str:
        path = f"{GAME_PATH}/resources/{folder}/prompt_guesser"
        prompt_guesser = self.load_template(path)

        board = ", ".join(self.board.get_all_hidden_words())
        instance_prompt_guesser = Template(prompt_guesser).substitute(board=board,
                                                                      clue=self.cluegiver.clue,
                                                                      number=self.cluegiver.number_of_targets)
        return instance_prompt_guesser

    def _on_before_game(self):
        # add initial cluegiver prompt
        self.set_context_for(self.cluegiver, self._get_cluegiver_prompt(True))

    def _on_before_round(self):
        # print(self.board.get_current_board())
        self.log_to_self(Turn_logs.BOARD_STATUS, self.board.get_current_board())

        self.cluegiver.retries = 0
        self.guesser.retries = 0
        # initial = True if self.number_of_turns == 1 else False
        # add new cluegiver prompt
        # self.add_user_message(self.cluegiver, self._get_cluegiver_prompt(initial))

    def _does_game_proceed(self) -> bool:
        continue_game = True
        if self.invalid_response:
            self.aborted = True
            continue_game = False

        # for the base version, a check is needed whether all team words from one team are revealed or the assassin is revealed
        if self.board.has_team_won():
            self.lost = False
            self.assassin_won = False
            continue_game = False
            self.log_to_self("game end", "team has won")
        elif self.board.has_opponent_won():
            self.lost = True
            self.assassin_won = False
            continue_game = False
            self.log_to_self("game end", "opponent has won")
        elif self.board.has_team_won_through_assassin():
            self.lost = False
            self.assassin_won = True
            continue_game = False
            self.log_to_self("game end", "team has won through assassin")
        elif self.board.has_opponent_won_through_assassin():
            self.lost = True
            self.assassin_won = True
            continue_game = False
            self.log_to_self("game end", "opponent has won through assassin")

        if not continue_game:
            self._log_game_end()
            return False
        return True

    def _validate_player_response(self, player: Union[ClueGiver, Guesser], utterance: str) -> bool:
        self.request_count += 1
        self.invalid_response = False
        if player == self.cluegiver:
            try:
                player.validate_response(utterance, self.board.get_revealed_words(TEAM),
                                         self.board.get_all_hidden_words())
            except ValidationError as error:
                self.log_to_self(Turn_logs.VALIDATION_ERROR, error.get_dict())
                self.invalid_response = True
                self.violated_request_count += 1
                self.last_error_message = error.attributes["message"]
        else:
            try:
                player.validate_response(utterance, self.board.get_revealed_words(TEAM),
                                         self.board.get_all_hidden_words(), self.cluegiver.number_of_targets,
                                         self.cluegiver.clue)
            except ValidationError as error:
                self.log_to_self(Turn_logs.VALIDATION_ERROR, error.get_dict())
                self.invalid_response = True
                self.violated_request_count += 1
                self.last_error_message = error.attributes["message"]

        return not self.invalid_response

    def _parse_response(self, player: Union[ClueGiver, Guesser], utterance: str) -> str:
        self.parsed_request_count += 1
        if player == self.cluegiver:
            utterance = player.parse_response(utterance, self.board.get_all_hidden_words())
            self.log_to_self(Turn_logs.CLUE, player.clue)
            self.log_to_self(Turn_logs.TARGETS, player.targets)
            return utterance
        else:
            parsed_utterance = player.parse_response(utterance, self.board.get_all_hidden_words())
            self.log_to_self(Turn_logs.GUESSES, player.guesses)

            return parsed_utterance

    def _should_pass_turn(self):
        player: Union[ClueGiver, Guesser] = self.current_player
        if player.flags["REPROMPT ON ERROR"]:
            if player.retries < MAX_RETRIES:
                if self.invalid_response:
                    logger.debug("Reprompting...")
                    player.retries += 1
                    player.flags_engaged["REPROMPT ON ERROR"] += 1
                    self.set_context_for(player,
                                         f"Your answer did not follow the requested format: {self.last_error_message}")
                    return False
        return True

    def _on_valid_player_response(self, player: Union[ClueGiver, Guesser], parsed_response: str):
        if player == self.cluegiver:
            # score cluegiver precision
            for target in player.targets:
                assignment = self.board.get_word_assignment(target)
                self.log_to_self(Turn_logs.WORD_TARGETED, {"word": target, "assignment": assignment})

            # add response of cluegiver embedded in guesser prompt to guesser history
            if self.current_round == 0:
                self.set_context_for(self.guesser, self._get_guesser_initial_prompt())
            else:
                context = self.get_context_for(self.guesser)
                context["content"] += (f"\n{self._get_guesser_intermittent_prompt()}")

        else:
            evaluated_guesses = []
            # reveal guesses in order
            for guess in player.guesses:
                assignment = self.board.reveal_word(guess)
                if not assignment:
                    continue
                evaluated_guesses.append((guess, assignment))

                # TODO: add player messages here, whether word was revealed and correct, or incorrect and all other guesses were ignored
                self.log_to_self(Turn_logs.TEAM_REVEALED, {"word": guess, "assignment": assignment})
                if self._was_target(guess):
                    self.log_to_self(Turn_logs.TARGET_REVEALED, {"word": guess, "assignment": assignment})
                if not self.board.should_continue_after_revealing(guess):
                    self.log_to_self("turn end after", guess)
                    break

            guess_feedback = ""
            if evaluated_guesses[-1][1] == TEAM:
                if len(evaluated_guesses) >= 2:
                    guess_feedback = f"The words {', '.join([guess for guess, assignment in evaluated_guesses])} were guessed correctly. "
                else:
                    guess_feedback = f"The word {evaluated_guesses[0][0]} was guessed correctly. "
            else:
                correct_guesses = evaluated_guesses[0:-1]
                incorrect_guess = evaluated_guesses[-1]
                if len(correct_guesses) >= 2:
                    guess_feedback += (
                        f"The words {', '.join([guess for guess, assignment in correct_guesses])} were guessed correctly. ")
                elif len(correct_guesses) == 1:
                    guess_feedback += (f"The word {correct_guesses[0][0]} was guessed correctly. ")
                guess_feedback += (f"The word {incorrect_guess[0]} was guessed but is an {incorrect_guess[1]} word. ")

            cluegiver_guess_feedback = copy.copy(guess_feedback)
            cluegiver_guess_feedback += ("Your teammate's turn ended there.")

            guesser_guess_feedback = copy.copy(guess_feedback)
            guesser_guess_feedback += ("Your turn ended there.")

            # add guess feedback to guesser history
            self.set_context_for(self.guesser, guesser_guess_feedback)

            # add guesser utterance to cluegiver history and new cluegiver prompt
            self.set_context_for(self.cluegiver, f"{cluegiver_guess_feedback}\n{self._get_cluegiver_prompt(False)}")

    def _on_after_round(self):
        # let mock opponent reveal their cards
        if self._does_game_proceed():
            self._opponent_turn()

    def _opponent_turn(self):
        # reveal as many opponent cards as the opponent difficulty
        hidden_opponent_words = self.board.get_hidden_words(OPPONENT)
        opponent_words = random.sample(hidden_opponent_words, min(self.opponent_difficulty, len(hidden_opponent_words)))
        for word in opponent_words:
            assignment = self.board.reveal_word(word, OPPONENT)
            self.log_to_self(Turn_logs.OPPONENT_REVEALED, {"word": word, "assignment": assignment})

    def _log_game_end(self):
        # log everything that is needed for score calculation and game evaluation
        self.log_key(BOARD_END_STATUS, self.board.get_current_board())
        self.log_key(NUMBER_OF_TURNS, self.current_round + 1)
        self.log_key(METRIC_ABORTED, self.aborted)
        self.log_key(METRIC_LOSE, self.lost)
        self.log_key(GAME_ENDED_THROUGH_ASSASSIN, self.assassin_won)
        # METRIC_SUCCESS does not need to be logged as it is inferred from ABORTED and LOSE
        self.log_key(METRIC_REQUEST_COUNT, self.request_count)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.parsed_request_count)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_count)
        self.log_key("Cluegiver engaged flags", self.cluegiver.flags_engaged)
        self.log_key("Guesser engaged flags", self.guesser.flags_engaged)


class CodenamesGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)
        random.seed(SEED)

    def create_game_master(self, experiment: Dict, player_models: List[backends.Model]) -> DialogueGameMaster:
        return CodenamesGame(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment_config, game_instance) -> GameScorer:
        return CodenamesScorer(self.game_name, experiment_config, game_instance)
