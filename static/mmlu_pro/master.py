import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from clemcore import backends
from clemcore.backends import Model
from clemcore.clemgame import Player, GameBenchmark, GameMaster, ParseError
from clemcore.clemgame.legacy.scorer import GameScorer
from clemcore.clemgame.legacy.master import DialogueGameMaster
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_LOSE, METRIC_SUCCESS, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_COUNT_VIOLATED, BENCH_SCORE
from jinja2 import Template


class Answerer(Player):
    def __init__(self, model: Model, target: str, choices: List[str]):
        super().__init__(model)
        self.target = target
        self.choices = choices

    def _custom_response(self, context: Dict) -> str:
        r = random.random()  # float from 0 to 1
        if r < 1 / 3:  # correct answer (SUCCESS)
            return f"The answer is ({self.target})"
        if r < 2 / 3:  # wrong answer (LOSE)
            possible_choices = self.choices.copy()
            possible_choices.remove(self.target)
            return f"The answer is ({random.choice(possible_choices)})"
        return "I don't know"  # ABORT


def parse_response(response: str, choices: List, regex_pattern) -> str:
    parsed_response = re.findall(regex_pattern, response)
    if len(parsed_response) == 0:
        raise ParseError("Response doesn't contain any answer, but the expected format give one.")
    if len(parsed_response) > 1:
        raise ParseError(
            "Response contains more than a one answer, but the expected format is to only give one.")
    parsed_response = parsed_response[0]
    if parsed_response not in choices:
        raise ParseError(f"Response contains none of the valid answer letters {choices}, "
                         f"but '{parsed_response}'.")
    return parsed_response


@dataclass
class GameState:
    target: str
    initial_prompt: str
    choices: List[str]
    regex_pattern: str
    parsed_response: Optional[str] = None
    success: bool = False  # When response format is adhered to and exact match is achieved
    failure: bool = False  # When response format is adhered to, but no exact match
    aborted: bool = False  # When response format is violated


class MMLUProGameMaster(DialogueGameMaster):
    def _on_setup(self, **instance):
        # Setup game state (arguments in same order as above)
        initial_prompt = Template(self.experiment["initial_prompt"]).render(input=instance["input"])
        self.state = GameState(instance["target"],
                               initial_prompt,
                               self.experiment["choices"],
                               self.experiment["regex_pattern"])  # pattern is case-sensitive!

        # Setup player
        self.answerer = Answerer(self.player_models[0], self.state.target, self.state.choices)
        self.add_player(self.answerer, initial_context=initial_prompt)

        # Setup game specific logging
        self.request_counts: int = 0
        self.parsed_request_counts: int = 0
        self.violated_request_counts: int = 0

    def _does_game_proceed(self):
        return not (self.state.aborted or self.state.failure or self.state.success)

    def _validate_player_response(self, player: Player, response: str) -> bool:
        self.request_counts += 1
        try:
            parsed_response = parse_response(response, self.state.choices, self.state.regex_pattern)
            self.parsed_request_counts += 1
            self.state.parsed_response = parsed_response
            self.log_to_self("parsed", parsed_response)
            return True
        except ParseError as e:
            self.violated_request_counts += 1
            self.log_to_self("metadata", f"ParseError: {e.reason}")
            self.state.aborted = True
            self.log_to_self("invalid format", "game_result = ABORT")
        return False

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        self.log_to_self("target", self.state.target)
        if self.state.parsed_response == self.state.target:  # case-sensitive comparison!
            self.log_to_self("correct answer", "game_result = WIN")
            self.state.success = True
        else:
            self.log_to_self("wrong answer", "game_result = LOSE")
            self.state.failure = True

    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, int(self.state.aborted))
        self.log_key(METRIC_LOSE, int(self.state.failure))
        self.log_key(METRIC_SUCCESS, int(self.state.success))

        self.log_key(METRIC_REQUEST_COUNT, self.request_counts)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.parsed_request_counts)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_counts)


class MMLUProGameScorer(GameScorer):

    def score_turns(self, episode_interactions: Dict) -> None:
        pass  # single-turn

    def log_main_score(self, episode_interactions: Dict):
        if episode_interactions[METRIC_ABORTED]:
            self.log_episode_score(BENCH_SCORE, np.nan)
            return
        accuracy = 100 if episode_interactions[METRIC_SUCCESS] else 0
        self.log_episode_score(BENCH_SCORE, accuracy)


class MMLUProGameBenchmark(GameBenchmark):

    def create_game_master(self, experiment: Dict, player_models: List[backends.Model]) -> GameMaster:
        return MMLUProGameMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return MMLUProGameScorer(self.game_name, experiment, game_instance)
