import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from clemcore import backends
from clemcore.backends import Model
from clemcore.clemgame import Player, DialogueGameMaster, GameBenchmark, GameMaster, GameScorer, ParseError, \
    RuleViolationError
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_LOSE, METRIC_SUCCESS, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_COUNT_VIOLATED


def to_response_format(target: Dict):
    return "\n".join([f"{emotion}: {score}" for emotion, score in target.items()])


class Answerer(Player):
    def __init__(self, model: Model, target: Dict):
        super().__init__(model)
        self.target = target

    def _custom_response(self, context: Dict) -> str:
        r = random.random()  # float from 0 to 1
        if r < 1 / 3:  # correct response (SUCCESS)
            return to_response_format(self.target)
        if r < 2 / 3:  # not exactly correct (LOSE)
            wrong_target = self.target.copy()
            key_to_scramble = random.choice(list(wrong_target.keys()))
            wrong_target[key_to_scramble] = random.randint(0, 10)
            return to_response_format(wrong_target)
        return "I don't know"  # ABORT


def parse_response(response: str, reference: Dict) -> Dict:
    parsed_response = dict(re.findall(r"(\w+):\s+(\d+)", response))
    if len(parsed_response) == 0:
        raise ParseError("Response doesn't contain any '<emotion>: <score>' pairs, "
                         "but this would be the expected format")
    if len(parsed_response) != len(reference):
        raise ParseError(f"Number of scored emotion is '{len(parsed_response)}', "
                         f"but the expected format lists '{len(reference)}'")
    missing_emotions = [emotion for emotion in reference if emotion not in parsed_response]
    if missing_emotions:
        raise ParseError(f"The mentioned emotions {', '.join(parsed_response.keys())}, "
                         f"do not match the ones in the expected format {', '.join(reference.keys())}")
    scores = [v for v in parsed_response.values()]
    if not all(str(v).isdigit() for v in scores):
        raise ParseError(f"Not all scores are positive integers, "
                         f"but the scores in the response are: {scores}")
    if not any(int(v) > 0 for v in scores):
        raise ParseError(f"At least one score in the response must be greater than 0,"
                         f"but the scores are: {scores}")
    parsed_response = {k: int(v) for k, v in parsed_response.items()}
    violated_scores = [f"{k}: {v}" for k, v in parsed_response.items() if not (0 <= v <= 10)]
    if violated_scores:
        raise ParseError(f"The scores must be given in range 0 to 10, "
                         f"but there are violations: {', '.join(violated_scores)}")
    return parsed_response


@dataclass
class GameState:
    target: Dict  # For EQBench, the target is a dict of emotion-score pairs
    initial_prompt: str  # For EQBench the input is the prompt (no templates)
    parsed_response: Optional[Dict] = None  # For EQBench, the response becomes a Dict after parsing
    success: bool = False  # When response format is adhered to and exact match is achieved
    failure: bool = False  # When response format is adhered to, but no exact match
    aborted: bool = False  # When response format is violated


class EQBenchGameMaster(DialogueGameMaster):
    def _on_setup(self, **instance):
        # Setup game state (arguments in same order as above)
        self.state = GameState(instance["target"], instance["input"])

        # Setup player
        self.answerer = Answerer(self.player_models[0], self.state.target)
        self.add_player(self.answerer, initial_context=self.state.initial_prompt)

        # Setup game specific logging
        self.request_counts: int = 0
        self.parsed_request_counts: int = 0
        self.violated_request_counts: int = 0

    def _does_game_proceed(self):
        return not (self.state.aborted or self.state.failure or self.state.success)

    def _validate_player_response(self, player: Player, response: str) -> bool:
        self.request_counts += 1
        try:
            parsed_response = parse_response(response, self.state.target)
            self.parsed_request_counts += 1
            self.state.parsed_response = parsed_response
            self.log_to_self("parsed", to_response_format(parsed_response))
            return True
        except ParseError as e:
            self.violated_request_counts += 1
            self.log_to_self("metadata", f"ParseError: {e.reason}")
            self.state.aborted = True
            self.log_to_self("invalid format", "game_result = ABORT")
        return False

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        self.log_to_self("target", to_response_format(self.state.target))
        if self.state.parsed_response == self.state.target:
            self.log_to_self("exact match", "game_result = WIN")
            self.state.success = True
        else:
            self.log_to_self("not exact match", "game_result = LOSE")
            self.state.failure = True

    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, int(self.state.aborted))
        self.log_key(METRIC_LOSE, int(self.state.failure))
        self.log_key(METRIC_SUCCESS, int(self.state.success))

        self.log_key(METRIC_REQUEST_COUNT, self.request_counts)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.parsed_request_counts)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_counts)


class EQBenchGameScorer(GameScorer):

    def score_turns(self, episode_interactions: Dict) -> None:
        pass

    def log_main_score(self, episode_interactions: Dict):
        pass  # see utils.py


class EQBenchGameBenchmark(GameBenchmark):

    def create_game_master(self, experiment: Dict, player_models: List[backends.Model]) -> GameMaster:
        return EQBenchGameMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return EQBenchGameScorer(self.game_name, experiment, game_instance)
