from dataclasses import dataclass
from typing import Dict, List
from clemcore import backends
from clemcore.backends import Model
from clemcore.clemgame import Player, DialogueGameMaster, GameBenchmark, GameMaster, GameScorer, ParseError
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_LOSE, METRIC_SUCCESS, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_COUNT_VIOLATED, BENCH_SCORE

import instructions_registry


class InstructionFollower(Player):
    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, context: Dict) -> str:
        return "I don't know"  # LOSE


@dataclass
class GameState:
    target: Dict
    initial_prompt: str
    success: bool = False  # When response format is adhered to and exact match is achieved
    failure: bool = False  # When response format is adhered to, but no exact match
    aborted: bool = False  # When response format is violated (not applicable to IFEval)


def is_successful(response: str, targets: Dict) -> bool:
    """This implements test_instruction_following_strict of the original work"""
    is_following_list = []
    for instruction_id, kwargs in targets.items():
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        instruction.build_description(**kwargs)
        is_following_list.append(response.strip()  # not empty string
                                 and instruction.check_following(response))
    return all(is_following_list)


class IFEvalGameMaster(DialogueGameMaster):
    def _on_setup(self, **instance):
        # Setup game state (arguments in same order as above)
        self.state = GameState(instance["target"], instance["input"])

        # Setup player
        self.follower = InstructionFollower(self.player_models[0])
        self.add_player(self.follower, initial_context=self.state.initial_prompt)

        # Setup game specific logging
        self.request_counts: int = 0
        self.parsed_request_counts: int = 0
        self.violated_request_counts: int = 0

    def _does_game_proceed(self):
        return not (self.state.aborted or self.state.failure or self.state.success)

    def _validate_player_response(self, player: Player, response: str) -> bool:
        self.request_counts += 1
        self.parsed_request_counts += 1
        return True  # Always valid

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        self.log_to_self("target", self.state.target)
        if is_successful(parsed_response, self.state.target):
            self.log_to_self("correct label", "game_result = WIN")
            self.state.success = True
        else:
            self.log_to_self("wrong label", "game_result = LOSE")
            self.state.failure = True

    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, int(self.state.aborted))
        self.log_key(METRIC_LOSE, int(self.state.failure))
        self.log_key(METRIC_SUCCESS, int(self.state.success))

        self.log_key(METRIC_REQUEST_COUNT, self.request_counts)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.parsed_request_counts)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_counts)


class IFEvalGameScorer(GameScorer):

    def score_turns(self, episode_interactions: Dict) -> None:
        pass  # single-turn

    def log_main_score(self, episode_interactions: Dict):
        accuracy = 1.0 if episode_interactions[METRIC_SUCCESS] else 0.0
        self.log_episode_score(BENCH_SCORE, accuracy)


class IFEvalGameBenchmark(GameBenchmark):

    def create_game_master(self, experiment: Dict, player_models: List[backends.Model]) -> GameMaster:
        return IFEvalGameMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return IFEvalGameScorer(self.game_name, experiment, game_instance)
