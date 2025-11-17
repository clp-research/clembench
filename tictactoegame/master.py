from typing import Dict, List, Optional, Tuple

import numpy as np
from clemcore.backends import Model
from clemcore.clemgame import (
    ActionSpace,
    EnvGameMaster,
    GameBenchmark,
    GameScorer,
    GameSpec,
    Observation,
)
from clemcore.clemgame.metrics import BENCH_SCORE, METRIC_ABORTED, METRIC_SUCCESS

from game_environment import (
    TicTacToeAction,
    TicTacToeEnvironment,
    TicTacToePlayer,
)


class TicTacToeGame(EnvGameMaster):

    def __init__(
        self,
        game_spec: GameSpec,
        experiment: Dict,
        player_models: List[Model],
    ):
        super().__init__(game_spec, experiment, player_models)

    def _on_setup(self, **game_instance):
        self.game_environment = TicTacToeEnvironment(game_instance)

        for player in self.player_models:
            self.add_player(TicTacToePlayer(player))

        self.game_environment.reset()

    def _response_valid(self, player: TicTacToePlayer, utterance: str) -> bool:
        try:
            row, col = map(int, utterance.strip().split())
            return 0 <= row < 3 and 0 <= col < 3
        except (ValueError, IndexError):
            return False

    def _parse_action_from_response(self, response: str) -> TicTacToeAction:
        row, col = map(int, response.strip().split())
        action: TicTacToeAction = {
            "action_type": "default",
            "row": row,
            "col": col,
        }
        return action


class TicTacToeGameScorer(GameScorer):

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_round_score(self, round_idx, round_events):
        self.log_round_score(
            round_idx,
            "Reward",
            round_events[-1]["action"]["content"],
        )

    def compute_episode_scores(self, interactions: Dict) -> None:
        aborted = interactions.get(METRIC_ABORTED, False)
        success = interactions.get(METRIC_SUCCESS, False)

        if aborted:
            bench_score = np.nan
        else:
            bench_score = 100.0 if success else 0.0

        self.log_episode_score(BENCH_SCORE, bench_score)


class TicTacToeGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> TicTacToeGame:
        return TicTacToeGame(
            self.game_spec,
            experiment,
            player_models,
        )

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return TicTacToeGameScorer(self.game_name, experiment, game_instance)
