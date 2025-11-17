from typing import Dict, List

import numpy as np
from clemcore.backends import Model
from clemcore.clemgame import (
    ActionSpace,
    EnvGameMaster,
    GameBenchmark,
    GameScorer,
    GameSpec,
    Observation,
    Player,
)
from clemcore.clemgame.metrics import BENCH_SCORE, METRIC_ABORTED, METRIC_SUCCESS

from game_environment import SudokuAction, SudokuEnvironment, SudokuPlayer


class SudokuGame(EnvGameMaster):

    def __init__(
        self,
        game_spec: GameSpec,
        experiment: Dict,
        player_models: List[Model],
    ):
        super().__init__(game_spec, experiment, player_models)

    def _on_setup(self, **game_instance):
        self.game_environment = SudokuEnvironment(game_instance)

        for player in self.player_models:
            self.add_player(SudokuPlayer(player))

        self.game_environment.reset()

    def _response_valid(self, player: Player, response: str) -> bool:
        try:
            parts = response.strip().split()

            if len(parts) != 3:
                return False

            row, col, value = map(int, parts)

            if not (0 <= row < 9 and 0 <= col < 9):
                return False

            if not (1 <= value <= 9):
                return False

            return True
        except (ValueError, TypeError):
            return False

    def _parse_action_from_response(self, response: str) -> SudokuAction:
        row, col, value = map(int, response.strip().split())
        action: SudokuAction = {
            "action_type": "default",
            "row": row,
            "col": col,
            "value": value,
        }
        return action


class SudokuGameScorer(GameScorer):

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_episode_scores(self, interactions: Dict) -> None:
        aborted = interactions.get(METRIC_ABORTED, False)
        success = interactions.get(METRIC_SUCCESS, False)

        if aborted:
            bench_score = np.nan
        else:
            bench_score = 100.0 if success else 0.0

        self.log_episode_score(BENCH_SCORE, bench_score)


class SudokuGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> SudokuGame:
        return SudokuGame(
            self.game_spec,
            experiment,
            player_models,
        )

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return SudokuGameScorer(self.game_name, experiment, game_instance)
