import random
import re
from typing import Dict, List

import numpy as np
from clemcore.backends import Model
from clemcore.clemgame import EnvGameMaster, GameBenchmark, GameScorer, GameSpec, Player
from clemcore.clemgame.metrics import (
    BENCH_SCORE,
    METRIC_ABORTED,
    METRIC_LOSE,
    METRIC_SUCCESS,
)

from game_environment import PortalAction, PortalGameEnvironment


class PortalPlayer(Player):

    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, context: Dict) -> str:
        direction = random.choice(["n", "s", "e", "w"])
        return f"DIRECTION: {direction}"


class PortalGame(EnvGameMaster):

    def __init__(
        self,
        game_spec: GameSpec,
        experiment: Dict,
        player_models: List[Model],
    ):
        super().__init__(game_spec, experiment, player_models)

    def _on_setup(self, **game_instance):
        self.game_environment = PortalGameEnvironment(config=game_instance)

        for player in self.player_models:
            self.add_player(PortalPlayer(player))

        self.game_environment.reset()

    def _response_valid(self, player: Player, utterance: str) -> bool:
        try:
            # look for 'DIRECTION:' and extract the last non-whitespace char after it

            match = re.search(
                r"DIRECTION:\s*([nsewNSEW])", utterance, re.IGNORECASE | re.DOTALL
            )
            if match:
                direction = match.group(1).lower()
            else:
                # fallback: last non-whitespace char
                direction = utterance.strip()[-1].lower()
            return direction in ["n", "s", "e", "w"]
        except Exception:
            return False

    def _parse_action_from_response(self, response: str) -> PortalAction:
        match = re.search(
            r"DIRECTION:\s*([nsewNSEW])", response, re.IGNORECASE | re.DOTALL
        )
        if match:
            direction = match.group(1).lower()
        else:
            direction = response.strip()[-1].lower()
        action: PortalAction = {
            "action_type": "default",
            "direction": direction,
        }
        return action


class PortalGameScorer(GameScorer):

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

        shortest_path = self.game_instance["shortest_path"]
        moves = sum(interactions["Request Count"])
        efficiency = shortest_path / moves

        if aborted:
            bench_score = np.nan
        else:
            bench_score = 100.0

            # the more efficient the player is, the higher the bench score
            bench_score = bench_score - (50 * (1 - efficiency))

        self.log_episode_score("Efficiency", efficiency)
        self.log_episode_score(BENCH_SCORE, bench_score)


class PortalGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> EnvGameMaster:
        return PortalGame(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return PortalGameScorer(self.game_name, experiment, game_instance)
