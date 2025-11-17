from typing import Dict, List

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
from clemcore.utils.logger import format_json, setup_logger

from hellogame.game_environment import HelloGameAction, HelloGameEnvironment

logger = setup_logger(__name__)


class HelloGamePlayer(Player):
    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, context: Dict) -> str:
        return "Hello, how are you?"


class HelloGame(EnvGameMaster):
    """This class implements a greeting game in which player A
    is greeting another player with a target name.

    This version uses the GameEnvironment approach for state management.
    """

    def __init__(
        self,
        game_spec: GameSpec,
        experiment: Dict,
        player_models: List[Model],
    ):
        logger.info(f"[_init] Initializing HelloGame GameMaster with spec={game_spec}")
        logger.debug(f"[_init] Experiment parameters: {experiment}")

        super().__init__(game_spec, experiment, player_models)
        logger.info("[_init] HelloGame initialization complete")

    def _on_setup(self, **game_instance):
        """
        Called during game setup. Configure game parameters and initialize players.

        Args:
            game_instance: Game instance parameters from instances.json
        """
        logger.info("[_on_setup] Setting up HelloGame")

        logger.debug(f"[_on_setup] Game instance: {game_instance}")
        self.game_environment = HelloGameEnvironment(config=game_instance)

        self.greeter = HelloGamePlayer(self.player_models[0])
        logger.debug(f"[_on_setup] Created players: greeter={self.greeter}")

        self.add_player(self.greeter)
        logger.info(f"[_on_setup] Added players: greeter={self.greeter.name}")

        self.game_environment.reset()

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        """
        Validate the player's response.

        Will be called in GameMaster.play(), after the response is received, and before it is parsed.
        """
        logger.debug(
            f"[_validate_player_response] Validating response from {player.name}: {utterance}"
        )

        if player == self.greeter:
            is_valid = bool(utterance.strip())
            logger.debug(
                f"[_validate_player_response] Greeter response validation result: {is_valid}"
            )
            return is_valid

        logger.debug(
            "[_validate_player_response] Greeted player response is always valid"
        )
        return True

    def compute_turn_score(self):
        """
        Compute a score for the player's response based on the environment state.
        """
        score = 1.0 if self.game_environment.state["success"] else 0.0
        return score

    def compute_episode_score(self):
        """
        Compute the overall episode score.
        In HelloGame, this is the same as the response score.
        """
        logger.info("[_compute_episode_score] Computing episode score")

        score = 1.0 if self.game_environment.state["success"] else 0.0
        logger.info(f"[_compute_episode_score] Episode score: {score}")
        return score

    def _response_valid(
        self, player: Player, response: str
    ) -> bool:
        """Check if player response is in expected format."""
        return bool(response.strip())

    def _parse_action_from_response(self, response: str) -> HelloGameAction:
        """Parse action from player response."""
        return {"action_type": "verbal_response", "message": response}


class HelloGameScorer(GameScorer):
    """Scorer for the Hello game."""

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.target_name = game_instance["target_name"]
        logger.debug(f"HelloGameScorer initialized for target: {self.target_name}")

    def compute_episode_scores(self, interactions: Dict) -> None:
        """Compute episode-level scores for the Hello game.

        Args:
            interactions: Dict containing the logged episode's interactions.
        """
        aborted = interactions.get(METRIC_ABORTED, False)
        success = interactions.get(METRIC_SUCCESS, False)

        if aborted:
            bench_score = 0.0
        else:
            bench_score = 100.0 if success else 0.0

        self.log_episode_score(BENCH_SCORE, bench_score)


class HelloGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)
        logger.info(f"HelloGameBenchmark initialized with game spec: {game_spec}")

    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> EnvGameMaster:
        logger.info(f"Creating HelloGame master with experiment: {experiment}")
        logger.debug(
            f"Player models: {[model.__class__.__name__ for model in player_models]}"
        )
        return HelloGame(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        """
        Create a scorer for the Hello Game.

        Args:
            experiment: Experiment configuration dictionary
            game_instance: Game instance dictionary with specific parameters

        Returns:
            A HelloGameScorer instance
        """
        logger.info(f"Creating HelloGameScorer with experiment: {experiment}")
        logger.debug(f"Game instance for scorer: \n{format_json(game_instance)}")
        return HelloGameScorer(self.game_name, experiment, game_instance)
