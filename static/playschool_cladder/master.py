import random
from dataclasses import dataclass
from typing import Dict, List

from clemcore import backends
from clemcore.backends import Model
from clemcore.clemgame import Player, DialogueGameMaster, GameBenchmark, GameMaster
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_LOSE, METRIC_SUCCESS, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_COUNT_VIOLATED
from jinja2 import Template


class Answerer(Player):
    def __init__(self, model: Model, target: str):
        super().__init__(model)
        self.target = target

    def _custom_response(self, context: Dict) -> str:
        if random.randint(0, 100) > 50:
            return self.target
        return "I don't know"


@dataclass
class GameState:
    target: str
    initial_prompt: str
    success: bool = False
    failure: bool = False
    aborted: bool = False


class CLadderGameMaster(DialogueGameMaster):
    def _on_setup(self, **instance):
        initial_prompt_template = Template(self.experiment["initial_prompt"])
        initial_prompt = initial_prompt_template.render(prompt=instance["input"])

        # Setup game state (arguments in same order as above)
        self.state = GameState(instance["target"].lower(), initial_prompt)

        # Setup player
        self.answerer = Answerer(self.player_models[0], self.state.target)
        self.add_player(self.answerer, initial_context=initial_prompt)

        # Setup game specific logging
        self.request_counts: int = 0
        self.parsed_request_counts: int = 0
        self.violated_request_counts: int = 0

    def _does_game_proceed(self):
        return not (self.state.aborted or self.state.failure or self.state.success)

    def _validate_player_response(self, player: Player, response: str) -> bool:
        self.request_counts += 1
        return True  # accept any response

    def _parse_response(self, player: Player, response: str) -> str:
        parsed_response = response.split(" ")[0].lower()
        self.log_to_self("metadata", f"Parsed: {parsed_response}")
        return parsed_response

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        self.parsed_request_counts += 1
        self.log_to_self("metadata", f"Target: {self.state.target}")
        if parsed_response.startswith(self.state.target):
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


class CLadderGameBenchmark(GameBenchmark):

    def create_game_master(self, experiment: Dict, player_models: List[backends.Model]) -> GameMaster:
        return CLadderGameMaster(self.game_spec, experiment, player_models)
