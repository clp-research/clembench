import string
from typing import Dict, List

from clemcore.backends import Model, CustomResponseModel
from clemcore.clemgame import GameBenchmark, Player, DialogueGameMaster, GameSpec, GameRecorder
import logging

logger = logging.getLogger(__name__)


class Greeted(Player):

    def __init__(self, name, game_recorder: GameRecorder):
        super().__init__(CustomResponseModel(), game_recorder)
        self.name = name

    def _custom_response(self, messages):
        return f"{self.name}: Hi, thanks for having me!"


class Greeter(Player):

    def __init__(self, model: Model, game_recorder: GameRecorder):
        super().__init__(model, game_recorder)

    def _custom_response(self, messages):
        return "GREET: Hello Ted!"


class HelloGame(DialogueGameMaster):
    """This class implements a greeting game in which player A
    is greeting another player with a target name.
    """

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)
        self.language: int = experiment["language"]  # fetch experiment parameters here
        self.turns = []
        self.required_words = ["welcome", "hello"]
        self.missing_words = []
        self.success = True
        self.aborted = False

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance  # fetch game parameters here

        # Create the players
        self.greeted = Greeted(game_instance["target_name"], self)
        self.greeter = Greeter(self.player_models[0], self)

        # Add the players: these will be logged to the records interactions.json
        # Note: During game play the players will be called in the order added here
        self.add_player(self.greeter)
        self.add_player(self.greeted)

        self.required_words.append(self.greeted.name.lower())

    def _on_before_game(self):
        # Do something before the game start e.g. add the initial prompts to the message list for the players
        self.add_user_message(self.greeter, self.game_instance["prompt"])

    def _does_game_proceed(self):
        # Determine if the game should proceed. This is also called once initially.
        if len(self.turns) == 0:
            return True
        if self.aborted:
            self.log_to_self("invalid format", "abort game")
        if self.success:
            self.log_to_self("greeting successful", "end game")
        else:
            self.log_to_self("greeting failed", f"missing words=[{','.join(self.missing_words)}]")
        return False

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        # Check responses for specific players
        if player == self.greeter:
            # Check rule: utterance starts with key word
            if not utterance.startswith("GREET:"):
                self.aborted = True
                self.success = False
                return True
            # Check rule: required words are included
            utterance = utterance.lower()
            utterance = utterance.translate(str.maketrans("", "", string.punctuation))
            for required_word in self.required_words:
                if required_word not in utterance:
                    self.success = False
                    self.missing_words.append(required_word)
        return True

    def _on_after_turn(self, turn_idx: int):
        self.turns.append(self.success)

    def _after_add_player_response(self, player: Player, utterance: str):
        if player == self.greeter:
            self.add_user_message(self.greeted, utterance)

    def compute_turn_score(self):
        return self.compute_episode_score()  # single turn game

    def compute_episode_score(self):
        score = 0
        if self.success:
            score = 1
        return score


class HelloGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> DialogueGameMaster:
        return HelloGame(self.game_name, self.game_path, experiment, player_models)
