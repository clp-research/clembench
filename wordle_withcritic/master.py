from typing import Dict, List

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameMaster, GameScorer, GameSpec
from wordle.master import WordleGameMaster, WordleGameScorer

GAME_NAME = "wordle_withcritic"


class WordleWithClueAndCriticGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def get_description(self):
        return "Wordle Game with a clue given to the guesser and a critic for the clue"

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return WordleGameMaster(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return WordleGameScorer(self.game_name, experiment, game_instance)
