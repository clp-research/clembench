"""
Generate instances for the TicTacToe game.

Creates `in/instances.json` with experiments and game instances that match the
TicTacToe environment requirements.
"""

import os
from typing import Dict

from clemcore.clemgame import GameInstanceGenerator

LANGUAGE = "en"
RENDER_AS = "human-readable"
WIDTH = 3
HEIGHT = 3
MAX_MOVES = 10


class TicTacToeGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed: int, **kwargs):
        """Create a standard experiment with configurable number of instances.

        Params (overridable via kwargs):
        - num_instances: number of game instances per experiment (default: 10)
        - width, height: grid size (defaults to 3x3)
        - max_moves: maximum number of turns in an episode (default: 10)
        - render_as: rendering mode (default: human-readable)
        """

        prompt_text = (
            "You are playing a TicTacToe game. TicTacToe is a two-player game played on a 3x3 grid. "
            "Players take turns placing their marks (X or O) in empty cells. The first player to get three of their "
            "marks in a row (horizontally, vertically, or diagonally) wins. If all cells are filled and no player has "
            "won, the game is a draw. See the board below. Make your move by specifying the row and column where you "
            "want to place your mark. Answer in the following format: '<row> <col>', example: '0 2' for the upper "
            "right cell. Return nothing but the '<row> <col>', otherwise the parsing won't work."
        )

        # parameters (override via kwargs, optional)
        num_instances: int = int(kwargs.get("num_instances", 10))
        width: int = int(kwargs.get("width", WIDTH))
        height: int = int(kwargs.get("height", HEIGHT))
        max_moves: int = int(kwargs.get("max_moves", MAX_MOVES))
        render_as: str = str(kwargs.get("render_as", RENDER_AS))

        experiment = self.add_experiment("tictactoe_standard")
        experiment["language"] = LANGUAGE

        for game_id in range(num_instances):
            config: Dict = {
                "game_name": "tictactoegame",
                "prompt": prompt_text,
                "render_as": render_as,
                "width": width,
                "height": height,
                "max_moves": max_moves,
            }

            game_instance = self.add_game_instance(experiment, game_id)
            for key, value in config.items():
                game_instance[key] = value


if __name__ == "__main__":
    TicTacToeGameInstanceGenerator().generate()
