"""
Generate instances for the Sudoku game.

Creates `in/instances.json` with experiments and game instances that match the
SudokuGame environment requirements.
"""

import os
import random
from typing import Dict, List

from clemcore.clemgame import GameInstanceGenerator
from sudoku import Sudoku

LANGUAGE = "en"
RENDER_AS = "string"
WIDTH = 9
HEIGHT = 9
MAX_MOVES = 20


class SudokuGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed: int, **kwargs):
        """Generate randomized Sudoku instances to avoid static leakage.

        Creates three experiments:
        - sudoku_easy: difficulty 0.1
        - sudoku_medium: difficulty 0.5
        - sudoku_hard: difficulty 0.9

        All experiments use 9x9 grid size.
        """

        prompt_text = (
            "You are playing a Sudoku game. Sudoku is a number puzzle played on a grid made up of subgrids. "
            "The goal is to fill the grid with numbers following these rules: "
            "1) Each row must contain all numbers 1 to n_size without repetition, "
            "2) Each column must contain all numbers 1 to n_size without repetition, "
            "3) Each subgrid must contain all numbers 1 to n_rows without repetition. "
            "Fill in the next number on your road to solve the puzzle, by replacing any of the 0s with the correct number. "
            "Answer in the following format: '<row> <col> <value>', example: '0 0 1'. "
            "Note that row and col values start with 0. Return nothing but the '<row> <col> <value>', "
            "otherwise the parsing won't work."
        )

        # parameters (override those via kwargs, optional)
        num_instances: int = int(kwargs.get("num_instances", 10))
        width: int = int(kwargs.get("width", WIDTH))
        height: int = int(kwargs.get("height", HEIGHT))
        max_moves: int = int(kwargs.get("max_moves", MAX_MOVES))
        render_as: str = str(kwargs.get("render_as", RENDER_AS))

        # define experiments with their difficulties
        experiments = [
            ("sudoku_easy", 0.1),
            ("sudoku_medium", 0.5),
            ("sudoku_hard", 0.9),
        ]

        for experiment_name, difficulty in experiments:
            experiment = self.add_experiment(experiment_name)
            experiment["language"] = LANGUAGE

            for game_id in range(num_instances):
                # create original puzzle grid based on difficulty
                puzzle = Sudoku(width // 3).difficulty(difficulty)
                original_grid = [
                    [0 if cell is None else int(cell) for cell in row]
                    for row in puzzle.board
                ]

                config: Dict = {
                    "game_name": "sudokugame",
                    "prompt": prompt_text,
                    "width": width,
                    "height": height,
                    "render_as": render_as,
                    "max_moves": max_moves,
                    "difficulty": difficulty,
                    "original_grid": original_grid,
                }

                game_instance = self.add_game_instance(experiment, game_id)
                for key, value in config.items():
                    game_instance[key] = value


if __name__ == "__main__":
    SudokuGameInstanceGenerator().generate(seed=42)
