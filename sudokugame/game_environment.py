import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from clemcore.backends import Model
from clemcore.clemgame import (
    Action,
    ActionSpace,
    GridEnvironment,
    Object,
    Observation,
    Player,
)

class SudokuPlayer(Player):

    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, context: Dict) -> str:
        i = random.randint(0, 8)
        j = random.randint(0, 8)
        number = random.randint(0, 8)

        return f"{i} {j} {number}"


class SudokuAction(Action):

    row: int
    col: int
    value: int


class SudokuObject(Object):

    def __init__(self, position: Tuple[int, int], value: int):
        symbol = str(value)
        emoji_numbers = ["0️⃣", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]
        pretty_symbol = emoji_numbers[value] if 0 <= value <= 9 else str(value)
        super().__init__(
            position, f"cell_{position[0]}_{position[1]}", symbol, pretty_symbol
        )


class SudokuEnvironment(GridEnvironment):

    def _initialize_state(self) -> None:
        super()._initialize_state()

        config_grid = self.config.get("original_grid")
        original_grid = [[int(v) for v in row] for row in config_grid]

        for i in range(self.width):
            for j in range(self.height):
                value = original_grid[i][j]
                object = SudokuObject((i, j), value)
                self._add_object(object)

    def _is_grid_valid(self, row: int, col: int, value: int) -> bool:
        object = self._get_objects_at((row, col))[0]
        if object.symbol != "0":
            return False

        for i in range(self.height):
            object = self._get_objects_at((i, col))[0]
            if object.symbol == str(value):
                return False

        for j in range(self.width):
            object = self._get_objects_at((row, j))[0]
            if object.symbol == str(value):
                return False

        block_size = 3
        block_row = (row // block_size) * block_size
        block_col = (col // block_size) * block_size
        for i in range(block_row, block_row + block_size):
            for j in range(block_col, block_col + block_size):
                object = self._get_objects_at((i, j))[0]
                if object.symbol == str(value):
                    return False

        return True

    def _action_valid_in_state(
        self, player: SudokuPlayer, action: SudokuAction
    ) -> tuple[bool, str]:
        row = action.get("row")
        col = action.get("col")
        value = action.get("value")

        if self._is_grid_valid(row, col, value):
            return True, ""
        else:
            return (
                False,
                f"Invalid move: cannot place {value} at position ({row}, {col})",
            )

    def _update_state_through_action(
        self, player: SudokuPlayer, action: SudokuAction
    ) -> None:
        row = action["row"]
        col = action["col"]
        value = action["value"]

        objects = self._get_objects_at((row, col))
        self._remove_object(objects[0])

        new_cell = SudokuObject((row, col), value)
        self._add_object(new_cell)

    def _check_won(self, player: Player) -> Tuple[bool, bool]:
        for i in range(self.width):
            for j in range(self.height):
                object = self._get_objects_at((i, j))[0]
                if object.symbol == "0":
                    return False, True

        return True, True

    def _compose_turn_prompt(self, player_name: Optional[str] = None) -> str:
        base_prompt = self.config.get("prompt", "")
        prefix = ""
        if base_prompt and self.state["moves"] < len(self.players):
            prefix += base_prompt + "\n\n"
        prefix += "The board is:"
        return prefix

    def _render_state_as_string(self, player_name: Optional[str] = None) -> str:
        board_size = self.width
        box_size = int(self.width / 3)

        output = []

        for i in range(board_size):
            row = []
            for j in range(board_size):
                object = self._get_objects_at((i, j))[0]
                val = object.symbol
                row.append(val)

                if (j + 1) % box_size == 0 and j < board_size - 1:
                    row.append("|")
            output.append("".join(row))

            if (i + 1) % box_size == 0 and i < board_size - 1:
                output.append("-" * (board_size + 2))

        return "\n".join(output)

    def _render_state_as_image(self, player_name: Optional[str] = None) -> bytes:
        return b"" # not implemented

    def _render_state_as_human_readable(self, player_name: Optional[str] = None) -> str:
        return self._render_state_as_string()
