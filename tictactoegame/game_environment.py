import random
from typing import Dict, List, Literal, Optional

import numpy as np
from clemcore.clemgame import (
    Action,
    ActionSpace,
    GridEnvironment,
    Object,
    Observation,
    Player,
)


class TicTacToeAction(Action):
    action_type: str
    row: int
    col: int


class TicTacToePlayer(Player):

    def __init__(self, model):
        super().__init__(model)

    def _custom_response(self, context: Dict) -> str:
        i = random.randint(0, 2)
        j = random.randint(0, 2)

        return f"{i} {j}"


class TicTacToeCell(Object):

    def __init__(self, position: tuple[int, int], value: str = " "):
        symbol = "X" if value == "X" else "O" if value == "O" else "empty"
        pretty_symbol = "❎" if value == "X" else "🅾️" if value == "O" else "⬜️"
        super().__init__(
            position, f"cell_{position[0]}_{position[1]}", symbol, pretty_symbol
        )


class TicTacToeEnvironment(GridEnvironment):

    def _action_valid_in_state(
        self, player: TicTacToePlayer, action: TicTacToeAction
    ) -> tuple[bool, str]:
        row = action.get("row")
        col = action.get("col")

        if row is None or col is None:
            return False, "Missing row or col in action"

        if not (0 <= row < 3 and 0 <= col < 3):
            return False, f"Position ({row}, {col}) is out of bounds"

        if self._get_objects_at((row, col)) != []:
            return False, f"Position ({row}, {col}) is already occupied"

        return True, ""

    def _check_won(self, player: Player) -> tuple[bool, bool]:
        board_np = np.array([["empty" for _ in range(3)] for _ in range(3)])
        for i in range(self.height):
            for j in range(self.width):
                if self._get_objects_at((i, j)) != []:
                    symbol = self._get_objects_at((i, j))[0].symbol
                    board_np[i, j] = symbol

        for i in range(3):
            if all(board_np[i, :] == "X") or all(board_np[:, i] == "X"):
                return True, True
            if all(board_np[i, :] == "O") or all(board_np[:, i] == "O"):
                return True, True

        if all(np.diag(board_np) == "X") or all(np.diag(np.fliplr(board_np)) == "X"):
            return True, True
        if all(np.diag(board_np) == "O") or all(np.diag(np.fliplr(board_np)) == "O"):
            return True, True

        if np.all(board_np != "empty"):
            return True, False

        return False, True

    def _get_current_symbol(self) -> str:
        symbol_count = sum(
            1 for r in self.state["_grid"] for c in r if c["objects"] != []
        )
        return "X" if symbol_count % 2 == 0 else "O"

    def _compose_turn_prompt(self, player_name: Optional[str] = None) -> str:
        non_empty_cells = sum(
            1 for row in self.state["_grid"] for cell in row if cell["objects"] != []
        )
        prefix = ""
        if non_empty_cells < 2:
            base_prompt = self.config.get("prompt", "")
            prefix += base_prompt + "\n\n"

            current_symbol = self._get_current_symbol()
            prefix += f"You are the player that plays {current_symbol}.\n\n"
        prefix += "The board is:"
        return prefix

    def _update_state_through_action(
        self, player: TicTacToePlayer, action: TicTacToeAction
    ) -> None:
        row = action["row"]
        col = action["col"]

        objects = self._get_objects_at((row, col))
        if objects:
            self._remove_object(objects[0])

        current_symbol = self._get_current_symbol()
        new_cell = TicTacToeCell((row, col), value=current_symbol)
        self._add_object(new_cell)
