from typing import Tuple

from clemcore.clemgame import Object


class Wall(Object):

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "Wall", "W", "⬛️")


class Portal(Object):

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "Portal", "O", "🌀")


class Door(Object):

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "Door", "D", "🚪")
        self.is_open = False

    def toggle_state(self) -> None:
        self.is_open = not self.is_open


class Switch(Object):

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position, "Switch", "S", "🔘")
