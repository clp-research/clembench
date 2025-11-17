from typing import Dict, Literal, Tuple

from clemcore.clemgame import Action, InclusiveGridEnvironment, Player

from objects import Door, Portal, Switch, Wall


class PortalAction(Action):
    action_type: str  # 'move
    direction: str  # 'n', 's', 'e', 'w'


class PortalGameEnvironment(InclusiveGridEnvironment):

    def _initialize_state(self) -> None:
        super()._initialize_state()

        if "grid" not in self.config:
            return

        grid_config = self.config["grid"]

        for wall_pos in grid_config.get("walls", []):
            row, col = wall_pos
            self._add_object(Wall(position=(row, col)))

        portal_pos = grid_config.get("portal")
        if portal_pos:
            row, col = portal_pos
            self._add_object(Portal(position=(row, col)))

        switch_pos = grid_config.get("switch")
        if switch_pos:
            row, col = switch_pos
            self._add_object(Switch(position=(row, col)))

        door_pos = grid_config.get("door")
        if door_pos:
            row, col = door_pos
            self._add_object(Door(position=(row, col)))

    def _update_state_through_action(
        self, player: Player, action: PortalAction
    ) -> None:
        direction = action.get("direction")
        self._move_player(player.name, direction)

        new_cell_objects = self._get_objects_at(self._get_player_position(player.name))

        if new_cell_objects != [] and isinstance(new_cell_objects[0], Switch):
            for y in self.state["_grid"]:
                for cell in y:
                    if cell["objects"] != [] and isinstance(cell["objects"][0], Door):
                        cell["objects"][0].toggle_state()

    def _check_won(self, player: Player) -> Tuple[bool, bool]:
        new_cell_objects = self._get_objects_at(self._get_player_position(player.name))

        if new_cell_objects != [] and isinstance(new_cell_objects[0], Portal):
            return True, True

        return False, True

    def _action_valid_in_state(
        self, player: Player, action: PortalAction
    ) -> Tuple[bool, str]:
        direction: Literal["n", "s", "e", "w"] = action.get("direction")  # type: ignore
        valid, reason = super()._action_valid_in_state(player, direction)
        if not valid:
            return valid, reason

        y, x = self._get_player_position(player.name)
        new_y = y - 1 if direction == "n" else y + 1 if direction == "s" else y
        new_x = x + 1 if direction == "e" else x - 1 if direction == "w" else x

        # check if the new position is a wall or closed door
        cell = self.state["_grid"][new_y][new_x]
        if cell["objects"] != [] and isinstance(cell["objects"][0], Wall):
            return (
                False,
                f"The object at cell ({new_y}, {new_x}) is a wall! You cannot pass through walls! Please try again.",
            )
        if (
            cell["objects"] != []
            and isinstance(cell["objects"][0], Door)
            and not cell["objects"][0].is_open
        ):
            return (
                False,
                f"The object at cell ({new_y}, {new_x}) is a closed door! You need to open it first.",
            )

        return True, ""

    def _compose_turn_prompt(self, player_name: str | None = None) -> str:
        door_state = None
        for row in self.state["_grid"]:
            for cell in row:
                if cell["objects"] != [] and isinstance(cell["objects"][0], Door):
                    door_state = cell["objects"][0].is_open
        prefix = ""
        if self.state.get("moves", 0) == 0:
            prompt = self.config["prompt"]
            prefix += prompt + "\n\nYou initially see the following grid layout:\n"
        else:
            player_pos = self._get_player_position(player_name) if player_name else None
            if player_pos is not None:
                prefix += f"Current position: {player_pos}\n"
            if door_state is not None:
                prefix += f"Door state: {'open' if door_state else 'closed'}\n"
            prefix += "\nGrid (Visible Area):"
        return prefix
