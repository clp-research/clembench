"""
Generate instances for the Portal game.

Creates `in/instances.json` with experiments and game instances that match the
PortalGame environment requirements.
"""

import os
import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from clemcore.clemgame import GameInstanceGenerator

LANGUAGE = "en"
RENDER_AS = "string"
LIMITED_VISIBILITY = True
SHOW_EXPLORED = True
WIDTH = 5
HEIGHT = 5
WALL_FRACTION = 0.2


class PortalGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed: int, **kwargs):
        """Algorithm per instance:
        - Choose random player start within borders.
        - Put the portal relatively nearby, then compute a shortest path.
        - Place a door on the last step of that path (cell before the portal).
        - Place a switch on a random adjacent spot on the path.
        - Add random walls to the rest of the interior excluding occupied cells.
        - Add the 2-step detour for hitting the switch to the shortest_path.
        """

        prompt_text = (
            "You are playing the Portal Game, a maze navigation challenge where your goal is to reach the portal.\n\n"
            "Game Elements:\n\n"
            "- Player (P): You. You can move around the grid.\n"
            "- Portal (O): Your goal. Reach it to win.\n"
            "- Door (D): A door that can be opened or closed.\n"
            "- Switch (S): Opens and closes the door.\n"
            "- Walls (W): You can't pass through them.\n"
            "- Empty cells (' '): You can pass through them.\n\n"
            "Response Format: Write one sentence of reasoning above your move. End your response with a line of the form:\n\n"
            "DIRECTION: <letter>\n\n"
            "where <letter> is one of n (to move north/up), s (to move south/down), e (to move east/right), or w (to move west/left)."
        )

        # parameters (override those via kwargs, optional)
        num_instances: int = int(kwargs.get("num_instances", 6))
        width: int = int(kwargs.get("width", WIDTH))
        height: int = int(kwargs.get("height", HEIGHT))
        wall_fraction: float = float(kwargs.get("wall_fraction", WALL_FRACTION))
        limited_visibility: bool = bool(
            kwargs.get("limited_visibility", LIMITED_VISIBILITY)
        )
        show_explored: bool = bool(kwargs.get("show_explored", SHOW_EXPLORED))
        render_as: str = str(kwargs.get("render_as", RENDER_AS))

        def border_walls(width: int, height: int) -> List[Tuple[int, int]]:
            """Create walls along the outer border of a grid."""
            walls: List[Tuple[int, int]] = []
            # top and bottom rows
            for x in range(width):
                walls.append((0, x))
                walls.append((height - 1, x))
            # left and right columns
            for y in range(1, height - 1):
                walls.append((y, 0))
                walls.append((y, width - 1))
            return walls

        def within_borders(cell: Tuple[int, int]) -> bool:
            y, x = cell
            return 1 <= y < height - 1 and 1 <= x < width - 1

        def neighbors(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
            y, x = cell
            return [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]

        def bfs_shortest_path(
            start: Tuple[int, int],
            goal: Tuple[int, int],
            blocked: Optional[Set[Tuple[int, int]]] = None,
        ) -> Optional[List[Tuple[int, int]]]:
            if blocked is None:
                blocked = set()

            q: deque[Tuple[int, int]] = deque([start])

            came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

            while q:
                cur = q.popleft()
                if cur == goal:
                    break
                for nb in neighbors(cur):
                    if not within_borders(nb) or nb in blocked or nb in came_from:
                        continue
                    came_from[nb] = cur
                    q.append(nb)

            if goal not in came_from:
                return None

            path: List[Tuple[int, int]] = []

            c: Optional[Tuple[int, int]] = goal

            while c is not None:
                path.append(c)
                c = came_from[c]
            path.reverse()

            return path

        # five distinct experiment configurations
        combos = [
            {
                "name": "6x6_unlimited",
                "width": 6,
                "height": 6,
                "limited_visibility": False,
                "show_explored": False,
            },
            {
                "name": "5x5_limited_explored",
                "width": 5,
                "height": 5,
                "limited_visibility": True,
                "show_explored": True,
            },
            {
                "name": "5x5_limited_hidden",
                "width": 5,
                "height": 5,
                "limited_visibility": True,
                "show_explored": False,
            },
            {
                "name": "7x7_limited_explored",
                "width": 7,
                "height": 7,
                "limited_visibility": True,
                "show_explored": True,
            },
            {
                "name": "7x7_limited_hidden",
                "width": 7,
                "height": 7,
                "limited_visibility": True,
                "show_explored": False,
            },
        ]

        for combo in combos:
            # set per-experiment parameters
            width = combo["width"]
            height = combo["height"]
            limited_visibility = combo["limited_visibility"]
            show_explored = combo["show_explored"]

            experiment = self.add_experiment(f"portalgame_{combo['name']}")
            experiment["language"] = LANGUAGE

            for game_id in range(num_instances):
                base_walls = set(border_walls(width, height))

                # random player start, within borders
                player_start = (
                    random.randint(1, height - 2),
                    random.randint(1, width - 2),
                )

                portal: Optional[Tuple[int, int]] = None
                portal_shortest_path: Optional[List[Tuple[int, int]]] = None

                # choose portal with valid empty-grid shortest path
                while not portal_shortest_path:
                    candidate_portal_pos = (
                        random.randint(1, height - 2),
                        random.randint(1, width - 2),
                    )

                    if candidate_portal_pos == player_start:
                        continue

                    candidate_path = bfs_shortest_path(
                        player_start, candidate_portal_pos, blocked=base_walls
                    )

                    if candidate_path is not None and len(candidate_path) >= 3:
                        portal = candidate_portal_pos
                        portal_shortest_path = candidate_path

                # this is just to suppress type errors
                assert portal_shortest_path is not None
                assert portal is not None

                base_shortest = len(portal_shortest_path) - 1

                door = portal_shortest_path[-2]

                # surround portal by walls except for door
                portal_neighbors = [
                    nb for nb in neighbors(portal) if within_borders(nb) and nb != door
                ]
                portal_surround_walls: Set[Tuple[int, int]] = set(portal_neighbors)

                # make sure not to place walls onto cells that are already occupied
                reserved: Set[Tuple[int, int]] = set(portal_shortest_path)
                reserved.add(player_start)
                reserved.add(portal)
                reserved.add(door)
                reserved.update(portal_surround_walls)

                # switch adjacent to the shortest path (random)
                candidates: List[Tuple[int, int]] = []
                for c in portal_shortest_path:
                    for nb in neighbors(c):
                        if within_borders(nb) and nb not in reserved:
                            candidates.append(nb)
                if not candidates:
                    for nb in neighbors(door):
                        if within_borders(nb) and nb not in reserved:
                            candidates.append(nb)
                switch = random.choice(candidates)
                switch_shortest_path = bfs_shortest_path(
                    player_start, switch, blocked=base_walls
                )

                assert switch_shortest_path is not None

                reserved.add(switch)
                reserved.update(switch_shortest_path)

                # add random walls
                available_cells: List[Tuple[int, int]] = [
                    (y, x)
                    for y in range(1, height - 1)
                    for x in range(1, width - 1)
                    if (y, x) not in reserved
                ]
                random.shuffle(available_cells)
                num_walls = int(wall_fraction * len(available_cells))
                random_walls = set(available_cells[:num_walls])
                walls = list(
                    base_walls.union(random_walls).union(portal_surround_walls)
                )

                if switch not in portal_shortest_path:
                    shortest_path_with_switch = (
                        base_shortest + 2
                    )  # step to switch and back
                else:
                    shortest_path_with_switch = base_shortest
                max_moves = min(width * height // 2, shortest_path_with_switch + 10)

                config: Dict = {
                    "game_name": "portalgame",
                    "prompt": prompt_text,
                    "height": height,
                    "width": width,
                    "max_moves": max_moves,
                    "shortest_path": shortest_path_with_switch,
                    "limited_visibility": limited_visibility,
                    "show_explored": show_explored,
                    "render_as": render_as,
                    "grid": {
                        "walls": walls,
                        "portal": portal,
                        "switch": switch,
                        "door": door,
                        "players_start": [player_start],
                    },
                }

                game_instance = self.add_game_instance(experiment, game_id)
                for key, value in config.items():
                    game_instance[key] = value


if __name__ == "__main__":
    PortalGameInstanceGenerator().generate()
