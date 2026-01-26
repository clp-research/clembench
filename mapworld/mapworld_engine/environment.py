## Working Grid environment


from typing import Dict, Tuple
import ast
import os
import logging
from datetime import datetime
import imageio.v2 as imageio
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


from mapworld.engine.map_utils import load_json


RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
env_config = load_json(RESOURCES_DIR / "env_config.json")
robot_image = os.path.join(RESOURCES_DIR, "robot.png")

logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("mapworld.environment")

class MapWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, render_mode: str = "human", size: int = 5, map_metadata: Dict = None,
                 agent_pos: str = None, target_pos: str = None):
        """
        Initialize mapworld as a Gymnasium environment.

        Args:
            render_mode: Type of rendering mode to use - human (renders the observation as a plot), rgb_array (returns an array without any plots)
            size: Grid size of the mapworld environment
            map_metadata: metadata from the ADEMap class for a graph
            agent_pos: Agent start room on the mapworld environment, assigns a random outdoor room if agent_pos is None
            target_pos: Target room on the mapworld environment (Think Escaperoom Base version),
                        assigns a random outdoor room if target_pos is None
        """

        self.size = size  # The size of the square grid
        self.window_size = 500  # The size of the PyGame window
        self.map_metadata = map_metadata

        self.start_pos = agent_pos if agent_pos is not None else np.array(ast.literal_eval(self.map_metadata["start_node"]))
        self.target_pos = target_pos if target_pos is not None else np.array(ast.literal_eval(self.map_metadata["target_node"]))
        self._agent_location = self.start_pos
        self._target_location = self.target_pos

        # Counters
        self.visited = set()
        self.visited.add(tuple(self.start_pos))
        self.reached_target = False

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2
        # Ref - https://gymnasium.farama.org/introduction/create_custom_env/
        # Ref - https://gymnasium.farama.org/introduction/basic_usage/#action-and-observation-spaces
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 5 actions, corresponding to "east", "north", "west", "south", "explore", and "escape"
        self.action_space = spaces.Discrete(6)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "down" etc.
        """

        self._action_to_direction = {
            0: np.array([1, 0]),  # East
            1: np.array([0, 1]),  # South
            2: np.array([-1, 0]), # West
            3: np.array([0, -1]), # North

            # Game Specific Actions. Set accordingly in engine/resources/env_config.json
            4: np.array([0, 0]),  # Action 5 (e.g. - Wait)
            5: np.array([1, 1])   # Action 6 (e.g. - Escape)
        }

        _action_to_move = env_config["action_to_move"]

        self._action_to_move = {int(k): v for k, v in _action_to_move.items()}
        self._move_to_action = {v: k for k, v in self._action_to_move.items()}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self._frames = []


    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        self._agent_location = self.start_pos
        self._target_location = self.target_pos

        observation = self._get_obs()
        info = self._get_info()

        self._frames = []
        self._render_frame()

        return observation, info

    def step(self, action):
        
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        if tuple(self._agent_location) == tuple(self.target_pos):
            self.reached_target = True

        if tuple(self._agent_location) not in self.visited:
            self.visited.add(tuple(self._agent_location))

        # An episode is done if the guide agent has generated the <escape> token
        terminated = 0
        reward = 0
        if action == 4:
            terminated = 1
            reward = 1
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    @staticmethod
    def _to_xy(node):
        """
        Accept (x,y) in many forms:
        - tuple/list/np.array of ints
        - string like "(3, 4)" or "3,4"
        Returns np.ndarray([x, y], dtype=int)
        """
        if isinstance(node, (tuple, list, np.ndarray)):
            a = np.asarray(node, dtype=int).reshape(2, )
            return a
        if isinstance(node, str):
            try:
                val = ast.literal_eval(node)  # handles "(3, 4)" or "[3,4]"
            except Exception:
                parts = node.split(",")  # handles "3,4"
                if len(parts) == 2:
                    val = (int(parts[0]), int(parts[1]))
                else:
                    raise ValueError(f"Cannot parse node coordinate: {node!r}")
            return np.asarray(val, dtype=int).reshape(2, )
        raise TypeError(f"Unsupported node type: {type(node)}")

    def _draw_rect(self, canvas, color, pos, pix_square_size, room_ratio, label):

        # Define rectangle dimensions
        pygame.draw.rect(
            canvas,
            color,
            pygame.Rect(
                (pos*pix_square_size + ((1-room_ratio)/2)*pix_square_size), # (left, top)
                (room_ratio*pix_square_size, room_ratio*pix_square_size), # (width, height)
            ),
        )


        text_pos = pos*pix_square_size + pix_square_size/2
        text_pos = [text_pos[0], text_pos[1] - pix_square_size/2 + 10]
        if self.render_mode != "rgb_array":
            text_surf = self.font.render(str(label), True, (0, 0, 0))
            # center it in the cell
            text_rect = text_surf.get_rect(center=text_pos)
            canvas.blit(text_surf, text_rect)

    def _draw_line(self, canvas, color, edge, pix_square_size, room_ratio):
        a = self._to_xy(edge[0])  # np.array([x,y], int)
        b = self._to_xy(edge[1])

        start_pos = a * pix_square_size + pix_square_size / 2
        end_pos = b * pix_square_size + pix_square_size / 2

        # offset amount in pixels so lines don't overlap room rectangles
        offset = (room_ratio / 2.0) * pix_square_size

        if a[0] < b[0]:
            # Horizontal left-to-right
            start_pos[0] += offset
            end_pos[0] -= offset
        elif a[0] > b[0]:
            # Horizontal right-to-left
            start_pos[0] -= offset
            end_pos[0] += offset
        elif a[1] < b[1]:
            # Vertical top-to-bottom
            start_pos[1] += offset
            end_pos[1] -= offset
        else:
            # Vertical bottom-to-top
            start_pos[1] -= offset
            end_pos[1] += offset

        # snap to integers to avoid sub-pixel “double line” artifacts
        start_pos = (int(round(start_pos[0])), int(round(start_pos[1])))
        end_pos = (int(round(end_pos[0])), int(round(end_pos[1])))

        pygame.draw.line(canvas, color, start_pos, end_pos, width=1)

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 10)
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = int(self.window_size / self.size)
        room_ratio = 0.6

        # Draw edges
        for edge in self.map_metadata["unnamed_edges"]:
            self._draw_line(canvas, (0, 0, 0), edge, pix_square_size, room_ratio)

        # Draw rooms
        for node in self.map_metadata["unnamed_nodes"]:
            self._draw_rect(
                canvas, (255, 0, 0), self._to_xy(node),
                pix_square_size, room_ratio, self.map_metadata["node_to_category"][node]
            )

        # Robot sprite
        self.robot_img = pygame.image.load(robot_image).convert_alpha()
        self.robot_img = pygame.transform.smoothscale(
            self.robot_img, (int(room_ratio * pix_square_size), int(room_ratio * pix_square_size))
        )
        canvas.blit(self.robot_img, self._agent_location * pix_square_size + 0.2 * pix_square_size)

        # --- capture a frame for GIFs (HxWx3 uint8) ---
        frame = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)).copy()
        self._frames.append(frame)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            # also return frame so caller can inspect if desired
            return frame
        else:  # rgb_array
            return frame


    @staticmethod
    def _get_direction(start_pos: Tuple, next_pos: Tuple) -> str:

        """
        Get the direction of next move
        Args:
            start_pos: current node of the agent inside mapworld
            next_pos: next possible node of the agent inside mapworld

        Returns:
            direction: direction of the next move as a string item
        """
        if next_pos[0] == start_pos[0] and next_pos[1] == start_pos[1] + 1:
            return "south"
        elif next_pos[0] == start_pos[0] and next_pos[1] == start_pos[1] - 1:
            return "north"
        elif next_pos[1] == start_pos[1] and next_pos[0] == start_pos[0] + 1:
            return "east"
        elif next_pos[1] == start_pos[1] and next_pos[0] == start_pos[0] - 1:
            return "west"
        else:
            raise ValueError("Invalid move! Check the node positions!")

    def get_next_moves(self):
        agent_node = self._agent_location
        moves = []
        edges = self.map_metadata['unnamed_edges']

        for edge in edges:
            start_pos = None
            next_pos = None
            # Check edges from current agent position
            # TODO: Save metadata containing edge from n1 to n2 and n2 to n1, instead of only one of em
            edge1 = ast.literal_eval(edge[0])
            edge2 = ast.literal_eval(edge[1])

            if np.array_equal(edge1, agent_node):
                start_pos = edge1
                next_pos = edge2
            elif np.array_equal(edge2, agent_node):
                start_pos = edge2
                next_pos = edge1

            if start_pos:
                direction = self._get_direction(start_pos, next_pos)
                # room = self.map_metadata['node_to_category'][str(tuple(next_pos))]
                moves.append(direction)

        return str(moves)


    def _check_room(self) -> str:
        """
        Checks if the current room of agent is ambiguous, target room, or other
        Returns:
            "ambiguous" if ambiguous room,
            "target" if target room,
            "other" if other room
        """
        current_node = self._agent_location

        room_name = self.map_metadata["node_to_category"][str(tuple(current_node))]
        last_char = room_name[-1]
        str_digits = [str(i) for i in range(10)]

        if last_char in str_digits:
            # Checks if room is ambiguous or not
            room_name = room_name[:-2].strip()
            target_name = self.map_metadata["node_to_category"][str(tuple(self.target_pos))]
            target_name = target_name[:-2].strip()
            if target_name == room_name:
                return "ambiguous"
            else:
                return "other"
        else:
            return "other"


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def record_video(self, out_path: str | None = None, fps: int = 2, loop: int = 0):
        """
        Save all buffered frames (captured during render calls) as a GIF.

        Args:
            out_path: destination .gif path. If None, saves under
                      mapworld/engine/resources/gifs/<timestamp>.gif
            fps: frames per second for the GIF
            loop: 0=infinite loop, or number of loops

        Returns:
            out_path (str): full path to the saved GIF
        """
        if not self._frames:
            raise RuntimeError("No frames recorded. Call env.render() during your episode first.")

        # default path
        if out_path is None:
            gifs_dir = RESOURCES_DIR / "gifs"
            gifs_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = gifs_dir / f"episode_{stamp}.gif"
        else:
            out_dir = Path(out_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)

        duration = 2.0 / max(1, fps)  # seconds per frame
        imageio.mimsave(
            str(out_path),
            self._frames,
            format="GIF",
            duration=duration,
            loop=loop,
        )
        return str(out_path)


if __name__ == '__main__':

    metadata = {'graph_id': '10b11r20c21b30a31r40r41m', 'm': 5, 'n': 5, 'named_nodes': ['Bedroom', 'Reception', 'Computer room', 'Bedroom', 'Art studio', 'Reading room', 'Reading room', 'Music studio'], 'unnamed_nodes': ['(1, 0)', '(1, 1)', '(2, 0)', '(2, 1)', '(3, 0)', '(3, 1)', '(4, 0)', '(4, 1)'], 'named_edges': [('Bedroom', 'Computer room'), ('Bedroom', 'Reception'), ('Reception', 'Bedroom'), ('Computer room', 'Art studio'), ('Bedroom', 'Reading room'), ('Art studio', 'Reading room'), ('Reading room', 'Music studio'), ('Reading room', 'Music studio')], 'unnamed_edges': [('(1, 0)', '(2, 0)'), ('(1, 0)', '(1, 1)'), ('(1, 1)', '(2, 1)'), ('(2, 0)', '(3, 0)'), ('(2, 1)', '(3, 1)'), ('(3, 0)', '(4, 0)'), ('(3, 1)', '(4, 1)'), ('(4, 0)', '(4, 1)')], 'node_to_category': {'(1, 0)': 'Bedroom', '(1, 1)': 'Reception', '(2, 0)': 'Computer room', '(2, 1)': 'Bedroom', '(3, 0)': 'Art studio', '(3, 1)': 'Reading room', '(4, 0)': 'Reading room', '(4, 1)': 'Music studio'}, 'node_to_image': {'(1, 0)': 'https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/home_or_hotel/bedroom/ADE_train_00003554.jpg', '(1, 1)': 'https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/work_place/reception/ADE_train_00015719.jpg', '(2, 0)': 'https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/work_place/computer_room/ADE_train_00005955.jpg', '(2, 1)': 'https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/home_or_hotel/bedroom/ADE_train_00003467.jpg', '(3, 0)': 'https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/cultural/art_studio/ADE_train_00001758.jpg', '(3, 1)': 'https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/work_place/reading_room/ADE_train_00015700.jpg', '(4, 0)': 'https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/work_place/reading_room/ADE_train_00015697.jpg', '(4, 1)': 'https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/cultural/music_studio/ADE_train_00012288.jpg'}, 'start_node': '(1, 1)', 'target_node': '(4, 0)'}
    env = MapWorldEnv(render_mode="human", size=5, map_metadata=metadata)
    env.reset()
    env.render()

    moves = [0, 0, 0, 3, 2, 2, 2, 1]
    for a in moves:
        env.render()  # <- buffers a frame each time
        env.step(a)

    # Save GIF under mapworld/engine/resources/gifs/
    gif_path = env.record_video()  # or env.record_video("mapworld/engine/resources/gifs/run.gif", fps=4)
    print("Saved:", gif_path)

    env.close()