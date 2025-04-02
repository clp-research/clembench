from typing import List, Dict, Tuple
import re
import os
import json
from queue import Queue
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import imageio
import shutil
import logging

import mm_mapworld_utils as utils
from clemcore.backends import Model, CustomResponseModel
from clemcore.clemgame import GameMaster, GameBenchmark, DialogueGameMaster, GameScorer, GameSpec
from clemcore.clemgame import Player
from clemcore.utils import file_utils
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE

logger = logging.getLogger(__name__)

DIRS = ["north", "south", "east", "west"]
GAME_NAME = 'mm_mapworld'
MAX_TURNS = 20

CARDINAL_TO_DELTA = {
    'north': (0, 1),
    'south': (0, -1),
    'east': (1, 0),
    'west': (-1, 0)
}
DELTA_TO_CARDINAL = {
    (0, 1): 'north',
    (0, -1): 'south',
    (1, 0): 'east',
    (-1, 0): 'west'
}


class PathWalker(Player):
    def __init__(self, model: Model):
        super().__init__(model, forget_extras=["image"])

    def _custom_response(self, context) -> str:
        """Return a random direction."""
        actions = ["GO: west", "GO: east", "GO: north", "GO: south", "DONE"]
        response = {
            "description": " ",
            "action": np.random.choice(actions)
        }
        return json.dumps(response)


class PathDescriber(Player):
    def __init__(self, game_instance):
        super().__init__(CustomResponseModel())
        instance_data = utils.load_instance(game_instance)
        self.imgs = instance_data["imgs"]
        self.nodes = instance_data["nodes"]
        self.edges = instance_data["edges"]
        self.start = instance_data["start"]
        self.current_room = instance_data["start"]
        self.success_response = game_instance["success_response"]
        self.invalid_response = game_instance["invalid_response"]
        self.init_prompt = game_instance["initial_prompt"]
        self.loop_response = game_instance["loop_warning"]
        self.limit_warning = game_instance["limit_warning"]
        self.visited_nodes = [self.current_room]
        self.use_loop_warning = game_instance["use_loop_warning"]
        self.use_turn_limit_warning = game_instance["use_turn_limit_warning"]

        self.is_first_turn = True
        self.invalid_move = False

    def get_available_moves(self, node):
        return [edge for edge in self.edges if node == edge[0]]

    def detect_loop(self):
        if len(self.visited_nodes) >= 4:
            if len(set(self.visited_nodes[-4:])) < 3:
                return True
        return False

    def get_available_directions(self, node):
        moves = self.get_available_moves(node)
        deltas = [utils.edge_to_delta(move) for move in moves]
        cardinals = [DELTA_TO_CARDINAL[delta] for delta in deltas]
        return cardinals

    def cardinal_room_change(self, cardinal):
        delta = CARDINAL_TO_DELTA[cardinal]
        new_room = (self.current_room[0] + delta[0], self.current_room[1] + delta[1])
        if (self.current_room, new_room) in self.edges:
            self.current_room = new_room

    def _custom_response(self, context) -> str:
        if self.is_first_turn:
            self.is_first_turn = False
            response = self.init_prompt
            available_directions = self.get_available_directions(self.current_room)
            response = response.replace('$INITIAL_DIRECTIONS$', ', '.join(available_directions))
        else:
            available_directions = self.get_available_directions(self.current_room)
            if self.invalid_move:
                response = self.invalid_response.replace("$DIRECTIONS$", ", ".join(available_directions))
            else:
                response = self.success_response.replace("$DIRECTIONS$", ", ".join(available_directions))
            # if self.detect_loop() and self.use_loop_warning:
            #     response = self.loop_response + response
            # if turn_idx == (MAX_TURNS - 1) and self.use_turn_limit_warning:
            #     response = self.limit_warning + response
        return response


class MmMapWorld(DialogueGameMaster):
    """Implement mechanisms for playing MM-MapWorld."""

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)

        self.turns = []
        self.aborted: bool = False
        self.stop: bool = False
        self.need_reprompt: bool = False
        self.did_reprompt: bool = False
        self.experiment = experiment['name']

    def get_available_moves(self, node):
        return [edge for edge in self.edges if node == edge[0]]

    def get_available_directions(self, node):
        moves = self.get_available_moves(node)
        deltas = [utils.edge_to_delta(move) for move in moves]
        cardinals = [DELTA_TO_CARDINAL[delta] for delta in deltas]
        return cardinals

    def cardinal_room_change(self, cardinal):
        delta = CARDINAL_TO_DELTA[cardinal]
        new_room = (self.current_room[0] + delta[0], self.current_room[1] + delta[1])
        if (self.current_room, new_room) in self.edges:
            self.current_room = new_room

    def _on_setup(self, **game_instance):
        """" sets the information you specify in instances.json """
        self.game_instance = game_instance
        instance_data = utils.load_instance(self.game_instance)
        self.imgs = instance_data["imgs"]
        self.nodes = instance_data["nodes"]
        self.edges = instance_data["edges"]
        self.start = instance_data["start"]
        self.cats = instance_data["cats"]
        self.current_room = instance_data["start"]
        self.visited_nodes = [self.current_room]

        self.response_regex = re.compile(game_instance["response_regex"], re.IGNORECASE)
        self.done_regex = re.compile(game_instance["done_regex"], re.IGNORECASE)
        self.move_regex = re.compile(game_instance["move_regex"], re.IGNORECASE)

        self.done_const = game_instance["stop_construction"]
        self.move_const = game_instance["move_construction"]

        self.use_images = game_instance["use_images"]

        self.do_reprompt = game_instance["reprompt"]
        self.reprompt_format = game_instance["reprompt_format"]

        self.describer = PathDescriber(game_instance)
        self.walker = PathWalker(self.player_models[0])
        self.add_player(self.describer)
        self.add_player(self.walker)

    def _on_before_game(self):
        begin_message = json.dumps({
            "start": self.start,
            "size": len(self.nodes),
            "game": GAME_NAME
        })
        self.set_context_for(self.describer, begin_message)

    def _on_before_turn(self, turn_idx: int):
        img_path = 'games/mm_mapworld/resources/images/'
        value = {
            "image": [img_path + os.path.split(self.imgs[self.current_room])[1]]
        }
        self.log_to_self("room_image", json.dumps(value))

    def _does_game_proceed(self):
        if not self.aborted and not self.stop and self.current_round < MAX_TURNS:
            return True
        if self.current_round >= MAX_TURNS:
            self.aborted = True
            self.log_to_self(type_="aborted", value=self.aborted)
            self.log_to_self("turn limit reached", True)
        return False

    def _parse_response(self, player: Player, utterance: str) -> str:
        if player == self.walker:
            utterance = utterance.replace("\n", "").strip()
            for word in DIRS:
                utterance = utterance.replace(word.capitalize(), word)
                utterance = utterance.replace(word.upper(), word)
            found = re.search(self.response_regex, utterance)
            if found:
                utterance = found.group()
            self.log_to_self("parse", utterance)
        return utterance

    def _validate_player_response(self, player: Player, answer: str) -> bool:
        if player == self.walker:
            answer = answer.replace("\n", "").strip()
            for word in DIRS:
                answer = answer.replace(word.capitalize(), word)
                answer = answer.replace(word.upper(), word)
            # in case we abort we set the next move to None
            self.move = None
            # Check if the answer begins with 'MOVE:'
            hit = re.search(self.response_regex, answer)
            if not hit:
                if self.do_reprompt:
                    if self.did_reprompt:
                        self.aborted = True
                        self.log_to_self("Invalid format", "Game aborted.")
                        return False
                    self.need_reprompt = True
                    self.log_to_self("reprompting", "invalid format")
                    return True
                self.aborted = True
                self.log_to_self("Invalid format", "Game aborted.")
                return False
            try:
                action = json.loads(hit.group())['action']
                action = action.lower()
            except json.decoder.JSONDecodeError:
                self.aborted = True
                self.log_to_self("JSON decode error", "Game aborted.")
                return False
            action_hit = re.search(self.done_regex, action)
            if action_hit:
                self.stop = True
                self.log_to_self("DONE", True)
                return True
            hit = re.search(self.move_regex, action)
            if not hit:
                if self.do_reprompt:
                    if self.did_reprompt:
                        self.aborted = True
                        self.log_to_self("Invalid format", "Game aborted.")
                        return False
                    self.need_reprompt = True
                    self.log_to_self("reprompting", "invalid format")
                    return True
                self.aborted = True
                self.log_to_self("Invalid format", "Game aborted.")
                return False
            new_dir = hit.group(1)
            self.move = new_dir
            self.log_to_self("Valid format", "Continue")
        return True

    def _on_valid_player_response(self, player: PathDescriber, utterance: str):
        if player == self.walker:
            if not self.need_reprompt or self.did_reprompt:
                self.set_context_for(self.describer, utterance)
        if player == self.describer:
            self.set_context_for(self.walker, utterance, image=[player.imgs[self.current_room]])

    def _should_pass_turn(self):
        if self.current_player == self.walker and self.need_reprompt and not self.did_reprompt:
            avail = self.get_available_directions(self.current_room)
            reprompt = self.reprompt_format
            reprompt = reprompt.replace("$DIRECTIONS$", ', '.join(avail))
            if self.use_images:
                self.set_context_for(self.walker, reprompt, image=[self.imgs[self.current_room]])
            else:
                self.set_context_for(self.walker, reprompt)
            self.did_reprompt = True
            return False
        return True

    def _on_after_round(self):
        if self.aborted:
            self.log_to_self(type_="aborted", value=self.aborted)
        elif self.stop:
            pass
        else:
            old_room = self.current_room
            if self.move is not None:
                self.cardinal_room_change(self.move)
                self.describer.cardinal_room_change(self.move)
            self.describer.invalid_move = old_room == self.current_room
            self.visited_nodes.append(self.current_room)
            self.describer.visited_nodes.append(self.current_room)
            self.log_to_self(type_="move", value=json.dumps({"old": old_room, "new": self.current_room}))
        self.need_reprompt = False
        self.did_reprompt = False


class MM_MapWorldScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        instance_data = utils.load_instance(self.game_instance)
        self.name = game_name
        self.imgs = instance_data["imgs"]
        self.nodes = instance_data["nodes"]
        self.edges = instance_data["edges"]
        self.start_node = instance_data["start"]

    def adj(self, node):
        return set([ed[1] for ed in self.edges if ed[0] == node])

    def visited_all(self, visited, to_visit):
        return all([n in visited for n in to_visit])

    def get_available_moves(self, node, visited):
        return [edge for edge in self.edges if node == edge[0] and (edge[0] in visited or edge[1] in visited)]

    def find_best_moves(self, current, visited):
        to_visit = [ed[1] for ed in self.edges if ed[0] in visited and ed[1] not in visited]
        start = [current]
        q = Queue()
        q.put(start)
        found = set()
        max_len = 100
        while True:
            if not q.qsize():
                break
            n = q.get()
            if len(n) > max_len:
                break
            if self.visited_all(n, to_visit):
                found.add((n[0], n[1]))
                max_len = len(n)
                continue
            if len(n) == max_len:
                continue
            avail = self.get_available_moves(n[-1], visited)
            if all([move[1] in n for move in avail]):
                for move in avail:
                    new = deepcopy(n)
                    new.append(move[1])
                    q.put(new)
            else:
                for move in avail:
                    if not move[1] in n:
                        new = deepcopy(n)
                        new.append(move[1])
                        q.put(new)
        return found

    def plot_path(self, path):
        offset = 0.05
        fig = plt.figure(figsize=(4, 4))
        for node in self.nodes:
            if node in path and node != path[-1]:
                plt.plot(node[0], node[1], 'o', color='brown',
                         linewidth=20, markersize=25, zorder=9, mfc='tab:olive')
            if node == path[-1]:
                plt.plot(node[0], node[1], 'o', color='brown',
                         linewidth=20, markersize=25, zorder=9, mfc='tab:cyan')
            if not node in path:
                plt.plot(node[0], node[1], 'o', color='brown',
                         linewidth=20, markersize=25, zorder=9, mfc='tab:gray')
        plt.xlim(-1, 4)
        plt.ylim(-1, 4)
        traveled = {node: 0 for node in self.nodes}
        traveled[self.start_node] += 1
        for edge in self.edges:
            plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color='gray', linestyle='--', zorder=5)
        last = path[0]
        if len(path) > 1:
            for i in range(1, len(path)):
                if path[i] == path[i - 1]:
                    continue
                x1, y1 = last
                x2, y2 = path[i]
                dx = x2 - x1
                dy = y2 - y1
                t = traveled[path[i]]
                traveled[path[i]] += 1
                color = "black"
                if i == len(path) - 1:
                    color = "red"
                t = sum([(1 / (1 + j)) for j in range(t)])
                plt.arrow(x1,
                          y1,
                          dx + t * offset,
                          dy + t * offset,
                          color=color,
                          width=0.005,
                          head_width=0.05,
                          length_includes_head=True,
                          zorder=10)
                last = (
                    x1 + dx + t * offset,
                    y1 + dy + t * offset
                )
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.35)
        return fig

    def compute_scores(self, episode_interactions) -> None:
        current = self.start_node
        seen = {self.start_node}
        seen.update(self.adj(self.start_node))
        visited = {self.start_node}
        self.path = [self.start_node]
        valid_moves = 0
        invalid_moves = 0
        aborted = False
        good_move = []

        for turn in episode_interactions["turns"]:

            for event in turn:
                action = event["action"]
                if action["type"] == "aborted":
                    if action["content"]:
                        aborted = True
                if action['type'] == "move":
                    cont = json.loads(action['content'])
                    old = tuple(cont["old"])
                    new = tuple(cont["new"])
                    if not old == new:
                        valid_moves += 1
                    else:
                        invalid_moves += 1

                    if not self.visited_all(visited, self.nodes) and not old == new:
                        best_moves = self.find_best_moves(old, visited)
                        #                         print(best_moves)
                        if (old, new) in best_moves:
                            good_move.append(True)

                        else:
                            good_move.append(False)
                    else:
                        good_move.append(False)
                    current = new
                    seen.update(self.adj(current))
                    visited.add(current)
                    self.path.append(current)

        # log all the scores
        if aborted:  # set all values to NaN if game is aborted
            for i, val in enumerate(good_move):
                self.log_turn_score(i, "effiencient_move", np.NaN)
            self.log_episode_score(METRIC_ABORTED, 1)
            self.log_episode_score(METRIC_SUCCESS, np.NaN)
            self.log_episode_score(METRIC_LOSE, np.NaN)
            self.log_episode_score('moves', np.NaN)
            self.log_episode_score('valid_moves', np.NaN)
            self.log_episode_score('invalid_moves', np.NaN)
            self.log_episode_score('visited', np.NaN)
            self.log_episode_score('seen', np.NaN)
            self.log_episode_score('effieciency', np.NaN)
            self.log_episode_score('exploration', np.NaN)
            self.log_episode_score(BENCH_SCORE, np.NaN)
        else:  # else set them to their respective values
            self.log_episode_score(METRIC_ABORTED, 0)
            if self.visited_all(visited, self.nodes):
                self.log_episode_score(METRIC_SUCCESS, 1)
                self.log_episode_score(METRIC_LOSE, 0)
            else:
                self.log_episode_score(METRIC_SUCCESS, 0)
                self.log_episode_score(METRIC_LOSE, 1)
            self.log_episode_score('moves', valid_moves + invalid_moves)
            self.log_episode_score('valid_moves', valid_moves)
            self.log_episode_score('invalid_moves', invalid_moves)
            self.log_episode_score('visited', len(visited))
            self.log_episode_score('seen', len(seen))
            eff = 100 * sum(good_move) / max([len(good_move), 1])
            self.log_episode_score('effieciency', eff)
            exp = 100 * len(visited) / len(self.nodes)
            self.log_episode_score('exploration', exp)
            if not eff and not exp:
                self.log_episode_score(BENCH_SCORE, 0)
            else:
                self.log_episode_score(BENCH_SCORE, (2 * exp * eff) / (eff + exp))

    def store_scores(self, results_root: str, dialogue_pair: str, game_record_dir: str):
        self.store_results_file(self.scores, "scores.json",
                                dialogue_pair=dialogue_pair,
                                sub_dir=game_record_dir,
                                results_dir=results_root)

        # plotting & animation
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        path_plot = self.plot_path(self.path)
        path_plot.savefig(os.path.join(results_root, dialogue_pair, self.name, game_record_dir, "path.png"))
        plt.close()
        if not os.path.exists("tmp/step_plots"):
            os.makedirs("tmp/step_plots")
        images = []
        for i in range(len(self.path)):
            step_plot = self.plot_path(self.path[:i + 1])
            step_plot.savefig(f"tmp/step_plots/{i}.png")
            images.append(imageio.imread(f"tmp/step_plots/{i}.png"))
            plt.close()
        imageio.mimsave(os.path.join(results_root, dialogue_pair, self.name, game_record_dir, "animation.gif"), images,
                        fps=1, loop=True)
        try:
            shutil.rmtree("tmp")
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


class MmMapWorldBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    # copy this, replacing the name of the game master in the return statement
    def create_game_master(self,
                           experiment: Dict,
                           player_models: List[Model]
                           ) -> GameMaster:
        return MmMapWorld(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return MM_MapWorldScorer(self.game_name, experiment, game_instance)


def main():
    game_path = os.path.dirname(os.path.abspath(__file__))
    experiments = file_utils.load_json("in/instances.json", game_path)
