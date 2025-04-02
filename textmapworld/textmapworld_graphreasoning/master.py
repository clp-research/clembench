from clemcore.backends import Model, CustomResponseModel
from clemcore.clemgame import GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer, GameSpec
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE
from clemcore.utils import file_utils, string_utils

from typing import Dict, Tuple, List
import json
import numpy as np
import ast
from queue import Queue
from copy import deepcopy
import re
import random
from logging import getLogger

logger = getLogger(__name__)

from textmapworld_utils import loop_identification, get_directions, string_available_directions, have_common_element, \
    get_nextnode_label, calculate_similarity, create_graph

INVALID = 0


class PathGuesser(Player):

    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, context):
        random_path = random.choice(["north", "south", "east", "west"])
        answer = {}
        answer["action"] = f"GO: {random_path}"
        answer["graph"] = {"nodes": [], "edges": {}}
        answer = str(answer).replace("'", "\"")  # we have to undo the quote inversion of str()
        return answer


class PathDescriber(Player):

    def __init__(self, game_instance: Dict):

        super().__init__(CustomResponseModel())
        self.graph_type = game_instance['Game_Type']
        self.ambiguity = game_instance["Ambiguity"]
        self.moves = ast.literal_eval(game_instance['Moves'])
        self.directions = ast.literal_eval(game_instance['Directions'])
        self.move_construction = game_instance["Move_Construction"]
        self.stop_construction = game_instance["Stop_Construction"]
        self.nodes = ast.literal_eval(game_instance['Graph_Nodes'])
        self.edges = ast.literal_eval(game_instance['Graph_Edges'])
        self.positive_answer = game_instance["Player2_positive_answer"]
        self.negative_answer = game_instance["Player2_negative_answer"]
        self.directions_next_node = None
        self.old_node = None
        self.graph_info = None
        self.move_type = None
        self.visited_nodes = []
        self.current_node = game_instance["Current_Position"]
        self.visited_nodes.append(self.current_node)

    def check_path_answer(self, utterance: str, directions: List[str], node, saved_node) -> List[Dict]:
        previous_direction = get_directions(node, directions, saved_node)
        previous_dirrection_changed = string_available_directions(previous_direction)
        previous_dirrection_no_pq = string_utils.remove_punctuation(previous_dirrection_changed)
        if not have_common_element(utterance, previous_dirrection_no_pq):
            return [{
                "message": f"The desired direction is not in available paths",
                "type": 0}]

    def validate_answer(self, utterance):
        "Check if the direction is valid"
        the_last_node = self.visited_nodes[-1]
        self.old_node = the_last_node
        errors = self.check_path_answer(utterance, self.directions, the_last_node, self.old_node)
        if errors:
            error = errors[0]
            self.game_error = error
            self.directions_next_node = get_directions(the_last_node, self.directions, the_last_node)
            self.directions_next_node = string_available_directions(self.directions_next_node)
            return "not valid"
        else:
            next_node_label, self.move_type = get_nextnode_label(self.moves, the_last_node, utterance,
                                                                 self.move_construction)
            self.current_node = next_node_label
            if next_node_label in self.nodes:
                self.visited_nodes.append(next_node_label)
                list_directions_nextnode = get_directions(next_node_label, self.directions, self.current_node)
                self.directions_next_node = string_available_directions(list_directions_nextnode)
                return True

    def turn_information(self):
        "Returns the information of the current turn"
        turn_info = {}
        turn_info["from"] = self.old_node
        turn_info["to"] = self.current_node
        turn_info["graph"] = self.graph_info
        return turn_info

    def _custom_response(self, context) -> str:
        "Generate the response for the player"
        try:
            content = ast.literal_eval(context["content"])["action"]
            self.graph_info = ast.literal_eval(context["content"])["graph"]
        except:
            return "Game needs to be aborted"
        utterance = None
        move = re.search(self.move_construction, content, re.IGNORECASE)
        if move:
            utterance = move.group(1)
        logger.info(f"context: {context}")
        logger.info(f"utterance: {utterance}")
        validation = self.validate_answer(utterance)
        if self.directions_next_node == None:
            return "Game needs to be aborted"
        if self.current_node == None:
            return "Game needs to be aborted"
        current_location = self.current_node
        if self.ambiguity != None:
            current_location = self.current_node.split("_")[
                0]  # because if there is ambiguity, the node is saved as "Kitchen_(1,2)"
        if validation != "not valid":
            positive_answer = self.positive_answer
            positive_answer = positive_answer.replace("$DIRECTIONS$", self.directions_next_node)
            positive_answer = positive_answer.replace("$ANOTHER_ROOM$", current_location)
            utterance = positive_answer
        else:
            negative_answer = self.negative_answer
            negative_answer = negative_answer.replace("$DIRECTIONS$", self.directions_next_node)
            negative_answer = negative_answer.replace("$SAME_ROOM$", current_location)
            utterance = negative_answer
        return utterance


class Graphreasoning(DialogueGameMaster):
    """This class implements a graph traversal game in which player A (DecisionMaker)."""

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)
        self.steps_made = 0
        self.max_turns = 20
        self.game_error = None
        self.game_stop = False
        self.invalid_response = False
        self.limit_reached = False
        self.non_processable = False

    def _on_setup(self, **game_instance):

        logger.info("_on_setup")
        self.graph_type = game_instance['Game_Type']
        self.initial_position = game_instance["Current_Position"]
        self.playerA_initial_prompt = game_instance["Prompt"]
        self.directions = ast.literal_eval(game_instance['Directions'])
        self.ambiguity = game_instance["Ambiguity"]
        self.move_construction = game_instance["Move_Construction"]
        self.stop_construction = game_instance["Stop_Construction"]
        self.response_regex = game_instance["Response_Construction"]

        self.guesser = PathGuesser(self.player_models[0])
        self.describer = PathDescriber(game_instance)
        self.add_player(self.guesser)
        self.add_player(self.describer)

        self.reprompting_parameter = game_instance["Loop_Reminder"]
        self.loop_reprompting = game_instance["Loop_Reminder_Text"]
        self.maxturns_parameter = game_instance["Max_Turns_Reminder"]
        self.max_turns_reprompting = game_instance["Max_Turns_Reminder_Text"]

    def _on_before_game(self):
        if "$INITIAL_ROOM$" in self.playerA_initial_prompt:
            initial_directions = self.initial_position
            if self.ambiguity != None:
                initial_directions = self.initial_position.split("_")[0]
            self.playerA_initial_prompt = self.playerA_initial_prompt.replace("$INITIAL_ROOM$", initial_directions)
        self.initial_directions = get_directions(self.initial_position, self.directions, self.initial_position)
        self.changed_initial_directions = string_available_directions(self.initial_directions)
        self.playerA_initial_prompt = self.playerA_initial_prompt.replace("$INITIAL_DIRECTIONS$",
                                                                          self.changed_initial_directions)
        self.set_context_for(self.guesser, self.playerA_initial_prompt)
        self.visited_nodes = []
        self.visited_nodes.append(self.initial_position)

    def _does_game_proceed(self):

        if self.non_processable:
            self.log_to_self("non_processable", "The answer is not processable")
            self.log_to_self("aborted", "abort game")
            return False

        if self.invalid_response:
            self.log_to_self("aborted", "abort game")
            return False

        if self.game_stop:
            self.log_to_self("stop", "The guesser decided to stop the game")
            return False

        if self.game_error is not None:
            error_type = self.game_error["type"]
            if error_type == 0:
                self.log_to_self("Direction not available", "The desired direction is not in available paths")
            return False

        if self.current_round >= self.max_turns:
            self.log_to_self("turns_limit", str(self.max_turns))
            self.limit_reached = True
            return False

        return True

    def _parse_response(self, player: Player, utterance: str) -> str:
        if player == self.guesser:
            found = None
            utterance = utterance.replace("\n", "").strip()
            if re.search(self.response_regex, utterance, re.IGNORECASE):
                found = re.search(self.response_regex, utterance, re.IGNORECASE)
            if found:
                utterance = found.group()
            self.log_to_self("parse", utterance)
        return utterance

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        if player == self.guesser:
            utterance = utterance.replace("\n", "").strip()
            first_filter = re.search(self.response_regex, utterance, re.IGNORECASE)
            if not first_filter:
                self.invalid_response = True
                return False
            else:
                action = first_filter.group(1)
                action_stop = re.search(self.stop_construction, action, re.IGNORECASE)
                action_move = re.search(self.move_construction, action, re.IGNORECASE)
                if action_move and action_stop:
                    self.invalid_response = True
                    return False
                if action_stop:
                    self.game_stop = True
                    return True
                count_go = re.findall(self.move_construction, action, re.IGNORECASE)
                if len(count_go) > 1:
                    self.invalid_response = True
                    return False
                if not action_move and not action_stop:
                    self.invalid_response = True
                    return False
                if action_move:
                    return True

        if player == self.describer:
            if utterance == "Game needs to be aborted":
                self.invalid_response = True
                return False
        self.log_to_self("Valid format", "Continue")
        return True

    def _on_valid_player_response(self, player: Player, utterance: str):
        """Add the utterance to other player's history, if necessary.
        To do this use the method add_user_message(other_player,utterance)."""
        if player == self.guesser:
            self.set_context_for(self.describer, utterance)
        if player == self.describer:
            if self.reprompting_parameter and loop_identification(self.visited_nodes, False):
                self.log_to_self("loop_detected", "Loop detected in the visited nodes list")
                self.reprompting_parameter = False
                utterance = self.loop_reprompting + "\n" + utterance
            if self.maxturns_parameter and self.current_round == self.max_turns - 2:
                self.maxturns_parameter = False
                utterance = self.max_turns_reprompting + "\n" + utterance
            self.set_context_for(self.guesser, utterance)

    def _on_after_round(self):
        turn_dict = self.describer.turn_information()
        old_node = turn_dict["from"]
        new_node = turn_dict["to"]
        graph_turn = turn_dict["graph"]
        if old_node != new_node:
            if not self.invalid_response and not self.limit_reached and not self.game_error:
                self.log_to_self(type_="move", value=json.dumps({"old": old_node, "new": new_node}))
                self.visited_nodes.append(new_node)
                if self.reprompting_parameter and loop_identification(self.visited_nodes, False):
                    self.visited_nodes.clear()
                    self.reprompting_parameter = True
                self.log_to_self(type_="graph", value=str(graph_turn))


class GraphGameScorer(GameScorer):

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.nodes = ast.literal_eval(game_instance['Graph_Nodes'])
        self.game_type = game_instance['Game_Type']
        self.ambiguity = game_instance['Ambiguity']
        self.old_edges = ast.literal_eval(game_instance['Graph_Edges'])
        new_edges = [(edge[1], edge[0]) for edge in self.old_edges]
        new_edges.extend(self.old_edges)
        self.edges = new_edges
        self.start = game_instance["Current_Position"]
        self.mapping = ast.literal_eval(game_instance['Mapping'])
        self.graph_data = {}

    def visited_all(self, visited, to_visit):
        return all([n in visited for n in to_visit])

    def get_available_moves(self, node):
        return [edge for edge in self.edges if node == edge[0]]

    def adj(self, node):
        return set([ed[1] for ed in self.edges if ed[0] == node])

    def find_best_moves(self, current, visited):

        to_visit = [ed[1] for ed in self.edges if ed[0] in visited and ed[1] not in visited]
        start = [current]
        q = Queue()
        q.put(start)
        found = set()
        max_len = 100
        while True:
            n = q.get()
            if len(n) > max_len:
                break
            if self.visited_all(n, to_visit):
                if len(n) > 1:
                    found.add((n[0], n[1]))
                    max_len = len(n)

            avail = self.get_available_moves(n[-1])
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

    def compute_scores(self, episode_interactions) -> None:

        current = self.start
        seen = {self.start}
        seen.update(self.adj(self.start))
        visited = {self.start}
        valid_moves = 0
        invalid_moves = 0
        stopped = False
        aborted = False
        turns_limit_reached = False
        good_move = []
        graphs_similarity = []
        graphs = []
        loops = []
        count_loops = 0
        count_graphs = 0
        content = []
        loops.append(self.start)

        for turn in episode_interactions["turns"]:
            for event in turn:
                action = event["action"]
                if action["type"] == "non_processable":
                    self.non_processable = True
                if action["type"] == "aborted":
                    if action["content"]:
                        aborted = True
                if action['type'] == "move":
                    cont = json.loads(action['content'])
                    if not cont["old"] == cont["new"]:
                        content.append(cont["old"])
                        valid_moves += 1
                    else:
                        invalid_moves += 1
                    best_moves = self.find_best_moves(current, visited)
                    new_move = cont["new"]
                    if (current, new_move) in best_moves:
                        if len(visited) != len(self.nodes):
                            good_move.append(True)
                        else:
                            good_move.append(False)
                    else:
                        good_move.append(False)
                    current = cont["new"]
                    seen.update(self.adj(current))
                    loops.append(current)
                    visited.add(current)
                    if loop_identification(loops):
                        count_loops += 1

                # if event['from']== "Player 1" and event["to"]== "GM" and action['type'] == "get message":
                # content.append(ast.literal_eval(action['content'])["action"])

                if action['type'] == "graph":
                    try:
                        count_graphs += 1
                        self.graph_data["goal"] = self.mapping
                        self.graph_data[count_graphs] = {}
                        cont = ast.literal_eval(action['content'])
                        G1 = create_graph(self.nodes, self.edges, "original")
                        G2 = create_graph(cont['nodes'], cont['edges'], "generated")
                        self.graph_data[count_graphs]["generated"] = cont
                        similarity_percent = calculate_similarity(G1, G2) * 100
                        self.graph_data[count_graphs]["similarity"] = similarity_percent
                        graphs_similarity.append(similarity_percent)
                        graphs.append(cont)
                    except:
                        graphs_similarity.append(0)

                if action['type'] == "stop":
                    if action["content"]:
                        stopped = True

                if action['type'] == "turns_limit":
                    turns_limit_reached = True

        for i, val in enumerate(good_move):
            self.log_turn_score(i, "effiencient_move", val)

        for i, val in enumerate(graphs_similarity):
            self.log_turn_score(i, "similarity", val)
        if aborted:
            self.log_episode_score(METRIC_ABORTED, 1)
            self.log_episode_score(METRIC_SUCCESS, 0)
            self.log_episode_score(METRIC_LOSE, 0)
        else:
            if not stopped:
                self.log_episode_score(METRIC_ABORTED, 1)
                self.log_episode_score(METRIC_SUCCESS, 0)
                self.log_episode_score(METRIC_LOSE, 0)
            else:
                if visited == set(self.nodes):
                    self.log_episode_score(METRIC_SUCCESS, 1)
                    self.log_episode_score(METRIC_ABORTED, 0)
                    self.log_episode_score(METRIC_LOSE, 0)
                else:
                    self.log_episode_score(METRIC_SUCCESS, 0)
                    self.log_episode_score(METRIC_ABORTED, 0)
                    self.log_episode_score(METRIC_LOSE, 1)

        exploration = (len(visited) / len(self.nodes) * 100) if len(self.nodes) else 0
        efficiency = (sum(good_move) / len(good_move) * 100) if good_move else 0
        bench_score = (2 * efficiency * exploration / (efficiency + exploration)) if (efficiency + exploration) else 0
        self.log_episode_score('moves', valid_moves + invalid_moves if stopped else np.NaN)
        self.log_episode_score('valid_moves', valid_moves if stopped else np.NaN)
        self.log_episode_score('invalid_moves', invalid_moves if stopped else np.NaN)
        self.log_episode_score('stopped', int(stopped) if stopped else np.NaN)
        self.log_episode_score('turns_limit', int(turns_limit_reached) if stopped else np.NaN)
        self.log_episode_score('loops', count_loops if stopped else np.NaN)
        self.log_episode_score('number_visited', len(visited) if stopped else np.NaN)
        self.log_episode_score('seen', len(seen) if stopped else np.NaN)
        self.log_episode_score('efficiency', efficiency if stopped else np.NaN)
        self.log_episode_score('exploration', exploration if stopped else np.NaN)
        if graphs_similarity and stopped:
            self.log_episode_score('graph_similarity', graphs_similarity[-1] if stopped else np.NaN)
        else:
            self.log_episode_score('graph_similarity', 0 if stopped else np.NaN)
        self.log_episode_score(BENCH_SCORE, bench_score if stopped else np.NaN)


class GraphGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return Graphreasoning(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return GraphGameScorer(self.game_name, experiment, game_instance)


def main():
    # select one experiment and instance
    experiments = file_utils.load_json("in/instances_graphreasoning.json", "textmapworld")
    experiment_1 = experiments["experiments"][0]
    game_1 = experiment_1["game_instances"]
    master = Graphreasoning(experiment_1, ["mock", "mock"])
    master.setup(**game_1)
    master.play()


if __name__ == '__main__':
    main()
