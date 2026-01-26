import json
from typing import Tuple, Dict, List
from collections import deque
import logging
import os
import ast

import numpy as np
from clemcore.clemgame import GameScorer
from clemcore.clemgame import metrics as ms

logger = logging.getLogger(__name__)

# Define min_q for ambiguous rooms
min_q_mapping = {
    "no_ambiguity": 1,
    "medium_ambiguity": 3,
    "high_ambiguity": 4,
    "low_dual_ambiguity": 4,
    "medium_dual_ambiguity": 6,
    "high_dual_ambiguity": 8,
}

def get_neighbors(current_node, edges):
    neighbors = []
    for edge in edges:
        node1 = ast.literal_eval(edge[0])
        node2 = ast.literal_eval(edge[1])
        if node1 == current_node:
            if node2 not in neighbors:
                neighbors.append(node2)
        if node2 == current_node:
            if node1 not in neighbors:
                neighbors.append(node1)

    return neighbors

def get_neighbors_str(agent_room: Tuple, edges: List):
    """

    Args:
        agent_room: Current position of Explorer agent
        edges: All edges in the given graph

    Returns:
        A list of neighboring rooms of agent_room in the given graph
    """

    neighbors = []

    for u, v in edges:
        if u == agent_room:
            neighbors.append(v)
        elif v == agent_room:
            neighbors.append(u)

    return neighbors


def normalize_edges(raw_edges: List[List[str]]):
    def parse(s: str):
        # from "(4, 0)" → (4,0)
        x, y = s.strip('()').split(',')
        return (int(x), int(y))
    return [(parse(a), parse(b)) for a,b in raw_edges]

def unexplored_distance(neighbors: List[Tuple[int, int]],
                        visited_rooms: List[Tuple[int, int]],
                        map_edges: List) -> List[Dict]:
    """
    Args:
        neighbors: Neighbors of Explorer
        visited_rooms: Rooms already visited by the explorer
        map_edges: All edges in the given graph

    Returns:
        dist_to_unexplored: A dict containing the distance to the unexplored rooms from each neighbor of the Explorer
    """
    logger.info(f"Finding distances to unexplored rooms for neighbors = {neighbors}"
                f"\nvisited_rooms = {visited_rooms}"
                f"\nmap_edges = {map_edges}")
    distances = []
    for nbr in neighbors:
        # BFS from this neighbor
        queue = deque([(nbr, 0)])
        seen = set(nbr)
        dist_to_unexplored = None
        while queue:
            room, d = queue.popleft()
            if room not in visited_rooms:
                dist_to_unexplored = d
                break
            for nxt in get_neighbors(room, map_edges):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, d + 1))

        distances.append({"neighbor": nbr, "dist": dist_to_unexplored})

    if not distances:
        raise RuntimeError(f"No unexplored rooms reachable !! Check efficient move method")

    return distances


def is_efficient_move(next_room: Tuple[int, int],
                      neighbors: List[Tuple[int, int]],
                      visited_rooms: List[Tuple[int, int]],
                      target_observed: bool,
                      map_edges) -> bool:

    """
    Args:
        next_room: Next position of the Explorer Agent after making move
        neighbors: Neighboring Rooms from the Explorer Agent's current position
        visited_rooms: rooms visited by explorer
        target_observed: A flag to indicate whether the target room has already been observed by the explorer
        map_edges: Edges of a networkX based graph

    NOTE: Presupposes that after making 'move' the agent ends up in one of the rooms in 'neighbors'
    TODO: Handle re-prompting when move made is to an invalid neighbor
    Returns:
        True if the move mase is efficient, False otherwise
    """

    # Check 1: Any move made after reaching target_room is inefficient
    # The agent, once reaching the target room, should ask questions and escape
    if target_observed:
        return False

    # Check 2: Any move made from a room with degree=1 is efficient
    # Only possible move
    if len(neighbors) == 1:
        return True

    # Check 3.a: Any move to an unexplored room is efficient
    if next_room not in visited_rooms:
        return True

    # Check 3.b: If any neighbor is unexplored, choosing a visited one is inefficient
    for nbr in neighbors:
        if nbr not in visited_rooms:
            return False


    # Check 4: Among visited neighbors, pick the one closest to any unexplored room
    dists = unexplored_distance(neighbors, visited_rooms, map_edges)
    # find minimal distance
    min_dist = min(d['dist'] for d in dists)
    # check if chosen next_room achieves that minimal
    for entry in dists:
        if entry['neighbor'] == next_room and entry['dist'] == min_dist:
            return True
    return False

def get_metadata(instances, exp_name, game_id):
    metadata = None
    for exp in instances["experiments"]:
        if exp["name"] == exp_name:
            all_instances = exp["game_instances"]
            for inst in all_instances:
                if inst["game_id"] == game_id:
                    metadata = inst

    return metadata

def get_next_node(current_node, edges, move):
    next_node = None
    if move == 'north':
        next_node = (current_node[0], current_node[1]-1)
    elif move == 'south':
        next_node = (current_node[0], current_node[1]+1)
    elif move == 'east':
        next_node = (current_node[0]+1, current_node[1])
    elif move == 'west':
        next_node = (current_node[0]-1, current_node[1])
    else:
        print("Invalid move! - Expected - north, south, east, west, got - {}".format(move))

    edge1 = [str(current_node), str(next_node)]
    edge2 = [str(next_node), str(current_node)]

    if edge1 in edges:
        return next_node
    if edge2 in edges:
        return next_node

    return None


def get_efficient_moves(instances, exp_name, game_id, moves_made):
    aborted = False
    metadata = get_metadata(instances, exp_name, game_id)
    unnamed_edges = metadata["unnamed_edges"]
    start_node = ast.literal_eval(metadata["start_node"])
    current_node = start_node
    target_observed = False

    visited_rooms = [list(start_node)]
    total_moves = 0
    eff_moves = 0
    for i in range(len(moves_made)):
        move = moves_made[i]
        next_node = get_next_node(current_node, unnamed_edges, move)
        if next_node is not None:
            neighbors = get_neighbors(current_node, unnamed_edges)
            eff_move = is_efficient_move(next_node, neighbors, visited_rooms, target_observed, unnamed_edges)
            if eff_move:
                eff_moves += 1
            if str(next_node) == metadata["target_node"]:
                target_observed = True
            current_node = next_node
            if next_node not in visited_rooms:
                visited_rooms.append(next_node)
            total_moves += 1
        else:
            if i!=len(moves_made)-1:
                print(f"Invalid response for a model - {exp_name, game_id, moves_made} for move {move}")
                aborted = True

    return total_moves, eff_moves, aborted


class EscapeRoomScorer(GameScorer):
    """
    Scorer class for Escape Room Game
    """

    def __init__(self, game_name:str, experiment:Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        """
        Method to compute scores for Escape Room Game

        Move_Efficiency = (efficient moves * 100) / (total number of moves)
        Question_Efficiency = 100 / Total Questions
        QualityScore = HarmonicMean(Move_Efficiency, Question_Efficiency)

        """
        total_moves = 0
        efficient_moves = 0
        total_questions = 0
        success = False
        aborted = False

        exp_name = episode_interactions['meta']["experiment_name"]
        game_id = episode_interactions['meta']["game_id"]
        instance_file = os.path.join("mapworld", "escapegame", "in", "instances.json")
        with open(instance_file, "r") as f:
            instances = json.load(f)

        if exp_name in min_q_mapping:
            min_q = min_q_mapping[exp_name]
        else:
            min_q = 2
        logger.info(f"Setting min_q to {min_q} for {exp_name}")

        all_turn_scores = []
        moves_made = []
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            turn_score_dict = {
                "request_count": 0,
                "violated_request_count": 0,
                "parsed_request_count": 0,
            }

            # Walk through log_to_self items from DGM
            for event in turn:
                action = event["action"]
                turn_score_dict["request_count"] += 1
                if action["type"] == "invalid value":
                    turn_score_dict["violated_request_count"] += 1
                    aborted = True
                elif action["type"] == "turns exceeded":
                    aborted = True
                else:
                    turn_score_dict["parsed_request_count"] += 1
                    if action["type"] == "question":
                        total_questions += 1
                    elif action["type"] == "escape" and action["content"] == "success":
                        success = True
                    elif action["type"] == "get message":
                        action_content = action["content"].lower()
                        if action_content.startswith("move"):
                            move = action_content[5:].strip()
                            moves_made.append(move)


            # log turn request scores
            self.log_round_score(turn_idx, ms.METRIC_REQUEST_COUNT_VIOLATED, turn_score_dict["violated_request_count"])
            self.log_round_score(turn_idx, ms.METRIC_REQUEST_COUNT_PARSED, turn_score_dict["parsed_request_count"])
            self.log_round_score(turn_idx, ms.METRIC_REQUEST_COUNT, turn_score_dict["request_count"])
            all_turn_scores.append(turn_score_dict)

        last_turn = episode_interactions["turns"][-1][-1]
        if not aborted and last_turn["action"]["type"] != "escape":
            aborted = True
        # Log episodic scores
        ep_request_count = 0
        ep_violated_request_count = 0
        ep_parsed_request_count = 0
        for s in all_turn_scores:
            ep_request_count += s["request_count"]
            ep_violated_request_count += s["violated_request_count"]
            ep_parsed_request_count += s["parsed_request_count"]

        self.log_episode_score(ms.METRIC_REQUEST_COUNT, ep_request_count)
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, ep_violated_request_count)
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, ep_parsed_request_count)

        total_moves, efficient_moves, aborted_temp = get_efficient_moves(instances, exp_name, game_id, moves_made)
        if aborted_temp and not aborted:
            aborted = True
        if aborted:
            self.log_episode_score(ms.METRIC_ABORTED, 1)
            self.log_episode_score(ms.METRIC_SUCCESS, 0)
            self.log_episode_score(ms.METRIC_LOSE, 0)
            self.log_episode_score(ms.BENCH_SCORE, np.nan)
        else:
            self.log_episode_score(ms.METRIC_ABORTED, 0)

            if success:
                self.log_episode_score(ms.METRIC_SUCCESS, 1)
                self.log_episode_score(ms.METRIC_LOSE, 0)


                total_questions = max(min_q, total_questions) # Set to min_q, if no questions asked
                if not total_moves:
                    self.log_episode_score(ms.BENCH_SCORE, 0)
                else:
                    move_efficiency = (efficient_moves * 100) /total_moves
                    if move_efficiency > 100:
                        logger.info(f"move_efficiency > 100 - {move_efficiency}")
                        logger.info(f"")
                    question_efficiency = 100*min_q/total_questions
                    if question_efficiency > 100:
                        logger.info(f"question_efficiency > 100 - {question_efficiency}, total questions: {total_questions}")
                    quality_score = 2*move_efficiency*question_efficiency/(move_efficiency + question_efficiency)
                    self.log_episode_score(ms.BENCH_SCORE, quality_score)

            else:
                self.log_episode_score(ms.METRIC_SUCCESS, 0)
                self.log_episode_score(ms.METRIC_LOSE, 1)

                self.log_episode_score(ms.BENCH_SCORE, 0)
