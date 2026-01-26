"""
Utils module to handle room type/image assignments to Graphs
"""
import json
from typing import Tuple, Dict, List
from collections import deque, defaultdict

import numpy as np

class MapConfigError(Exception):
    """Base class for all map config errors."""
    pass

class NodesExhaustedError(MapConfigError):
    """Raised when there aren’t enough nodes left for the requested ambiguity."""
    def __init__(self, nodes_available: List, ambiguity: List, ambiguity_region: str):
        msg = (f"Cannot assign ambiguous nodes in region - {ambiguity_region}"
               f"\nPassed ambiguity is {ambiguity} that requires at least {sum(ambiguity)} nodes in "
               f"{ambiguity_region} region but only {len(nodes_available)} node(s): {nodes_available} are available."
               f"\nSet another start/end type, set another ambiguity region "
               f"or reduce ambiguity for the selected graph type.")
        super().__init__(msg)


def load_json(json_path: str):
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    return data

def print_mapping(nx_graph):
    """
    Print a mapping of node: room_type - image_url for all nodes in the graph
    """
    for this_node in nx_graph.nodes():
        print('{}: {} - {:>50}'.format(this_node,
                                       nx_graph.nodes[this_node]['type'],
                                       nx_graph.nodes[this_node]['image']))

def select_random_type(room_types_assigned: List, category_list: List, rng: np.random.default_rng):
    """
    Select a random room type not already assigned
    Args:
        room_types_assigned: List of room types already assigned
        category_list: A list of all available room types
        rng: Random number generator
    """
    assert len(room_types_assigned) < len(category_list), \
        "Maximum number of room types already assigned, increase room categories!"

    random_room_type = rng.choice(category_list)
    while random_room_type in room_types_assigned:
        random_room_type = rng.choice(category_list)

    room_types_assigned.append(random_room_type)
    return random_room_type


def select_random_room(available_rooms: list, occupied: Tuple | None, rng: np.random.default_rng):
    """
    Pick a random room from a list of (available rooms - occupied)
    Args:
        available_rooms: List of available nodes that can be assigned a room
        occupied: A node that already has been assigned a room
        rng: Random number generator

    Returns:
        A random room chosen
    """

    if occupied in available_rooms:
        available_rooms.remove(occupied)
    return available_rooms[rng.choice(len(available_rooms))]


def find_distance(edges: List[Tuple], nodes: List) -> Dict:
    """
    Given the edges and nodes of a graph, generate distances between every node using BFS.

    Args:
        edges: List of tuples representing edges in the graph.
        nodes: List of nodes in the graph.

    Returns:
        A dictionary where distances[start][end] gives the shortest distance from start to end.
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # Dictionary to hold distances between all node pairs
    distances = {}

    # BFS from each node
    for start in nodes:
        queue = deque([(start, 0)])
        visited = set()
        dist_map = {}

        while queue:
            current, dist = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            dist_map[current] = dist

            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))

        distances[start] = dist_map

    return distances


def get_next_node(start_pos: Tuple, move: str) -> Tuple:
    """
    Get the next node after making move from a given start node
    Args:
        start_pos: current node of the agent inside MapWorld
        move: move as a string item

    Returns:
        node: node of the move as a string item
    """
    if move == "north":
        return start_pos[0], start_pos[1] - 1
    elif move == "south":
        return start_pos[0], start_pos[1] + 1
    elif move == "east":
        return start_pos[0] + 1, start_pos[1]
    elif move == "west":
        return start_pos[0] - 1, start_pos[1]
    else:
        raise ValueError(f"Invalid move! Check the parsed response! Passed value for move - {move}")

