"""Util module to handle room type/image assignments to Graphs"""

import json
import os
from typing import Tuple, Dict, List
import logging

import numpy as np
import networkx as nx

import mapworld.engine.map_utils as map_utils

logger = logging.getLogger(__name__)
# Categories.json/images.json Paths
RESOURCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
config_path = os.path.join(RESOURCES_DIR, "map_config.json")

config = map_utils.load_json(config_path)

CATEGORIES_PATH = os.path.join(RESOURCES_DIR, config["CATEGORIES_FILE"])
IMAGES_PATH = os.path.join(RESOURCES_DIR, config["IMAGES_FILE"])

# Assign the category keys from "/categories.json" here.
CATEGORY_OUTDOORS = config["OUTDOOR_CATS"]
CATEGORY_TARGETS = config["TARGET_CATS"]
CATEGORY_DISTRACTORS = config["DISTRACTOR_CATS"]


def _assign_node_degree(nx_graph: nx.Graph, node: Tuple):
    """
    Assign degree_category to each node
    degree_category: A str representing the category of node based on its degree - "indoor" or "outdoor"
    """
    node_degree = nx_graph.degree(node)
    if node_degree == 1:
        degree_category = "outdoor"
    else:
        degree_category = "indoor"
    return degree_category

def _split_nodes(nx_graph: nx.Graph) -> Tuple[List, List]:
    """
    Split nodes of the graph into indoors and outdoors based on degree of the node
    """
    indoor_nodes = []
    outdoor_nodes = []
    for node in nx_graph.nodes:
        degree = _assign_node_degree(nx_graph, node)
        if degree == "outdoor":
            outdoor_nodes.append(node)
        else:
            indoor_nodes.append(node)

    return indoor_nodes, outdoor_nodes

def _set_categories_and_nodes(nx_graph: nx.Graph, ambiguity: List, ambiguity_area: str, categories: Dict):
    """
    Set available nodes and available room categories for a given type of ambiguity_area
    Args:
        nx_graph: A networkx graph
        ambiguity: A list of ambiguity config
        ambiguity_area (object) : A string representing the ambiguity region - "random"/"indoor"/"outdoor"
        categories: A dictionary with category names as keys

    Returns:
        nodes_available: A list of available nodes based on required ambiguity area
        category_list: A list of category names to assign to nodes_available
    """
    if ambiguity_area == "random":
        # Consider all nodes and all available categories
        category_list = categories[CATEGORY_TARGETS] + categories[CATEGORY_OUTDOORS] + categories[CATEGORY_DISTRACTORS]
        nodes_available = list(nx_graph.nodes())
    elif ambiguity_area == "indoor":
        category_list = categories[CATEGORY_TARGETS] + categories[CATEGORY_DISTRACTORS]
        indoor_nodes, _ = _split_nodes(nx_graph)
        nodes_available = indoor_nodes
    else:
        category_list = categories[CATEGORY_OUTDOORS]
        _, outdoor_nodes = _split_nodes(nx_graph)
        nodes_available = outdoor_nodes

    logger.info(f"Categories available: {category_list} \n for nodes {nodes_available} \n for the chosen ambiguity "
                f"region: {ambiguity_area}")

    if len(nodes_available) < sum(ambiguity):
        raise map_utils.NodesExhaustedError(nodes_available, ambiguity, ambiguity_area)



    return category_list, nodes_available



def _assign_non_ambiguous_room_categories(
       nx_graph: nx.Graph,
       category_list: List,
       nodes_assigned: List,
       nodes_available: List,
       room_categories_assigned: List,
       rng: np.random.default_rng
)-> None:
    """
    Args:
        nx_graph: A networkx graph with exactly 1 connected component
        category_list: A list containing different room types for a given category based from
                       categories.json (targets,distractors,outdoors)
        nodes_assigned: List of nodes assigned as rooms
        nodes_available: List of nodes without any room type assignment
        room_categories_assigned: List of rooms already assigned to a category from categories_list
        rng: Random number generators
    """

    for node in nodes_available:
        degree_category = _assign_node_degree(nx_graph, node)
        random_room_type = map_utils.select_random_type(room_categories_assigned, category_list, rng)
        nodes_assigned.append(node) # Update state for later checks
        nx_graph.nodes[node]['base_type'] = degree_category
        nx_graph.nodes[node]['room_type'] = random_room_type
        nx_graph.nodes[node]['ambiguous'] = False
        logger.info(f"Assigned node - {node} with degree {degree_category} as {random_room_type} (non-ambiguous)")

def _assign_ambiguous_room_categories(
    nx_graph: nx.Graph,
    category_list: List,
    nodes_assigned: List,
    nodes_available: List,
    room_categories_assigned: List,
    rng: np.random.default_rng,
    ambiguity: List,
):
    """
     Args:
        nx_graph: A networkx graph with exactly 1 connected component
        category_list: A list containing different room types for a given category based from
                       categories.json (targets,distractors,outdoors)
        nodes_assigned: List of nodes already assigned with a room type
        nodes_available: List of nodes without any room type assignment
        room_categories_assigned: List of rooms assigned to a category from categories_list
        rng: Random number generators
        ambiguity: config list for ambiguous rooms
    """

    rng.shuffle(nodes_available)
    start_index = 0
    for amb in ambiguity:
        # pick a random room type for each val in ambiguity
        random_room_type = map_utils.select_random_type(room_categories_assigned, category_list, rng)
        room_categories_assigned.append(random_room_type)
        for i in range(amb):
            node_picked = nodes_available[start_index]
            node_degree = _assign_node_degree(nx_graph, node_picked)
            start_index += 1
            nodes_assigned.append(node_picked)
            nx_graph.nodes[node_picked]['room_type'] = random_room_type
            nx_graph.nodes[node_picked]['base_type'] = node_degree
            nx_graph.nodes[node_picked]['ambiguous'] = True
            logger.info(f"Assigned node - {node_picked} with degree {node_degree} as {random_room_type} (ambiguous)")



def _assign_room_categories(
    nx_graph: nx.Graph,
    ambiguity: list[int] = None,
    ambiguity_region: str = "random",
    categories: Dict = None,
    use_outdoor_categories: bool = False,
    rng: np.random.default_rng = None
):
    """
    Assign room categories and room type to the nodes in the generated graph.
    Example nx_graph.nodes[room] = {
        'base_type': 'indoor' or 'outdoor' based on degree of the node
        'room_type': 'k/kitchen',
        'ambiguous': True/False
    }

    Args:
        nx_graph: Generated graph. (via BaseGraph methods)
        ambiguity: List of integers to control ambiguity. Example: [3,2] means - two (len(ambiguity)) types room
        categories, in which the first category is assigned to three different nodes, and the second to two nodes.
        ambiguity_region: A str that specifies ambiguous rooms distribution between indoor nodes, outdoor nodes
                                or both
        categories: Loaded json file containing "targets", "outdoors" and "distractors" categories
        use_outdoor_categories: Assign room types from "outdoors" category to nodes with degree==1 if True,
                                else use room types from "targets"+"distractors".
        rng: Random number generator
    """

    room_categories_assigned = []  # Collect all the room types assigned
    nodes_assigned = []  # Collect all nodes that have been already assigned a node

    category_list, nodes_available = _set_categories_and_nodes(nx_graph, ambiguity, ambiguity_region, categories)
    _assign_ambiguous_room_categories(nx_graph=nx_graph,
                                 category_list=category_list,
                                 nodes_assigned=nodes_assigned,
                                 nodes_available=nodes_available,
                                 room_categories_assigned=room_categories_assigned,
                                 rng=rng,
                                 ambiguity=ambiguity)

    nodes_available = list(set(nx_graph.nodes()) - set(nodes_assigned))
    logger.info(f"Nodes available after setting ambiguous nodes: {nodes_available}")

    if use_outdoor_categories:
        logger.info(f"Assigning rooms with degree==1 from {CATEGORY_OUTDOORS} category from given "
                    f"{CATEGORIES_PATH} file. The graph can be a ladder or cycle with no nodes having degree==1.\n"
                    f"For such cases it will assign remaining available nodes with degree>1, with room types "
                    f"from {CATEGORY_OUTDOORS} category.\n"
                    f"To avoid this behaviour set use_outdoor_categories to False.")
        category_list = categories[CATEGORY_OUTDOORS]
    else:
        category_list = categories[CATEGORY_TARGETS]+categories[CATEGORY_DISTRACTORS]

    _assign_non_ambiguous_room_categories(nx_graph=nx_graph,
                                          category_list=category_list,
                                          nodes_assigned=nodes_assigned,
                                          nodes_available=nodes_available,
                                          room_categories_assigned=room_categories_assigned,
                                          rng=rng)

    nodes_available = list(set(nx_graph.nodes()) - set(nodes_assigned))
    assert len(set(nodes_available)) == 0, (f"All nodes were not assigned a room type!"
                                            f"Remaining available nodes: {nodes_available}")


def assign_room_categories(
    nx_graph: nx.Graph,
    ambiguity: list[int] = None,
    ambiguity_region: str = "random",
    use_outdoor_categories: bool = False,
    json_path: str = CATEGORIES_PATH,
    rng: np.random.default_rng = None
):
    """
    Assign room categories and room type to the nodes in the generated graph.
    Example nx_graph.nodes[room] = {
        'room_category_name': 'k/kitchen',
        'room_type': 'ambiguous', 'indoor' or 'outdoor'
    }

    Args:
        nx_graph: Generated graph. (via BaseGraph methods)
        ambiguity: List of integers to control ambiguity. Example: [3,2] means - two (len(ambiguity)) types room
        categories, in which the first category is assigned to three different nodes, and the second to two nodes.
        ambiguity_region: A str that specifies ambiguous rooms distribution between indoor nodes, outdoor nodes
                                or both
        use_outdoor_categories: Assign room types from "outdoors" category to nodes with degree==1 if True,
                                else use room types from "targets"+"distractors".
        json_path: Path to a json file containing "targets", "outdoors" and "distractors" categories
        rng: Random number generator

    Raises:
        ValueError: if total number of rooms < sum of ambiguity
    """

    logger.info(f"Assigning room categories for ambiguity = {ambiguity}, ambiguity_region = {ambiguity_region}")

    # Fixes ambiguity = None case
    if not ambiguity:
        ambiguity = [1]

    num_nodes = len(nx.nodes(nx_graph))
    if num_nodes < sum(ambiguity):
        raise ValueError(f"Total number of nodes in the map ({num_nodes}) "
                         f"is less than sum of required ambiguity ({sum(ambiguity)}).)"
                         f"\nIncrease number of rooms or set lesser ambiguity")

    categories = map_utils.load_json(json_path)
    total_categories = 0
    for k,v in categories.items():
        total_categories += len(v)

    required_categories = num_nodes - sum(ambiguity) + len(ambiguity)
    if required_categories > total_categories:
        raise ValueError(f"Total number of available categories ({total_categories}) in JSON File - {json_path}"
                         f"is less than required categories ({required_categories}), "
                         f"given the ambiguity of {ambiguity} and numer of nodes in the graph ({num_nodes})"
                         f"\nIncrease number of nodes, decrease ambiguity, or provide more categories\n")

    _assign_room_categories(nx_graph=nx_graph,
                       ambiguity=ambiguity,
                       ambiguity_region=ambiguity_region,
                       categories=categories,
                       use_outdoor_categories=use_outdoor_categories,
                       rng=rng)

    logger.info(f"Successfully assigned room categories for the required config")


def assign_images(nx_graph, json_path: str = IMAGES_PATH, rng: np.random.default_rng = None):
    """
    Assign Images from ADE20k dataset to a graph whose nodes have already been assigned a specific room type

    Args:
        nx_graph: networkx type graph containing node info - {type, base_type, target}
        json_path: Path to a jsonn file containing mapping of room_types to various images
        rng: Random number generator

    Return:
        nx_graph: Graph with updated nodes with randomly assigned image of a specific room_type
    """

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    images_assigned = []
    for node in nx_graph.nodes():
        room_type = nx_graph.nodes[node]['room_type']
        room_type = room_type.split(" ")[0]  # For ambiguous cases - remove assigned number
        random_image = rng.choice(json_data[room_type])
        while random_image in images_assigned:
            random_image = rng.choice(json_data[room_type])
        images_assigned.append(random_image)
        nx_graph.nodes[node]['image'] = random_image
