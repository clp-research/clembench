from typing import Any
import logging

import numpy as np

from mapworld.engine.graphs import BaseGraph
from mapworld.engine.map_assignments import assign_images, assign_room_categories
from mapworld.engine.map_utils import select_random_room, find_distance

logger = logging.getLogger(__name__)

class BaseMap(BaseGraph):

    def __init__(self, m: int = 3, n: int = 3, n_rooms: int = 9, graph_type: str = None, seed: int = None):
        """
        Set up a base 2-D map whose rooms are based on a given image (ADE20k) dataset.

        Args:
            m: Number of rows in the graph.
            n: Number of columns in the graph
            n_rooms: Required number of rooms. Should be less than n*m
            graph_type: Type of graph from BaseGraph methods
            seed: Random seed

        Raises:
            ValueError: If any value is unset
            AssertionError: If `n_rooms` > `n*m`
        """
        super().__init__(m, n, n_rooms, seed)
        self.graph_type = graph_type


    def set_positions(
        self,
        ambiguous_rooms: list,
        indoor_rooms: list,
        outdoor_rooms: list,
        start_type: str = "random",
        end_type: str = "random",
        distance: int = 2,
        edges: list = None
    ) -> Any | None:
        """
        Set agent position/target position.
        Based on the list of rooms provided and required room type for the node

        Args:
            ambiguous_rooms: List of rooms assigned as ambiguous
            indoor_rooms: List of rooms assigned as indoor
            outdoor_rooms: List of rooms assigned as outdoor
            start_type: Type of room that needs to be assigned from
                        - {indoor/outdoor/random/ambiguous} to agents start position
            end_type: Type of room that needs to be assigned to the target room
            distance: Distance between start and target rooms.
            edges: List of edges in the graph.

        Return:
            start_pos, target_pos - in position: (x, y)
        """
        
        all_rooms = ambiguous_rooms + indoor_rooms + outdoor_rooms
        start_pos = None
        logging.info(f"Ambiguous Rooms: {ambiguous_rooms}"
                     f"\nIndoor Rooms: {indoor_rooms}"
                     f"\nOutdoor Rooms: {outdoor_rooms}")

        if end_type == "random":
            available_rooms = all_rooms

        elif end_type == "ambiguous":
            if ambiguous_rooms:
                available_rooms = ambiguous_rooms
            else:
                logging.info(f"No ambiguous rooms available! Setting a random room as target position. "
                             f"Check graph configuration!!")
                available_rooms = indoor_rooms+outdoor_rooms

        elif end_type == "indoor":
            if indoor_rooms:
                available_rooms = indoor_rooms
            else:
                logging.info(f"No indoor rooms available! Setting a random room as target position. "
                             f"Check graph configuration!!")
                available_rooms = ambiguous_rooms + outdoor_rooms
        else:
            if end_type == "outdoor":
                available_rooms = outdoor_rooms
            else:
                logging.info(f"No outdoor rooms available! Setting a random room as target position. "
                             f"Check graph configuration!!")

                available_rooms = ambiguous_rooms + indoor_rooms

        target_pos = select_random_room(available_rooms=available_rooms, occupied=None, rng=self.graph_rng)

        node_distances = find_distance(edges, all_rooms)[target_pos]
        logging.info(f"Node distances from target position: {node_distances}")

        ## Next, find nodes at `distance` from target_pos and then look if expected start_type is available
        exact_nodes = []
        for k, v in node_distances.items():
            if v == distance:
                exact_nodes.append(k)

        if not exact_nodes:
            raise RuntimeError(f"No node found at distance {distance} from selected target_node type!"
                               f"\nAvailable nodes at distances - {node_distances}"
                               f"\nSelected node - {target_pos}")

        if len(exact_nodes) == 1:
            start_pos = exact_nodes[0]
        else:
            if start_type == "random":
                start_pos = exact_nodes[self.graph_rng.integers(len(exact_nodes))]
            elif start_type == "ambiguous":
                for node in exact_nodes:
                    if node in ambiguous_rooms:
                        start_pos = node
                if not start_pos:
                    logging.info(f"No ambiguous room found at distance {distance} from selected target_node type! "
                                 f"Setting a random room as start position at distance {distance}! ")
                    start_pos = exact_nodes[self.graph_rng.integers(len(exact_nodes))]
            elif start_type == "indoor":
                for node in exact_nodes:
                    if node in indoor_rooms:
                        start_pos = node
                if not start_pos:
                    logging.info(f"No indoor room found at distance {distance} from selected target_node type! "
                                 f"Setting a random room as start position at distance {distance}! ")
                    start_pos = exact_nodes[self.graph_rng.integers(len(exact_nodes))]
            else:
                for node in exact_nodes:
                    if node in outdoor_rooms:
                        start_pos = node
                if not start_pos:
                    logging.info(f"No outdoor room found at distance {distance} from selected target_node type! "
                                 f"Setting a random room as start position at distance {distance}! ")
                    start_pos = exact_nodes[self.graph_rng.integers(len(exact_nodes))]

        logging.info(f"Selected start node: {start_pos}, end node: {target_pos} at distance {distance}")
        return start_pos, target_pos


    def metadata(
        self,
        start_type: str = "outdoor",
        end_type: str = "outdoor",
        ambiguity: list = None,
        ambiguity_region: str = "random",
        distance: int = 2
    ) -> dict:
        """
        Generate metadata for the Graph incl. start/end points
        Args:

            start_type: "outdoor"/"indoor"/"random"/"ambiguous" - Defining agent start position
            end_type: "outdoor"/"indoor"/"random"/"ambiguous" - Defining target room position
            ambiguity: Type of ambiguity to use
            ambiguity_region: A str that specifies ambiguous rooms distribution between indoor nodes, outdoor nodes
                                or both
            distance: Distance between start and target node.
        """
        if self.graph_type=="cycle":
            nx_graph = self.create_cycle_graph()
        elif self.graph_type=="tree":
            nx_graph = self.create_tree_graph()
        elif self.graph_type=="star":
            nx_graph = self.create_star_graph()
        elif self.graph_type=="path":
            nx_graph = self.create_path_graph()
        elif self.graph_type=="ladder":
            nx_graph = self.create_ladder_graph()
        else:
            raise ValueError(f"Graph type {self.graph_type} is not supported.")

        assign_room_categories(nx_graph=nx_graph,
                               ambiguity=ambiguity,
                               ambiguity_region=ambiguity_region,
                               rng=self.graph_rng)
        assign_images(nx_graph=nx_graph, rng=self.graph_rng)

        # Metadata values
        graph_id = ""
        named_nodes = []
        unnamed_nodes = []
        named_edges = []
        unnamed_edges = []
        node_to_category = {}
        category_to_node = {}
        node_to_image = {}
        category_to_image = {}

        # Additional values
        ambiguous_rooms = []
        indoor_rooms = []
        outdoor_rooms = []

        # Nodes Metadata
        # NOTE: Every node is saved by typecasting into a string - str(node) -
        #       As this metadata gets dumped into instances.json, does not accept np.int64/tuples
        for node in nx_graph.nodes():
            # Clean Node name
            node_name = nx_graph.nodes[node]['room_type']
            node_name = " ".join(node_name.split("__"))
            node_name = " ".join(node_name.split("_"))
            node_name = node_name.capitalize()

            # Create a graph id based on node position and node category
            graph_id += str(list(node)[0]) + str(list(node)[1]) + node_name[0].lower()

            named_nodes.append(node_name)
            unnamed_nodes.append(str(node))

            node_to_category[str(node)] = node_name
            category_to_node[node_name] = str(node)
            node_to_image[str(node)] = nx_graph.nodes[node]['image']
            category_to_image[node_name] = nx_graph.nodes[node]['image']

            # Additional info
            if nx_graph.nodes[node]['ambiguous']:
                ambiguous_rooms.append(node)
            else:
                if nx_graph.nodes[node]['base_type'] == "indoor":
                    indoor_rooms.append(node)
                else:
                    outdoor_rooms.append(node)

        # Edge Metadata
        for edge in nx_graph.edges():
            named_edge = []
            unnamed_edge = []
            for e in edge:
                named_edge.append(node_to_category[str(e)])
                unnamed_edge.append(str(e))
            named_edges.append(tuple(named_edge))
            unnamed_edges.append(tuple(unnamed_edge))

        # Set Random start and Target positions
        # Can be overridden by the experiment config/game dev if required
        start_pos, target_pos = self.set_positions(ambiguous_rooms=ambiguous_rooms,
                                                   indoor_rooms=indoor_rooms,
                                                   outdoor_rooms=outdoor_rooms,
                                                   start_type=start_type,
                                                   end_type=end_type,
                                                   distance=distance,
                                                   edges=nx_graph.edges())

        map_metadata = {
            "graph_id": graph_id,
            "m": int(self.m),
            "n": int(self.n),
            "named_nodes": named_nodes,
            "unnamed_nodes": unnamed_nodes,
            "named_edges": named_edges,
            "unnamed_edges": unnamed_edges,
            "node_to_category": node_to_category,
            "node_to_image": node_to_image,
            "start_node": str(start_pos),
            "target_node": str(target_pos),
        }

        return map_metadata
