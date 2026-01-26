from typing import List, Set
from collections import deque
import logging

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class BaseGraph:

    def __init__(self, m: int = 3, n: int = 3, n_rooms: int = 9, seed: int = None):
        """
        Set up a base layout for a 2-D graph, for a given type

        Args:
            m: Number of rows in the graph.
            n: Number of columns in the graph
            n_rooms: Required number of nodes. Should be less than n*m

        Raises:
            ValueError: If any value is unset
            AssertionError: If `n_rooms` > `n*m`
        """

        assert n_rooms <= m * n, "Number of rooms cannot exceed grid size"

        logger.info(f"Initializing graph with {n_rooms} rooms, in a {m} x {n} grid")
        self.m = m
        self.n = n
        self.n_rooms = n_rooms
        self.graph_rng = np.random.default_rng(seed)


    @staticmethod
    def get_valid_neighbors(current_pos: np.array = np.array([0, 0]), visited: List|Set = None, m: int = 3, n: int = 3):
        """
        Get a list of all 'Valid' neighboring nodes.
        Valid neighboring room is defined as a node in the grid that has not been set as a room,
        and is at distance=1 from curr_pos.
        This is used by create_*graph methods to search a candidate for extending a graph

        Args:
            current_pos: Position of the current/given room.
            visited: List of visited rooms.
            m: Number of rows in the graph.
            n: Number of columns in the graph
        """

        if visited is None:
            visited = []
        possible_moves = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        valid_neighbors = []
        for move in possible_moves:
            next_pos = (current_pos[0] + move[0], current_pos[1] + move[1])

            if 0 <= next_pos[0] < m and 0 <= next_pos[1] < n and tuple(next_pos) not in visited:
                valid_neighbors.append(next_pos)

        return valid_neighbors

    def _has_star_motif(self, G: nx.Graph) -> bool:
        """Return True if any node has the 4-way star (up, down, left, right)."""
        for (x, y) in G.nodes:
            if G.degree((x, y)) == 4:
                nbrs = set(G.neighbors((x, y)))
                expected = {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)}
                if expected.issubset(nbrs):
                    return True
        return False

    def create_tree_graph(self):
        """
        Returns:
            tree_graph: nx.Graph of Tree type created using BFS,
                        guaranteed to have no 4-way star (center+4 arms).
        """
        logger.info(f"Creating tree graph with {self.m} x {self.n} rooms (star-free)")
        tree_graph = nx.Graph()
        visited = set()

        # Start node
        start_node = (self.graph_rng.integers(0, self.m), self.graph_rng.integers(0, self.n))
        queue = deque([start_node])
        visited.add(start_node)
        tree_graph.add_node(start_node)

        while len(visited) < self.n_rooms and queue:
            current_node = queue.popleft()

            # If current_node already has degree 3, do not add more children from it
            if tree_graph.degree(current_node) >= 3:
                continue

            neighbors = self.get_valid_neighbors(current_node, visited, self.m, self.n)
            self.graph_rng.shuffle(neighbors)

            for next_node in neighbors:
                if len(visited) >= self.n_rooms:
                    break

                # Adding this edge would make current_node degree +1 and next_node degree +1.
                # Since next_node is unvisited, its current degree is 0.
                if tree_graph.degree(current_node) >= 3:
                    # adding would create degree 4 => potential star center; skip
                    continue

                # Safe to add
                visited.add(next_node)
                tree_graph.add_node(next_node)
                tree_graph.add_edge(current_node, next_node)
                queue.append(next_node)

        # Final sanity check (defensive)
        assert not self._has_star_motif(tree_graph), "Tree graph accidentally contains a 4-way star."

        return tree_graph


    def create_star_graph(self):
        """
        Create a Generalized “star” configuration:
        Create an S5 star configuration then recursively add a room to a random arm of the star.

        Returns:
            star_graph: networkx Graph with exactly self.n_rooms.

        Raises:
            ValueError: If the grid or n_rooms aren’t compatible.
        """
        # Basic checks
        if self.m < 3 or self.n < 3:
            raise ValueError(f"Grid must be at least 3×3 for a star (got {self.m}×{self.n}).")
        if self.n_rooms < 5:
            raise ValueError(f"Need at least 5 rooms for a star (got {self.n_rooms}).")

        logger.info(f"Creating star graph with {self.m} x {self.n} rooms")

        star_graph = nx.Graph()
        visited = set()

        # Pick a random room with padding of 1 on the borders
        center = (self.graph_rng.integers(1, self.m-1), self.graph_rng.integers(1, self.n-1))
        star_graph.add_node(center)
        visited.add(center)

        # Add the 4 orthogonal neighbors/arms
        arms = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nb = (center[0] + dx, center[1] + dy)
            if 0 <= nb[0] < self.m and 0 <= nb[1] < self.n:
                star_graph.add_node(nb)
                star_graph.add_edge(center, nb)
                visited.add(nb)
                arms.append(nb)
        assert len(arms) == 4, "Check the configuration for arms/central room"

        # If more n_rooms remain, attach them one by one to a random arm
        while len(visited) < self.n_rooms:
            # pick a random endpoint from the current arms
            endpoint =tuple(self.graph_rng.choice(arms))
            # find its valid unvisited neighbors
            vn = [tuple(p) for p in self.get_valid_neighbors(endpoint, visited, self.m, self.n)]
            if not vn:
                # if this arm is stuck, remove it from arms and continue
                arms.remove(endpoint)
                if not arms:
                    raise ValueError("No more possible extensions; cannot place all rooms!"
                                     f"For the given {self.n_rooms}, try increasing the grid size")
                continue

            new_room = tuple(self.graph_rng.choice(vn))
            star_graph.add_node(new_room)
            star_graph.add_edge(endpoint, new_room)
            visited.add(new_room)
            arms.append(new_room)

        return star_graph


    def _create_path_graph(self):
        """
        Create a simple path (chain) of self.n_rooms nodes on the grid.

        Returns:
          path_graph: A networkx Graph whose nodes form a single chain.
        Raises:
          ValueError: if it gets stuck before placing all nodes.
        """

        path_graph = nx.Graph()
        visited = []
        # start somewhere random
        # start = (random.randrange(self.m), random.randrange(self.n))
        start = (int(self.m/2), int(self.n/2)) # Hardcode start pos for more flexible walk, more space to explore
        visited.append(start)
        path_graph.add_node(start)

        while len(visited) < self.n_rooms:
            curr = visited[-1]
            nbrs = [tuple(nb) for nb in self.get_valid_neighbors(curr, visited, self.m, self.n)]
            if not nbrs:
                raise ValueError(
                    f"Stuck at {curr} after {len(visited)} nodes; Likely due to a spiral config "
                    f"\nVisited the following nodes - {visited}"
                    f"\nCannot extend to {self.n_rooms}. Try another random seed"
                )
            nxt = tuple(self.graph_rng.choice(nbrs))
            visited.append(nxt)
            path_graph.add_node(nxt)
            path_graph.add_edge(curr, nxt)

        return path_graph

    def create_path_graph(self):
        """
        Attempt to create a path graph up to max_attempts times.
        Returns:
            path_graph: A valid path graph.
        Raises:
            RuntimeError: If all attempts fail.
        """
        max_attempts = 20
        logger.info(f"Creating path graph with {self.m} x {self.n} rooms, with max_attempts {max_attempts}")

        for attempt in range(max_attempts):
            try:
                logger.info(f"Attempt {attempt+1}...")
                path_graph = self._create_path_graph()
                logger.info(f"Success on attempt {attempt+1}")
                return path_graph
            except ValueError as e:
                logger.info(f"Attempt {attempt+1} failed: {e}")

        raise RuntimeError(
            f"Failed to create a path graph after {max_attempts} attempts. "
            f"Try adjusting parameters!!"
        )


    def create_cycle_graph(self):
        """
        Create a simple (horizontal) cycle of self.n_rooms rooms on the grid. 2 rows and n_rooms/2 cols
        Returns:
            cycle_graph: A networkx Graph whose nodes form a single loop of self.n_rooms
        """
        if self.n_rooms%2 !=0:
            raise ValueError(f"Number of rooms must be even (got {self.n_rooms}).")

        cycle_graph = nx.Graph()

        start_row = self.graph_rng.integers(0, self.n-1) # Leave 1 row
        num_cols = int(self.n_rooms/2)
        threshold_col = self.m - num_cols
        start_col = self.graph_rng.integers(0, threshold_col+1)

        # Add nodes
        for i in range(start_col, start_col+num_cols):
            for j in (start_row, start_row+1):
                cycle_graph.add_node((i, j))

        # Add vertical edges
        for i in range(start_col, start_col+num_cols-1):
            room1 = (i, start_row)
            room2 = (i+1, start_row)
            room3 = (i, start_row+1)
            room4 = (i+1, start_row+1)

            cycle_graph.add_edge(room1, room2)
            cycle_graph.add_edge(room3, room4)

        # Add 2 horizontal edges to complete the loop
        cycle_graph.add_edge((start_col, start_row), (start_col, start_row+1))
        cycle_graph.add_edge((start_col+num_cols-1, start_row), (start_col+num_cols-1, start_row+1))

        return cycle_graph


    def create_ladder_graph(self):
        """
        Create a ladder graph by adding vertical edges to a cycle graph
        Returns:
            ladder_graph: A networkx Graph in a ladder configurations
        """

        ladder_graph = self.create_cycle_graph()
        nodes = ladder_graph.nodes()
        list_nodes = list(nodes)
        edges = ladder_graph.edges()
        for node in list_nodes:
            # Check if there are any nodes 'above' the current node, if yes, then add edge if it doesn't exist
            curr_node = list(node)
            top_node = (curr_node[0], curr_node[1]+1)
            if top_node in nodes:
                edge1 = (top_node, node)
                edge2 = (node, top_node)

                if edge1 not in edges and edge2 not in edges:
                    ladder_graph.add_edge(node, top_node)

        return ladder_graph

    @staticmethod
    def plot_graph(nx_graph):
        nx.draw_networkx(nx_graph, pos={n: n for n in nx_graph.nodes()})
        plt.show()

    @staticmethod
    def save_graph(nx_graph, path: str):
        nx.draw_networkx(nx_graph, pos={n: n for n in nx_graph.nodes()})
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def __repr__(self) -> str:
        return f"<BaseMap({self.m}, {self.n}, {self.n_rooms})>"
