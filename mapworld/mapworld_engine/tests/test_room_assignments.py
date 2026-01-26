import os
import json
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from engine.map_assignments import assign_room_categories, assign_images
from engine.map_utils import load_json
from engine.maps import BaseMap

class TestRoomAssignments(unittest.TestCase):
    def setUp(self):
        self.n_rooms = 5
        self.map = BaseMap(10,10, self.n_rooms, 'path', 42)
        self.G = self.map.create_path_graph()
        self.json_path = os.path.join("engine", "resources", "categories.json")
        self.categories = load_json(self.json_path)

    def test_basic_assignment_all_nodes_get_type(self):
        assign_room_categories(
            self.G,
            ambiguity=[2,1],
            ambiguity_region="random",
            use_outdoor_categories=False,
            json_path=self.json_path,
            rng=self.map.rng
        )
        # every node has type/base_type/ambiguous
        for n in self.G.nodes:
            node = self.G.nodes[n]
            self.assertIn("room_type", node)
            self.assertIn("base_type", node)
            self.assertIn("ambiguous", node)

        # exactly 3 ambiguous
        amb_count = sum(self.G.nodes[n]["ambiguous"] for n in self.G.nodes)
        self.assertEqual(amb_count, 3)
        # remainder non-ambiguous
        non_amb = self.n_rooms - 3
        non_amb_count = sum(not self.G.nodes[n]["ambiguous"] for n in self.G.nodes)
        self.assertEqual(non_amb_count, non_amb)

    def test_use_outdoor_categories_nodes_only(self):
        # nodes have degree==1
        assign_room_categories(
            self.G,
            ambiguity=[0],                   # no ambiguous
            ambiguity_region="random",
            use_outdoor_categories=True,
            json_path=self.json_path,
            rng=self.map.rng
        )
        # check that both nodes picked from outdoors
        nodes = [n for n,d in self.G.degree() if d==1]
        for node in nodes:
            typ = self.G.nodes[node]["room_type"]
            self.assertIn(typ, self.categories["outdoors"])

    def test_ambiguity_region_indoor_only(self):
        assign_room_categories(
            self.G,
            ambiguity=[2],
            ambiguity_region="indoor",       # only interior
            use_outdoor_categories=False,
            json_path=self.json_path,
            rng=self.map.rng
        )
        for n in self.G.nodes:
            if self.G.nodes[n]["ambiguous"]:
                self.assertTrue(self.G.degree[n] > 1)

    def test_error_when_ambiguity_too_large(self):
        with self.assertRaises(ValueError):
            assign_room_categories(
                nx.path_graph(2),
                ambiguity=[2,1],             # 3 > 2 nodes
                rng=self.map.rng
            )

    def test_assign_images_default_path_unique_and_correct(self):
        assign_room_categories(
            self.G,
            ambiguity=[2, 1],
            ambiguity_region="random",
            use_outdoor_categories=False,
            json_path=self.json_path,
            rng=self.map.rng
        )
        assign_images(self.G, rng=self.map.rng)

        # Verify every node got an image
        imgs = [self.G.nodes[n]["image"] for n in sorted(self.G.nodes)]
        self.assertEqual(len(imgs), self.n_rooms)
        # All images are unique
        self.assertEqual(len(set(imgs)), self.n_rooms)


if __name__ == "__main__":
    unittest.main()
