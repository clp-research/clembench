import unittest
import os
import logging
import networkx as nx

from mapworld.engine.graphs import BaseGraph

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=os.path.join('mapworld', 'engine', 'tests', 'test_graphs.log'),
                    filemode='w')

IMG_PTH = os.path.join('mapworld', 'engine', 'tests', 'graph_images')
os.makedirs(IMG_PTH, exist_ok=True)

"""
Fix graph size at 10x10
"""

class GraphGenerationTest(unittest.TestCase):

    def setUp(self):
        self.grid_size = 10
        self.seed_val = 42

    def test_tree_graph(self):
        for n_rooms in range(4,11):
            for seed in range(1,self.seed_val+1):
                base_graph = BaseGraph(m=self.grid_size, n=self.grid_size, n_rooms=n_rooms, seed=seed)
                tree_graph = base_graph.create_tree_graph()
                cycles = nx.cycle_basis(tree_graph)
                connected_comps = nx.number_connected_components(tree_graph)
                assert len(cycles) == 0
                assert connected_comps == 1

    def test_path_graph(self):
        for n_rooms in range(4,11):
            for seed in range(1,self.seed_val+1):
                base_graph = BaseGraph(m=self.grid_size, n=self.grid_size, n_rooms=n_rooms, seed=seed)
                path_graph = base_graph.create_path_graph()
                cycles = nx.cycle_basis(path_graph)
                connected_comps = nx.number_connected_components(path_graph)
                assert len(cycles) == 0
                assert connected_comps == 1

    def test_star_graph(self):
        for n_rooms in range(5,11): # at least 5 required for a star
            for seed in range(1,self.seed_val+1):
                base_graph = BaseGraph(m=self.grid_size, n=self.grid_size, n_rooms=n_rooms, seed=seed)
                star_graph = base_graph.create_star_graph()
                cycles = nx.cycle_basis(star_graph)
                connected_comps = nx.number_connected_components(star_graph)
                assert len(cycles) == 0
                assert connected_comps == 1

    def test_cycle_graph(self):
        for n_rooms in range(4, 11, 2):
            for seed in range(1,self.seed_val+1):
                base_graph = BaseGraph(m=self.grid_size, n=self.grid_size, n_rooms=n_rooms, seed=seed)
                cycle_graph = base_graph.create_cycle_graph()
                cycles = nx.cycle_basis(cycle_graph)
                assert len(cycles) == 1
                connected_comps = nx.number_connected_components(cycle_graph)
                assert connected_comps == 1

    def test_ladder_graph(self):
        cycle_map = {
            4: 1,
            6: 2,
            8: 3,
            10: 4
        }
        for n_rooms in range(4, 11, 2):
            for seed in range(1,self.seed_val+1):
                base_graph = BaseGraph(m=self.grid_size, n=self.grid_size, n_rooms=n_rooms, seed=seed)
                ladder_graph = base_graph.create_ladder_graph()
                cycles = nx.cycle_basis(ladder_graph)
                assert len(cycles) == cycle_map[n_rooms]
                connected_comps = nx.number_connected_components(ladder_graph)
                assert connected_comps == 1

if __name__ == '__main__':
    unittest.main()







