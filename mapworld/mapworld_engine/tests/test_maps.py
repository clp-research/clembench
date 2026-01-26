import os
import logging
import unittest

from engine.maps import BaseMap


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=os.path.join('engine', 'tests', 'test_maps.log'),
                    filemode='w')


class MapGenerationTest(unittest.TestCase):

    def setUp(self):
        self.start_types = ["ambiguous", "indoor", "outdoor", "random"]
        self.target_types = ["ambiguous", "indoor", "outdoor", "random"]
        self.seed = 10


    def testPathMap(self):
        for n_rooms in range(8,9):
            for dist in range(1,5):
                for amb in ([1], [2], [3], [4]):
                    for seed in range(1,self.seed+1):
                        map = BaseMap(10,10, n_rooms=n_rooms, graph_type="path", seed=seed)
                        if sum(amb) <= n_rooms:
                            map_metadata = map.metadata(start_type="indoor",
                                                        end_type="ambiguous",
                                                        ambiguity=amb,
                                                        ambiguity_region="random",
                                                        distance=dist)


    def testStarMap(self):
        for n_rooms in range(6,7):
            for dist in range(2,3):
                for amb in [[2,2]]:
                    for seed in range(1,self.seed+1):
                        map = BaseMap(10,10, n_rooms=n_rooms, graph_type="star", seed=seed)
                        if sum(amb) <= n_rooms:
                            map_metadata = map.metadata(start_type="ambiguous",
                                                        end_type="ambiguous",
                                                        ambiguity=amb,
                                                        ambiguity_region="indoor",
                                                        distance=dist)

    def testCycleMap(self):
        for n_rooms in range(8,9):
            for dist in range(1,5):
                for amb in ([1], [2], [3], [4]):
                    for seed in range(1,self.seed+1):
                        map = BaseMap(10,10, n_rooms=n_rooms, graph_type="cycle", seed=seed)
                        cycle_graph = map.create_cycle_graph()
                        if sum(amb) <= n_rooms:
                            map_metadata = map.metadata(start_type="indoor",
                                                        end_type="ambiguous",
                                                        ambiguity=amb,
                                                        ambiguity_region="indoor",
                                                        distance=dist)


if __name__ == '__main__':
    unittest.main()