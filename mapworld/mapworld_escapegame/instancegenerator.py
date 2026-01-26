import os

import numpy as np
from clemcore.clemgame import GameInstanceGenerator

from mapworld.engine.maps import BaseMap

# CONFIG
N = 10 # Number of instances per experiment
np_rng = np.random.default_rng(seed=12)
random_seeds = [np_rng.integers(1,1000) for i in range(N)]
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")


def _make_native(obj):
    if isinstance(obj, dict):
        return { _make_native(k): _make_native(v) for k, v in obj.items() }
    elif isinstance(obj, list):
        return [ _make_native(i) for i in obj ]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class EscapeRoomInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(os.path.abspath(__file__)))


    def on_generate(self, seed=None, **kwargs):
        seeker_prompt = self.load_template(os.path.join(RESOURCES_DIR, "initial_prompts", "seeker.template"))
        guide_prompt = self.load_template(os.path.join(RESOURCES_DIR, "initial_prompts", "guide.template"))
        seeker_reprompt = self.load_template(
            os.path.join(RESOURCES_DIR, "re_prompts", "seeker_correct_move.template")
        )
        seeker_fail_reprompt = self.load_template(
            os.path.join(RESOURCES_DIR, "re_prompts", "seeker_incorrect_move.template")
        )

        experiments = self.load_json(os.path.join(RESOURCES_DIR, "experiment_config.json"))

        for exp in experiments.keys():

            experiment = self.add_experiment(exp)
            game_id = 0
            size = experiments[exp]["size"]
            rooms = experiments[exp]["rooms"]
            graph_type = experiments[exp]["type"]
            ambiguity = experiments[exp]["ambiguity"]
            ambiguity_region = experiments[exp]["ambiguity_region"]
            distance = experiments[exp]["distance"]
            start_type = experiments[exp]["start_type"]
            end_type = experiments[exp]["end_type"]

            for i in range(N):
                base_map = BaseMap(m=size, n=size, n_rooms=rooms, graph_type=graph_type, seed=random_seeds[i])
                map_metadata = base_map.metadata(start_type=start_type,
                                             end_type=end_type,
                                             ambiguity=ambiguity,
                                             ambiguity_region=ambiguity_region,
                                             distance=distance)
                map_metadata["seeker_prompt"] = seeker_prompt
                map_metadata["guide_prompt"] = guide_prompt
                map_metadata["seeker_reprompt"] = seeker_reprompt
                map_metadata["seeker_failed_reprompt"] = seeker_fail_reprompt

                escape_room_instance = self.add_game_instance(experiment, game_id)

                for orig_k, orig_v in map_metadata.items():
                    # 1) normalize the key
                    k = str(orig_k) if isinstance(orig_k, tuple) else orig_k

                    # 2) convert the value:
                    if isinstance(orig_v, tuple):
                        v = str(orig_v)
                    elif isinstance(orig_v, (list, dict)):
                        v = orig_v  # those are already JSON‐safe
                    elif isinstance(orig_v, np.generic):
                        # catches numpy ints/floats/bools, etc.
                        v = orig_v.item()
                    else:
                        v = orig_v

                    escape_room_instance[k] = v

                game_id += 1


if __name__ == '__main__':
    EscapeRoomInstanceGenerator().generate()