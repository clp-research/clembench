"""
Take a random instance from each experiment as a test sample.
"""

import os
import json
import numpy as np

SEED = 42
np.random.seed(SEED)

def create_test(json_path: str) -> None:
    """
    Create a test file for a given instance*.json file
    Args:
        json_path: path to an instance*.json file
    """

    with open(json_path, "r", encoding="utf-8") as f:
        instance_data = json.load(f)

    test_data = {
        "experiments": []
    }
    experiments = instance_data["experiments"]
    for exp in experiments:

        instance_data = {
            "name": exp["name"],
            "game_instances": [exp["game_instances"][np.random.randint(len(exp["game_instances"]))]],
        }

        test_data["experiments"].append(instance_data)


    save_path = json_path.replace(".json", "_test.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    json_path = os.path.join("mapworld", "escapegame", "in", "instances.json")
    create_test(json_path)

    json_path_local = os.path.join("mapworld", "escapegame", "in", "instances_local.json")
    if os.path.exists(json_path_local):
        create_test(json_path_local)