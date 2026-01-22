from copy import deepcopy
from resources.game_state.game_state import GridState
import json
import os

def find_experiment_dirs(base_dir: str, game_name: str = ""):
    experiment_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if "interactions.json" in files and "instance.json" in files and game_name in root:
            experiment_dirs.append(root)
    return experiment_dirs

def add_grids_to_interactions(experiment_dir: str):
    interaction_file = os.path.join(experiment_dir, "interactions.json")
    instance_file = os.path.join(experiment_dir, "instance.json")
    with open(instance_file, "r") as f:
        instance = json.load(f)
    interaction_file = os.path.join(experiment_dir, "interactions.json")
    with open(interaction_file, "r") as f:
        interactions = json.load(f)
    init_states = interactions["Init States"]
    end_states = interactions["End States"]
    grids = {
        "Player 1": [
            str(GridState(background=instance["background"], objects=init_states["state_1"])),
            str(GridState(background=instance["background"], objects=end_states["state_1"]))
        ],
        "Player 2": [
            str(GridState(background=instance["background"], objects=init_states["state_2"])),
            str(GridState(background=instance["background"], objects=end_states["state_2"]))
        ]
    }

    if grids["Player 2"][0] == grids["Player 2"][1]:
        print(f"WARNING: Player 2's initial and final grids are the same in {experiment_dir}")
    else:
        print(f"Player 2's initial and final grids are different in {experiment_dir}")

    initial_grids = f"Initial Grids:\n```\n{'Player 1:'.ljust(14)}Player 2:"
    for line_1, line_2 in zip(grids["Player 1"][0].split('\n'), grids["Player 2"][0].split('\n')):
        if line_1.strip() == "" and line_2.strip() == "":
            continue
        initial_grids += f"\n{line_1.ljust(14)}{line_2}"
    initial_grids += "\n```"
    end_grids = f"Final Grids:\n```\n{'Player 1:'.ljust(14)}Player 2:"
    for line_1, line_2 in zip(grids["Player 1"][1].split('\n'), grids["Player 2"][1].split('\n')):
        if line_1.strip() == "" and line_2.strip() == "":
            continue
        end_grids += f"\n{line_1.ljust(14)}{line_2}"
    end_grids += "\n```"
    grid_log = {
        "from": "GM",
        "to": "GM",
        "timestamp": "",
        "action": {
            "type": "grid logs",
            "content": f"{initial_grids}\n\n{end_grids}"
        }
    }
    for event in interactions["turns"][-1]:
        if event["action"]["type"] == "grid logs":
            # remove existing grid logs
            print(f"Removing existing grid logs in {interaction_file}")
            interactions["turns"][-1].remove(event)
    interactions["turns"][-1].append(grid_log)
    with open(f"{interaction_file}", "w") as f:
        json.dump(interactions, f, indent=4)
    print(f"updated {interaction_file}")

def remove_dev_log(experiment_dir: str):
    interaction_file = os.path.join(experiment_dir, "interactions.json")
    with open(interaction_file, "r") as f:
        interactions = json.load(f)
    dev_log = interactions["turns"][-1][-2]
    if dev_log["action"]["type"] == "dev:game_finished":
        interactions["turns"][-1].pop(-2)
        with open(f"{interaction_file}", "w") as f:
            json.dump(interactions, f, indent=4)
        print(f"removed dev log from {interaction_file}")

def count_penalty_types(experiment_dir: str):
    interaction_file = os.path.join(experiment_dir, "interactions.json")
    with open(interaction_file, "r") as f:
        interactions = json.load(f)

    keys_to_log = {
        "invalid move": "Invalid Moves",
        "valid move": "Valid Moves",
        "parse_error": "Parse Errors"
    }
    stats = {v: 0 for v in keys_to_log.values()}
    for turn in interactions["turns"]:
        for event in turn:
            if event["action"]["type"] in keys_to_log:
                stats[keys_to_log[event["action"]["type"]]] += 1
    print("Stats for ", experiment_dir)
    for k, v in stats.items():
        interactions[k] = v
    
    with open(f"{interaction_file}", "w") as f:
        json.dump(interactions, f, indent=4)
    print(f"updated {interaction_file} with penalty stats")

def add_markdown_key(experiment_dir: str):
    interaction_file = os.path.join(experiment_dir, "interactions.json")
    with open(interaction_file, "r") as f:
        interactions = json.load(f)
    if "markdown" not in interactions:
        interactions["markdown"] = True
        with open(f"{interaction_file}", "w") as f:
            json.dump(interactions, f, indent=4)
        print(f"added markdown key to {interaction_file}")


if __name__ == "__main__":
    clean_up_dirs = find_experiment_dirs("/project/kosswald/benchmark3.0/results", game_name="clean_up")
    # clean_up_dirs += find_experiment_dirs("/Users/karlosswald/repositories/clemclass/negotiation-games/results_de", game_name="clean_up")
    # clean_up_dirs += find_experiment_dirs("/Users/karlosswald/repositories/clemclass/negotiation-games/results_it", game_name="clean_up")
    for dir in clean_up_dirs:
        add_grids_to_interactions(dir)
        remove_dev_log(dir)
        count_penalty_types(dir)
        add_markdown_key(dir)