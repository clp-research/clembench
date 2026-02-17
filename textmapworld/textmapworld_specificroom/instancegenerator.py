import os
import shutil
import random
import networkx as nx

from typing import Dict, List
from textmapworld.graph_generator import GraphGenerator
from clemcore.clemgame import GameInstanceGenerator
from textmapworld.config_languages import LANG_CONFIG


def create_graph_file_name(game_type, graph_size, cycle_type, ambiguity):
    if cycle_type == "cycle_true":
        cycle = "True"
    elif cycle_type == "cycle_false":
        cycle = "False"
    if graph_size == None:
        filename_parts = [game_type.capitalize().split("_graph")[0], cycle, str(ambiguity)]
        filename = "_".join(filename_parts)
    else:
        filename_parts = [game_type.capitalize().split("_graph")[0], str(graph_size), cycle, str(ambiguity)]
        filename = "_".join(filename_parts) + ".txt"
    return filename


def check_graphs(graphs, instance_number, game_type):
    grids = []
    check_set = set()
    for c, graph in enumerate(graphs):
        nodes = graph.get('Graph_Nodes', [])
        if c < instance_number:
            if all(isinstance(item, tuple) for item in nodes):
                checked_graph_type = "unnamed_graph"
                check_set.add(checked_graph_type)
            elif all(isinstance(item, str) for item in nodes):
                checked_graph_type = "named_graph"
                check_set.add(checked_graph_type)
            grids.append(graph)
    if check_set != {str(game_type)}:
        raise ValueError("Graph type does not match the specified type")
    return grids


def create_graphs(num_graphs, graph_type, n, m, rooms, cycle_bool, abiguity, game_name, game_path) -> List[Dict]:
    descriptors = []
    num_retries = 0
    while len(descriptors) < num_graphs:
        new_instance = GraphGenerator(graph_type, n, m, rooms, cycle_bool, abiguity, game_name, game_path)
        descriptor = new_instance.generate_instance()
        if descriptor != "No graph generated":  # ps: this seems rather weird, b.c. there can be more error types
            descriptors.append(descriptor)
            num_retries += 1
        assert num_retries <= 100, f"Counted {num_retries} while generating {num_graphs} graphs. Abort."
    return descriptors


def create_nxgraph(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


"Enter the parameters for the game instance generator"
"-------------------------------------------------------------------------------------------------------------"
"°°°°°°°changeable parameters°°°°°°°"

strict = True
size = 8  # "large"
n = 4
m = 4
instance_number = 10
game_type = "named_graph"  # "named_graph" or "unnamed_graph"
cycle_type = "cycle_false"  # "cycle_true" or "cycle_false"
ambiguity = None  # (repetition_rooms, repetition_times) or None
loop_reminder = False
max_turns_reminder = False
experiments = {"on": [0], "close": [1, 2], "far": [3, 4]}

"°°°°°°°imported parameters°°°°°°°"
game_name = "textmapworld_specificroom"

"-------------------------------------------------------------------------------------------------------------"


class TextMapWorldRoomGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self, ):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed: int, **kwargs):
        language = kwargs.get("language") or kwargs.get("lang", "en")
        lang_cfg = LANG_CONFIG[language]
        prompt_dir = lang_cfg["prompt_dir"]
        if strict:
            MOVE_REGEX = f'{lang_cfg["MOVE"]}:\s*({"|".join(lang_cfg["DIRECTIONS"])})'
            DONE_REGEX = f'^{lang_cfg["DONE"]}$'
        else:
            MOVE_REGEX = f'{lang_cfg["MOVE"]}:\s*({"|".join(lang_cfg["DIRECTIONS"])})'
            DONE_REGEX = f'^{lang_cfg["DONE"]}$'

        prompt_template_name = ('PromptNamedGame.template' if game_type == "named_graph" else 'PromptUnnamedGame.template')
        prompt_file_path = os.path.join('resources', 'initial_prompts', prompt_dir, prompt_template_name)
        # prepare folder for generated files
        generated_dir = os.path.join(self.game_path, "generated")
        print("Prepare", generated_dir)
        if os.path.exists(generated_dir):
            shutil.rmtree(generated_dir)
        os.makedirs(os.path.join(generated_dir, "images"))
        os.makedirs(os.path.join(generated_dir, "graphs"))
        # perform the instance generation
        answers_file = self.load_json(f"resources/initial_prompts/{prompt_dir}/answers.json")
        reminders_file = self.load_json(f"resources/initial_prompts/{prompt_dir}/reminders.json")
        player_a_prompt_header = self.load_template(prompt_file_path)
        Player2_positive_answer = answers_file["PositiveAnswerNamedGame"]
        Player2_negative_answer = answers_file["NegativeAnswerNamedGame"]
        # create only a single graphs file
        graph_file_name = create_graph_file_name(game_type, size, cycle_type, ambiguity)
        graphs = create_graphs(instance_number, game_type, n, m, size, cycle_type, ambiguity,
                               game_name, self.game_path)
        self.store_file("\n".join([str(g) for g in graphs]), graph_file_name, "generated/graphs")
        grids = check_graphs(graphs, instance_number, game_type)
        game_id = 0
        # the experiments test model performances for various distances on the same graphs
        for experiment_name, distances in experiments.items():
            experiment = self.add_experiment(experiment_name)
            for grid in grids:
                print(f"Add instance {game_id}")
                game_instance = self.add_game_instance(experiment, game_id)
                game_id += 1
                game_instance["Prompt"] = player_a_prompt_header
                game_instance["Player2_positive_answer"] = Player2_positive_answer
                game_instance["Player2_negative_answer"] = Player2_negative_answer
                game_instance["Move_Construction"] = MOVE_REGEX
                game_instance["Stop_Construction"] = DONE_REGEX
                game_instance["Grid_Dimension"] = str(grid["Grid_Dimension"])
                game_instance['Graph_Nodes'] = str(grid['Graph_Nodes'])
                game_instance['Graph_Edges'] = str(grid['Graph_Edges'])
                game_instance['Current_Position'] = str(grid['Initial_Position'])
                game_instance['Picture_Name'] = grid['Picture_Name']
                game_instance["Directions"] = str(grid["Directions"])
                game_instance["Moves"] = str(grid["Moves"])
                game_instance['Cycle'] = grid['Cycle']
                game_instance['Ambiguity'] = grid['Ambiguity']
                game_instance['Game_Type'] = game_type
                game_instance["Loop_Reminder"] = loop_reminder
                game_instance["Loop_Reminder_Text"] = reminders_file["loop_reminder"]
                game_instance["Max_Turns_Reminder"] = max_turns_reminder
                game_instance["Max_Turns_Reminder_Text"] = reminders_file["max_turns_reminder"]
                game_instance["Mapping"] = str(grid["Mapping"])
                game_instance["Strict"] = strict
                game_instance["Lang"] = language
                generated_graph = create_nxgraph(grid["Graph_Nodes"], grid["Graph_Edges"])
                dists = dict(nx.all_pairs_shortest_path_length(generated_graph))
                random_distance = random.choice(distances)
                distance_found = False
                for k, v in dists.items():
                    if k == grid["Initial_Position"]:
                        for neighbor, distance in v.items():
                            if distance == random_distance:
                                game_instance["Specific_Room"] = neighbor
                                game_instance["Specific_Room_Distance"] = str(random_distance)
                                distance_found = True
                                break
                if not distance_found:
                    for room, val in dists.items():
                        for neighbor, distance in val.items():
                            if distance == random_distance:
                                game_instance['Current_Position'] = room
                                game_instance["Specific_Room"] = neighbor
                                game_instance["Specific_Room_Distance"] = str(random_distance)
                                break


if __name__ == '__main__':
    import json

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    with open(os.path.join(ROOT, "SUPPORTED_LANGUAGES.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    GAME_NAME = "textmapworld_specificroom"
    supported_languages = [lang for lang, data in config["languages"].items() if GAME_NAME in data["games"]]
    for lang in supported_languages:
        print(f"Generating instances for language '{lang}'")
        TextMapWorldRoomGameInstanceGenerator().generate(filename=f"instances_{lang}.json", seed=123, lang=lang)
