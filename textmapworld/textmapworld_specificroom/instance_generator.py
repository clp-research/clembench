import os
import shutil
import random
import networkx as nx

from typing import Dict, List
from textmapworld.graph_generator import GraphGenerator
from clemcore.clemgame import GameInstanceGenerator


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
if strict:
    DONE_REGEX = '^DONE$'
    MOVE_REGEX = '^GO:\s*(north|east|west|south)$'
else:
    DONE_REGEX = 'DONE'
    MOVE_REGEX = 'GO:\s*(north|east|west|south)'
loop_reminder = False
max_turns_reminder = False
experiments = {"on": [0], "close": [1, 2], "far": [3, 4]}

"°°°°°°°imported parameters°°°°°°°"
prompt_file_name = 'PromptNamedGame.template' if game_type == "named_graph" else 'PromptUnnamedGame.template'
prompt_file_name = os.path.join('resources', 'initial_prompts', prompt_file_name)
game_name = "textmapworld_specificroom"

"-------------------------------------------------------------------------------------------------------------"


class TextMapWorldRoomGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self, ):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed: int, **kwargs):
        # prepare folder for generated files
        generated_dir = os.path.join(self.game_path, "generated")
        print("Prepare", generated_dir)
        if os.path.exists(generated_dir):
            shutil.rmtree(generated_dir)
        os.makedirs(os.path.join(generated_dir, "images"))
        os.makedirs(os.path.join(generated_dir, "graphs"))
        # perform the instance generation
        answers_file = self.load_json("resources/initial_prompts/answers.json")
        reminders_file = self.load_json("resources/initial_prompts/reminders.json")
        player_a_prompt_header = self.load_template(prompt_file_name)
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
    TextMapWorldRoomGameInstanceGenerator().generate(seed=42)
