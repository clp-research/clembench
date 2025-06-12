import random
import networkx as nx
import numpy as np

"----------------------------------------------------"
"The functions used in master.py"

def get_directions(node, direction_list, saved_node):

    if saved_node != node:
        node = saved_node
    node_directions = None  
    for i in direction_list:
        if i[0]==node:
            node_directions=i[1]
            break
    return node_directions

def get_directions_main(node, direction_list, saved_node, graph_type):

    if saved_node != node:
        node = saved_node
    node_directions = None  
    for i in direction_list:
        if graph_type == "named_graph":
            if i[0].lower()==node.lower():
                node_directions=i[1]
                break
        elif graph_type == "unnamed_graph":
            if i[0]==node:
                node_directions=i[1]
                break
    return node_directions

def string_available_directions(word_list):
    return ', '.join(word_list)

def have_common_element(str1, str2):
    common_elements = ["east", "west", "north", "south"]
    # Convert strings to sets of words
    words1 = set(str1.split())
    words2 = set(str2.split())
    # Check for exact matches
    common_matches = words1.intersection(words2).intersection(common_elements)
    # Return True if there is at least one exact match
    return any(match in common_elements for match in common_matches)


def get_nextnode_label(moves, node, utterance, move_construction):
    next_label=None
    utterance = utterance.strip()
    for move in moves:
        if move["node"]==node:
            moves_node=move['node_moves']
            for step in moves_node:
                move_type = step[0]
                if move_type ==utterance:
                    next_label=step[1]
    return next_label, move_type

def ambiguity_move(old_one, new_one, mapping, moves, move_type):
    
    mapping_dict, label_moves = mapping
    for label, name in mapping_dict.items():
        if name == old_one:
            for move in moves:
                if name == move["node"]:
                    for label_move in label_moves:
                        if label_move["node"] == label:
                            new_labelled_moves = [(m[0], mapping_dict[m[1]]) for m in label_move["node_moves"]]
                            for m in new_labelled_moves:
                                if m[1] == new_one and m[0] == move_type:
                                    for s in label_move["node_moves"]:
                                        if s[0] == move_type:
                                            return label, s[1]
                                        
def loop_identification(visited_nodes):
    if len(visited_nodes) >= 4:
        if len(set(visited_nodes[-4:])) < 3:
            return True
    return False

"----------------------------------------------------"
"The functions used in instance_generator.py"

# Function to get nodes at a certain distance from the initial node
def get_nodes_at_distance(graph, initial_node, distance):
    return [node for node in nx.single_source_shortest_path_length(graph, initial_node) if nx.single_source_shortest_path_length(graph, initial_node)[node] == distance]

# Randomly select nodes at various distances from the initial node
def select_nodes_at_distances(G, initial_position, max_distance):
    chosen_nodes = {}
    for distance in range(max_distance):
        nodes_at_distance = get_nodes_at_distance(G, initial_position, distance)
        if nodes_at_distance:
            random_node = random.choice(nodes_at_distance)
            chosen_nodes[str(distance)] = random_node
        else:
            print(f"No nodes found at distance {distance} from {initial_position}")
    return chosen_nodes

def lowercase_list_strings(original_list):
    return [item.lower() for item in original_list]

def lowercase_tuple_strings(original_list, type):
    if type == "generated":
        combined_list = [value for sublist in original_list.values() for value in sublist]
        return [(item[0].lower(), item[1].lower()) for item in combined_list]
    elif type == "original":
        return [(item[0].lower(), item[1].lower()) for item in original_list]
    elif type == "none":
        return original_list
    
def create_graph(nodes, edges, type):
    G = nx.Graph()
    if type == "generated" or type == "original":
        nodes= lowercase_list_strings(nodes)
    edges= lowercase_tuple_strings(edges, type)
    G.add_nodes_from(nodes)
    for edge in edges:
        if len(edge) == 2:
            G.add_edge(edge[0], edge[1])
    return G

#Create a networkx graph from the given graph data.

def normalize(distance):
        normalized_distance = 1 / (1 + np.exp(-0.5 * distance))
        normalized_distance = 2*(normalized_distance - 0.5)
        return normalized_distance
    
def calculate_similarity(graph1, graph2):
    distance = nx.graph_edit_distance(graph1, graph2)
    normalized_distance = normalize(distance)
    similarity = 1 - normalized_distance
    return similarity

def count_word_in_sentence(sentence, word):
    # Split the sentence into words
    words = sentence.split()
    
    # Convert the words and the target word to lowercase for case insensitive comparison
    word = word.lower()
    words = [w.lower() for w in words]
    
    # Count the occurrences of the word
    count = words.count(word)
    return count