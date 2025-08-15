"""
Script using GraphViz to create graph plots of initial world states.
"""

import json

import graphviz


def split_state_string(state_string: str, value_delimiter: str = "(", value_separator: str = ","):
    """
    Split a state predicate string and return its values as a tuple.
    """
    first_split = state_string.split(value_delimiter, 1)
    predicate_type = first_split[0]
    if value_separator in first_split[1]:
        values_split = first_split[1][:-1].split(value_separator, 1)
        return predicate_type, values_split[0], values_split[1]
    else:
        return predicate_type, first_split[1][:-1]


# adventure source file with initial state and goals:
source_file_path = "layout_source2.json"
# load initial state facts:
with open(source_file_path, 'r', encoding='utf-8') as source_file:
    adventure_source = json.load(source_file)
adventure_facts = adventure_source['initial_state']
adventure_goals = adventure_source['goal_state']
# split facts into tuples:
split_facts = [split_state_string(fact) for fact in adventure_facts]
# split goals into tuples:
split_goals = [split_state_string(goal) for goal in adventure_goals]
# directed graph:
dot = graphviz.Digraph('room_layout', format='png')

# dot.attr('node', shape='house')
dot.attr('node', shape='box')

for fact in split_facts:
    if fact[0] == "room":
        dot.node(fact[1], fact[1])

dot.attr('node', shape='box')
"""
for fact in split_facts:
    if fact[0] == "type":
        for fact2 in split_facts:
            if fact2[1] == fact[1]:
                if fact2[0] == "container" or fact2[0] == "support":
                    dot.node(fact[1], fact[1])
"""
dot.attr('node', shape='ellipse')

for fact in split_facts:
    """
    if fact[0] == "type":
        for fact2 in split_facts:
            if fact2[1] == fact[1]:
                if fact2[0] not in ["container", "support"]:
                    dot.node(fact[1], fact[1])
    """
    if fact[0] == "exit":
        dot.edge(fact[1], fact[2])
    """
    if fact[0] == "at":
        dot.edge(fact[1], fact[2], "at")
    
    if fact[0] == "on":
        dot.edge(fact[1], fact[2], "on")
    if fact[0] == "in":
        dot.edge(fact[1], fact[2], "in")
    """

dot.attr('edge', style='dashed')
"""
for goal in split_goals:
    if goal[0] == "on":
        dot.edge(goal[1], goal[2], "on")
    if goal[0] == "in":
        dot.edge(goal[1], goal[2], "in")
"""
dot.render()
