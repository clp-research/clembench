"""
Functions to convert PDDL definitions to ASP for instance generation.
"""
import json

import lark
from lark import Lark

from pddl_util import PDDLActionTransformer, PDDLDomainTransformer

# TODO: properly save mutable states and mutability state in definitions; attach to entity definitions or add to domain as in proper PDDL?

# TODO?: expand PDDL domain usage to incorporate mutable states better; ie types tied to mutable states centrally?

# load lark grammars and init lark transformers:
with open("pddl_actions.lark", 'r', encoding='utf-8') as actions_grammar_file:
    action_def_grammar = actions_grammar_file.read()
action_def_parser = Lark(action_def_grammar, start="action")

action_def_transformer = PDDLActionTransformer()

with open("pddl_domain.lark", 'r', encoding='utf-8') as domain_grammar_file:
    domain_def_grammar = domain_grammar_file.read()
domain_def_parser = Lark(domain_def_grammar, start="define")

domain_def_transformer = PDDLDomainTransformer()


def action_to_asp(action_def: dict, trait_dict: dict):
    """Create ASP encoding rules from action definition PDDL.
    Args:
        action_def: A dict action definition.
    Returns:
        A list of ASP encoding rule strings.
    """
    action_asp_rules = list()
    # parse and process action definition PDDL:
    parsed_action_pddl = action_def_parser.parse(action_def['pddl'])
    processed_action_pddl = action_def_transformer.transform(parsed_action_pddl)

    # PARAMETERS AND PRECONDITION
    params = processed_action_pddl['parameters']
    # filter out always-present default parameters:
    important_params = list()
    for param in params['type_list']:
        if param['type_list_element'] not in ['player', 'room']:
            important_params.append(param)
    # important parameters are mutability trait predicates/facts

    action_mutabilities = [mutability['type_list_element'] for mutability in important_params]
    # print(f"action mutabilities for {action_def['type_name']}:", action_mutabilities)

    mutability_asp_strings = [f"{mutability['type_list_element']}(THING)" for mutability in important_params]

    # print(f"action to ASP mutabilities for {action_def['type_name']}:", mutability_asp_strings)

    # TODO: make mutability type facts in base ASP solver script

    # catch at condition to put on ASP RHS:
    # check for default at facts (player at same as argument)
    precon = processed_action_pddl['precondition']
    precon_and = precon[0]['and']
    important_precon_facts = list()
    for precon_fact in precon_and:
        if precon_fact['predicate'] == "at":
            # assume that there are no interactions with other locations and the at conditions are always the same
            continue
        else:
            important_precon_facts.append(precon_fact)
    important_precon_mutables = [precon['predicate'] for precon in important_precon_facts]

    # TODO: handle more complex preconditions with OR etc

    # insert into ASP encoding rule template:
    asp_potential_action = "{ action_t(TURN,ACTION_TYPE,THING):at_t(TURN,THING,ROOM),PRECON_FACTS } 1 :- turn(TURN), at_t(TURN,player1,ROOM), not turn_limit(TURN)."
    # asp_potential_action = "{ action_t(TURN,ACTION_TYPE,THING):PRECON_FACTS } 1 :- turn(TURN), at_t(TURN,player1,ROOM), at_t(TURN,THING,ROOM), not turn_limit(TURN)."
    # asp_potential_action = "{ action_t(TURN,ACTION_TYPE,THING):at_t(TURN,THING,ROOM);PRECON_FACTS } 1 :- turn(TURN), at_t(TURN,player1,ROOM), not turn_limit(TURN)."
    # asp_potential_action = "{ action_t(TURN,ACTION_TYPE,THING):at_t(TURN,THING,ROOM) } 1 :- turn(TURN), at_t(TURN,player1,ROOM), PRECON_FACTS, not turn_limit(TURN)."
    asp_potential_action = asp_potential_action.replace("ACTION_TYPE", processed_action_pddl['action_name'])
    mutable_asp_strings = [f"{mutable}_t(TURN,THING)" for mutable in important_precon_mutables]
    asp_potential_action = asp_potential_action.replace("PRECON_FACTS",
                                                        ",".join(mutability_asp_strings + mutable_asp_strings))
    # collect:
    action_asp_rules.append(asp_potential_action)

    # EFFECT
    effect = processed_action_pddl['effect'][0]['and']
    # differentiate between facts added by action (ie existing next turn) and subtracted by action (ie existing next turn unless removed by action)
    effect_add_predicates = list()
    effect_sub_predicates = list()
    # iterate over effect predicates:
    for effect_pred in effect:
        if 'not' in effect_pred:
            effect_sub_predicates.append(effect_pred['not'])
        else:
            effect_add_predicates.append(effect_pred)

    # TODO: handle more complex effects with WHEN etc

    # facts added:
    # insert into ASP encoding rule template:
    asp_next_turn_add = "MUTABLE_FACTS :- action_t(TURN,ACTION_TYPE,THING)."
    asp_next_turn_add = asp_next_turn_add.replace("ACTION_TYPE", processed_action_pddl['action_name'])
    effect_add_asp_strings = [f"{pred}_t(TURN+1,THING)" for pred in
                              [effect_pred['predicate'] for effect_pred in effect_add_predicates]]
    asp_next_turn_add = asp_next_turn_add.replace("MUTABLE_FACTS", ",".join(effect_add_asp_strings))
    # collect:
    action_asp_rules.append(asp_next_turn_add)

    # persist irreversible end mutable state:
    if trait_dict[action_mutabilities[0]]['interaction'] == "irreversible":
        # print(trait_dict[action_mutabilities[0]]) # template for irreversible mutables:
        asp_next_turn_irreversible = "MUTABLE_FACT_t(TURN+1,THING) :- turn(TURN), MUTABLE_FACT_t(TURN,THING)."
        asp_next_turn_irreversible = asp_next_turn_irreversible.replace(
            "MUTABLE_FACT",
            f"{trait_dict[action_mutabilities[0]]['mutable_states'][-1]}")
        action_asp_rules.append(asp_next_turn_irreversible)

    # TODO: figure out what goes wrong with the action ASP that prevents adventures from being solvable if there are different mutabilities for goals

    # facts removed:
    for sub_pred in effect_sub_predicates:
        # template for reversible mutables:
        asp_next_turn_sub = "MUTABLE_FACT_t(TURN+1,THING) :- turn(TURN), MUTABLE_FACT_t(TURN,THING), not action_t(TURN,ACTION_TYPE,THING)."
        # insert into ASP encoding rule template:
        asp_next_turn_sub = asp_next_turn_sub.replace("ACTION_TYPE", processed_action_pddl['action_name'])
        effect_sub_asp_string = f"{sub_pred['predicate']}"
        asp_next_turn_sub = asp_next_turn_sub.replace("MUTABLE_FACT", effect_sub_asp_string)

        # collect:
        action_asp_rules.append(asp_next_turn_sub)

    return action_asp_rules


def actions_file_to_asp(action_defs_file_path: str = "new_word_generation/new_actions_test.json"):
    """Get ASP encoding rules from action definitions file."""
    with open(action_defs_file_path, 'r', encoding='utf-8') as action_defs_file:
        action_defs = json.load(action_defs_file)

    all_action_asp_rules = list()

    for action_def in action_defs:
        action_asp_rules = action_to_asp(action_def)
        all_action_asp_rules += action_asp_rules

    return all_action_asp_rules


def augment_action_defs_with_asp(action_defs: list, trait_dict: dict):
    """Create ASP encoding rules for action definitions and add them to existing definitions."""
    for action_def_idx, action_def in enumerate(action_defs):
        action_asp_rules = action_to_asp(action_def, trait_dict)
        action_defs[action_def_idx]['asp'] = "\n".join(action_asp_rules)

    return action_defs


def augment_actions_file_with_asp(action_defs_file_path: str):
    """Create ASP encoding rules for action definitions and add them to an existing action definitions file."""
    with open(action_defs_file_path, 'r', encoding='utf-8') as action_defs_file_in:
        action_defs = json.load(action_defs_file_in)

    action_defs = augment_action_defs_with_asp(action_defs)

    with open(action_defs_file_path, 'w', encoding='utf-8') as action_defs_file_out:
        json.dump(action_defs, action_defs_file_out, indent=2)




if __name__ == "__main__":
    # augment_actions_file_with_asp("new_word_generation/new_actions_test.json")
    augment_actions_file_with_asp("definitions/witch_actions_core_testing.json")