"""
    IF interpreter for adventuregame.
"""

import json
import lark
from lark import Lark, Transformer
import jinja2

import numpy as np
import os
from copy import deepcopy
from typing import List, Set, Union
import itertools

from clemcore.clemgame import GameResourceLocator

import logging

from adv_util import fact_str_to_tuple, fact_tuple_to_str

PATH = "games/adventuregame/"
RESOURCES_SUBPATH = "resources/"

GAME_NAME = "adventuregame"

logger = logging.getLogger(__name__)


class IFTransformer(Transformer):
    """
    IF action grammar transformer to convert Lark parse to python dict for further use.
    """
    # since this is solely for action command parse conversion, any input is converted to a parsed action dict:
    def action(self, content):
        action: lark.Tree = content[0]
        # print("IFTransformer action:", action)
        action_type = action.data  # main grammar rule the input was parsed as
        # print("IFTransformer action_type:", action_type)
        action_content = action.children  # all parsed arguments of the action 'VP'
        action_dict = {'type': action_type.value}  # value = string name of rule in grammar

        arguments = []
        arg_idx = 1

        for child in action_content:
            # handle potentially multi-word 'thing' arguments; roughly equivalent to generic 'NP':
            if type(child) == lark.Tree and child.data == 'thing':
                argument_words = [word.value for word in child.children if word.type == 'WORD']
                arguments.append(" ".join(argument_words))
                # extract defined adjectives:
                argument_adjs = [adj.value.strip() for adj in child.children[:-1] if adj.type == 'ADJ']
                if argument_adjs:
                    action_dict[f'arg{arg_idx}_adjs'] = argument_adjs
                action_dict[f'arg{arg_idx}'] = arguments[-1]
                arg_idx += 1
            # extract defined prepositions:
            if type(child) == lark.Token and child.type == 'PREP':
                action_dict['prep'] = child.value.strip()
            # if the input can't be parsed as a defined action command, the grammar labels it as 'unknown'
            # in this case, the first word is assumed to be the verb and is returned for feedback:
            if action_type.value == 'unknown' and type(child) == lark.Token and child.type == 'WORD':
                action_dict[f'arg{arg_idx}'] = child.value
                break

        return action_dict


class PDDLBaseTransformer(Transformer):
    """PDDL parse transformer for shared base methods/grammar rules.
    Method names must match grammar rule names, thus some rules have an added -p to distinguish their name from a python
    constant/type/default term string."""
    def forall(self, content):
        # print("forall cont:", content)
        iterated_object = content[2]
        # print("iterated object:", iterated_object)
        forall_body = content[4:]
        # print("forall body:", forall_body)

        forall_dict = {'forall': iterated_object, 'body': forall_body}
        # print("forall returns:", forall_dict)
        return forall_dict

    def when(self, content):
        # print("when cont:", content)
        when_items = list()
        for when_item in content:
            # ignore delimiters and whitespace:
            if type(when_item) == lark.Token and when_item.type in ["WHENL", "WS"]:
                pass
            else:
                when_items.append(when_item)
        when_dict = {'when': when_items}
        # print("when returns:", when_dict)
        return when_dict

    def andp(self, content):
        # print("andp cont:", content)
        and_items = list()
        for and_item in content:
            # ignore delimiters and whitespace:
            if type(and_item) == lark.Token and and_item.type in ["ANDL", "WS"]:
                pass
            else:
                and_items.append(and_item)
        and_dict = {'and': and_items}
        # print("andp returns:", and_dict, "\n")
        return and_dict

    def orp(self, content):
        # print("orp cont:", content)
        or_items = list()
        for or_item in content:
            # ignore delimiters and whitespace:
            if type(or_item) == lark.Token and or_item.type in ["ORL", "WS"]:
                pass
            else:
                or_items.append(or_item)
        or_dict = {'or': or_items}
        # print("orp returns:", or_dict, "\n")
        return or_dict

    def notp(self, content):
        # print("notp cont:", content)
        # (not X) always wraps only one item, hence:
        return {'not': content[2]}

    def type_list(self, content):
        # print(content)
        return {'type_list': content}

    def type_list_element(self, content):
        # print("type_list_item cont:", content)
        type_list_items = list()
        for item_element in content:
            if 'variable' in item_element:
                type_list_items.append(item_element)
            elif type(item_element) == lark.Token:
                if item_element.type == "WORDP":
                    type_list_items.append(item_element.value)
                elif item_element.type == "DASH":
                    break

        if content[-1].type == "WS":
            cat_name = content[-2].value
        else:
            cat_name = content[-1].value

        return {'type_list_element': cat_name, 'items': type_list_items}

    def pred(self, content):
        # print("pred content:", content)
        if type(content[0]) == lark.Token:
            pred_type = content[0].value
        else:
            pred_type = content[0]
        # valence up to three, using None assignments to avoid downstream checks
        pred_arg1 = None
        pred_arg2 = None
        pred_arg3 = None

        if len(content) >= 3:
            # print('pred arg 1:', content[2])
            if type(content[2]) == lark.Token:
                pred_arg1 = content[2].value
            else:
                pred_arg1 = content[2]
        if len(content) >= 5:
            if type(content[4]) == lark.Token:
                pred_arg2 = content[4].value
            else:
                pred_arg2 = content[4]
        if len(content) >= 7:
            if type(content[6]) == lark.Token:
                pred_arg3 = content[6].value
            else:
                pred_arg3 = content[6]

        pred_dict = {'predicate': pred_type, 'arg1': pred_arg1, 'arg2': pred_arg2, 'arg3': pred_arg3}
        # print(pred_dict, "\n")

        return pred_dict

    def var(self, content):
        # print(content[0])
        return {'variable': content[0].value}

    def function(self, content):
        # print("function content:", content)

        function_dict = dict()

        if content[0].type == 'NUMBER':
            # print("function NUMBER:", content[0].value)
            function_dict['function_number'] = content[0].value
        else:
            function_dict['function_id'] = content[0].value
            function_dict['function_variable'] = content[2]

        # print("function_dict:", function_dict)

        return function_dict

    def equal(self, content):
        # print("greq content:", content)

        equal_dict = {'num_comp': "equal"}

        equal_dict['arg1'] = content[2]
        equal_dict['arg2'] = content[4]

        # print("greq_dict:", greq_dict)

        return equal_dict

    def greater(self, content):
        # print("greq content:", content)

        greater_dict = {'num_comp': "greater"}

        greater_dict['arg1'] = content[2]
        greater_dict['arg2'] = content[4]

        # print("greq_dict:", greq_dict)

        return greater_dict

    def greq(self, content):
        # print("greq content:", content)

        greq_dict = {'num_comp': "greq"}

        greq_dict['arg1'] = content[2]
        greq_dict['arg2'] = content[4]

        # print("greq_dict:", greq_dict)

        return greq_dict

    def less(self, content):
        # print("greq content:", content)

        less_dict = {'num_comp': "less"}

        less_dict['arg1'] = content[2]
        less_dict['arg2'] = content[4]

        # print("greq_dict:", greq_dict)

        return less_dict

    def leq(self, content):
        # print("greq content:", content)

        leq_dict = {'num_comp': "leq"}

        leq_dict['arg1'] = content[2]
        leq_dict['arg2'] = content[4]

        # print("greq_dict:", greq_dict)

        return leq_dict

    def assign(self, content):
        # print("greq content:", content)

        assign_dict = {'function_change': "assign"}

        assign_dict['arg1'] = content[2]
        assign_dict['arg2'] = content[4]

        # print("greq_dict:", greq_dict)

        return assign_dict

    def increase(self, content):
        # print("greq content:", content)

        increase_dict = {'function_change': "increase"}

        increase_dict['arg1'] = content[2]
        increase_dict['arg2'] = content[4]

        # print("greq_dict:", greq_dict)

        return increase_dict

    def decrease(self, content):
        # print("greq content:", content)

        decrease_dict = {'function_change': "decrease"}

        decrease_dict['arg1'] = content[2]
        decrease_dict['arg2'] = content[4]

        # print("greq_dict:", greq_dict)

        return decrease_dict


class PDDLActionTransformer(PDDLBaseTransformer):
    """PDDL action definition transformer to convert Lark parse to python dict for further use.
    Method names must match grammar rule names, thus some rules have an added -p to distinguish their name from a python
    constant/type/default term string.
    """
    def action(self, content):
        # print("action cont:", content)

        # action_def_dict = {'action_name': content[1].value, 'content': content[3:]}
        action_def_dict = {'action_name': content[1].value.lower()}

        for cont in content:
            # print(type(cont))
            if type(cont) == lark.Token:
                # print(cont.type, cont.value)
                pass
            else:
                # print("non-Token", cont)
                if 'parameters' in cont:
                    action_def_dict['parameters'] = cont['parameters']
                elif 'precondition' in cont:
                    action_def_dict['precondition'] = cont['precondition']
                elif 'effect' in cont:
                    action_def_dict['effect'] = cont['effect']


        # action: lark.Tree = content[0]
        # action_type = action.data  # main grammar rule the input was parsed as
        # action_content = action.children  # all parsed arguments of the action 'VP'

        # print("action returns:", action_def_dict)
        return action_def_dict
        # return content
        # pass

    def parameters(self, content):
        parameter_list = None
        if type(content[0]) == lark.Token and content[0].type == "WS":
            parameter_list = content[1]
        # print("parameters:", parameter_list)

        return {'parameters': parameter_list}

    def precondition(self, content):
        # print("precond cont:", content)
        # print("precond cont:", content[1][1:])
        return {'precondition': content[1:-1]}

    def effect(self, content):
        # print("effect cont:", content)
        effect_dict = {'effect': content[1:-1]}
        # print("effect returns:", effect_dict)
        return effect_dict


class PDDLDomainTransformer(PDDLBaseTransformer):
    """PDDL domain definition transformer to convert Lark parse to python dict for further use.
    Method names must match grammar rule names, thus some rules have an added -p to distinguish their name from a python
    constant/type/default term string.
    """
    def define(self, content):
        # print("define cont:", content)

        # domain_def_dict = {'domain_name': content[1].value.lower()}
        domain_def_dict = dict()

        for cont in content:
            # print(type(cont))
            # print("define item:", cont)
            if type(cont) == lark.Token:
                # print("lark token:", cont.type, cont.value)
                pass
            else:
                # print("non-Token:", cont)
                if 'domain_id' in cont:
                    domain_def_dict['domain_id'] = cont['domain_id']
                if 'types' in cont:
                    domain_def_dict['types'] = cont['types']
                if 'functions' in cont:
                    domain_def_dict['functions'] = cont['functions']
                if 'event_id' in cont:
                    if not 'events' in domain_def_dict:
                        domain_def_dict['events'] = [cont]
                    else:
                        domain_def_dict['events'].append(cont)

        # action: lark.Tree = content[0]
        # action_type = action.data  # main grammar rule the input was parsed as
        # action_content = action.children  # all parsed arguments of the action 'VP'

        # print("define returns:", domain_def_dict)
        return domain_def_dict
        # return content
        # pass

    def domain_id(self, content):
        # print("domain_id cont:", content)
        # print("domain_id return:", {'domain_id': content[-1].value})
        return {'domain_id': content[-1].value}

    def types(self, content):
        # print("types cont:", content)
        types_list = list()
        for cont in content:
            if 'type_list_element' in cont:
                types_list.append(cont)
        types_dict = dict()
        for type_list in types_list:
            # print(type_list)
            types_dict[f'{type_list["type_list_element"]}'] = type_list['items']
        # print("types return:", {'types': types_list})
        return {'types': types_dict}

    def parameters(self, content):
        parameter_list = None
        if type(content[0]) == lark.Token and content[0].type == "WS":
            parameter_list = content[1]
        # print("parameters:", parameter_list)

        return {'parameters': parameter_list}

    def precondition(self, content):
        # print("precond cont:", content)
        # print("precond cont:", content[1][1:])
        return {'precondition': content[1:-1]}

    def effect(self, content):
        # print("effect cont:", content)
        effect_dict = {'effect': content[1:-1]}
        # print("effect returns:", effect_dict)
        return effect_dict

    def functions(self, content):
        # print("functions content:", content)
        functions_dict = {'functions': list()}
        for functions_item in content:
            if 'function_def_predicate' in functions_item:
                functions_dict['functions'].append(functions_item)

        return functions_dict

    def function_list_element(self, content):
        # print("function_list_element content:", content)

        # for function_item in content:
        #    print("function_list_element item:", function_item)

        function_def_predicate = content[0].value
        # print("function_predicate:", function_predicate)

        function_def_variable = content[2]
        # print("function_variable:", function_variable)

        function_def_type = content[6].value
        # print("function_type:", function_type)

        function_dict = {'function_def_predicate': function_def_predicate,
                         'function_def_variable': function_def_variable,
                         "function_def_type": function_def_type}

        return function_dict

    def event(self, content):
        # print("event content:", content)

        event_id = content[2].value
        # print("event_id:", event_id)

        event_dict = {'event_id': content[2].value}

        for event_item in content[4:]:
            # print("event_item:", event_item)
            if 'parameters' in event_item:
                event_dict['event_parameters'] = event_item['parameters']
            if 'precondition' in event_item:
                event_dict['event_precondition'] = event_item['precondition']
            if 'effect' in event_item:
                event_dict['event_effect'] = event_item['effect']

        # print("event_dict:", event_dict)

        return event_dict


class PDDLEventTransformer(PDDLBaseTransformer):
    """PDDL event definition transformer to convert Lark parse to python dict for further use.
    Method names must match grammar rule names, thus some rules have an added -p to distinguish their name from a python
    constant/type/default term string.
    """
    def event(self, content):
        # print("event content:", content)

        event_id = content[2].value
        # print("event_id:", event_id)

        event_dict = {'event_id': content[2].value}

        for event_item in content[4:]:
            # print("event_item:", event_item)
            if 'parameters' in event_item:
                # event_dict['event_parameters'] = event_item['parameters']
                event_dict['parameters'] = event_item['parameters']
            if 'precondition' in event_item:
                # event_dict['event_precondition'] = event_item['precondition']
                event_dict['precondition'] = event_item['precondition']
            if 'effect' in event_item:
                # event_dict['event_effect'] = event_item['effect']
                event_dict['effect'] = event_item['effect']

        # print("event_dict:", event_dict)

        return event_dict

    def types(self, content):
        # print("types cont:", content)
        types_list = list()
        for cont in content:
            if 'type_list_element' in cont:
                types_list.append(cont)
        types_dict = dict()
        for type_list in types_list:
            # print(type_list)
            types_dict[f'{type_list["type_list_element"]}'] = type_list['items']
        # print("types return:", {'types': types_list})
        return {'types': types_dict}

    def parameters(self, content):
        parameter_list = None
        if type(content[0]) == lark.Token and content[0].type == "WS":
            parameter_list = content[1]
        # print("parameters:", parameter_list)

        return {'parameters': parameter_list}

    def precondition(self, content):
        # print("precond cont:", content)
        # print("precond cont:", content[1][1:])
        return {'precondition': content[1:-1]}

    def effect(self, content):
        # print("effect cont:", content)
        effect_dict = {'effect': content[1:-1]}
        # print("effect returns:", effect_dict)
        return effect_dict

    def functions(self, content):
        # print("functions content:", content)
        functions_dict = {'functions': list()}
        for functions_item in content:
            if 'function_def_predicate' in functions_item:
                functions_dict['functions'].append(functions_item)

        return functions_dict

    def function_list_element(self, content):
        # print("function_list_element content:", content)

        # for function_item in content:
        #    print("function_list_element item:", function_item)

        function_def_predicate = content[0].value
        # print("function_predicate:", function_predicate)

        function_def_variable = content[2]
        # print("function_variable:", function_variable)

        function_def_type = content[6].value
        # print("function_type:", function_type)

        function_dict = {'function_def_predicate': function_def_predicate,
                         'function_def_variable': function_def_variable,
                         "function_def_type": function_def_type}

        return function_dict



class AdventureIFInterpreter(GameResourceLocator):
    """
    IF interpreter for adventuregame.
    Holds game world state and handles all interaction and feedback.
    """
    def __init__(self, game_path, game_instance: dict, name: str = GAME_NAME, verbose: bool = False, rng_seed: int = 42):
        super().__init__(name, game_path)

        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(seed=self.rng_seed)

        # game instance is the instance data as passed by the GameMaster class
        self.game_instance: dict = game_instance
        # surface strings (repr_str here) to spaceless internal identifiers:
        self.repr_str_to_type_dict: dict = dict()

        self.entity_types = dict()
        self.initialize_entity_types()

        self.room_types = dict()
        self.initialize_room_types()

        self.action_def_parser = None
        self.action_def_transformer = PDDLActionTransformer()
        self.domain_def_parser = None
        self.domain_def_transformer = PDDLDomainTransformer()
        if "event_definitions" in game_instance:
            self.event_def_parser = None
            self.event_def_transformer = PDDLEventTransformer()
        self.initialize_pddl_definition_parsing()

        self.act_parser = None
        self.act_transformer = IFTransformer()
        self.action_types = dict()
        self.initialize_action_types()

        self.domain = dict()
        self.initialize_domain()

        if "event_definitions" in game_instance:
            self.event_types = dict()
            self.initialize_event_types()

        self.world_state: set = set()
        self.world_state_history: list = list()
        self.goal_state: set = set()
        self.goals_achieved: set = set()
        self.initialize_states_from_strings()

        self.initialize_action_parsing(print_lark_grammar=verbose)

        self.exploration_history = list()
        self.exploration_state = set()
        # start tracking exploration:
        self.track_exploration()

    def initialize_entity_types(self):
        """
        Load and process entity types in this adventure.
        Definitions are loaded from external files.
        """
        # load entity type definitions in game instance:
        entity_definitions: list = list()

        for entity_def in self.game_instance["entity_definitions"]:
            # check if entity definition is file name string:
            if type(entity_def) == str:
                entities_file = self.load_json(f"resources{os.sep}definitions{os.sep}{entity_def[:-5]}")
                entity_definitions += entities_file
            # check if entity definition is direct dict:
            elif type(entity_def) == dict:
                entity_definitions.append(entity_def)

        for entity_definition in entity_definitions:
            self.entity_types[entity_definition['type_name']] = dict()
            for entity_attribute in entity_definition:
                if entity_attribute == 'type_name':
                    # assign surface strings:
                    self.repr_str_to_type_dict[entity_definition['repr_str']] = entity_definition[entity_attribute]
                else:
                    # get all other attributes:
                    self.entity_types[entity_definition['type_name']][entity_attribute] = entity_definition[
                        entity_attribute]

    def initialize_room_types(self):
        """
        Load and process room types in this adventure.
        Definitions are loaded from external files.
        """
        # load room type definitions in game instance:
        room_definitions: list = list()

        for room_def in self.game_instance["room_definitions"]:
            # check if room definition is file name string:
            if type(room_def) == str:
                rooms_file = self.load_json(f"resources{os.sep}definitions{os.sep}{room_def[:-5]}")
                room_definitions += rooms_file
            # check if room definition is direct dict:
            elif type(room_def) == dict:
                room_definitions.append(room_def)

        for room_definition in room_definitions:
            self.room_types[room_definition['type_name']] = dict()
            for room_attribute in room_definition:
                if room_attribute == 'type_name':
                    # assign surface strings:
                    self.repr_str_to_type_dict[room_definition['repr_str']] = room_definition[room_attribute]
                else:
                    # get all other attributes:
                    self.room_types[room_definition['type_name']][room_attribute] = room_definition[
                        room_attribute]

    def initialize_pddl_definition_parsing(self):
        action_def_grammar = self.load_file(f"resources{os.sep}pddl_actions.lark")
        self.action_def_parser = Lark(action_def_grammar, start="action")
        domain_def_grammar = self.load_file(f"resources{os.sep}pddl_domain.lark")
        self.domain_def_parser = Lark(domain_def_grammar, start="define")
        if "event_definitions" in self.game_instance:
            event_def_grammar = self.load_file(f"resources{os.sep}pddl_events.lark")
            self.event_def_parser = Lark(event_def_grammar, start="event")

    def initialize_action_types(self):
        """
        Load and process action types in this adventure.
        Definitions are loaded from external files.
        """
        # load action type definitions in game instance:
        action_definitions: list = list()

        for action_def in self.game_instance["action_definitions"]:
            # check if action definition is file name string:
            if type(action_def) == str:
                actions_file = self.load_json(f"resources{os.sep}definitions{os.sep}{action_def[:-5]}")
                action_definitions += actions_file
            # check if room definition is direct dict:
            elif type(action_def) == dict:
                action_definitions.append(action_def)

        for action_definition in action_definitions:
            self.action_types[action_definition['type_name']] = dict()
            # get all action attributes:
            for action_attribute in action_definition:
                if not action_attribute == 'type_name':
                    self.action_types[action_definition['type_name']][action_attribute] = action_definition[
                        action_attribute]

        for action_type in self.action_types:
            cur_action_type = self.action_types[action_type]
            if 'pddl' in cur_action_type:
                # print(cur_action_type['pddl'])
                parsed_action_pddl = self.action_def_parser.parse(cur_action_type['pddl'])
                processed_action_pddl = self.action_def_transformer.transform(parsed_action_pddl)
                # print(processed_action_pddl)
                cur_action_type['interaction'] = processed_action_pddl
            else:
                raise KeyError

    def initialize_domain(self):
        """Load and process the domain(s) used in this adventure.
        Definitions are loaded from external files.
        """
        # load domain definitions in game instance:
        domain_definitions: list = list()

        # domain_def = self.game_instance["domain_definition"]
        domain_def = self.game_instance["domain_definitions"][0]

        # print("domain_def:", domain_def)

        domain_preparsed = False

        # check if action definition is file name string:
        if type(domain_def) == str:
            # print("domain_def is str")
            domain_file = self.load_json(f"resources{os.sep}definitions{os.sep}{domain_def[:-5]}")
            # print("domain_file:", domain_file)
            # domain_definitions += domain_file
            domain_definitions.append(domain_file)
            # print("domain_definitions:", domain_definitions)
        # check if room definition is direct dict:
        elif type(domain_def) == dict:
            domain_definitions.append(domain_def)
            domain_preparsed = True
            # TODO: make this more robust

        # print("domain_definitions", domain_definitions)

        self.domain['mutable_states']: list = list()

        if not domain_preparsed:
            for domain_definition in domain_definitions:
                # print("domain_definition:", domain_definition)
                # parsed_domain_pddl = self.domain_def_parser.parse(domain_definition)
                parsed_domain_pddl = self.domain_def_parser.parse(domain_definition['pddl_domain'])
                processed_domain_pddl = self.domain_def_transformer.transform(parsed_domain_pddl)
                # print("processed_domain_pddl:", processed_domain_pddl)
                # for now assume only one domain definition:
                self.domain = processed_domain_pddl
                if 'mutable_states' in domain_definition:
                    # print("mutable states in domain file")
                    self.domain['mutable_states'] = domain_definition['mutable_states']
                else:
                    self.domain['mutable_states'] = ["open", "closed", "at", "in", "on"]
        else:
            processed_domain_pddl = domain_definitions[-1]
            # for now assume only one domain definition:
            self.domain = processed_domain_pddl

        # for now assume only one domain definition:
        # self.domain = processed_domain_pddl
        # multiple domain definitions would need proper checks/unification

        # TODO?: full type inheritance as dict or the like?

        # print("domain:", self.domain)
        # print("domain types:", self.domain['types'])

        # TRAIT TYPES FROM ENTITY DEFINITIONS
        # print("self.entity_types:", self.entity_types)
        # print("self.domain:", self.domain)
        # print(self.domain['types']['entity'])
        trait_type_dict = dict()
        for entity_type in self.domain['types']['entity']:

            # print("domain entity type:", entity_type, "; type defined:", self.entity_types[entity_type])
            if 'traits' in self.entity_types[entity_type]:
                # print("defined type traits:", self.entity_types[entity_type]['traits'])
                for trait in self.entity_types[entity_type]['traits']:
                    if trait not in trait_type_dict:
                        trait_type_dict[trait] = [entity_type]
                    else:
                        trait_type_dict[trait].append(entity_type)
                    if trait not in self.domain['types']:
                        self.domain['types'][trait] = [entity_type]
                    else:
                        self.domain['types'][trait].append(entity_type)
        # print("trait type dict:", trait_type_dict)
        # print(self.domain['types'])

        # REVERSE SUBTYPE/SUPERTYPE DICT
        supertype_dict = dict()
        for supertype, subtypes in self.domain['types'].items():
            # print("supertype:", supertype, "subtypes:", subtypes)
            for subtype in subtypes:
                if subtype not in supertype_dict:
                    supertype_dict[subtype] = [supertype]
                else:
                    supertype_dict[subtype].append(supertype)

        # print(supertype_dict)
        self.domain['supertypes'] = supertype_dict
        # print("domain:", self.domain)

        if domain_preparsed:
            # mutable states/predicates set from domain:
            mutable_states: list = list()
            for predicate in self.game_instance['domain_definitions'][0]['predicates']:
                mutable_states.append(predicate['predicate_id'])
            self.domain['mutable_states'] = mutable_states

    def initialize_event_types(self):
        """Load and process the event(s) used in this adventure.
        Definitions are loaded from external files.
        """
        # load event definitions in game instance:
        event_definitions: list = list()

        for event_def in self.game_instance["event_definitions"]:
            # check if event definition is file name string:
            if type(event_def) == str:
                events_file = self.load_json(f"resources{os.sep}definitions{os.sep}{event_def[:-5]}")
                event_definitions += events_file
            # check if event definition is direct dict:
            elif type(event_def) == dict:
                event_definitions.append(event_def)

        # print("event_definitions", event_definitions)

        for event_definition in event_definitions:
            self.event_types[event_definition['type_name']] = dict()
            # get all action attributes:
            for event_attribute in event_definition:
                if not event_attribute == 'type_name':
                    self.event_types[event_definition['type_name']][event_attribute] = event_definition[
                        event_attribute]

        for event_type in self.event_types:
            cur_event_type = self.event_types[event_type]
            if 'pddl' in cur_event_type:
                # logger.info(f"event pddl:\n{cur_event_type['pddl']}")
                parsed_event_pddl = self.event_def_parser.parse(cur_event_type['pddl'])
                processed_event_pddl = self.event_def_transformer.transform(parsed_event_pddl)
                # print(processed_event_pddl)
                cur_event_type['interaction'] = processed_event_pddl
            else:
                raise KeyError

        # print(self.event_types)

    def initialize_action_parsing(self, print_lark_grammar: bool = False):
        """
        Initialize the lark action input parser and transformer.
        Constructs a lark grammar string from action definition lark snippets.
        """
        act_grammar_rules = list()
        act_grammar_larks = list()

        for action_type in self.action_types:
            cur_action_type = self.action_types[action_type]
            action_rule = cur_action_type['lark'].split(":")[0]
            act_grammar_rules.append(action_rule)
            act_grammar_larks.append(cur_action_type['lark'])
        # root rule to parse any action command input, with fallback 'unknown':
        act_grammar_action_line = f"action: {' | '.join(act_grammar_rules)} | unknown\n"
        # append all individual action lark grammar snippets:
        act_grammar_larks_str = "\n".join(act_grammar_larks)
        # gather all possible adjectives from entity definitions:
        all_adjs = set()
        for entity_type, entity_def in self.entity_types.items():
            if 'possible_adjs' in entity_def:
                new_adj_set = set(entity_def['possible_adjs'])
                all_adjs.update(new_adj_set)
        all_adjs = [f'"{adj}"' for adj in all_adjs]
        # adjective rule:
        act_grammar_adj_line = f"ADJ.1: ({' | '.join(all_adjs)}) WS\n"
        # load the core grammar from file:
        grammar_core = self.load_json(f"resources{os.sep}grammar_core")
        grammar_head = grammar_core['grammar_head']
        grammar_foot = grammar_core['grammar_foot']
        # combine adventure-specific grammar rules with core grammar:
        act_grammar = (f"{grammar_head}{act_grammar_action_line}"
                       f"{act_grammar_larks_str}\n{act_grammar_adj_line}{grammar_foot}")
        # print grammar in verbose mode for inspection:
        if print_lark_grammar:
            print(act_grammar)
        # initialize lark parser with the combined grammar:
        self.act_parser = Lark(act_grammar, start='action')

    def initialize_states_from_strings(self):
        """
        Initialize the world state set from instance data.
        Converts List[Str] world state format into Set[Tuple].
        """
        # INITIAL STATE:
        for fact_string in self.game_instance['initial_state']:
            self.world_state.add(fact_str_to_tuple(fact_string))

        # NOTE: The following world state augmentations are left in here to make manual adventure creation/modification
        # convenient. Initial adventure world states generated with the clingo adventure generator already cover these
        # augmentations. Due to the world state being a set of tuples, the augmentations done here simply unify.

        # facts to add are gathered in a set to prevent duplicates
        facts_to_add = set()

        # add trait facts for objects:
        for fact in self.world_state:
            if fact[0] == 'type':
                # logger.info({"type fact for trait assignment": fact})
                # add trait facts by entity type:
                if 'traits' in self.entity_types[fact[2]]:
                    type_traits: list = self.entity_types[fact[2]]['traits']
                    for type_trait in type_traits:
                        facts_to_add.add((type_trait, fact[1]))

        # add floors to rooms:
        for fact in self.world_state:
            if fact[0] == 'room':
                facts_to_add.add(('type', f'{fact[1]}floor1', 'floor'))
                # add floor:
                facts_to_add.add(('at', f'{fact[1]}floor1', fact[1]))

        self.world_state = self.world_state.union(facts_to_add)

        # dict with the type for each entity instance in the adventure:
        self.inst_to_type_dict = dict()
        # get entity instance types from world state:
        for fact in self.world_state:
            # entity instance to entity type mapping:
            if fact[0] == 'type':
                self.inst_to_type_dict[fact[1]] = fact[2]

        # dict with the type for each room instance in the adventure:
        self.room_to_type_dict = dict()
        # get room instance types from world state:
        for fact in self.world_state:
            # room instance to room type mapping:
            if fact[0] == 'room':
                self.room_to_type_dict[fact[1]] = fact[2]

        # put 'supported' items on the floor if they are not 'in' or 'on':
        for fact in self.world_state:
            if fact[1] in self.inst_to_type_dict:
                if self.inst_to_type_dict[fact[1]] in self.entity_types:
                    pass
            if fact[0] == 'at' and ('needs_support', fact[1]) in self.world_state:
                currently_supported = False
                for state_pred2 in self.world_state:
                    if state_pred2[0] == 'on' and state_pred2[1] == fact[1]:
                        currently_supported = True
                        break
                    if state_pred2[0] == 'in' and state_pred2[1] == fact[1]:
                        currently_supported = True
                        break
                if not currently_supported:
                    facts_to_add.add(('on', fact[1], f'{fact[2]}floor'))

        # make items that are not 'in' closed containers or 'in' inventory or 'on' supports 'accessible':
        for fact in self.world_state:
            if fact[1] in self.inst_to_type_dict:
                if self.inst_to_type_dict[fact[1]] in self.entity_types:
                    pass
            if fact[0] == 'in' and ('container', fact[2]) in self.world_state:
                container_currently_open = False
                for state_pred2 in self.world_state:
                    if state_pred2[0] == 'open' and state_pred2[1] == fact[2]:
                        container_currently_open = True
                        break
                if container_currently_open:
                    facts_to_add.add(('accessible', fact[1]))
            if fact[0] == 'in' and fact[2] == 'inventory':
                # print(f"{fact[1]} in inventory!")
                facts_to_add.add(('accessible', fact[1]))
            if fact[0] == 'on' and ('support', fact[2]) in self.world_state:
                facts_to_add.add(('accessible', fact[1]))
            if fact[0] == 'type' and ('needs_support', fact[1]) not in self.world_state and fact[2] not in (
            "floor", "player"):
                facts_to_add.add(('accessible', fact[1]))
        # make inventory 'accessible' from the start:
        facts_to_add.add(('accessible', 'inventory'))

        self.world_state = self.world_state.union(facts_to_add)

        # FUNCTIONS
        if 'functions' in self.domain:
            # logger.info(f"Domain functions: {self.domain['functions']}")
            # convert premade initial_state function fact string numbers to proper numbers:
            for function_def in self.domain['functions']:
                # logger.info(f"Checking domain function: {function_def}")
                found_function_facts = list()
                for fact in self.world_state:
                    if fact[0] == function_def['function_def_predicate']:
                        found_function_facts.append(fact)
                # logger.info(f"Found function facts: {found_function_facts}")
                # remove non-number function facts and replace with number function facts:
                for found_function_fact in found_function_facts:
                    self.world_state.remove(found_function_fact)
                    found_function_fact_list = list(found_function_fact)
                    if "." in found_function_fact_list[2]:
                        found_function_fact_list[2] = float(found_function_fact_list[2])
                    else:
                        found_function_fact_list[2] = int(found_function_fact_list[2])
                    found_function_fact_tuple = tuple(found_function_fact_list)
                    self.world_state.add(found_function_fact_tuple)

                # add function facts with value 0 for defined functions in the domain for corresponding type instances:
                for fact in self.world_state:
                    # TODO?: use domain type inheritance to augment in addition to direct type?
                    if fact[0] == 'type' and fact[2] == function_def['function_def_type']:
                        augmentable_function_fact = [function_def['function_def_predicate'], fact[1], 0]
                        function_fact_already_exists = False
                        for fact2 in self.world_state:
                            if fact2[0] == function_def['function_def_predicate'] and fact2[1] == fact[1]:
                                function_fact_already_exists = True
                        if not function_fact_already_exists:
                            self.world_state.add(tuple(augmentable_function_fact))

                # add missing function fact(s) with value 0 for inventory as there is no type fact for inventory:
                if function_def['function_def_type'] == "inventory":
                    inventory_function_fact_already_exists = False
                    for fact in self.world_state:
                        if fact[0] == function_def['function_def_predicate'] and fact[1] == "inventory":
                            inventory_function_fact_already_exists = True
                    if not inventory_function_fact_already_exists:
                        self.world_state.add((function_def['function_def_predicate'], "inventory", 0))

        # add initial world state to world state history:
        self.world_state_history.append(deepcopy(self.world_state))

        # GOALS
        # get goal state fact set:
        for fact_string in self.game_instance['goal_state']:
            self.goal_state.add(fact_str_to_tuple(fact_string))

    def _get_inst_str(self, inst) -> str:
        """
        Get a full string representation of an entity or room instance with adjectives.
        Args:
            inst: The object instance ID of the object instance to get the surface string representation for.
                Ex: 'apple1', 'livingroom1'
        Returns:
            Full surface string representation of the object instance. Ex: 'red apple', 'living room'
        """
        inst_adjs = list()
        # get instance adjectives from adj facts:
        for fact in self.world_state:
            if fact[0] == 'adj' and fact[1] == inst:
                inst_adjs.append(fact[2])

        # get type of instance:
        inst_type = self._inst_to_type(inst)

        # get surface string for instance type:
        if inst_type in self.entity_types:
            inst_str: str = self.entity_types[inst_type]['repr_str']
        elif inst_type in self.room_types:
            inst_str: str = self.room_types[inst_type]['repr_str']
        else:  # fallback for potential edge cases
            inst_str: str = inst_type
        # combine into full surface string:
        inst_adjs.append(inst_str)
        adj_str = " ".join(inst_adjs)

        return adj_str

    def _inst_to_type(self, inst) -> str:
        """Get the type name of an entity or room instance.
        Args:
            inst: The instance ID string.
        Returns:
            The type name string for the entity or room instance.
        """
        # get type of instance:
        if inst in self.inst_to_type_dict:
            inst_type: str = self.inst_to_type_dict[inst]
        elif inst in self.room_to_type_dict:
            inst_type: str = self.room_to_type_dict[inst]
        else:  # fallback for potential edge cases
            # TODO: retrace why this can fail
            logger.info(f"_inst_to_type got {inst}, which is not in the _to_type dicts! "
                        f"Heuristically culling numbers from inst string end as fallback...")
            inst_type = deepcopy(inst)
            while inst_type.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                inst_type = inst_type[:-1]
            logger.info(f"inst_type after heuristic culling: {inst_type}")

        return inst_type

    def get_player_room(self) -> str:
        """
        Get the current player location's internal room string ID.
        """
        for fact in self.world_state:
            if fact[0] == 'at' and fact[1] == 'player1':
                player_room = fact[2]
                break

        return player_room

    def get_player_room_contents(self) -> List:
        """
        Get all contents of the current player location room.
        """
        player_room = self.get_player_room()
        room_contents = list()
        for fact in self.world_state:
            # get all entities 'at' the player's location, except the player themselves:
            if fact[0] == 'at' and fact[2] == player_room and not fact[1] == 'player1':
                room_contents.append(fact[1])

        return room_contents

    def get_player_room_contents_visible(self) -> List:
        """
        Get the visible contents of the current room.
        Entities 'in' closed entities are not returned.
        In v2, this is NO LONGER used to determine if an entity is accessible for interaction - this is handled via PDDL
        action definition now.
        """
        room_contents = self.get_player_room_contents()
        visible_contents = list()
        for thing in room_contents:
            # do not access entities that are hidden by type:
            if 'hidden' in self.entity_types[self._inst_to_type(thing)]:
                continue

            # do not access entities inside closed containers:
            contained_in = None
            for fact in self.world_state:
                # check if entity is 'in' closed container:
                if fact[0] == 'in' and fact[1] == thing:
                    contained_in = fact[2]
                    # print(f"{thing} is contained in {contained_in}")
                    for state_pred2 in self.world_state:
                        if state_pred2[1] == contained_in:
                            if state_pred2[0] == 'closed':
                                # not visible/accessible in closed container
                                break
                            elif state_pred2[0] == 'open':
                                visible_contents.append(thing)
                                break
                            elif state_pred2[1] == 'inventory':
                                # inventory content is not visible
                                break
                            else:
                                visible_contents.append(thing)
                                break

            if contained_in:
                continue
            visible_contents.append(thing)

        return visible_contents

    def get_player_room_exits(self) -> List:
        """
        Get all passages in the current room.
        """
        player_room = self.get_player_room()
        room_exits = list()
        for fact in self.world_state:
            # passage facts are 'exit' in the adventure/instance format
            if fact[0] == 'exit' and fact[1] == player_room:
                room_exits.append(fact[2])

        return room_exits

    def get_full_room_desc(self) -> str:
        """
        Creates and returns full description of the room the player is at.
        """
        # get player room:
        player_room = self.get_player_room()
        # create room description start:
        room_repr_str = self.room_types[self.room_to_type_dict[player_room]]['repr_str']
        # using simple type surface string due to v1 not having multiple rooms of the same type:
        player_at_str = f"You are in a {room_repr_str} now."

        # get visible room content:
        internal_visible_contents = self.get_player_room_contents_visible()
        # print("internal_visible_contents:", internal_visible_contents)

        # convert to types:
        visible_contents = [self._get_inst_str(instance) for instance in internal_visible_contents]

        # create visible room content description:
        visible_contents_str = str()
        if len(visible_contents) >= 3:
            comma_list = ", a ".join(visible_contents[:-1])
            and_last = f"and a {visible_contents[-1]}"
            visible_contents_str = f"There are a {comma_list} {and_last}."
            visible_contents_str = " " + visible_contents_str
        elif len(visible_contents) == 2:
            visible_contents_str = f"There are a {visible_contents[0]} and a {visible_contents[1]}."
            visible_contents_str = " " + visible_contents_str
        elif len(visible_contents) == 1:
            visible_contents_str = f"There is a {visible_contents[0]}."
            visible_contents_str = " " + visible_contents_str

        # get predicate state facts of visible objects and create textual representations:
        # TODO: de-hardcode this, anticipate localization
        visible_content_state_strs = list()
        for thing in internal_visible_contents:
            for fact in self.world_state:
                if fact[0] == 'closed' and fact[1] == thing:
                    visible_content_state_strs.append(f"The {self._get_inst_str(thing)} is closed.")
                elif fact[0] == 'open' and fact[1] == thing:
                    visible_content_state_strs.append(f"The {self._get_inst_str(thing)} is open.")
                if fact[0] == 'in' and fact[1] == thing:
                    visible_content_state_strs.append(
                        f"The {self._get_inst_str(thing)} is in the {self._get_inst_str(fact[2])}.")
                if fact[0] == 'on' and fact[1] == thing:
                    visible_content_state_strs.append(
                        f"The {self._get_inst_str(thing)} is on the {self._get_inst_str(fact[2])}.")

        if visible_content_state_strs:
            visible_content_state_combined = " ".join(visible_content_state_strs)
            visible_content_state_combined = " " + visible_content_state_combined
        else:
            visible_content_state_combined = str()

        # get room passages and create textual representation:
        room_exits = self.get_player_room_exits()
        exits_str = str()
        if len(room_exits) == 1:
            exits_str = f" There is a passage to a {self._get_inst_str(room_exits[0])} here."
        elif len(room_exits) == 2:
            exits_str = (f" There are passages to a {self._get_inst_str(room_exits[0])} and a "
                         f"{self._get_inst_str(room_exits[1])} here.")
        elif len(room_exits) >= 3:
            comma_exits = ", a ".join([self._get_inst_str(room_exit) for room_exit in room_exits[:-1]])
            exits_str = f" There are passages to a {comma_exits} and a {self._get_inst_str(room_exits[-1])} here."

        # combine full room description:
        room_description = f"{player_at_str}{visible_contents_str}{visible_content_state_combined}{exits_str}"

        return room_description

    def get_inventory_content(self) -> list:
        """Get list of inventroy content."""
        inventory_content = list()
        for fact in self.world_state:
            if fact[0] == 'in' and fact[2] == 'inventory':
                inventory_content.append(fact[1])

        return inventory_content

    def get_inventory_desc(self) -> str:
        """Get a text description of the current inventory content.
        Used for feedback for 'take' action.
        """
        inventory_content: list = self.get_inventory_content()
        inv_list = inventory_content
        inv_item_cnt = len(inv_list)
        if inv_item_cnt == 0:
            inv_desc = "Your inventory is empty."
            return inv_desc
        elif inv_item_cnt == 1:
            inv_str = f"a {self._get_inst_str(inv_list[0])}"
        else:
            inv_strs = [f"a {self._get_inst_str(inv_item)}" for inv_item in inv_list]
            inv_str = ", ".join(inv_strs[:-1])
            inv_str += f" and {inv_strs[-1]}"
        inv_desc = f"In your inventory you have {inv_str}."

        return inv_desc

    def get_container_content(self, container_id) -> list:
        container_content = list()
        for fact in self.world_state:
            if fact[0] == 'in' and fact[2] == container_id:
                container_content.append(fact[1])

        return container_content

    def get_container_content_desc(self, container_id) -> str:
        container_repr = self._get_inst_str(container_id)
        container_content = self.get_container_content(container_id)
        container_item_cnt = len(container_content)
        if container_item_cnt == 0:
            content_desc = f"The {container_repr} is empty."
            return content_desc
        elif container_item_cnt == 1:
            inv_str = f"a {self._get_inst_str(container_content[0])}"
            content_desc = f"In the {container_repr} there is {inv_str}."
        else:
            content_strs = [f"a {self._get_inst_str(container_item)}" for container_item in container_content]
            content_str = ", ".join(content_strs[:-1])
            content_str += f" and {content_strs[-1]}"
            content_desc = f"In the {container_repr} there are {content_str}."

        return content_desc

    def get_entity_desc(self, entity) -> str:
        """Get a full description of an entity.
        Used for the EXAMINE action.
        """
        # get inventory description if inventory is examined:
        if entity == "inventory":
            return self.get_inventory_desc()

        # get entity ID:
        # NOTE: This assumes only one instance of any entity type is in any adventure!
        entity_id = str()
        for fact in self.world_state:
            if fact[0] == 'type' and fact[2] == entity:
                entity_id = fact[1]
                break
        # print("entity ID found:", entity_id)
        entity_desc_list = list()

        # basic type description:
        """
        stripped_entity_id = deepcopy(entity_id)
        while stripped_entity_id.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
            stripped_entity_id = stripped_entity_id[:-1]
        base_entity_desc = f"This is a {stripped_entity_id}."
        """
        base_entity_desc = f"This is a {self._get_inst_str(entity_id)}."
        entity_desc_list.append(base_entity_desc)

        # get all entity states to describe:
        for fact in self.world_state:
            if fact[1] == entity_id:
                # print("entity state fact:", fact)
                if fact[0] == "text":
                    entity_desc_list.append("There is writing on it.")
                # describe 'openable' entity states:
                if fact[0] == "openable":
                    openable_entity: str = fact[1]
                    while openable_entity.endswith(("0","1","2","3","4","5","6","7","8","9")):
                        openable_entity = openable_entity[:-1]
                    # print("openable_entity:", openable_entity)
                    for fact2 in self.world_state:
                        if fact2[1] == entity_id and fact2[0] in ("open", "closed"):
                            openable_state = fact2[0]
                            # print("openable_state:", openable_state)
                            break
                    openable_desc = f"The {self.entity_types[openable_entity]['repr_str']} is openable and currently {openable_state}."
                    entity_desc_list.append(openable_desc)
                # describe 'takeable' entities:
                if fact[0] == "takeable":
                    takeable_entity: str = fact[1]
                    while takeable_entity.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                        takeable_entity = takeable_entity[:-1]
                    # print("takeable_entity:", takeable_entity)
                    takeable_desc = f"The {self.entity_types[takeable_entity]['repr_str']} is takeable."
                    entity_desc_list.append(takeable_desc)
                # describe the container or support state of 'needs_support' entities:
                if fact[0] == "needs_support":
                    needs_support_entity: str = fact[1]
                    while needs_support_entity.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                        needs_support_entity = needs_support_entity[:-1]
                    # print("needs_support_entity:", needs_support_entity)

                    for fact2 in self.world_state:
                        if fact2[1] == entity_id and fact2[0] in ("on", "in"):
                            support_state = fact2[0]
                            # print("support_state:", support_state)
                            supporter_entity = fact2[2]
                            # print("supporter_entity:", supporter_entity)
                            break

                    if supporter_entity == "inventory":
                        supporter_entity = "inventory"
                    else:
                        while supporter_entity.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                            supporter_entity = supporter_entity[:-1]
                        # supporter_entity = f"the {self.entity_types[supporter_entity]['repr_str']}"
                    if supporter_entity.endswith("floor"):
                        needs_support_desc = f"The {self.entity_types[needs_support_entity]['repr_str']} is {support_state} the floor."
                    elif supporter_entity.endswith("ceiling"):
                        needs_support_desc = f"The {self.entity_types[needs_support_entity]['repr_str']} is {support_state} the ceiling."
                    else:
                        needs_support_desc = f"The {self.entity_types[needs_support_entity]['repr_str']} is {support_state} the {self.entity_types[supporter_entity]['repr_str']}."
                    entity_desc_list.append(needs_support_desc)

                if fact[0] == "container":
                    container_entity: str = fact[1]
                    while container_entity.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                        container_entity = container_entity[:-1]
                    # print("container_entity:", container_entity)

                    contained_entities = list()

                    for fact2 in self.world_state:
                        if len(fact2) == 3:
                            if fact2[2] == entity_id and fact2[0] == "in":
                                # print(fact2)
                                contained_entity = fact2[1]
                                # print("contained_entity:", contained_entity)
                                # check if contained entity is accessible:
                                if ('accessible', contained_entity) not in self.world_state:
                                    # print()
                                    continue
                                # print("contained_entity is accessible")

                                while contained_entity.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                                    contained_entity = contained_entity[:-1]
                                # print("contained_entity:", contained_entity)
                                contained_entities.append(f"a {self.entity_types[contained_entity]['repr_str']}")

                    if ('closed', fact[1]) in self.world_state:
                        container_content_desc = f"You can't see the {self.entity_types[container_entity]['repr_str']}'s contents because it is closed."
                    else:
                        if len(contained_entities) == 0:
                            container_content_desc = f"The {self.entity_types[container_entity]['repr_str']} is empty."
                        elif len(contained_entities) == 1:
                            container_content_desc = f"There is {contained_entities[0]} in the {self.entity_types[container_entity]['repr_str']}."
                        elif len(contained_entities) == 2:
                            container_content_desc = f"There are {contained_entities[0]} and {contained_entities[1]} in the {self.entity_types[container_entity]['repr_str']}."
                        elif len(contained_entities) >= 3:
                            container_content_desc = f"There are {', '.join(contained_entities[:-1])} and {contained_entities[-1]} in the {self.entity_types[container_entity]['repr_str']}."

                    entity_desc_list.append(container_content_desc)

                if fact[0] == "support":
                    support_entity: str = fact[1]
                    while support_entity.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                        support_entity = support_entity[:-1]
                    # print("support_entity:", support_entity)

                    supported_entities = list()

                    for fact2 in self.world_state:
                        if len(fact2) == 3:
                            if fact2[2] == entity_id and fact2[0] == "on":
                                # print(fact2)
                                supported_entity = fact2[1]
                                # print("supported_entity:", supported_entity)

                                while supported_entity.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                                    supported_entity = supported_entity[:-1]
                                # print("supported_entity:", supported_entity)
                                supported_entities.append(f"a {self.entity_types[supported_entity]['repr_str']}")

                    if len(supported_entities) == 0:
                        support_content_desc = f"There is nothing on the {self.entity_types[support_entity]['repr_str']}."
                    elif len(supported_entities) == 1:
                        support_content_desc = f"There is {supported_entities[0]} on the {self.entity_types[support_entity]['repr_str']}."
                    elif len(supported_entities) == 2:
                        support_content_desc = f"There are {supported_entities[0]} and {supported_entities[1]} on the {self.entity_types[support_entity]['repr_str']}."
                    elif len(supported_entities) >= 3:
                        support_content_desc = f"There are {', '.join(supported_entities[:-1])} and {supported_entities[-1]} on the {self.entity_types[support_entity]['repr_str']}."

                    entity_desc_list.append(support_content_desc)

                if self.domain['mutable_states'] and fact[0] in self.domain['mutable_states']:
                    if not fact[0] == "at":
                        mutable_state_entity: str = fact[1]
                        while mutable_state_entity.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                            mutable_state_entity = mutable_state_entity[:-1]
                        mutable_state_desc = f"The {self.entity_types[mutable_state_entity]['repr_str']} is {fact[0]}."
                        entity_desc_list.append(mutable_state_desc)

        return " ".join(entity_desc_list)

    def get_entity_text(self, entity) -> str:
        """Get the readable text of an entity.
        Used for the READ action.
        """
        # get entity ID:
        # NOTE: This assumes only one instance of any entity type is in any adventure!
        entity_id = str()
        for fact in self.world_state:
            if fact[0] == 'type' and fact[2] == entity:
                entity_id = fact[1]
                break

        # get entity's text fact:
        for fact in self.world_state:
            if fact[1] == entity_id:
                # return text fact content:
                if fact[0] == "text":
                    return fact[2]

    def get_current_perceived(self) -> set:
        current_perceived: set = set()

        # get player room at fact
        for fact in self.world_state:
            if fact[0] == 'at' and fact[1] == 'player1':
                current_perceived.add(fact)

        visible_room_contents = self.get_player_room_contents_visible()
        for fact in self.world_state:
            # TODO: de-hardcode mutable predicates tracked here
            # if fact[1] in visible_room_contents and fact[0] in self.domain['mutable_states']:
            if fact[1] in visible_room_contents and fact[0] in ["open", "closed", "at", "in", "on"]:
                current_perceived.add(fact)

        inventory_content = self.get_inventory_content()
        for fact in self.world_state:
            if fact[1] in inventory_content and fact[0] in ("at", "in"):
                current_perceived.add(fact)
            if fact[1] == "inventory" and fact[0] == "itemcount":  # TODO: de-hardcode this
                current_perceived.add(fact)

        # current_room_exits = self.get_player_room_exits()
        for fact in self.world_state:
            # if fact[0] == "exit" and fact[1] in current_room_exits:
            if fact[0] == "exit" and fact[1] == self.get_player_room():
                current_perceived.add(fact)

        # logger.info(f"current_perceived: {current_perceived}")

        return current_perceived

    def track_exploration(self, world_state_effects: dict = None):
        """Track exploration of the world state.
        Updates the exploration state with what the player perceives at the current turn and records it.
        """
        # logger.info(f"len(self.exploration_history): {len(self.exploration_history)}")

        if len(self.exploration_history) >= 1:
            # logger.info("len(self.exploration_history) >= 2")

            current_perceived: set = self.get_current_perceived()
            prior_known: set = self.exploration_history[-1]
            # logger.info(f"prior_known: {prior_known}")

            # changes:
            current_set_difference = current_perceived.difference(prior_known)  # newly perceived
            # logger.info(f"current_set_difference: {current_set_difference}")
            prior_set_difference = prior_known.difference(current_perceived)  # perceived before
            # logger.info(f"prior_set_difference: {prior_set_difference}")

            # not changed:
            current_set_intersect = current_perceived.intersection(prior_known)
            # logger.info(f"current_set_intersect: {current_set_intersect}")

            # logger.info(f"Exploration state before update: {self.exploration_state}")
            self.exploration_state = self.exploration_state.union(current_set_difference)
            # self.exploration_state = self.exploration_state.union(current_set_difference).difference(prior_set_difference)
            # logger.info(f"Exploration state after update: {self.exploration_state}")

            new_exploration_state_difference = self.exploration_state.difference(prior_known)
            # logger.info(f"new_exploration_state_difference: {new_exploration_state_difference}")

            # logger.info(f"Action resolution world_state_effects: {world_state_effects}")
            """
            for added_fact in world_state_effects['added']:
                if added_fact in self.exploration_state:
                    logger.info(f"Added fact {added_fact} is in exploration state.")
                else:
                    logger.info(f"Added fact {added_fact} NOT in exploration state!")
            for removed_fact in world_state_effects['removed']:
                if removed_fact in self.exploration_state:
                    logger.info(f"Removed fact {removed_fact} IS in exploration state!")
                else:
                    logger.info(f"Removed fact {removed_fact} not in exploration state.")
            """
            # remove facts from exploration state based on just-performed action:
            # NOTE: This is done this way to assure that actions like GO don't result in 'loss of exploration' as using
            # set operations would
            if world_state_effects:
                for removed_fact in world_state_effects['removed']:
                    if removed_fact in self.exploration_state:
                        # logger.info(f"Removing fact {removed_fact} from exploration state.")
                        self.exploration_state.remove(removed_fact)
                        # logger.info(f"Fact {removed_fact} in exploration state: {removed_fact in self.exploration_state}")

            # logger.info(f"Current exploration_state: {self.exploration_state}")
            # record current exploration state:
            self.exploration_history.append(deepcopy(self.exploration_state))
            # logger.info(f"Current exploration_history: {self.exploration_history}")

        # record initial exploration state:
        if not self.exploration_state:
            logger.info("Recording initial exploration state.")
            self.exploration_state = self.get_current_perceived()
            self.exploration_history.append(self.exploration_state)

    def parse_action_input(self, action_input: str) -> [bool, Union[dict, str], Union[dict, Set]]:
        """
        Parse input action command string to action dict.
        Input is cleaned by removing trailing punctuation and lower-casing it.
        Fail if action/entities are not defined or input command is not covered by grammar.
        This method is effectively the parsing phase mentioned in the paper.
        Returns tuple of: failure bool, parsed action dict or failure feedback, failure information dict or empty set.
        """
        # remove final punctuation:
        if action_input.endswith(".") or action_input.endswith("!"):
            action_input = action_input[:-1]
        # lower for proper parsing:
        action_input = action_input.lower()

        logger.info(f"Cleaned action input: {action_input}")

        # try parsing input, return lark_exception failure if parsing fails:
        try:
            parsed_command = self.act_parser.parse(action_input)
        except Exception as exception:
            logger.info(f"Parsing lark exception")
            fail_dict: dict = {'phase': "parsing", 'fail_type': "lark_exception", 'arg': str(exception)}
            return False, f"I don't know what you mean.", fail_dict
        action_dict = self.act_transformer.transform(parsed_command)

        # catch 'unknown' action parses:
        if action_dict['type'] == "unknown":
            if action_dict['arg1'] in self.action_types:
                logger.info(f"Parsing unknown action with defined verb")
                logger.info(f"{action_dict}")
                fail_dict: dict = {'phase': "parsing", 'fail_type': "malformed_command", 'arg': str(action_dict)}
                return False, f"I don't know what you mean.", fail_dict

        if action_dict['type'] not in self.action_types:
            if 'arg1' in action_dict:
                logger.info(f"Parsing undefined action with undefined verb")
                fail_dict: dict = {'phase': "parsing", 'fail_type': "undefined_action_verb", 'arg': action_dict['arg1']}
                return False, f"I don't know how to interpret this '{action_dict['arg1']}' action.", fail_dict
            else:
                logger.info(f"Parsing undefined action without verb")
                fail_dict: dict = {'phase': "parsing", 'fail_type': "undefined_action", 'arg': action_input}
                return False, f"I don't know what you mean.", fail_dict

        logger.info(f"current parsed action_dict: {action_dict}")

        if action_dict['type'] == "done":
            return True, action_dict, {}

        if 'arg1' in action_dict:
            if action_dict['arg1'] in self.repr_str_to_type_dict:
                # convert arg1 from repr to internal type:
                action_dict['arg1'] = self.repr_str_to_type_dict[action_dict['arg1']]
            else:
                # in this case, the action is defined, but the first argument isn't, leading to corresponding feedback
                fail_dict: dict = {'phase': "parsing", 'fail_type': "undefined_repr_str", 'arg': action_dict['arg1']}
                return False, f"I don't know what '{action_dict['arg1']}' means.", fail_dict

            # TODO?: Remove action-type specific hardcode below?; should be handled by PDDL-based resolution now

            if action_dict['arg1'] not in self.entity_types:
                logger.info(f"Action arg1 '{action_dict['arg1']}' is not an entity")
                # handle manipulating rooms, ie "> take from kitchen":
                if action_dict['arg1'] in self.room_types:
                    if action_dict['type'] in ["take", "put", "open", "close"]:
                        logger.info(f"Action type is '{action_dict['type']}', manipulating room")
                        fail_dict: dict = {'phase': "parsing", 'fail_type': "manipulating_room",
                                           'arg': action_dict['arg1']}
                        if action_dict['type'] == "take":
                            fail_response = f"You can't {action_dict['type']} the '{action_dict['arg1']}'."
                        elif action_dict['type'] == "put":
                            fail_response = f"You can't {action_dict['type']} the '{action_dict['arg1']}' anywhere."
                        elif action_dict['type'] == "open":
                            fail_response = f"You don't need to {action_dict['type']} the '{action_dict['arg1']}'."
                        elif action_dict['type'] == "close":
                            fail_response = f"You can't {action_dict['type']} the '{action_dict['arg1']}'."
                        return False, fail_response, fail_dict
                else:
                    logger.info(f"Action arg1 {action_dict['arg1']} is not a room either")
                    fail_dict: dict = {'phase': "parsing", 'fail_type': "undefined_argument_type",
                                       'arg': action_dict['arg1']}
                    return False, f"I don't know what a '{action_dict['arg1']}' is.", fail_dict


        if 'arg2' in action_dict:
            if action_dict['type'] == "take":
                # handle unnecessary inventory interaction:
                if action_dict['arg2'] == "inventory":
                    # TODO: remove 'taking from inventory', now handled via PDDL precondition
                    #  but PDDL handling does it via precondition (not (in <item> inventory)), not by checking for the
                    #  second argument, so check if this handling here might still be useful
                    logger.info("Taking from inventory")
                    # get inventory content:
                    inventory_content = self.get_inventory_content()
                    for inventory_item in inventory_content:
                        if self.inst_to_type_dict[inventory_item] == action_dict['arg1']:
                            fail_dict: dict = {'phase': "resolution", 'fail_type': "taking_from_inventory",
                                               'arg': action_dict['arg1']}
                            return False, f"The {self.entity_types[action_dict['arg1']]['repr_str']} is already in your inventory.", fail_dict
                    fail_dict: dict = {'phase': "parsing", 'fail_type': "taking_from_inventory", 'arg': action_dict['arg2']}
                    return False, f"You don't need to take things from your inventory.", fail_dict
            if action_dict['arg2'] in self.repr_str_to_type_dict:
                # convert arg1 from repr to internal type:
                action_dict['arg2'] = self.repr_str_to_type_dict[action_dict['arg2']]
                # handle other room interaction attempts; ie "> take plate from kitchen" while player is elsewhere:
                if action_dict['arg2'] in self.room_types:
                    cur_room_str = self.room_types[self.room_to_type_dict[self.get_player_room()]]['repr_str']
                    if not action_dict['arg2'] == cur_room_str:
                        fail_dict: dict = {'phase': "parsing", 'fail_type': "other_room_argument",
                                           'arg': action_dict['arg2']}
                        return False, f"You are not in a {action_dict['arg2']}.", fail_dict
            else:
                fail_dict: dict = {'phase': "parsing", 'fail_type': "undefined_repr_str", 'arg': action_dict['arg2']}
                return False, f"I don't know what '{action_dict['arg2']}' means.", fail_dict

        return True, action_dict, {}

    def check_fact(self, fact_tuple) -> bool:
        """Check if a fact tuple is in the world state."""
        # logger.info(f"IF.check_fact() checking for {fact_tuple}")
        # always return True for fact tuples with None, as this marks optional action arguments
        if None in fact_tuple:
            return True

        if fact_tuple in self.world_state:
            # print(fact_tuple, "in world state!")
            return True
        else:
            return False

    def predicate_to_tuple(self, predicate, variable_map) -> tuple:
        """Convert a PDDL predicate object to a world state tuple.
        Resolves variables as well.
        Resolves type action/predicate arguments assuming single type instance.
        """
        # logger.info(f"predicate_to_tuple input predicate: {predicate}")
        # logger.info(f"predicate_to_tuple input variable_map: {variable_map}")

        predicate_type = predicate['predicate']
        # print("predicate_type:", predicate_type)

        predicate_arg1 = predicate['arg1']
        if 'variable' in predicate_arg1:
            predicate_arg1 = variable_map[predicate_arg1['variable']]
            # print("filled when condition variable:", when_condition_arg1)
            # for now:
            if type(predicate_arg1) == list:
                when_condition_arg1 = predicate_arg1[0]

        predicate_list = [predicate_type, predicate_arg1]
        # print("predicate_list after arg1:", predicate_list)

        predicate_arg2 = None
        if predicate['arg2']:
            predicate_arg2 = predicate['arg2']
            if 'variable' in predicate_arg2:
                predicate_arg2 = variable_map[predicate_arg2['variable']]
                # print("filled when condition variable:", when_condition_arg2)
            predicate_list.append(predicate_arg2)

        # print("predicate_list after arg2:", predicate_list)

        predicate_arg3 = None
        if predicate['arg3']:
            predicate_arg3 = predicate['arg3']
            if 'variable' in predicate_arg3:
                predicate_arg3 = variable_map[predicate_arg3['variable']]
                # print("filled when condition variable:", when_condition_arg3)
            predicate_list.append(predicate_arg3)

        # print("when_condition_list:", when_condition_list)
        predicate_tuple = tuple(predicate_list)
        # print("when_condition_tuple:", when_condition_tuple)

        # logger.info(f"predicate_to_tuple predicate_tuple intermediate: {predicate_tuple}")
        #rint(f"predicate_to_tuple predicate_tuple intermediate: {predicate_tuple}")

        # if not predicate_type == "type":
        if predicate_type not in ["type", "room"]:
            # assume that action arguments that don't end in numbers or "floor" are type words:
            for tuple_idx, tuple_arg in enumerate(predicate_tuple[1:]):  # first tuple item is always a predicate
                # print("tuple_arg:", tuple_arg)
                # logger.info(f"tuple_arg: {tuple_arg}")
                type_matched_instances = list()
                if tuple_arg:
                    # logger.info(f"predicate_to_tuple tuple_arg intermediate: {tuple_arg}")
                    # hotfix for witch outhouse teleport:
                    # if type(tuple_arg) == list and len(tuple_arg) == 1 and tuple_arg[0] == "inventory":
                    #    tuple_arg = tuple_arg[0]

                    # if not tuple_arg.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                    if not tuple_arg.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "inventory")):
                        # print(f"{tuple_arg} is not a type instance ID!")
                        # go over world state facts to find room or type predicate:
                        for fact in self.world_state:
                            # check for predicate fact matching action argument:
                            if fact[0] == "room" or fact[0] == "type":
                                if fact[2] == tuple_arg:
                                    # print(f"{fact[0]} predicate fact found:", fact)
                                    type_matched_instances.append(fact[1])

                        # logger.info(f"type_matched_instances: {type_matched_instances}")

                        # fallback to prevent list index out-of-range exceptions:
                        if not type_matched_instances:
                            logger.info(f"Empty type_matched_instances for predicate_to_tuple tuple_arg intermediate: {tuple_arg}")
                            type_matched_instances = [None]
                            # there will be no matching facts for the condition check, since none were found
                            # so the check result will be negative as intended
                            # TODO?: cascade fail with proper missing type if there is no type-fitting instance in world state?

                        # NOTE: This assumes all room and entity types have only a single instance in the adventure!

                        # replace corresponding variable_map value with instance ID:
                        for variable in variable_map:
                            if variable_map[variable] == tuple_arg:
                                variable_map[variable] = type_matched_instances[0]
                                # print("instance-filled variable_map:", variable_map)

                        # create fact tuple to check for:
                        match len(predicate_tuple):
                            case 2:
                                predicate_tuple = (predicate_tuple[0], type_matched_instances[0])
                            case 3:
                                if tuple_idx == 0:
                                    predicate_tuple = (
                                    predicate_tuple[0], type_matched_instances[0], predicate_tuple[2])
                                elif tuple_idx == 1:
                                    predicate_tuple = (
                                    predicate_tuple[0], predicate_tuple[1], type_matched_instances[0])
                            case 4:
                                if tuple_idx == 0:
                                    predicate_tuple = (
                                        predicate_tuple[0], type_matched_instances[0], predicate_tuple[2],
                                        predicate_tuple[3])
                                elif tuple_idx == 1:
                                    predicate_tuple = (
                                        predicate_tuple[0], predicate_tuple[1], type_matched_instances[0],
                                        predicate_tuple[3])
                                elif tuple_idx == 2:
                                    predicate_tuple = (
                                        predicate_tuple[0], predicate_tuple[1], predicate_tuple[2],
                                        type_matched_instances[0])

            # print("predicate_tuple post-instance resolution:", predicate_tuple)

        return predicate_tuple

    def check_conditions(self, conditions, variable_map, check_precon_idx = True, precon_trace = True) -> bool:
        """Check if a passed condition 'and'/'or' clause is true.
        Full action preconditions must have a root 'and' clause!
        """
        # print()
        # print("check_conditions input conditions:", conditions)

        if 'not' in conditions:
            # print("'Not' phrase condition.")

            not_dict = {'not': dict()}

            conditions_polarity = False
            inner_condition = conditions['not']
            # print("'not' phrase inner_condition:", inner_condition)
            inner_condition_is_fact = self.check_conditions(inner_condition, variable_map, check_precon_idx=check_precon_idx, precon_trace=precon_trace)
            # print("inner_condition_is_fact:", inner_condition_is_fact)

            if precon_trace:
                not_dict['not'] = inner_condition_is_fact
                not_true = False
                if inner_condition_is_fact['fulfilled'] == conditions_polarity:
                    not_true = True
                not_dict['fulfilled'] = not_true
                self.precon_trace.append(not_dict)

                # print("not_dict:", not_dict)

                return not_dict

            if inner_condition_is_fact == conditions_polarity:
                return True
            else:
                return False

        if 'predicate' in conditions:
            # logger.info(f"IF.check_conditions() bare predicate condition: {conditions}")
            # print(f"IF.check_conditions() bare predicate condition: {conditions}")
            predicate_tuple = self.predicate_to_tuple(conditions, variable_map)
            # print("predicate_tuple:", predicate_tuple)
            if check_precon_idx:
                # print("Current self.precon_idx:", self.precon_idx)
                pass
            is_fact = self.check_fact(predicate_tuple)
            # print("is_fact:", is_fact)
            if check_precon_idx:
                self.precon_idx += 1
                self.precon_tuples.append((predicate_tuple, is_fact, self.precon_idx))

            if precon_trace:
                predicate_dict = {'predicate_tuple': predicate_tuple, 'fulfilled': is_fact, 'precon_idx': self.precon_idx}
                # logger.info(f"predicate condition precon_trace predicate_dict: {predicate_dict}")
                return predicate_dict

            return is_fact

        if 'num_comp' in conditions:
            # logger.info(f"IF.check_conditions() num_comp condition: {conditions}")

            # get direct number argument values or function fact argument values:
            arg1_function_list = list()
            arg1_is_number = False
            if 'function_number' in conditions['arg1']:
                arg1_is_number = True
            elif 'function_id' in conditions['arg1']:
                arg1_function_list.append(conditions['arg1']['function_id'])
                # logger.info(f"arg1_function_list: {arg1_function_list}")
                arg1_function_var = conditions['arg1']['function_variable']['variable']
                # logger.info(f"arg1_function_var: {arg1_function_var}")
                arg1_function_object = variable_map[arg1_function_var]
                # logger.info(f"num_comp condition arg1 function object: {arg1_function_object}")
                arg1_function_list.append(arg1_function_object)

            if not arg1_is_number:
                # logger.info(f"num_comp condition arg1 is function")
                arg1_function_fact_found = False
                # get numerical value of first argument from function fact:
                for fact in self.world_state:
                    if fact[0] == arg1_function_list[0] and fact[1] == arg1_function_list[1]:
                        # logger.info(f"Found world state fact '{fact}' matching arg1_function_list '{arg1_function_list}")
                        arg1_function_fact_found = True
                        arg1_function_list.append(fact[2])
                        arg1_value = fact[2]
                if not arg1_function_fact_found:
                    # logger.info(f"No world state fact matching arg1_function_list '{arg1_function_list}' found!")
                    pass
            else:
                arg1_value = conditions['arg1']['function_number']
                if "." in arg1_value:
                    arg1_value = float(arg1_value)
                else:
                    arg1_value = int(arg1_value)

            arg2_function_list = list()
            arg2_is_number = False
            if 'function_number' in conditions['arg2']:
                arg2_is_number = True
            elif 'function_id' in conditions['arg2']:
                arg2_function_list.append(conditions['arg2']['function_id'])
                arg2_function_var = conditions['arg2']['function_variable']['variable']
                arg2_function_object = variable_map[arg2_function_var]
                # logger.info(f"num_comp condition arg2 function object: {arg2_function_object}")
                arg2_function_list.append(arg2_function_object)

            if not arg2_is_number:
                # get numerical value of second argument from function fact:
                for fact in self.world_state:
                    if fact[0] == arg2_function_list[0] and fact[1] == arg2_function_list[1]:
                        arg2_function_list.append(fact[2])
                        arg2_value = fact[2]
            else:
                arg2_value = conditions['arg2']['function_number']
                if "." in arg2_value:
                    arg2_value = float(arg2_value)
                else:
                    arg2_value = int(arg2_value)

            # get numerical comparison type:
            num_comp_type = conditions['num_comp']

            # numerical comparison:
            predicate_tuple = list()
            fulfilled = False
            match num_comp_type:
                case "equal":
                    if arg1_value == arg2_value:
                        fulfilled = True
                    else:
                        if arg1_function_list:
                            predicate_tuple = arg1_function_list
                        elif arg2_function_list:
                            predicate_tuple = arg2_function_list
                case "less":
                    if arg1_value < arg2_value:
                        fulfilled = True
                    else:
                        if arg1_function_list:
                            predicate_tuple = arg1_function_list
                        elif arg2_function_list:
                            predicate_tuple = arg2_function_list
                case "leq":
                    if arg1_value <= arg2_value:
                        fulfilled = True
                    else:
                        if arg1_function_list:
                            predicate_tuple = arg1_function_list
                        elif arg2_function_list:
                            predicate_tuple = arg2_function_list
                case "greater":
                    if arg1_value > arg2_value:
                        fulfilled = True
                    else:
                        if arg1_function_list:
                            predicate_tuple = arg1_function_list
                        elif arg2_function_list:
                            predicate_tuple = arg2_function_list
                case "greq":
                    if arg1_value >= arg2_value:
                        fulfilled = True
                    else:
                        if arg1_function_list:
                            predicate_tuple = arg1_function_list
                        elif arg2_function_list:
                            predicate_tuple = arg2_function_list

            if check_precon_idx:
                self.precon_idx += 1
                self.precon_tuples.append((tuple(predicate_tuple), fulfilled, self.precon_idx))

            if precon_trace:
                predicate_dict = {'predicate_tuple': tuple(predicate_tuple), 'fulfilled': fulfilled,
                                  'precon_idx': self.precon_idx}
                # logger.info(f"num_comp condition precon_trace predicate_dict: {predicate_dict}")
                return predicate_dict

            return fulfilled

        if 'and' in conditions:
            and_dict = {'and': list()}

            and_conditions_checklist = list()
            # print("And conditions:", conditions)
            conditions = conditions['and']
            # print("Extracted and conditions list:", conditions)
            for and_condition in conditions:
                # print("and_condition:", and_condition)

                # fulfilled = self.check_conditions(and_condition, variable_map, check_precon_idx=check_precon_idx, precon_trace=precon_trace)

                fulfilled = self.check_conditions(and_condition, variable_map, check_precon_idx=check_precon_idx, precon_trace=precon_trace)
                # print("and item fulfilled:", fulfilled)
                # since all facts need to check out for 'and' clauses, immediately return failure:
                # if not fulfilled:
                #    return False
                if precon_trace:
                    and_conditions_checklist.append(fulfilled['fulfilled'])
                    and_dict['and'].append(fulfilled)
                else:
                    and_conditions_checklist.append(fulfilled)
                # print()
            # print("and_conditions_checklist:", and_conditions_checklist)

            # check if all conditions are true:
            if precon_trace:
                and_phrase_true = False
                if not False in and_conditions_checklist:
                    and_phrase_true = True
                and_dict['fulfilled'] = and_phrase_true
                self.precon_trace.append(and_dict)
                # print("and_dict:", and_dict)
                return and_dict
            else:
                if not False in and_conditions_checklist:
                    return True
                else:
                    return False

        if 'or' in conditions:

            or_dict = {'or': list()}

            or_conditions_checklist = list()
            # print("Or conditions:", conditions)
            conditions = conditions['or']
            # print("Extracted or conditions list:", conditions)
            for or_condition in conditions:
                # print("or_condition:", or_condition)
                fulfilled = self.check_conditions(or_condition, variable_map, check_precon_idx=check_precon_idx, precon_trace=precon_trace)
                # print("or item fulfilled:", fulfilled)

                if precon_trace:
                    or_conditions_checklist.append(fulfilled['fulfilled'])
                    # print("or_conditions_checklist:", or_conditions_checklist)
                    or_dict['or'].append(fulfilled)
                else:
                    or_conditions_checklist.append(fulfilled)
                # print()
            # print("or_conditions_checklist:", or_conditions_checklist)

            # check if any condition is true:
            if precon_trace:
                or_phrase_true = False
                # print("or_conditions_checklist:", or_conditions_checklist)
                if True in or_conditions_checklist:
                    or_phrase_true = True
                or_dict['fulfilled'] = or_phrase_true
                self.precon_trace.append(or_dict)
                return or_dict
            else:
                if True in or_conditions_checklist:
                    return True
                else:
                    return False


        # NOTE: Handling forall conditions not implemented due to time constraints.

        # print()

        return False

    def resolve_forall(self, forall_clause, variable_map):
        # print("forall effect:", forall_clause)
        forall_type = forall_clause['forall']
        # print("forall_type:", forall_type)

        forall_results = {'added': [], 'removed': []}

        forall_variable_map = dict()  # all values can be expected to be lists

        # handle single-predicate forall:
        if 'predicate' in forall_type:
            # print()
            forall_predicate = forall_type['predicate']
            if 'variable' in forall_predicate:
                # since this is no type_list, supply list of all __entities__:
                all_entities_list = [fact[1] for fact in self.world_state if fact[0] == 'type']

                # NOTE: This assumes that forall clauses will only iterate over entities, NOT rooms!

                # print("all_entities_list:", all_entities_list)
                forall_variable_map[forall_predicate['variable']] = all_entities_list
                forall_items = all_entities_list
        elif 'type_list' in forall_type:
            # TODO?: iterate over multiple type list variables?
            # print("Type list forall_type:", forall_type)
            forall_type_list = forall_type['type_list']
            # print("forall_type_list:", forall_type_list)
            for type_list_element in forall_type_list:
                type_list_type = type_list_element['type_list_element']
                for type_list_item in type_list_element['items']:
                    # print("type_list_item:", type_list_item)
                    if 'variable' in type_list_item:
                        type_list_item_variable = type_list_item['variable']
                        # print("type_list_item_variable:", type_list_item_variable)
                        # get all type-matched objects:
                        type_matched_objects = list()
                        for fact in self.world_state:
                            # TODO?: use domain type definitions, employ object type inheritance?
                            if fact[0] == type_list_type:  # relies on type facts for now
                                type_matched_objects.append(fact[1])
                        # assign all matched objects to forall variable map:
                        forall_variable_map[type_list_item_variable] = type_matched_objects

        # print("forall_variable_map:", forall_variable_map)

        # NOTE: For now only covering forall with a single variable/type to iterate over, due to time constraints.

        for iterated_variable, iterated_values in forall_variable_map.items():
            # print("iterated_variable:", iterated_variable)
            # print("iterated_values:", iterated_values)
            for iterated_object in iterated_values:
                # print("iterated_object:", iterated_object)
                # create individual variable map for this iterated object:
                iteration_forall_variable_map = dict()
                for key, value in variable_map.items():
                    iteration_forall_variable_map[key] = value
                iteration_forall_variable_map[iterated_variable] = iterated_object
                # print("iteration_forall_variable_map:", iteration_forall_variable_map)

                # resolve forall body for iterated object:
                forall_body = forall_clause['body']
                # print("forall_body:", forall_body)

                for forall_body_element in forall_body:
                    # print("forall_body_element:", forall_body_element)
                    if 'when' in forall_body_element:
                        # print("When clause in forall body element:", forall_body_element['when'])
                        # print("When clause forall body element:", forall_body_element)
                        when_results = self.resolve_when(forall_body_element, iteration_forall_variable_map)
                        # print("when_results:", when_results)
                        forall_results['added'] += when_results['added']
                        forall_results['removed'] += when_results['removed']

                    if 'and' in forall_body_element:
                        # print("And clause forall body element:", forall_body_element)
                        # print("And clause in forall body element(s):", forall_body_element['and'])

                        and_items = forall_body_element['and']

                        for and_item in and_items:
                            # print("and_item:", and_item)
                            if 'predicate' in and_item or 'not' in and_item:
                                # print("Bare predicate or 'not' predicate item.")
                                and_item_results = self.resolve_effect(and_item, iteration_forall_variable_map)
                                # print("and_item_results:", and_item_results)
                                forall_results['added'] += and_item_results['added']
                                forall_results['removed'] += and_item_results['removed']
                            if 'when' in and_item:
                                when_results = self.resolve_when(and_item, iteration_forall_variable_map)
                                forall_results['added'] += when_results['added']
                                forall_results['removed'] += when_results['removed']
                            # print()

                    # TODO?: handle single-predicate forall bodies? -> would need grammar coverage

        # print("forall_results:", forall_results)

        return forall_results

    def resolve_when(self, when_clause, variable_map):
        when_results = {'added': [], 'removed': []}

        # when_items = when_clause['when']
        # get actual content:
        when_clause = when_clause['when']

        # print("when_clause:", when_clause)
        # print("variable_map for when clause:", variable_map)

        when_conditions_fulfilled = False

        when_conditions = when_clause[0]
        # print("when_conditions pre-and/or:", when_conditions)

        checked_conditions = self.check_conditions(when_conditions, variable_map, check_precon_idx=False, precon_trace=False)
        # print("checked_conditions:", checked_conditions)

        # if when_conditions_fulfilled:
        if checked_conditions:
            # print("When conditions fulfilled!")
            when_effects = when_clause[1]
            # print("when_effects:", when_effects)
            if 'and' in when_effects:
                when_effects = when_effects['and']
            else:
                # put single-predicate effect in list for uniform handling:
                when_effects = [when_effects]
            # print("when_effects after 'and' handling:", when_effects)

            for when_effect in when_effects:
                # print("when_effect:", when_effect)
                resolve_effect_results = self.resolve_effect(when_effect, variable_map)
                # print("resolve_effect_results", resolve_effect_results)
                when_results['added'] += resolve_effect_results['added']
                when_results['removed'] += resolve_effect_results['removed']
        else:
            # print("When conditions NOT fulfilled!")
            pass

        # print("when_results:", when_results)

        return when_results

    def resolve_effect(self, effect, variable_map):
        """Add or remove fact from world state based on passed effect object."""
        # logger.info(f"effect passed to resolve_effect: {effect}")

        resolve_effect_results = {'added': [], 'removed': []}

        # catch 'not' effects:
        effect_polarity = True
        if 'not' in effect:
            effect_polarity = False
            effect = effect['not']

        if 'predicate' in effect:
            effect_list = [effect['predicate'], effect['arg1']]  # effect predicates always have at least one argument
            if effect['arg2']:
                # print("effect['arg2']:", effect['arg2'])
                effect_list.append(effect['arg2'])
                if effect['arg3']:
                    effect_list.append(effect['arg3'])

            # apply variable map:
            for effect_arg_idx, effect_arg in enumerate(effect_list):
                if effect_arg_idx == 0:  # predicate does not need variable value application
                    continue
                # print(f"effect_arg {effect_arg_idx}:", effect_arg)
                if type(effect_arg) == dict and 'variable' in effect_arg:
                    # print("effect_arg['variable']:", effect_arg['variable'])
                    effect_list[effect_arg_idx] = variable_map[effect_arg['variable']]

            effect_tuple = tuple(effect_list)
            # print("effect_tuple:", effect_tuple)

            # return unfilled dict for fact tuples with None, as this marks optional action arguments:
            if None in effect_tuple:
                return resolve_effect_results

            if effect_polarity:
                # print(f"Adding {effect_tuple} to world state.")
                self.world_state.add(effect_tuple)
                resolve_effect_results['added'].append(effect_tuple)
            elif not effect_polarity:
                # print(f"Removing {effect_tuple} from world state.")
                if effect_tuple in self.world_state:
                    self.world_state.remove(effect_tuple)
                resolve_effect_results['removed'].append(effect_tuple)
        elif 'function_change' in effect:
            # logger.info(f"function_change effect passed to resolve_effect: {effect}")

            # get direct number argument values or function fact argument values:
            arg1_function_list = list()
            arg1_is_number = False
            if 'function_number' in effect['arg1']:
                arg1_is_number = True
            elif 'function_id' in effect['arg1']:
                arg1_function_list.append(effect['arg1']['function_id'])
                arg1_function_var = effect['arg1']['function_variable']['variable']
                arg1_function_object = variable_map[arg1_function_var]
                logger.info(f"num_comp condition arg1 function object: {arg1_function_object}")
                arg1_function_list.append(arg1_function_object)

            if not arg1_is_number:
                # get numerical value of first argument from function fact:
                for fact in self.world_state:
                    if fact[0] == arg1_function_list[0] and fact[1] == arg1_function_list[1]:
                        arg1_function_list.append(fact[2])
                        arg1_value = fact[2]
            else:
                arg1_value = effect['arg1']['function_number']
                if "." in arg1_value:
                    arg1_value = float(arg1_value)
                else:
                    arg1_value = int(arg1_value)

            arg2_function_list = list()
            arg2_is_number = False
            if 'function_number' in effect['arg2']:
                arg2_is_number = True
            elif 'function_id' in effect['arg2']:
                arg2_function_list.append(effect['arg2']['function_id'])
                arg2_function_var = effect['arg2']['function_variable']['variable']
                arg2_function_object = variable_map[arg2_function_var]
                logger.info(f"num_comp condition arg2 function object: {arg2_function_object}")
                arg2_function_list.append(arg2_function_object)

            if not arg2_is_number:
                # get numerical value of second argument from function fact:
                for fact in self.world_state:
                    if fact[0] == arg2_function_list[0] and fact[1] == arg2_function_list[1]:
                        arg2_function_list.append(fact[2])
                        arg2_value = fact[2]
            else:
                arg2_value = effect['arg2']['function_number']
                if "." in arg2_value:
                    arg2_value = float(arg2_value)
                else:
                    arg2_value = int(arg2_value)

            # remove old function value fact:
            if tuple(arg1_function_list) in self.world_state:
                self.world_state.remove(tuple(arg1_function_list))
            resolve_effect_results['removed'].append(tuple(arg1_function_list))

            # get function change type:
            function_change_type = effect['function_change']

            # numerical change:
            match function_change_type:
                case "increase":
                    arg1_function_list[2] += arg2_value
                case "decrease":
                    arg1_function_list[2] -= arg2_value
                case "assign":
                    arg1_function_list[2] = arg2_value

            self.world_state.add(tuple(arg1_function_list))
            resolve_effect_results['added'].append(tuple(arg1_function_list))

        # logger.info(f"resolve_effect results: {resolve_effect_results}")

        return resolve_effect_results

    def resolve_action(self, action_dict: dict) -> [bool, Union[Set, str], Union[dict, Set]]:
        # print("resolve_action input action_dict:", action_dict)

        # deepcopy the world state to prevent referential interaction:
        prior_world_state = deepcopy(self.world_state)

        # get current action definition:
        cur_action_def = self.action_types[action_dict['type']]
        # print("cur_action_def:", cur_action_def)
        # pretty_action(cur_action_def)
        # get current action PDDL parameter mapping:
        cur_action_pddl_map = self.action_types[action_dict['type']]['pddl_parameter_mapping']
        # print("cur_action_pddl_map:", cur_action_pddl_map)

        # PARAMETERS
        variable_map = dict()
        parameters_base = cur_action_def['interaction']['parameters']
        # print(parameters_base)
        # print("cur_action_def:", cur_action_def)
        # check that parameters key correctly contains a PDDL type_list:
        if not 'type_list' in parameters_base:
            raise KeyError
        # type match fail variable to allow parameter failure feedback after precon failures:
        type_match_fails: list = list()
        # get parameters list:
        parameters = parameters_base['type_list']
        for param_idx, parameter in enumerate(parameters):
            # print("\nparameter:", parameter, "idx:", param_idx)
            cur_parameter_type = parameter['type_list_element']
            # print("cur_parameter_type:", cur_parameter_type)
            # go over variables in parameter:
            for variable in parameter['items']:
                # print("variable:", variable)
                var_id = variable["variable"]
                # print("var_id:", var_id)
                # use parameter mapping to resolve variable:
                cur_var_map = cur_action_pddl_map[f'?{var_id}']
                # print("cur_var_map:", cur_var_map)
                match cur_var_map[0]:
                    # assign action arguments:
                    case 'arg1':
                        variable_map[var_id] = action_dict['arg1']
                    case 'arg2':
                        if 'arg2' in action_dict:
                            # print("arg2 in action_dict")
                            variable_map[var_id] = action_dict['arg2']
                        else:
                            # print("arg2 NOT in action_dict")
                            # check alternate mapping:
                            if len(cur_var_map) == 2:
                                # print("Alternate mapping:", cur_var_map[1])
                                if cur_var_map[1] == "arg1_receptacle":
                                    # print("variable_map:", variable_map)
                                    arg1_variable = None
                                    for assigned_variable, assigned_value in cur_action_pddl_map.items():
                                        # print("Checking", assigned_variable, assigned_value)
                                        # print("assigned_variable value:", cur_action_pddl_map[assigned_variable])
                                        if assigned_value[0] == "arg1":
                                            # print("arg1 variable:", assigned_variable)
                                            arg1_variable = assigned_variable[1:]
                                            # print("arg1_variable:", arg1_variable)
                                            break
                                    arg1_value = variable_map[arg1_variable]
                                    # print(arg1_value)
                                    arg1_receptacle = None
                                    for fact in self.world_state:
                                        if fact[0] in ["in", "on"]:
                                            if fact[1] == f"{arg1_value}1":  # assume only one instance of each type
                                                arg1_receptacle = fact[2]
                                                # print("arg1_receptacle:", arg1_receptacle)
                                                break
                                    variable_map[var_id] = arg1_receptacle
                            else:
                                variable_map[var_id] = None
                    # assign default wildcards:
                    case 'current_player_room':
                        variable_map[var_id] = self.get_player_room()
                    case 'player':
                        # for now only single-player, so the current player is always player1:
                        variable_map[var_id] = "player1"
                    case 'inventory':
                        # for now only single-player, so the current player inventory is always 'inventory':
                        variable_map[var_id] = "inventory"
                    case 'current_room_floor':
                        # floors are always room ID + floor1
                        variable_map[var_id] = self.get_player_room() + "floor1"

                # check type match:
                # assume all world state instance IDs end in numbers:

                # logger.info(variable_map)

                if variable_map[var_id]:
                    if variable_map[var_id].endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                        # logger.info(self.inst_to_type_dict)
                        # logger.info(f"{variable_map[var_id]} in self.inst_to_type_dict: {variable_map[var_id] in self.inst_to_type_dict}")
                        # logger.info(f"{variable_map[var_id]} in self.room_to_type_dict: {variable_map[var_id] in self.room_to_type_dict}")
                        if variable_map[var_id] in self.inst_to_type_dict:
                            var_type = self.inst_to_type_dict[variable_map[var_id]]
                        elif variable_map[var_id] in self.room_to_type_dict:
                            var_type = self.room_to_type_dict[variable_map[var_id]]
                    else:
                        # assume that other strings are essentially type strings:
                        var_type = variable_map[var_id]
                else:
                    var_type = variable_map[var_id]

                # print("var_type:", var_type)

                # DOMAIN TYPE CHECK
                type_matched = False
                if type(var_type) == str:
                    # NOTE: Inventory contents are handled via effects PDDL forall now.
                    if var_type in self.domain['supertypes']:
                        # print("domain supertypes for current var_type:", self.domain['supertypes'][var_type])
                        pass
                    # check if type matches directly:
                    if var_type == cur_parameter_type:
                        type_matched = True
                    # check if type matches through supertype:
                    elif var_type in self.domain['supertypes'] and cur_parameter_type in self.domain['supertypes'][
                        var_type]:
                        type_matched = True
                    # print("type matched:", type_matched)
                else:
                    # Fallback for edge cases
                    type_matched = True

                if not type_matched:
                    # get the index of the mismatched variable:
                    var_idx = list(cur_action_pddl_map.keys()).index(f"?{var_id}")
                    # get fail feedback template using mismatched variable index:
                    feedback_template = cur_action_def['failure_feedback']['parameters'][var_idx][0]
                    feedback_jinja = jinja2.Template(feedback_template)
                    # fill feedback template:
                    jinja_args = {var_id: variable_map[var_id]}
                    feedback_str = feedback_jinja.render(jinja_args)
                    feedback_str = feedback_str.capitalize()
                    # use action def feedback fail type:
                    fail_type = cur_action_def['failure_feedback']['parameters'][var_idx][1]

                    failed_action_info = {'failed_action_type': action_dict['type'],
                                          'failed_parameter': variable_map[var_id]}

                    fail_dict: dict = {'phase': "resolution", 'fail_type': fail_type, 'arg': failed_action_info}

                    type_match_fails.append((False, feedback_str, fail_dict))

                    # return False, feedback_str, fail_dict

        # variable map is filled during parameter checking
        # print("variable_map pre-preconditions:", variable_map)

        # PRECONDITION
        preconditions: list = cur_action_def['interaction']['precondition'][0]
        # print("preconditions/cur_action_def['interaction']['precondition'][0]:", preconditions)
        self.precon_idx = -1
        # self.precon_idx = 0
        self.precon_tuples = list()
        self.precon_trace = list()
        checked_conditions = self.check_conditions(preconditions, variable_map)
        # print("Main action checked_conditions:",checked_conditions)
        # print("Checked precon tuples:", self.precon_tuples)

        # if checked_conditions:
        if self.precon_trace[-1]['fulfilled']:
            logger.info("Preconditions fulfilled!")
            pass
        else:
            logger.info("Preconditions not fulfilled!")

            # NOTE: The first precondition fact that does not check out is used for feedback. This means that the order
            # of predicates (and clauses) in the precondition PDDL for the action determines feedback priority!

            logger.info(f"precon_trace: {self.precon_trace}")

            def feedback_idx_from_precon_trace(precon_trace):
                # iterate over precon trace:
                for item in precon_trace[-1]['and']:
                    # print("precon_trace item:", item)
                    # print("Checks out:", item['fulfilled'])
                    if not item['fulfilled']:
                        # print("Precon trace item does not check out:")
                        # print(item)
                        if 'or' in item:
                            # print("or clause:", item)
                            for or_item in item['or']:
                                # print("or_item:", or_item)
                                if 'and' in or_item:
                                    for and_item in or_item['and']:
                                        if not and_item['fulfilled']:
                                            # print("or and_item does not check out:", and_item)
                                            if 'not' in and_item:
                                                feedback_idx = and_item['not']['precon_idx']
                                                return feedback_idx, and_item
                                            feedback_idx = and_item['precon_idx']
                                            return feedback_idx, and_item
                                elif 'predicate_tuple' in or_item:
                                    if not or_item['fulfilled']:
                                        if 'not' in or_item:
                                            feedback_idx = or_item['not']['precon_idx']
                                            return feedback_idx, or_item
                                        feedback_idx = or_item['precon_idx']
                                        return feedback_idx, or_item
                        elif 'and' in item:
                            for and_item in item['and']:
                                if not and_item['fulfilled']:
                                    # print("or and_item does not check out:", and_item)
                                    if 'not' in and_item:
                                        feedback_idx = and_item['not']['precon_idx']
                                        return feedback_idx, and_item
                                    feedback_idx = and_item['precon_idx']
                                    return feedback_idx, and_item
                        elif 'predicate_tuple' in item:
                            if not item['fulfilled']:
                                feedback_idx = item['precon_idx']
                                return feedback_idx, item
                        elif 'not' in item:
                            if not item['fulfilled']:
                                feedback_idx = item['not']['precon_idx']
                                return feedback_idx, item

            # TODO?: Make feedback_idx extraction from precon_trace recursive for optimal robustness?

            feedback_idx, failed_precon_predicate = feedback_idx_from_precon_trace(self.precon_trace)
            logger.info(f"Precondition fail feedback_idx: {feedback_idx}")

            # get textual failure feedback template:
            feedback_template = cur_action_def['failure_feedback']['precondition'][feedback_idx][0]
            feedback_jinja = jinja2.Template(feedback_template)
            # fill feedback template:
            clean_feedback_variable_map = deepcopy(variable_map)
            logger.info(f"Precondition fail clean_feedback_variable_map: {clean_feedback_variable_map}")
            for key in clean_feedback_variable_map:
                # erroneous non-supported/non-contained TAKE hotfix:
                # if key == 's' and clean_feedback_variable_map[key] is None:
                if clean_feedback_variable_map[key] is None:
                    feedback_str = "You can't do that."
                    failed_action_info = {'failed_action_type': action_dict['type'],
                                          'failed_precon_predicate': "?s - receptacle"}
                    # use action def feedback fail type:
                    fail_dict: dict = {'phase': "resolution", 'fail_type': "take_non_takeable_object", 'arg': failed_action_info}
                    return False, feedback_str, fail_dict
                    # TODO: handle this more generically in parameter handling instead
                if clean_feedback_variable_map[key].endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                    clean_feedback_variable_map[key] = self._get_inst_str(clean_feedback_variable_map[key])
            jinja_args = clean_feedback_variable_map
            feedback_str = feedback_jinja.render(jinja_args)
            feedback_str = feedback_str.capitalize()
            # failed action information for records:
            failed_action_info = {'failed_action_type': action_dict['type'],
                                  'failed_precon_predicate': failed_precon_predicate}
            # use action def feedback fail type:
            fail_type = cur_action_def['failure_feedback']['precondition'][feedback_idx][1]
            fail_dict: dict = {'phase': "resolution", 'fail_type': fail_type, 'arg': failed_action_info}

            return False, feedback_str, fail_dict

        # if there were type match fails, return the first including feedback:
        if type_match_fails:
            return type_match_fails[0]

        # print("variable_map post-preconditions:", variable_map)

        # EFFECT

        effects: list = cur_action_def['interaction']['effect']
        if 'and' in effects[0]:  # handle multi-predicate effect, but allow non-and single predicate effect
            effects: list = cur_action_def['interaction']['effect'][0]['and']
        # print("effects:", effects)

        world_state_effects = {'added': [], 'removed': []}

        for effect in effects:
            # print("effect:", effect)
            if 'forall' in effect:
                forall_results = self.resolve_forall(effect, variable_map)
                world_state_effects['added'] += forall_results['added']
                world_state_effects['removed'] += forall_results['removed']
            elif 'when' in effect:
                when_results = self.resolve_when(effect, variable_map)
                world_state_effects['added'] += when_results['added']
                world_state_effects['removed'] += when_results['removed']
            else:
                resolve_effect_results = self.resolve_effect(effect, variable_map)
                world_state_effects['added'] += resolve_effect_results['added']
                world_state_effects['removed'] += resolve_effect_results['removed']

        # print("world_state_effects:", world_state_effects)

        # print("World state after effects:", self.world_state)

        # add deepcopy of new current world state to world state history:
        self.world_state_history.append(deepcopy(self.world_state))

        # get all changed facts:
        post_world_state = deepcopy(self.world_state)
        post_resolution_changes = post_world_state.difference(prior_world_state)
        if prior_world_state == self.world_state_history[-2]:
            logger.info(f"Prior world state matches second to last world state in history")
        logger.info(f"Resolution world state changes: {post_resolution_changes}")


        # SUCCESS FEEDBACK

        # type word variable map instead of instance ID:
        clean_feedback_variable_map = deepcopy(variable_map)
        for key in clean_feedback_variable_map:
            if clean_feedback_variable_map[key]:
                if clean_feedback_variable_map[key].endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                    clean_feedback_variable_map[key] = self._get_inst_str(clean_feedback_variable_map[key])

        success_feedback_template = cur_action_def['success_feedback']
        # print("success_feedback_template:", success_feedback_template)

        feedback_jinja = jinja2.Template(success_feedback_template)

        # jinja_args: dict = {}
        jinja_args: dict = clean_feedback_variable_map
        if "room_desc" in success_feedback_template:
            jinja_args["room_desc"] = self.get_full_room_desc()
        if "inventory_desc" in success_feedback_template:
            jinja_args["inventory_desc"] = self.get_inventory_desc()
        if "prep" in success_feedback_template:
            if "prep" in action_dict:
                jinja_args["prep"] = action_dict['prep']
            else:
                # get preposition fact from world state effects:
                for added_fact in world_state_effects['added']:
                    if added_fact[0] in ['in', 'on']:
                        jinja_args["prep"] = added_fact[0]
                        break
        if "container_content" in success_feedback_template:
            # get opened container fact from world state effects:
            for added_fact in world_state_effects['added']:
                if added_fact[0] == "open":
                    opened_container_id = added_fact[1]
                    break
            jinja_args["container_content"] = self.get_container_content_desc(opened_container_id)
        if "arg1_desc" in success_feedback_template:
            # get description of arg1 entity:
            entity_desc = self.get_entity_desc(action_dict['arg1'])
            # print()
            jinja_args["arg1_desc"] = entity_desc
        if "arg2_desc" in success_feedback_template:
            # get description of arg2 entity:
            entity_desc = self.get_entity_desc(action_dict['arg2'])
            # print()
            jinja_args["arg2_desc"] = entity_desc

        if "arg1_text" in success_feedback_template:
            # get text of arg1 entity:
            entity_text = self.get_entity_text(action_dict['arg1'])
            # print()
            jinja_args["arg1_text"] = entity_text

        feedback_str = feedback_jinja.render(jinja_args)
        # feedback_str = feedback_str.capitalize()

        # print("feedback_str:", feedback_str)

        return True, feedback_str, {'world_state_effects': world_state_effects}

    def run_events(self):
        # deepcopy the world state to prevent referential interaction:
        prior_world_state = deepcopy(self.world_state)

        # iterate over defined events:
        for cur_event_type in self.event_types:
            cur_event_def = self.event_types[cur_event_type]
            # print("cur_event_def:", cur_event_def)
            # logger.info(f"cur_event_type: {cur_event_type}")
            # print(f"cur_event_type: {cur_event_type}")

            # PARAMETERS
            variable_map = dict()
            parameters_base = cur_event_def['interaction']['parameters']
            # print("parameters_base:", parameters_base)
            # check that parameters key correctly contains a PDDL type_list:
            if not 'type_list' in parameters_base:
                raise KeyError

            # get parameters list:
            parameters = parameters_base['type_list']
            # print("parameters:", parameters)
            # logger.info(f"parameters: {parameters}")
            cur_var_type_map: dict = dict()
            for parameter in parameters:
                # print("\nparameter:", parameter)
                cur_parameter_type = parameter['type_list_element']
                # print("cur_parameter_type:", cur_parameter_type)

                # go over variables in parameter:
                for variable in parameter['items']:
                    # print("variable:", variable)
                    # logger.info(f"variable: {variable}")
                    var_id = variable["variable"]
                    # print("var_id:", var_id)

                    # use parameter mapping to resolve variable:
                    # cur_var_map = cur_event_pddl_map[f'?{var_id}']
                    # print("cur_var_map:", cur_var_map)
                    if var_id not in cur_var_type_map:
                        # print(f"{var_id} not in cur_var_type_map")
                        cur_var_type_map[var_id] = [cur_parameter_type]
                    else:
                        cur_var_type_map[var_id].append(cur_parameter_type)

            # print("cur_var_type_map:", cur_var_type_map)
            # get all type-matching entities:
            cur_var_candidates: dict = dict()
            for key, value in cur_var_type_map.items():
                # print(key, value)
                candidate_types: list = list()
                for cur_type in value:
                    # print(self.domain['types'])
                    # print(type(self.domain['types']))
                    # get domain supertypes:
                    if cur_type in self.domain['types']:
                        # value is domain supertype
                        candidate_types += self.domain['types'][cur_type]
                    else:
                        candidate_types.append(cur_type)
                cur_var_candidates[key] = [type_fact[1] for type_fact in self.world_state
                                           if type_fact[0] in ["type", "room"] and type_fact[2] in candidate_types]

            # print("cur_var_candidates:", cur_var_candidates)
            candidates_lists = list(cur_var_candidates.values())
            # logger.info(f"candidates_lists before cleanup: {candidates_lists}")
            for candidates_list_idx, candidates_list in enumerate(candidates_lists):
                if not candidates_list:
                    candidates_lists.pop(candidates_list_idx)
            # logger.info(f"candidates_lists after cleanup: {candidates_lists}")

            # print(candidates_lists)
            candidate_combos = list(itertools.product(*candidates_lists))
            # print("candidate_combos:", candidate_combos)
            # logger.info(f"candidate_combos: {candidate_combos}")
            # iterate over candidate combos:
            for candidate_combo in candidate_combos:
                variable_map = deepcopy(cur_var_type_map)
                key_iterator = 0
                for value in candidate_combo:
                    variable_map[list(variable_map.keys())[key_iterator]] = value
                    key_iterator += 1
                # print("variable_map:", variable_map)
                # check event precondition:
                # PRECONDITION
                preconditions: list = cur_event_def['interaction']['precondition'][0]
                # if cur_event_type in ["outhouse_teleport"]:
                #     logger.info(f"Event preconditions before check_preconditions: {preconditions}")
                # print("preconditions/cur_action_def['interaction']['precondition'][0]:", preconditions)
                self.precon_idx = -1
                # self.precon_idx = 0
                self.precon_tuples = list()
                self.precon_trace = list()
                checked_conditions = self.check_conditions(preconditions, variable_map)
                # print("Event checked_conditions:",checked_conditions)
                # if cur_event_type in ["outhouse_teleport"]:
                #    logger.info(f"Event checked_conditions: {checked_conditions}")
                # print("Checked precon tuples:", self.precon_tuples)
                # if cur_event_type in ["outhouse_teleport"]:
                #    logger.info(f"Checked precon tuples: {self.precon_tuples}")

                # if checked_conditions:
                if self.precon_trace[-1]['fulfilled']:
                    # if cur_event_type not in ["outhouse_teleport", "workshop_antigravity_objects",
                    #                          "workshop_antigravity_player_float_start",
                    #                          "workshop_antigravity_player_float_stop"]:
                    #     logger.info(f"{cur_event_type}: Event preconditions fulfilled!")
                    # print("Event preconditions fulfilled!")
                    # logger.info(f"{cur_event_type}: Event preconditions fulfilled!")

                    # EFFECT
                    # IMPORTANT: Events MUST change their precondition to not be true (in the same way) after they have
                    # been triggered!
                    effects: list = cur_event_def['interaction']['effect']
                    if 'and' in effects[0]:  # handle multi-predicate effect, but allow non-and single predicate effect
                        effects: list = cur_event_def['interaction']['effect'][0]['and']
                    # print("effects:", effects)

                    world_state_effects = {'added': [], 'removed': []}

                    for effect in effects:
                        # if cur_event_type in ["outhouse_teleport"]:
                        #    logger.info(f"{cur_event_type} event effect: {effect}")
                        # print("effect:", effect)
                        if 'forall' in effect:
                            forall_results = self.resolve_forall(effect, variable_map)
                            world_state_effects['added'] += forall_results['added']
                            world_state_effects['removed'] += forall_results['removed']
                        elif 'when' in effect:
                            when_results = self.resolve_when(effect, variable_map)
                            world_state_effects['added'] += when_results['added']
                            world_state_effects['removed'] += when_results['removed']
                        else:
                            resolve_effect_results = self.resolve_effect(effect, variable_map)
                            world_state_effects['added'] += resolve_effect_results['added']
                            world_state_effects['removed'] += resolve_effect_results['removed']

                    # print("world_state_effects:", world_state_effects)

                    # print("World state after effects:", self.world_state)

                    # add deepcopy of new current world state to world state history:
                    self.world_state_history.append(deepcopy(self.world_state))

                    # TODO?: handle world state history properly; only needed for planning, ie not for current work

                    # get all changed facts:
                    post_world_state = deepcopy(self.world_state)
                    post_resolution_changes = post_world_state.difference(prior_world_state)
                    if prior_world_state == self.world_state_history[-2]:
                        logger.info(f"Prior world state matches second to last world state in history")
                    logger.info(f"Event world state changes: {post_resolution_changes}")

                    # EVENT FEEDBACK

                    # type word variable map instead of instance ID:
                    clean_feedback_variable_map = deepcopy(variable_map)
                    for key in clean_feedback_variable_map:
                        if clean_feedback_variable_map[key]:
                            if clean_feedback_variable_map[key].endswith(
                                    ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                                clean_feedback_variable_map[key] = self._get_inst_str(clean_feedback_variable_map[key])

                    event_feedback_template = cur_event_def['event_feedback']
                    # print("event_feedback_template:", event_feedback_template)

                    feedback_jinja = jinja2.Template(event_feedback_template)

                    # jinja_args: dict = {}
                    jinja_args: dict = clean_feedback_variable_map
                    if "room_desc" in event_feedback_template:
                        jinja_args["room_desc"] = self.get_full_room_desc()
                    if "inventory_desc" in event_feedback_template:
                        jinja_args["inventory_desc"] = self.get_inventory_desc()
                    if "prep" in event_feedback_template:
                        # get preposition fact from world state effects:
                        for added_fact in world_state_effects['added']:
                            if added_fact[0] in ['in', 'on']:
                                jinja_args["prep"] = added_fact[0]
                                break
                    if "container_content" in event_feedback_template:
                        # get opened container fact from world state effects:
                        for added_fact in world_state_effects['added']:
                            if added_fact[0] == "open":
                                opened_container_id = added_fact[1]
                                break
                        jinja_args["container_content"] = self.get_container_content_desc(opened_container_id)

                    feedback_str = feedback_jinja.render(jinja_args)
                    # feedback_str = feedback_str.capitalize()

                    # print("feedback_str:", feedback_str)

                    # randomize event effect after first trigger:
                    # currently tailor-made for potion brewing adventures teleporting outhouse
                    # would need to be made a lot more extensive and properly recursive to handle any possible event
                    if 'randomize' in cur_event_def:
                        if not hasattr(self, 'event_randomization'):
                            self.event_randomization: dict = dict()
                        if not cur_event_type in self.event_randomization:
                            # set predefined initial value as prior value to replace next time:
                            self.event_randomization[cur_event_type] = cur_event_def['randomize']['initial_value']
                        if 'replace_type' in cur_event_def['randomize']:
                            replace_candidates: list = list()
                            for fact in self.world_state:
                                if fact[0] == cur_event_def['randomize']['replace_type']:
                                    if fact[1] not in cur_event_def['randomize']['not_replacer']:
                                        replace_candidates.append(fact[1])
                            random_replacement = self.rng.choice(replace_candidates)
                            # replace prior value with random replacement in effects:
                            effects: list = cur_event_def['interaction']['effect']
                            if 'and' in effects[0]:
                                # handle multi-predicate effect, but allow non-and single predicate effect
                                effects: list = cur_event_def['interaction']['effect'][0]['and']
                            # print("effects:", effects)
                            for effect in effects:
                                # print("effect:", effect)
                                if 'predicate' in effect:
                                    if type(effect['arg1']) == str:
                                        if effect['arg1'] == self.event_randomization[cur_event_type]:
                                            effect['arg1'] = random_replacement
                                    if type(effect['arg2']) == str:
                                        if effect['arg2'] == self.event_randomization[cur_event_type]:
                                            effect['arg2'] = random_replacement
                                    if type(effect['arg3']) == str:
                                        if effect['arg3'] == self.event_randomization[cur_event_type]:
                                            effect['arg3'] = random_replacement
                                if 'forall' in effect:
                                    forall_body = effect['body']
                                    # logger.info(f"forall_body: {forall_body}")
                                    for forall_effect in forall_body:
                                        if 'when' in forall_effect:
                                            # logger.info(f"when forall_effect: {forall_effect}")
                                            # when_effects = forall_effect[1]
                                            when_effects = forall_effect['when']
                                            for when_effect in when_effects:
                                                if 'predicate' in when_effect:
                                                    if type(when_effect['arg1']) == str:
                                                        if when_effect['arg1'] == self.event_randomization[cur_event_type]:
                                                            when_effect['arg1'] = random_replacement
                                                    if type(when_effect['arg2']) == str:
                                                        if when_effect['arg2'] == self.event_randomization[cur_event_type]:
                                                            when_effect['arg2'] = random_replacement
                                                    if type(when_effect['arg3']) == str:
                                                        if when_effect['arg3'] == self.event_randomization[cur_event_type]:
                                                            when_effect['arg3'] = random_replacement
                            # store this randomization's random value for the next time to replace:
                            self.event_randomization[cur_event_type] = random_replacement

                    return True, feedback_str, {'world_state_effects': world_state_effects}
                else:
                    # if cur_event_type in ["outhouse_teleport"]:
                    #    logger.info(f"{cur_event_type}: Event preconditions not fulfilled.")
                    # logger.info(f"{cur_event_type}: Event preconditions not fulfilled!")
                    # if cur_event_type not in ["outhouse_teleport", "workshop_antigravity_objects",
                    #                          "workshop_antigravity_player_float_start",
                    #                          "workshop_antigravity_player_float_stop"]:
                    #     logger.info(f"{cur_event_type}: Event preconditions not fulfilled!")
                    pass

        # since no event triggered, return[0] = False:
        return False, "", {}

    def get_exploration_info(self, action_type = None, full_exploration_state = False, full_exploration_history = False):
        exploration_info = dict()

        if full_exploration_state:
            exploration_info['exploration_state'] = list(self.exploration_state)
        if full_exploration_history:
            exploration_info['exploration_history'] = list(self.exploration_history)

        # get epistemic/pragmatic info for action:
        if action_type:
            # logger.info(f"action_type: {action_type}")
            # logger.info(f"self.action_types[action_type]['epistemic']: {self.action_types[action_type]['epistemic']}")
            exploration_info['action_epistemic'] = self.action_types[action_type]['epistemic']
            exploration_info['action_pragmatic'] = self.action_types[action_type]['pragmatic']
        else:
            exploration_info['action_epistemic'] = False
            exploration_info['action_pragmatic'] = False

        epistemic_gain_removed = self.exploration_history[-2].difference(self.exploration_state)
        epistemic_gain_added = self.exploration_state.difference(self.exploration_history[-2])
        logger.info(f"Epistemic gain; Added: {epistemic_gain_added}; Removed: {epistemic_gain_removed}")

        if exploration_info['action_epistemic']:
            effective_epistemic_gain = epistemic_gain_added
            effective_epistemic_gain_amount = len(effective_epistemic_gain)
            if action_type:
                logger.info(f"Epistemic action '{action_type}' resulted in effective epistemic gain: {effective_epistemic_gain}")
            exploration_info['effective_epistemic_gain_facts'] = list(effective_epistemic_gain)
            logger.info(f"Epistemic gain amount: {effective_epistemic_gain_amount}")
            exploration_info['effective_epistemic_gain_amount'] = effective_epistemic_gain_amount
        else:
            exploration_info['effective_epistemic_gain_amount'] = 0

        # all entities:
        all_entities = set()
        for fact in self.world_state:
            if fact[0] == 'type':
                all_entities.add(fact[1])

        # known entities:
        known_entities = set()
        for fact in self.exploration_state:
            if fact[0] == "at":
                known_entities.add(fact)
        logger.info(f"Known entities: {known_entities}")
        exploration_info['known_entities'] = list(known_entities)

        known_entities_ratio = len(known_entities) / len(all_entities)
        logger.info(f"Known entities ratio: {known_entities_ratio}")
        exploration_info['known_entities_ratio'] = known_entities_ratio

        # all rooms:
        all_rooms = set()
        for fact in self.world_state:
            if fact[0] == 'room':
                all_rooms.add(fact[1])

        # visited rooms:
        visited_rooms = set()
        for exploration_state in self.exploration_history:
            for exploration_fact in exploration_state:
                if exploration_fact[0] == 'at' and exploration_fact[1] == 'player1':
                    visited_rooms.add(exploration_fact[2])
        logger.info(f"Visited rooms: {visited_rooms}")
        exploration_info['visited_rooms'] = list(visited_rooms)

        visited_rooms_ratio = len(visited_rooms) / len(all_rooms)
        logger.info(f"Visited rooms ratio: {visited_rooms_ratio}")
        exploration_info['visited_rooms_ratio'] = visited_rooms_ratio

        # get goal entitiy set:
        goal_entities = set()
        # TODO: expand to handle new-words goal tuples/any goal tuples
        for goal_fact in self.goal_state:
            goal_entities.add(goal_fact[1])
            if len(goal_fact) >= 3:
                goal_entities.add(goal_fact[2])
        logger.info(f"Goal entities: {goal_entities}")

        # check which goal-relevant entities are known:
        known_goal_entities = set()
        for goal_fact in self.goal_state:
            for known_entity in known_entities:
                if known_entity[1] in goal_fact:
                    known_goal_entities.add(known_entity)
        logger.info(f"Known goal entities: {known_goal_entities}")
        exploration_info['known_goal_entities'] = list(known_goal_entities)

        # ratio of known goal-relevant entities:
        known_goal_entities_ratio = len(known_goal_entities) / len(goal_entities)
        logger.info(f"Known goal entities ratio: {known_goal_entities_ratio}")
        exploration_info['known_goal_entities_ratio'] = known_goal_entities_ratio

        return exploration_info

    def process_action(self, action_input: str):
        """
        Fully process an action input.
        First parses the input, then resolves the parsed action, and returns feedback and goal achievement.
        """
        # logger.info(f"Pre-process_action world state:\n{self.world_state}")
        # logger.info(f"itemcount 0 in world state pre-process_action: {('itemcount', 'inventory', 0) in self.world_state}")
        # PARSING PHASE
        parsed, parse_result, fail = self.parse_action_input(action_input)
        if not parsed:
            self.track_exploration()
            fail['exploration_info'] = self.get_exploration_info()
            # logger.info(f"Post-process_action world state:\n{self.world_state}")
            # logger.info(f"itemcount 0 in world state post-process_action: {('itemcount', 'inventory', 0) in self.world_state}")

            events_feedback: list = list()
            events_world_state_changes: list = list()
            event_triggered, event_feedback, event_world_state_changes = self.run_events()
            while event_triggered:
                events_feedback.append(event_feedback)
                events_world_state_changes.append(event_world_state_changes)
                event_triggered, event_feedback, event_world_state_changes = self.run_events()

            results_feedback = parse_result + "\n" + "\n".join(events_feedback)

            return self.goals_achieved, results_feedback, fail
        else:
            # RESOLUTION PHASE
            # get visible/accessible entities before resolution:
            prior_visibles = set(self.get_player_room_contents_visible())

            # resolve action:
            resolved, resolution_result, fail = self.resolve_action(parse_result)
            if not resolved:
                self.track_exploration()
                 # get exploration info and add to fail dict:
                fail['exploration_info'] = self.get_exploration_info(parse_result['type'])
                # logger.info(f"Post-process_action world state:\n{self.world_state}")
                # logger.info(f"itemcount 0 in world state post-process_action: {('itemcount', 'inventory', 0) in self.world_state}")

                events_feedback: list = list()
                events_world_state_changes: list = list()
                event_triggered, event_feedback, event_world_state_changes = self.run_events()
                while event_triggered:
                    events_feedback.append(event_feedback)
                    events_world_state_changes.append(event_world_state_changes)
                    event_triggered, event_feedback, event_world_state_changes = self.run_events()

                results_feedback = resolution_result + "\n" + "\n".join(events_feedback)

                return self.goals_achieved, results_feedback, fail
            else:
                logger.info(f"Resolution result: {resolution_result}")
                base_result_str = resolution_result

                # print("world state effects:", fail['world_state_effects'])

                # check goal achievement:
                self.goals_achieved = self.goal_state & self.world_state
                goals_achieved_response = list(self.goal_state & self.world_state)
                # convert to goal states to string version:
                for goal_state_idx, goal_state in enumerate(goals_achieved_response):
                    goals_achieved_response[goal_state_idx] = fact_tuple_to_str(goal_state)
                goals_achieved_response = set(goals_achieved_response)
                logger.info(f"Achieved goal states: {goals_achieved_response}")

                # EXPLORATION TRACKING
                self.track_exploration(fail['world_state_effects'])

                # successful action returns extra information instead of failure information:
                extra_action_info = dict()
                extra_action_info['action_type'] = parse_result['type']
                # exploration info:
                extra_action_info['exploration_info'] = self.get_exploration_info(parse_result['type'])

                # handle DONE action:
                if parse_result['type'] == "done":
                    extra_action_info['done_action'] = True
                # logger.info(f"Post-process_action world state:\n{self.world_state}")
                # logger.info(f"itemcount 0 in world state post-process_action: {('itemcount', 'inventory', 0) in self.world_state}")

                events_feedback: list = list()
                events_world_state_changes: list = list()
                event_triggered, event_feedback, event_world_state_changes = self.run_events()
                if event_world_state_changes:
                    # print("event_world_state_changes:", event_world_state_changes)
                    pass
                while event_triggered:
                    events_feedback.append(event_feedback)
                    events_world_state_changes.append(event_world_state_changes)
                    event_triggered, event_feedback, event_world_state_changes = self.run_events()
                    if event_world_state_changes:
                        # print("event_world_state_changes:", event_world_state_changes)
                        pass

                results_feedback = base_result_str + "\n" + "\n".join(events_feedback)

                # TODO?: only give feedback for events happening in current player room? -> shouldn't be necessary for now

                return goals_achieved_response, results_feedback, extra_action_info

    def execute_optimal_solution(self):
        """
        Run through the game_instance's optimal solution.
        Used to verify parity of IF interpreter and solution generation.
        """
        print(self.get_full_room_desc())
        for command in self.game_instance["optimal_commands"]:
            print(f"> {command}")
            goals_achieved, response, fail = self.process_action(command)
            print(response)
            print("Goals achieved:", goals_achieved)
            print("Fail:", fail)
            print()

    def execute_plan_sequence(self, command_sequence: list) -> List:
        """
        Execute a command sequence plan and return results up to first failure.
        Used for plan logging and evaluation.
        Returns a list of action processing results including first failed plan action.
        """
        logger.info(f"Plan command sequence: {command_sequence}")
        # deepcopy world state before plan execution to assure proper reversion:
        pre_plan_world_state = deepcopy(self.world_state)
        pre_plan_exploration_state = deepcopy(self.exploration_state)

        result_sequence: list = list()
        world_state_change_count: int = 0
        for cmd_idx, command in enumerate(command_sequence):
            logger.info(f"Resolving plan action {cmd_idx}: {command}")
            # get result as list for mutability:
            result = list(self.process_action(command))
            # convert result goals achieved to list for JSON dumping:
            result[0] = list(result[0])
            result_sequence.append(result)
            # check for command failure:
            # result[2] is fail info; if it is truthy, the command failed
            # if result[2]:
            if 'fail_type' in result[2]:
                # stop executing commands at the first failure
                logger.info(f"Plan sequence failed at step {cmd_idx}")
                logger.info(f"Plan sequence fail dict: {result[2]}")
                logger.info(f"Plan world state change count at failure: {world_state_change_count}")
                break
            else:
                world_state_change_count += 1
                logger.info(f"New plan world state change count: {world_state_change_count}")

        # revert the world state to before plan execution if it changed:
        if world_state_change_count:
            logger.info(f"Plan world state change count: {world_state_change_count}; reverting changes")
            # deepcopy world state after plan execution to prevent reference issues:
            post_plan_world_state = deepcopy(self.world_state)
            post_plan_exploration_state = deepcopy(self.exploration_state)
            # logger.info(f"World state history before reverting: {self.world_state_history}")
            logger.info(f"World state history length before reverting: {len(self.world_state_history)}")
            logger.info(f"Exploration history length before reverting: {len(self.exploration_history)}")
            # reset world state history to before executed plan:
            self.world_state_history = self.world_state_history[:-world_state_change_count]
            self.exploration_history = self.exploration_history[:-world_state_change_count]
            # logger.info(f"World state history after reverting: {self.world_state_history}")
            logger.info(f"World state history length after reverting: {len(self.world_state_history)}")
            logger.info(f"Exploration history length after reverting: {len(self.exploration_history)}")
            # check that world state has been properly reset:
            if self.world_state_history[-1] == pre_plan_world_state:
                logger.info(f"Last world state history item matches pre-plan world state")
            else:
                logger.info(f"Last world state history item DOES NOT match pre-plan world state")
            if self.world_state_history[-1] == post_plan_world_state:
                logger.info(f"Last world state history item DOES match post-plan world state")
            else:
                logger.info(f"Last world state history item does not match post-plan world state")
            # reset world state to before plan execution:
            self.world_state = deepcopy(self.world_state_history[-1])
            self.exploration_state = deepcopy(self.exploration_history[-1])
            # double-check that world state has been reset properly:
            if self.world_state == pre_plan_world_state:
                logger.info(f"Pre-plan world state matches reverted post-plan world state")
            else:
                logger.info(f"Pre-plan world state does not match reverted post-plan world state")
            # log specific reverted fact changes from plan:
            post_plan_changes = post_plan_world_state.difference(self.world_state)
            logger.info(f"Reverted plan world state changes: {post_plan_changes}")
        else:
            logger.info(f"Plan world state change count: {world_state_change_count}; no changes to revert")

        logger.info(f"Plan result sequence: {result_sequence}")

        return result_sequence


if __name__ == "__main__":
    PATH = ""
    # example game instance:
    """
    game_instance_exmpl = {
          "game_id": 0,
          "variant": "basic",
          "prompt": "You are playing a text adventure game. I will describe what you can perceive in the game. You write the single action you want to take in the game starting with >. Only reply with an action.\nFor example:\n> open cupboard\n\nIn addition to common actions, you can ortid, enerk and alism.\n\nYour goal for this game is: Put the delly in the thot, the plate on the micon and the penol on the table.\n\nOnce you have achieved your goal, write \"> done\" to end the game.\n\n",
          "initial_state": [
            "at(kitchen1floor1,kitchen1)",
            "at(table1,kitchen1)",
            "at(player1,kitchen1)",
            "at(toad1,kitchen1)",
            "at(greenbeetle1,kitchen1)",
            "at(cauldron1,kitchen1)",

            "at(plasmbucket1,kitchen1)",
            "at(waterbucket1,kitchen1)",
            "at(nightshade1,kitchen1)",
            "at(plinklecrystal1,kitchen1)",
            "at(spideregg1,kitchen1)",
            "at(icewand1,kitchen1)",
            "at(firewand1,kitchen1)",
            "at(fairywand1,kitchen1)",
            "at(zulpowand1,kitchen1)",
            "at(ladle1,kitchen1)",
            "at(potionrecipe1,kitchen1)",

            "type(kitchen1floor1,floor)",
            "type(player1,player)",
            "type(table1,table)",
            "type(toad1,toad)",
            "type(greenbeetle1,greenbeetle)",
            "type(cauldron1,cauldron)",

            "type(plasmbucket1,plasmbucket)",
            "type(waterbucket1,waterbucket)",
            "type(nightshade1,nightshade)",
            "type(plinklecrystal1,plinklecrystal)",
            "type(spideregg1,spideregg)",
            "type(icewand1,icewand)",
            "type(firewand1,firewand)",
            "type(fairywand1,fairywand)",
            "type(zulpowand1,zulpowand)",
            "type(ladle1,ladle)",
            "type(potionrecipe1,potionrecipe)",

            # "readable(potionrecipe1)",
            "text(potionrecipe1,This is a potion recipe.\nIt has multiple lines!\nWith all the ingredients\nlike beetles, conkies, woppahs, etc)",

            "room(kitchen1,kitchen)",
            "support(kitchen1floor1)",
            "support(table1)",
            "on(toad1,kitchen1floor1)",
            "on(greenbeetle1,table1)",

            "on(plasmbucket1,kitchen1floor1)",
            "on(waterbucket1,kitchen1floor1)",
            "on(nightshade1,table1)",
            "on(plinklecrystal1,table1)",
            "on(spideregg1,table1)",
            "on(icewand1,table1)",
            "on(firewand1,table1)",
            "on(fairywand1,table1)",
            "on(zulpowand1,table1)",
            "on(ladle1,table1)",
            "on(potionrecipe1,table1)",

            "receptacle(table1)",
            "takeable(toad1)",
            "takeable(greenbeetle1)",
            "movable(toad1)",
            "movable(greenbeetle1)",
            "needs_support(toad1)",
            "needs_support(greenbeetle1)",
            "hungry(toad1)"
          ],
          "goal_state": [
            "in(delly1,thot1)",
            "on(plate1,micon1)",
            "on(penol1,table1)"
          ],
          "max_turns": 50,
          "optimal_turns": 14,
          "optimal_solution": [
            [
              "take",
              "delly1"
            ],
            [
              "go",
              "het1"
            ],
            [
              "go",
              "broomcloset1"
            ],
            [
              "take",
              "penol1"
            ],
            [
              "go",
              "het1"
            ],
            [
              "go",
              "kitchen1"
            ],
            [
              "take",
              "plate1"
            ],
            [
              "open",
              "thot1"
            ],
            [
              "go",
              "ousnee1"
            ],
            [
              "put",
              "plate1",
              "micon1"
            ],
            [
              "go",
              "kitchen1"
            ],
            [
              "put",
              "delly1",
              "thot1"
            ],
            [
              "go",
              "ousnee1"
            ],
            [
              "put",
              "penol1",
              "table1"
            ]
          ],
          "optimal_commands": [
            "read potion recipe",
            # "dump bucket of ectoplasm into cauldron",
            # "wave ice wand at cauldron",
            # "put nightshade in cauldron",
            # "put spider egg in cauldron",
            # "stir liquid with ladle",
            # "put plinkle crystal in cauldron",
            # "look around"
          ],
          "action_definitions": ["witch_actions_core.json"],
          "room_definitions": ["witch_rooms.json"],
          "entity_definitions": ["witch_entities.json"],
          "domain_definitions": ["witch_domain_core.json"],
          "event_definitions": ["witch_events_core.json"]
        }
    """
    game_instance_exmpl = {
          "game_id": 2,
          "variant": "basic",
          "prompt": "You are playing a text adventure game. I will describe what you can perceive in the game. You write the single action you want to take in the game starting with >. Only reply with an action.\nFor example:\n> open cupboard\n\nYou can have up to two objects in your inventory at the same time.\n\nYour goal for this game is to brew a potion following the potion recipe.\n\nOnce you have achieved your goal, write \"> done\" to end the game.\n\n",
          "initial_state": [
		    "type(crate1,crate)",
            "at(crate1,storage1)",
            "support(crate1)",
		    "type(workbench1,workbench)",
            "at(workbench1,workshop1)",
            "support(workbench1)",
			"type(gardenbench1,gardenbench)",
            "at(gardenbench1,garden1)",
            "support(gardenbench1)",
			"type(plantingtable1,plantingtable)",
            "at(plantingtable1,conservatory1)",
            "support(plantingtable1)",
			"type(utilityshelf1,utilityshelf)",
            "at(utilityshelf1,broomcloset1)",
            "support(utilityshelf1)",
			"type(ingredientrack1,ingredientrack)",
            "at(ingredientrack1,pantry1)",
            "support(ingredientrack1)",
			"type(orrery1,orrery)",
            "at(orrery1,workshop1)",
            "support(orrery1)",
            "type(kitchen1floor1,floor)",
            "type(pantry1floor1,floor)",
            "type(hallway1floor1,floor)",
            "type(broomcloset1floor1,floor)",
            "type(bedroom1floor1,floor)",
            "type(storage1floor1,floor)",
            "type(readingroom1floor1,floor)",
            "type(workshop1floor1,floor)",
            "type(workshop1ceiling1,ceiling)",
            "type(cellar1floor1,floor)",
            "type(conservatory1floor1,floor)",
            "type(garden1floor1,floor)",
            "type(outhouse1floor1,floor)",
            "type(player1,player)",
            "type(table1,table)",
            "type(sidetable1,sidetable)",
            "type(counter1,counter)",
            "type(icebox1,icebox)",
            "type(cupboard1,cupboard)",
            "type(wardrobe1,wardrobe)",
            "type(shelf1,shelf)",
            "type(chair1,chair)",
            "type(bed1,bed)",
            "type(couch1,couch)",
            "type(broom1,broom)",
            "type(mop1,mop)",
            "type(sandwich1,sandwich)",
            "type(apple1,apple)",
            "type(banana1,banana)",
            "type(orange1,orange)",
            "type(peach1,peach)",
            "type(plate1,plate)",
            "type(book1,book)",
            "type(pillow1,pillow)",
            "type(toad1,toad)",
            "type(greenbeetle1,greenbeetle)",
            "type(redbeetle1,redbeetle)",
            "type(eyeofnewt1,eyeofnewt)",
            "type(plinklecrystal1,plinklecrystal)",
            "type(plonklecrystal1,plonklecrystal)",
            "type(plunklecrystal1,plunklecrystal)",
            "type(spider1,spider)",
            "type(spiderweb1,spiderweb)",
            "type(spideregg1,spideregg)",
            "type(lilyofthevalley1,lilyofthevalley)",
            "type(nightshade1,nightshade)",
            "type(pricklypear1,pricklypear)",
            "type(waterbucket1,waterbucket)",
            "type(plasmbucket1,plasmbucket)",
            "type(cauldron1,cauldron)",
            "needs_support(cauldron1)",
            "on(cauldron1,kitchen1floor1)",
            "type(ladle1,ladle)",
            "type(whisk1,whisk)",
            "type(spoon1,spoon)",
            "type(borpulus1,borpulus)",
            "type(firewand1,firewand)",
            "type(icewand1,icewand)",
            "type(fairywand1,fairywand)",
            "type(zulpowand1,zulpowand)",
            "type(liquid1,liquid)",
            "type(potion1,potion)",
            "type(potionrecipe1,potionrecipe)",
            "room(kitchen1,kitchen)",
            "room(pantry1,pantry)",
            "room(hallway1,hallway)",
            "room(broomcloset1,broomcloset)",
            "room(bedroom1,bedroom)",
            "room(storage1,storage)",
            "room(readingroom1,readingroom)",
            "room(workshop1,workshop)",
            "room(cellar1,cellar)",
            "room(conservatory1,conservatory)",
            "room(garden1,garden)",
            "room(outhouse1,outhouse)",
            "at(kitchen1floor1,kitchen1)",
            "at(pantry1floor1,pantry1)",
            "at(hallway1floor1,hallway1)",
            "at(broomcloset1floor1,broomcloset1)",
            "at(bedroom1floor1,bedroom1)",
            "at(storage1floor1,storage1)",
            "at(readingroom1floor1,readingroom1)",
            "at(workshop1floor1,workshop1)",
            "at(workshop1ceiling1,workshop1)",
            "at(cellar1floor1,cellar1)",
            "at(conservatory1floor1,conservatory1)",
            "at(garden1floor1,garden1)",
            "at(outhouse1floor1,outhouse1)",
            "at(player1,bedroom1)",
            "at(potionrecipe1,bedroom1)",
            "at(table1,readingroom1)",
            "at(sidetable1,readingroom1)",
            "at(counter1,kitchen1)",
            "at(icebox1,pantry1)",
            "at(cupboard1,kitchen1)",
            "at(wardrobe1,bedroom1)",
            "at(shelf1,readingroom1)",
            "at(chair1,readingroom1)",
            "at(bed1,bedroom1)",
            "at(couch1,readingroom1)",
            "at(broom1,broomcloset1)",
            "at(mop1,broomcloset1)",
            "at(sandwich1,pantry1)",
            "at(apple1,pantry1)",
            "at(banana1,pantry1)",
            "at(orange1,pantry1)",
            "at(peach1,pantry1)",
            "at(plate1,kitchen1)",
            "at(book1,bedroom1)",
            "at(pillow1,bedroom1)",
            "at(toad1,pantry1)",
            "at(greenbeetle1,cellar1)",
            "at(redbeetle1,cellar1)",
            "at(eyeofnewt1,pantry1)",
            "at(plinklecrystal1,storage1)",
            "at(plonklecrystal1,storage1)",
            "at(plunklecrystal1,storage1)",
            "at(spider1,cellar1)",
            "at(spiderweb1,cellar1)",
            "at(spideregg1,cellar1)",
            "at(lilyofthevalley1,garden1)",
            "at(nightshade1,garden1)",
            "at(pricklypear1,conservatory1)",
            "at(waterbucket1,garden1)",
            "at(plasmbucket1,cellar1)",
            "at(cauldron1,kitchen1)",
            "at(ladle1,storage1)",
            "at(whisk1,storage1)",
            "at(spoon1,storage1)",
            "at(borpulus1,storage1)",
            "at(firewand1,workshop1)",
            "at(icewand1,workshop1)",
            "at(fairywand1,workshop1)",
            "at(zulpowand1,workshop1)",
            "support(kitchen1floor1)",
            "support(pantry1floor1)",
            "support(hallway1floor1)",
            "support(broomcloset1floor1)",
            "support(bedroom1floor1)",
            "support(storage1floor1)",
            "support(readingroom1floor1)",
            "support(workshop1floor1)",
            "support(workshop1ceiling1)",
            "support(cellar1floor1)",
            "support(conservatory1floor1)",
            "support(garden1floor1)",
            "support(outhouse1floor1)",
            "support(table1)",
            "support(sidetable1)",
            "support(counter1)",
            "support(shelf1)",
            "support(chair1)",
            "support(bed1)",
            "support(couch1)",
            "on(zulpowand1,workshop1ceiling1)",
            "on(fairywand1,workshop1ceiling1)",
            "on(icewand1,workshop1ceiling1)",
            "on(firewand1,workshop1ceiling1)",
            "on(borpulus1,storage1floor1)",
            "on(spoon1,storage1floor1)",
            "on(whisk1,storage1floor1)",
            "on(ladle1,storage1floor1)",
            "on(plasmbucket1,garden1floor1)",
            "on(waterbucket1,garden1floor1)",
            "on(pricklypear1,conservatory1floor1)",
            "on(nightshade1,garden1floor1)",
            "on(lilyofthevalley1,garden1floor1)",
            "on(spideregg1,cellar1floor1)",
            "on(spiderweb1,cellar1floor1)",
            "on(spider1,cellar1floor1)",
            "on(plunklecrystal1,storage1floor1)",
            "on(plonklecrystal1,storage1floor1)",
            "on(plinklecrystal1,storage1floor1)",
            "on(eyeofnewt1,pantry1floor1)",
            "on(redbeetle1,cellar1floor1)",
            "on(greenbeetle1,cellar1floor1)",
            "on(pillow1,bedroom1floor1)",
            "on(potionrecipe1,bed1)",
            "on(book1,bed1)",
            "on(plate1,kitchen1floor1)",
            "on(mop1,broomcloset1floor1)",
            "on(broom1,broomcloset1floor1)",
            "container(icebox1)",
            "container(cupboard1)",
            "container(wardrobe1)",
            "container(cauldron1)",
            "in(toad1,icebox1)",
            "in(peach1,icebox1)",
            "in(orange1,icebox1)",
            "in(banana1,icebox1)",
            "in(apple1,icebox1)",
            "in(sandwich1,icebox1)",
            "exit(kitchen1,pantry1)",
            "exit(kitchen1,conservatory1)",
            "exit(pantry1,kitchen1)",
            "exit(hallway1,readingroom1)",
            "exit(hallway1,broomcloset1)",
            "exit(hallway1,bedroom1)",
            "exit(hallway1,workshop1)",
            "exit(broomcloset1,hallway1)",
            "exit(bedroom1,readingroom1)",
            "exit(storage1,hallway1)",
            "exit(storage1,cellar1)",
            "exit(readingroom1,conservatory1)",
            "exit(readingroom1,garden1)",
            "exit(workshop1,kitchen1)",
            "exit(workshop1,bedroom1)",
            "exit(cellar1,hallway1)",
            "exit(conservatory1,hallway1)",
            "exit(garden1,storage1)",
            "exit(garden1,outhouse1)",
            "exit(outhouse1,garden1)",
            "receptacle(table1)",
            "receptacle(sidetable1)",
            "receptacle(counter1)",
            "receptacle(icebox1)",
            "receptacle(cupboard1)",
            "receptacle(wardrobe1)",
            "receptacle(shelf1)",
            "receptacle(chair1)",
            "receptacle(bed1)",
            "receptacle(couch1)",
            "receptacle(cauldron1)",
            "openable(icebox1)",
            "openable(cupboard1)",
            "openable(wardrobe1)",
            "closed(icebox1)",
            "closed(cupboard1)",
            "closed(wardrobe1)",
            "takeable(broom1)",
            "takeable(mop1)",
            "takeable(sandwich1)",
            "takeable(apple1)",
            "takeable(banana1)",
            "takeable(orange1)",
            "takeable(peach1)",
            "takeable(plate1)",
            "takeable(book1)",
            "takeable(pillow1)",
            "takeable(toad1)",
            "takeable(greenbeetle1)",
            "takeable(redbeetle1)",
            "takeable(eyeofnewt1)",
            "takeable(plinklecrystal1)",
            "takeable(plonklecrystal1)",
            "takeable(plunklecrystal1)",
            "takeable(spider1)",
            "takeable(spiderweb1)",
            "takeable(spideregg1)",
            "takeable(lilyofthevalley1)",
            "takeable(nightshade1)",
            "takeable(pricklypear1)",
            "takeable(waterbucket1)",
            "takeable(plasmbucket1)",
            "takeable(ladle1)",
            "takeable(whisk1)",
            "takeable(spoon1)",
            "takeable(borpulus1)",
            "takeable(firewand1)",
            "takeable(icewand1)",
            "takeable(fairywand1)",
            "takeable(zulpowand1)",
            "movable(broom1)",
            "movable(mop1)",
            "movable(sandwich1)",
            "movable(apple1)",
            "movable(banana1)",
            "movable(orange1)",
            "movable(peach1)",
            "movable(plate1)",
            "movable(book1)",
            "movable(pillow1)",
            "movable(toad1)",
            "movable(greenbeetle1)",
            "movable(redbeetle1)",
            "movable(eyeofnewt1)",
            "movable(plinklecrystal1)",
            "movable(plonklecrystal1)",
            "movable(plunklecrystal1)",
            "movable(spider1)",
            "movable(spiderweb1)",
            "movable(spideregg1)",
            "movable(lilyofthevalley1)",
            "movable(nightshade1)",
            "movable(pricklypear1)",
            "movable(waterbucket1)",
            "movable(plasmbucket1)",
            "movable(ladle1)",
            "movable(whisk1)",
            "movable(spoon1)",
            "movable(borpulus1)",
            "movable(firewand1)",
            "movable(icewand1)",
            "movable(fairywand1)",
            "movable(zulpowand1)",
            "needs_support(broom1)",
            "needs_support(mop1)",
            "needs_support(sandwich1)",
            "needs_support(apple1)",
            "needs_support(banana1)",
            "needs_support(orange1)",
            "needs_support(peach1)",
            "needs_support(plate1)",
            "needs_support(book1)",
            "needs_support(pillow1)",
            "needs_support(toad1)",
            "needs_support(greenbeetle1)",
            "needs_support(redbeetle1)",
            "needs_support(eyeofnewt1)",
            "needs_support(plinklecrystal1)",
            "needs_support(plonklecrystal1)",
            "needs_support(plunklecrystal1)",
            "needs_support(spider1)",
            "needs_support(spiderweb1)",
            "needs_support(spideregg1)",
            "needs_support(lilyofthevalley1)",
            "needs_support(nightshade1)",
            "needs_support(pricklypear1)",
            "needs_support(waterbucket1)",
            "needs_support(water1)",
            "needs_support(plasmbucket1)",
            "needs_support(ectoplasm1)",
            "needs_support(ladle1)",
            "needs_support(whisk1)",
            "needs_support(spoon1)",
            "needs_support(borpulus1)",
            "needs_support(firewand1)",
            "needs_support(icewand1)",
            "needs_support(fairywand1)",
            "needs_support(zulpowand1)",
            "ingredient(sandwich1)",
            "ingredient(apple1)",
            "ingredient(banana1)",
            "ingredient(orange1)",
            "ingredient(peach1)",
            "ingredient(greenbeetle1)",
            "ingredient(redbeetle1)",
            "ingredient(eyeofnewt1)",
            "ingredient(plinklecrystal1)",
            "ingredient(plonklecrystal1)",
            "ingredient(plunklecrystal1)",
            "ingredient(spider1)",
            "ingredient(spiderweb1)",
            "ingredient(spideregg1)",
            "ingredient(lilyofthevalley1)",
            "ingredient(nightshade1)",
            "ingredient(pricklypear1)",
            "ingredient(waterbucket1)",
            "ingredient(water1)",
            "ingredient(plasmbucket1)",
            "ingredient(ectoplasm1)",
            "bucket(waterbucket1)",
            "bucket(plasmbucket1)",
            "tool(ladle1)",
            "tool(whisk1)",
            "tool(spoon1)",
            "tool(borpulus1)",
            "tool(firewand1)",
            "tool(icewand1)",
            "tool(fairywand1)",
            "tool(zulpowand1)",
            "readable(potionrecipe1)",
            "text(potionrecipe1,Hungro's potion\nIngredients: Water, spider egg, sandwich, eye of newt.\n1. Pour water into your cauldron.\n2. Swirl a fire wand at your cauldron.\n3. Add the sandwich into your cauldron.\n4. Wave a ice wand at your cauldron.\n5. Add the eye of newt into your cauldron.\n6. Add the spider egg into your cauldron.\n)"
          ],
          "goal_state": [
            "at(potion1,kitchen1)"
          ],
          "max_turns": 50,
          "optimal_turns": 25,
          "optimal_solution": [
            [
              "go",
              "readingroom1"
            ],
            [
              "go",
              "garden1"
            ],
            [
              "take",
              "waterbucket1"
            ],
            [
              "go",
              "storage1"
            ],
            [
              "go",
              "cellar1"
            ],
            [
              "take",
              "spideregg1"
            ],
            [
              "go",
              "hallway1"
            ],
            [
              "go",
              "workshop1"
            ],
            [
              "take",
              "icewand1"
            ],
            [
              "take",
              "firewand1"
            ],
            [
              "go",
              "kitchen1"
            ],
            [
              "dump",
              "waterbucket1",
              "cauldron1"
            ],
            [
              "go",
              "pantry1"
            ],
            [
              "open",
              "icebox1"
            ],
            [
              "take",
              "eyeofnewt1"
            ],
            [
              "go",
              "kitchen1"
            ],
            [
              "wave",
              "cauldron1",
              "firewand1"
            ],
            [
              "go",
              "pantry1"
            ],
            [
              "take",
              "sandwich1"
            ],
            [
              "go",
              "kitchen1"
            ],
            [
              "wave",
              "cauldron1",
              "icewand1"
            ],
            [
              "put",
              "sandwich1",
              "cauldron1"
            ],
            [
              "put",
              "spideregg1",
              "cauldron1"
            ],
            [
              "put",
              "eyeofnewt1",
              "cauldron1"
            ],
            [
              "put",
              "spideregg1",
              "cauldron1"
            ]
          ],
          "optimal_commands": [
            "take potion recipe",
            "read potion recipe",
            "go to reading room",
            "examine shelf",
            "go to garden",
            "take bucket of water",
            "go to storage room",
            "take ladle",
            "drop potion recipe",
            "take ladle",
            "go to cellar",
            "take spider egg",
            "drop ladle",
            "take spider egg",
            "go to hallway",
            "go to magic workshop",
            "take fire wand",
            "drop spider egg",
            "take fire wand",
            "go to kitchen",
            "pour water into cauldron",
            "use bucket of water to fill cauldron",
            "pour bucket of water into cauldron",
            "swirl fire wand at cauldron",
            "go to pantry",
            "take eye of newt",
            "drop fire wand",
            "take eye of newt",
            "go to kitchen",
            "add eye of newt to cauldron",
            "put eye of newt in cauldron",
            "go to pantry",
            "open ice box",
            "take sandwich",
            "go to kitchen",
            "put sandwich in cauldron",
            "go to magic workshop",
            "go to conservatory",
            "go to hallway",
            "go to magic workshop",
            "take ice wand",
            "go to kitchen",
            "wave ice wand at cauldron",
            "go to pantry",
            "go to kitchen",
            "drop bucket of water",
            "go to pantry",
            "go to kitchen",
            "go to hallway",
            "go to pantry"
          ],
          "action_definitions": [
            "witch_actions_core.json"
          ],
          "room_definitions": [
            "witch_rooms.json"
          ],
          "entity_definitions": [
            "witch_entities.json"
          ],
		  "domain_definitions": ["witch_domain_core.json"],
          "event_definitions": [
            {
              "type_name": "workshop_antigravity_objects",
              "pddl": "(:event ANTIGRAVITY_OBJECTS\n    :parameters (?e - entity ?h - receptacle ?c - ceiling ?r - room ?p - player)\n    :precondition (and\n        (room ?r workshop)\n        (at ?e ?r)\n        (at ?p ?r)\n        (not (type ?h ceiling))\n        (or\n            (on ?e ?h)\n            (and\n                (in ?e ?h)\n                (open ?h)\n                )\n            )\n        (type ?c ceiling)\n        )\n    :effect (and\n        (when\n            (and\n                (in ?e ?h)\n                (open ?h)\n                )\n            (and\n                (not (in ?e ?h))\n                (on ?e ?c)\n                )\n            )\n        (when\n            (on ?e ?h)\n            (and\n                (not (on ?e ?h))\n                (on ?e ?c)\n                )\n            )\n        )\n    )",
              "event_feedback": "The {{ e }} floats up to the ceiling.",
              "asp": ""
            },
            {
              "type_name": "workshop_antigravity_player_float_start",
              "pddl": "(:event ANTIGRAVPLAYERSTART\n    :parameters (?r - room ?p - player)\n    :precondition (and\n        (room ?r workshop)\n        (at ?p ?r)\n        (not (floating ?p))\n        )\n    :effect (and\n        (floating ?p)\n        )\n    )",
              "event_feedback": "Magical experiments have reversed and dampened gravity in this room. You begin to float in the air.",
              "asp": ""
            },
            {
              "type_name": "workshop_antigravity_player_float_stop",
              "pddl": "(:event ANTIGRAVPLAYERSTOP\n    :parameters (?r - room ?p - player)\n    :precondition (and\n        (not (room ?r workshop))\n        (at ?p ?r)\n        (floating ?p)\n        )\n    :effect (and\n        (not (floating ?p))\n        )\n    )",
              "event_feedback": "Gravity behaves normally in this room. You stop floating in the air.",
              "asp": ""
            },
            {
              "type_name": "outhouse_teleport",
              "pddl": "(:event OUTHOUSETELEPORT\n    :parameters (?r - room ?p - player ?i - inventory)\n    :precondition (and\n        (room ?r outhouse)\n        (at ?p ?r)\n        )\n    :effect (and\n        (at ?p kitchen1)\n        (not (at ?p ?r))\n        (forall (?e - takeable)\n            (when\n                (in ?e ?i)\n                (and\n                    (at ?e kitchen1)\n                    (not (at ?e ?r))\n                )\n            )\n        )\n    )\n)",
              "event_feedback": "The world twists around you, and you suddenly find yourself elsewhere. {{ room_desc }}",
              "asp": "",
              "randomize": {
                "initial_value": "kitchen1",
                "replace_type": "room",
                "not_replacer": [
                  "outhouse1"
                ]
              }
            },
            {
              "type_name": "potion_step_1",
              "pddl": "(:event POTIONSTEP1\n\t:parameters (?l - liquid ?c - container ?r - room)\n\t:precondition (and\n\t\t(at ?l ?r)\n\t\t(at ?c ?r)\n\t\t(in ?l ?c)\n\t\t(type ?c cauldron)\n\t\t(type ?l water)\n\t\t(hot ?c)\n\t\t)\n\t:effect (and\n\t\t(not (at ?l ?r))\n\t\t(not (in ?l ?c))\n\t\t(not (accessible ?l))\n\t\t(type liquid1 liquid)\n\t\t(at liquid1 ?r)\n\t\t(in liquid1 ?c)\n\t\t(accessible liquid1)\n\t\t)\n\t)",
              "event_feedback": "The water in the hot cauldron bubbles with a puff of vaporleaving a liquid.",
              "asp": "event_t(TURN,potion_step_1,ROOM) :- turn(TURN), room(ROOM,_), at_t(TURN,water1,ROOM), at_t(TURN,cauldron1,ROOM),in_t(TURN,water1,cauldron1), hot_t(TURN,cauldron1).\ntype(liquid1,liquid) :- event_t(TURN,potion_step_1,ROOM).\nat_t(TURN,liquid1,ROOM) :- event_t(TURN,potion_step_1,ROOM).\nin_t(TURN,liquid1,cauldron1) :- event_t(TURN,potion_step_1,ROOM).\naccessible_t(TURN,liquid1) :- event_t(TURN,potion_step_1,ROOM).\nat_t(TURN+1,liquid1,ROOM) :- at_t(TURN,liquid1,ROOM), not event_t(TURN,potion_step_2,ROOM), turn(TURN), room(ROOM,_).\nin_t(TURN+1,liquid1,cauldron1) :- in_t(TURN,liquid1,cauldron1), not event_t(TURN,potion_step_2,ROOM), turn(TURN), room(ROOM,_).\naccessible_t(TURN+1,liquid1) :- accessible_t(TURN,liquid1), not event_t(TURN,potion_step_2,ROOM), turn(TURN), room(ROOM,_).\nat_t(TURN+1,water1,ROOM) :- turn(TURN), at_t(TURN,water1,ROOM), not event_t(TURN,potion_step_1,ROOM), room(ROOM,_).\nin_t(TURN+1,water1,cauldron1) :- turn(TURN), in_t(TURN,water1,cauldron1), not event_t(TURN,potion_step_1,ROOM), room(ROOM,_).\naccessible_t(TURN+1,water1) :- turn(TURN), accessible_t(TURN,water1), not event_t(TURN,potion_step_1,ROOM), room(ROOM,_).\nhot_t(TURN+1,cauldron1) :- turn(TURN), hot_t(TURN,cauldron1), not event_t(TURN,potion_step_1,ROOM), room(ROOM,_)."
            },
            {
              "type_name": "potion_step_2",
              "pddl": "(:event POTIONSTEP2\n\t:parameters (?l - liquid ?i - ingredient ?c - container ?r - room)\n\t:precondition (and\n\t\t(at liquid1 ?r)\n\t\t(at ?c ?r)\n\t\t(at ?i ?r)\n\t\t(type ?c cauldron)\n\t\t(type ?i sandwich)\n\t\t(in ?i ?c)\n\t\t(in liquid1 ?c)\n\t\t)\n\t:effect (and\n\t\t(not (type liquid1 liquid))\n\t\t(not (at liquid1 ?r))\n\t\t(not (in liquid1 ?c))\n\t\t(not (accessible liquid1))\n\t\t(not (at ?i ?r))\n\t\t(not (in ?i ?c))\n\t\t(not (type ?i sandwich))\n\t\t(type liquid2 liquid)\n\t\t(at liquid2  ?r)\n\t\t(in liquid2  ?c)\n\t\t(accessible liquid2 )\n\t\t)\n\t)",
              "event_feedback": "The sandwich combines with the liquid in the cauldron with a swirling pattern.",
              "asp": "event_t(TURN,potion_step_2,ROOM) :- turn(TURN), room(ROOM,_), at_t(TURN,sandwich1,ROOM), at_t(TURN,cauldron1,ROOM),in_t(TURN,liquid1,cauldron1), in_t(TURN,sandwich1,cauldron1).\ntype(liquid2,liquid) :- event_t(TURN,potion_step_2,ROOM).\nat_t(TURN,liquid2,ROOM) :- event_t(TURN,potion_step_2,ROOM).\nin_t(TURN,liquid2,cauldron1) :- event_t(TURN,potion_step_2,ROOM).\naccessible_t(TURN,liquid2) :- event_t(TURN,potion_step_2,ROOM).\nat_t(TURN+1,liquid2,ROOM) :- at_t(TURN,liquid2,ROOM), not event_t(TURN,potion_step_3,ROOM), turn(TURN), room(ROOM,_).\nin_t(TURN+1,liquid2,cauldron1) :- in_t(TURN,liquid2,cauldron1), not event_t(TURN,potion_step_3,ROOM), turn(TURN), room(ROOM,_).\naccessible_t(TURN+1,liquid2) :- accessible_t(TURN,liquid2), not event_t(TURN,potion_step_3,ROOM), turn(TURN), room(ROOM,_).\nat_t(TURN+1,liquid1,ROOM) :- turn(TURN), at_t(TURN,liquid1,ROOM), not event_t(TURN,potion_step_2,ROOM), room(ROOM,_).\nin_t(TURN+1,liquid1,cauldron1) :- turn(TURN), in_t(TURN,liquid1,cauldron1), not event_t(TURN,potion_step_2,ROOM), room(ROOM,_).\naccessible_t(TURN+1,liquid1) :- turn(TURN), accessible_t(TURN,liquid1), not event_t(TURN,potion_step_2,ROOM), room(ROOM,_).\nat_t(TURN+1,sandwich1,ROOM) :- turn(TURN), at_t(TURN,sandwich1,ROOM), not event_t(TURN,potion_step_2,ROOM), room(ROOM,_).\nin_t(TURN+1,sandwich1,cauldron1) :- turn(TURN), in_t(TURN,sandwich1,cauldron1), not event_t(TURN,potion_step_2,ROOM), room(ROOM,_)."
            },
            {
              "type_name": "potion_step_3",
              "pddl": "(:event POTIONSTEP3\n\t:parameters (?l - liquid ?c - container ?r - room)\n\t:precondition (and\n\t\t(at liquid2 ?r)\n\t\t(at ?c ?r)\n\t\t(in liquid2 ?c)\n\t\t(type ?c cauldron)\n\t\t(type liquid2 liquid)\n\t\t(cold ?c)\n\t\t)\n\t:effect (and\n\t\t(not (at liquid2 ?r))\n\t\t(not (in liquid2 ?c))\n\t\t(not (accessible liquid2))\n\t\t(type liquid3 liquid)\n\t\t(at liquid3 ?r)\n\t\t(in liquid3 ?c)\n\t\t(accessible liquid3)\n\t\t)\n\t)",
              "event_feedback": "The liquid in the cold cauldron swirls with a puff of vapor.",
              "asp": "event_t(TURN,potion_step_3,ROOM) :- turn(TURN), room(ROOM,_), at_t(TURN,liquid2,ROOM), at_t(TURN,cauldron1,ROOM),in_t(TURN,liquid2,cauldron1), cold_t(TURN,cauldron1).\ntype(liquid3,liquid) :- event_t(TURN,potion_step_3,ROOM).\nat_t(TURN,liquid3,ROOM) :- event_t(TURN,potion_step_3,ROOM).\nin_t(TURN,liquid3,cauldron1) :- event_t(TURN,potion_step_3,ROOM).\naccessible_t(TURN,liquid3) :- event_t(TURN,potion_step_3,ROOM).\nat_t(TURN+1,liquid3,ROOM) :- at_t(TURN,liquid3,ROOM), not event_t(TURN,potion_step_4,ROOM), turn(TURN), room(ROOM,_).\nin_t(TURN+1,liquid3,cauldron1) :- in_t(TURN,liquid3,cauldron1), not event_t(TURN,potion_step_4,ROOM), turn(TURN), room(ROOM,_).\naccessible_t(TURN+1,liquid3) :- accessible_t(TURN,liquid3), not event_t(TURN,potion_step_4,ROOM), turn(TURN), room(ROOM,_).\nat_t(TURN+1,liquid2,ROOM) :- turn(TURN), at_t(TURN,liquid2,ROOM), not event_t(TURN,potion_step_3,ROOM), room(ROOM,_).\nin_t(TURN+1,liquid2,cauldron1) :- turn(TURN), in_t(TURN,liquid2,cauldron1), not event_t(TURN,potion_step_3,ROOM), room(ROOM,_).\naccessible_t(TURN+1,liquid2) :- turn(TURN), accessible_t(TURN,liquid2), not event_t(TURN,potion_step_3,ROOM), room(ROOM,_)."
            },
            {
              "type_name": "potion_step_4",
              "pddl": "(:event POTIONSTEP4\n\t:parameters (?l - liquid ?i - ingredient ?c - container ?r - room)\n\t:precondition (and\n\t\t(at liquid3 ?r)\n\t\t(at ?c ?r)\n\t\t(at ?i ?r)\n\t\t(type ?c cauldron)\n\t\t(type ?i eyeofnewt)\n\t\t(in ?i ?c)\n\t\t(in liquid3 ?c)\n\t\t)\n\t:effect (and\n\t\t(not (type liquid3 liquid))\n\t\t(not (at liquid3 ?r))\n\t\t(not (in liquid3 ?c))\n\t\t(not (accessible liquid3))\n\t\t(not (at ?i ?r))\n\t\t(not (in ?i ?c))\n\t\t(not (type ?i eyeofnewt))\n\t\t(type liquid4 liquid)\n\t\t(at liquid4  ?r)\n\t\t(in liquid4  ?c)\n\t\t(accessible liquid4 )\n\t\t)\n\t)",
              "event_feedback": "The eye of newt mingles with the liquid in the cauldron with a gloopy sound.",
              "asp": "event_t(TURN,potion_step_4,ROOM) :- turn(TURN), room(ROOM,_), at_t(TURN,eyeofnewt1,ROOM), at_t(TURN,cauldron1,ROOM),in_t(TURN,liquid3,cauldron1), in_t(TURN,eyeofnewt1,cauldron1).\ntype(liquid4,liquid) :- event_t(TURN,potion_step_4,ROOM).\nat_t(TURN,liquid4,ROOM) :- event_t(TURN,potion_step_4,ROOM).\nin_t(TURN,liquid4,cauldron1) :- event_t(TURN,potion_step_4,ROOM).\naccessible_t(TURN,liquid4) :- event_t(TURN,potion_step_4,ROOM).\nat_t(TURN+1,liquid4,ROOM) :- at_t(TURN,liquid4,ROOM), not event_t(TURN,potion_step_5,ROOM), turn(TURN), room(ROOM,_).\nin_t(TURN+1,liquid4,cauldron1) :- in_t(TURN,liquid4,cauldron1), not event_t(TURN,potion_step_5,ROOM), turn(TURN), room(ROOM,_).\naccessible_t(TURN+1,liquid4) :- accessible_t(TURN,liquid4), not event_t(TURN,potion_step_5,ROOM), turn(TURN), room(ROOM,_).\nat_t(TURN+1,liquid3,ROOM) :- turn(TURN), at_t(TURN,liquid3,ROOM), not event_t(TURN,potion_step_4,ROOM), room(ROOM,_).\nin_t(TURN+1,liquid3,cauldron1) :- turn(TURN), in_t(TURN,liquid3,cauldron1), not event_t(TURN,potion_step_4,ROOM), room(ROOM,_).\naccessible_t(TURN+1,liquid3) :- turn(TURN), accessible_t(TURN,liquid3), not event_t(TURN,potion_step_4,ROOM), room(ROOM,_).\nat_t(TURN+1,eyeofnewt1,ROOM) :- turn(TURN), at_t(TURN,eyeofnewt1,ROOM), not event_t(TURN,potion_step_4,ROOM), room(ROOM,_).\nin_t(TURN+1,eyeofnewt1,cauldron1) :- turn(TURN), in_t(TURN,eyeofnewt1,cauldron1), not event_t(TURN,potion_step_4,ROOM), room(ROOM,_)."
            },
            {
              "type_name": "potion_step_5",
              "pddl": "(:event POTIONSTEP5\n\t:parameters (?l - liquid ?i - ingredient ?c - container ?r - room)\n\t:precondition (and\n\t\t(at ?c ?r)\n\t\t(at ?i ?r)\n\t\t(type ?c cauldron)\n\t\t(type ?i spideregg)\n\t\t(in ?i ?c)\n\t\t(in liquid4 ?c)\n\t\t)\n\t:effect (and\n\t\t(not (type spideregg1 spideregg))\n\t\t(not (at spideregg1 ?r))\n\t\t(not (in spideregg1 ?c))\n\t\t(not (accessible spideregg1))\n\t\t(type potion1 potion)\n\t\t(at potion1 ?r)\n\t\t(in potion1 ?c)\n\t\t(accessible potion1)\n\t\t)\n\t)",
              "event_feedback": "The spider egg mingles with the liquid with a swirling pattern leaving the finished potion in the cauldron.",
              "asp": "event_t(TURN,potion_step_5,ROOM) :- turn(TURN), room(ROOM,_), at_t(TURN,spideregg1,ROOM), at_t(TURN,cauldron1,ROOM),in_t(TURN,liquid4,cauldron1), in_t(TURN,spideregg1,cauldron1).\ntype(potion1,potion) :- event_t(TURN,potion_step_5,ROOM).\nat_t(TURN,potion1,ROOM) :- event_t(TURN,potion_step_5,ROOM).\nin_t(TURN,potion1,cauldron1) :- event_t(TURN,potion_step_5,ROOM).\naccessible_t(TURN,potion1) :- event_t(TURN,potion_step_5,ROOM).\nat_t(TURN+1,potion1,ROOM) :- at_t(TURN,potion1,ROOM), turn(TURN), room(ROOM,_).\nin_t(TURN+1,potion1,cauldron1) :- in_t(TURN,potion1,cauldron1), turn(TURN), room(ROOM,_).\naccessible_t(TURN+1,potion1) :- accessible_t(TURN,potion1), turn(TURN), room(ROOM,_).\nat_t(TURN+1,liquid4,ROOM) :- turn(TURN), at_t(TURN,liquid4,ROOM), not event_t(TURN,potion_step_5,ROOM), room(ROOM,_).\nin_t(TURN+1,liquid4,cauldron1) :- turn(TURN), in_t(TURN,liquid4,cauldron1), not event_t(TURN,potion_step_5,ROOM), room(ROOM,_).\naccessible_t(TURN+1,liquid4) :- turn(TURN), accessible_t(TURN,liquid4), not event_t(TURN,potion_step_5,ROOM), room(ROOM,_).\nat_t(TURN+1,spideregg1,ROOM) :- turn(TURN), at_t(TURN,spideregg1,ROOM), not event_t(TURN,potion_step_5,ROOM), room(ROOM,_).\nin_t(TURN+1,spideregg1,cauldron1) :- turn(TURN), in_t(TURN,spideregg1,cauldron1), not event_t(TURN,potion_step_5,ROOM), room(ROOM,_)."
            }
          ]
        }
    # initialize test interpreter:
    test_interpreter = AdventureIFInterpreter("D:/AdventureClemGame/adventuregame", game_instance_exmpl)

    # print("Intialized domain:", test_interpreter.domain)
    # print("Intialized domain mutable states:", test_interpreter.domain['mutable_states'])
    """
    for event_type in test_interpreter.event_types:
        print(event_type)
        print(test_interpreter.event_types[event_type])
        print(test_interpreter.event_types[event_type]['interaction'])
        print(test_interpreter.event_types[event_type]['interaction']['effect'])
    """
    """
    for action_type in test_interpreter.action_types:
        print(action_type, test_interpreter.action_types[action_type])
    print()
    """

    wob: str = ""

    # run optimal solution:
    test_interpreter.execute_optimal_solution()
