"""
    IF interpreter for adventuregame.
"""

import json
import lark
from lark import Lark, Transformer
import jinja2

import os
from copy import deepcopy
from typing import List, Set, Union

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
        action_type = action.data  # main grammar rule the input was parsed as
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


class PDDLActionTransformer(Transformer):
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


class PDDLDomainTransformer(Transformer):
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
        # print("type_list_item return:", {'type_list_item': cat_name, 'items': type_list_items})
        return {'type_list_element': cat_name, 'items': type_list_items}

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


class AdventureIFInterpreter(GameResourceLocator):
    """
    IF interpreter for adventuregame.
    Holds game world state and handles all interaction and feedback.
    """
    def __init__(self, game_path, game_instance: dict, name: str = GAME_NAME, verbose: bool = False):
        super().__init__(name, game_path)
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
        self.initialize_pddl_definition_parsing()

        self.act_parser = None
        self.act_transformer = IFTransformer()
        self.action_types = dict()
        self.initialize_action_types()

        self.domain = dict()
        self.initialize_domain()

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
        for entity_def_source in self.game_instance["entity_definitions"]:
            entities_file = self.load_json(f"resources{os.sep}definitions{os.sep}{entity_def_source[:-5]}")
            entity_definitions += entities_file

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
        for room_def_source in self.game_instance["room_definitions"]:
            rooms_file = self.load_json(f"resources{os.sep}definitions{os.sep}{room_def_source[:-5]}")
            room_definitions += rooms_file

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

    def initialize_action_types(self):
        """
        Load and process action types in this adventure.
        Definitions are loaded from external files.
        """
        # load action type definitions in game instance:
        action_definitions: list = list()
        for action_def_source in self.game_instance["action_definitions"]:
            actions_file = self.load_json(f"resources{os.sep}definitions{os.sep}{action_def_source[:-5]}")
            action_definitions += actions_file

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
        for domain_def_source in self.game_instance["domain_definitions"]:
            domain_def_raw = self.load_json(f"resources{os.sep}definitions{os.sep}{domain_def_source[:-5]}")
            # print("domain_def_raw:", domain_def_raw)
            # print("domain_def_raw pddl_domain:", domain_def_raw['pddl_domain'])
            domain_definitions.append(domain_def_raw['pddl_domain'])

        # print("domain_definitions", domain_definitions)

        for domain_definition in domain_definitions:
            # print("domain_definition:", domain_definition)
            parsed_domain_pddl = self.domain_def_parser.parse(domain_definition)
            processed_domain_pddl = self.domain_def_transformer.transform(parsed_domain_pddl)
            # print("processed_domain_pddl:", processed_domain_pddl)

        # for now assume only one domain definition:
        self.domain = processed_domain_pddl
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
        if inst in self.inst_to_type_dict:
            inst_type: str = self.inst_to_type_dict[inst]
        elif inst in self.room_to_type_dict:
            inst_type: str = self.room_to_type_dict[inst]
        else:  # fallback for potential edge cases
            # TODO: retrace why this can fail
            logger.info(f"_get_inst_str got {inst}, which is not in the _to_type dicts! "
                        f"Heuristically culling numbers from inst string end as fallback...")
            inst_type = deepcopy(inst)
            while inst_type.endswith(("0","1","2","3","4","5","6","7","8","9")):
                inst_type = inst_type[:-1]
            logger.info(f"inst_type after heuristic culling: {inst_type}")
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
            if 'hidden' in self.entity_types[self.inst_to_type_dict[thing]]:
                continue

            # do not access entities inside closed containers:
            contained_in = None
            for fact in self.world_state:
                # check if entity is 'in' closed container:
                if fact[0] == 'in' and fact[1] == thing:
                    contained_in = fact[2]
                    # print(f"{thing} is contained in {contained_in}")
                    for state_pred2 in self.world_state:
                        if state_pred2[0] == 'closed' and state_pred2[1] == contained_in:
                            # not visible/accessible in closed container
                            break
                        elif state_pred2[0] == 'open' and state_pred2[1] == contained_in:
                            visible_contents.append(thing)
                            break
                        elif state_pred2[1] == 'inventory' and state_pred2[1] == contained_in:
                            # inventory content is not visible
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
        # NOTE: This assumes only one instance of any entity type is in the adventure!
        entity_id = str()
        for fact in self.world_state:
            if fact[0] == 'type' and fact[2] == entity:
                entity_id = fact[1]
                break
        # print("entity ID found:", entity_id)
        entity_desc_list = list()
        # get all entity states to describe:
        for fact in self.world_state:
            if fact[1] == entity_id:
                # print("entity state fact:", fact)
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
                    openable_desc = f"The {openable_entity} is openable and currently {openable_state}."
                    entity_desc_list.append(openable_desc)
                # describe 'takeable' entities:
                if fact[0] == "takeable":
                    takeable_entity: str = fact[1]
                    while takeable_entity.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                        takeable_entity = takeable_entity[:-1]
                    # print("takeable_entity:", takeable_entity)
                    takeable_desc = f"The {takeable_entity} is takeable."
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
                        supporter_entity = "your inventory"
                    else:
                        while supporter_entity.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                            supporter_entity = supporter_entity[:-1]
                        supporter_entity = f"the {supporter_entity}"

                    needs_support_desc = f"The {needs_support_entity} is {support_state} {supporter_entity}."
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
                                    continue
                                # print("contained_entity is accessible")

                                while contained_entity.endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                                    contained_entity = contained_entity[:-1]
                                # print("contained_entity:", contained_entity)
                                contained_entities.append(f"a {contained_entity}")

                    if ('closed', fact[1]) in self.world_state:
                        container_content_desc = f"You can't see the {container_entity}'s contents because it is closed."
                    else:
                        if len(contained_entities) == 0:
                            container_content_desc = f"The {container_entity} is empty."
                        elif len(contained_entities) == 1:
                            container_content_desc = f"There is {contained_entities[0]} in the {container_entity}."
                        elif len(contained_entities) == 2:
                            container_content_desc = f"There are {contained_entities[0]} and {contained_entities[1]} in the {container_entity}."
                        elif len(contained_entities) >= 3:
                            container_content_desc = f"There are {', '.join(contained_entities[:-1])} and {contained_entities[-1]} in the {container_entity}."

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
                                supported_entities.append(f"a {supported_entity}")

                    if len(supported_entities) == 0:
                        support_content_desc = f"There is nothing on the {support_entity}."
                    elif len(supported_entities) == 1:
                        support_content_desc = f"There is {supported_entities[0]} on the {support_entity}."
                    elif len(supported_entities) == 2:
                        support_content_desc = f"There are {supported_entities[0]} and {supported_entities[1]} on the {support_entity}."
                    elif len(supported_entities) >= 3:
                        support_content_desc = f"There are {', '.join(supported_entities[:-1])} and {supported_entities[-1]} on the {support_entity}."

                    entity_desc_list.append(support_content_desc)

                # TODO?: room description?

        return " ".join(entity_desc_list)

    def get_current_perceived(self) -> set:
        current_perceived: set = set()

        # get player room at fact
        for fact in self.world_state:
            if fact[0] == 'at' and fact[1] == 'player1':
                current_perceived.add(fact)

        visible_room_contents = self.get_player_room_contents_visible()
        for fact in self.world_state:
            # TODO: de-hardcode mutable predicates tracked here
            if fact[1] in visible_room_contents and fact[0] in ("open", "closed", "at", "in", "on"):
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
            # logger.info("Recording initial exploration state.")
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
                    fail_dict: dict = {'phase': "parsing", 'fail_type': "manipulating_room", 'arg': action_dict['arg1']}
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
                fail_dict: dict = {'phase': "parsing", 'fail_type': "undefined_argument_type", 'arg': action_dict['arg1']}
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

        predicate_arg1 = predicate['arg1']
        if 'variable' in predicate_arg1:
            predicate_arg1 = variable_map[predicate_arg1['variable']]
            # print("filled when condition variable:", when_condition_arg1)
            # for now:
            if type(predicate_arg1) == list:
                when_condition_arg1 = predicate_arg1[0]

        predicate_list = [predicate_type, predicate_arg1]

        predicate_arg2 = None
        if predicate['arg2']:
            predicate_arg2 = predicate['arg2']
            if 'variable' in predicate_arg2:
                predicate_arg2 = variable_map[predicate_arg2['variable']]
                # print("filled when condition variable:", when_condition_arg2)
            predicate_list.append(predicate_arg2)

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

        # assume that action arguments that don't end in numbers or "floor" are type words:
        for tuple_idx, tuple_arg in enumerate(predicate_tuple[1:]):  # first tuple item is always a predicate
            # print("tuple_arg:", tuple_arg)
            type_matched_instances = list()
            if tuple_arg:
                # logger.info(f"predicate_to_tuple tuple_arg intermediate: {tuple_arg}")
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
                        # TODO?: fail if there is no type-fitting instance in world state?

                    # logger.info(f"type_matched_instances: {type_matched_instances}")

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
                                predicate_tuple = (predicate_tuple[0], type_matched_instances[0], predicate_tuple[2])
                            elif tuple_idx == 1:
                                predicate_tuple = (predicate_tuple[0], predicate_tuple[1], type_matched_instances[0])
                        case 4:
                            if tuple_idx == 0:
                                predicate_tuple = (
                                predicate_tuple[0], type_matched_instances[0], predicate_tuple[2], predicate_tuple[3])
                            elif tuple_idx == 1:
                                predicate_tuple = (
                                predicate_tuple[0], predicate_tuple[1], type_matched_instances[0], predicate_tuple[3])
                            elif tuple_idx == 2:
                                predicate_tuple = (
                                predicate_tuple[0], predicate_tuple[1], predicate_tuple[2], type_matched_instances[0])

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
        # check that parameters key correctly contains a PDDL type_list:
        if not 'type_list' in parameters_base:
            raise KeyError
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
                    return False, feedback_str, fail_dict

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
            for key in clean_feedback_variable_map:
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

        feedback_str = feedback_jinja.render(jinja_args)
        # feedback_str = feedback_str.capitalize()

        # print("feedback_str:", feedback_str)

        return True, feedback_str, {'world_state_effects': world_state_effects}

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
        for goal_fact in self.goal_state:
            goal_entities.add(goal_fact[1])
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
            return self.goals_achieved, parse_result, fail
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
                return self.goals_achieved, resolution_result, fail
            else:
                logger.info(f"Resolution result: {resolution_result}")
                base_result_str = resolution_result

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
                return goals_achieved_response, base_result_str, extra_action_info

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
    game_instance_exmpl = {"game_id": 11, "variant": "basic",
     "prompt": "You are playing a text adventure game. I will describe what you can perceive in the game. You write the single action you want to take in the game starting with >. Only reply with actions.\nFor example:\n> examine cupboard\n\nYour goal for this game is: Put the book on the table, the plate on the table and the mop on the table.\n\n",
     "initial_state": ["at(kitchen1floor,kitchen1)", "at(pantry1floor,pantry1)", "at(hallway1floor,hallway1)",
                       "at(livingroom1floor1,livingroom1)", "at(broomcloset1floor1,broomcloset1)",
                       "at(bedroom1floor1,bedroom1)", "at(table1,livingroom1)", "at(sidetable1,livingroom1)",
                       "at(counter1,kitchen1)", "at(refrigerator1,pantry1)", "at(cupboard1,kitchen1)",
                       "at(wardrobe1,bedroom1)", "at(shelf1,livingroom1)", "at(freezer1,pantry1)",
                       "at(pottedplant1,hallway1)", "at(chair1,livingroom1)", "at(bed1,bedroom1)",
                       "at(couch1,livingroom1)", "at(broom1,broomcloset1)", "at(mop1,broomcloset1)",
                       "at(sandwich1,pantry1)", "at(apple1,pantry1)", "at(banana1,pantry1)", "at(orange1,pantry1)",
                       "at(peach1,pantry1)", "at(plate1,kitchen1)", "at(book1,livingroom1)", "at(pillow1,bedroom1)",
                       "at(player1,bedroom1)", "type(kitchen1floor,floor)", "type(pantry1floor,floor)",
                       "type(hallway1floor1,floor)", "type(livingroom1floor1,floor)", "type(broomcloset1floor1,floor)",
                       "type(bedroom1floor1,floor)", "type(player1,player)", "type(table1,table)",
                       "type(sidetable1,sidetable)", "type(counter1,counter)", "type(refrigerator1,refrigerator)",
                       "type(cupboard1,cupboard)", "type(wardrobe1,wardrobe)", "type(shelf1,shelf)",
                       "type(freezer1,freezer)", "type(pottedplant1,pottedplant)", "type(chair1,chair)",
                       "type(bed1,bed)", "type(couch1,couch)", "type(broom1,broom)", "type(mop1,mop)",
                       "type(sandwich1,sandwich)", "type(apple1,apple)", "type(banana1,banana)", "type(orange1,orange)",
                       "type(peach1,peach)", "type(plate1,plate)", "type(book1,book)", "type(pillow1,pillow)",
                       "room(kitchen1,kitchen)", "room(pantry1,pantry)", "room(hallway1,hallway)",
                       "room(livingroom1,livingroom)", "room(broomcloset1,broomcloset)", "room(bedroom1,bedroom)",
                       "support(kitchen1floor1)", "support(pantry1floor1)", "support(hallway1floor1)",
                       "support(livingroom1floor1)", "support(broomcloset1floor1)", "support(bedroom1floor1)",
                       "support(table1)", "support(sidetable1)", "support(counter1)", "support(shelf1)",
                       "support(bed1)", "on(book1,sidetable1)", "on(plate1,kitchen1floor)",
                       "on(mop1,broomcloset1floor1)", "on(broom1,broomcloset1floor1)", "on(pottedplant1,hallway1floor1)",
                       "container(refrigerator1)", "container(cupboard1)", "container(wardrobe1)",
                       "container(freezer1)", "in(pillow1,wardrobe1)", "in(peach1,refrigerator1)",
                       "in(orange1,refrigerator1)", "in(banana1,refrigerator1)", "in(apple1,refrigerator1)",
                       "in(sandwich1,refrigerator1)", "exit(kitchen1,pantry1)", "exit(kitchen1,livingroom1)",
                       "exit(kitchen1,hallway1)", "exit(pantry1,kitchen1)", "exit(hallway1,kitchen1)",
                       "exit(hallway1,livingroom1)", "exit(hallway1,broomcloset1)", "exit(livingroom1,kitchen1)",
                       "exit(livingroom1,hallway1)", "exit(broomcloset1,hallway1)", "exit(bedroom1,livingroom1)",
                       "exit(livingroom1,bedroom1)", "openable(refrigerator1)", "openable(cupboard1)",
                       "openable(wardrobe1)", "openable(freezer1)", "closed(refrigerator1)", "closed(cupboard1)",
                       "closed(wardrobe1)", "closed(freezer1)", "takeable(pottedplant1)", "takeable(broom1)",
                       "takeable(mop1)", "takeable(sandwich1)", "takeable(apple1)", "takeable(banana1)",
                       "takeable(orange1)", "takeable(peach1)", "takeable(plate1)", "takeable(book1)",
                       "takeable(pillow1)", "movable(pottedplant1)", "movable(broom1)", "movable(mop1)",
                       "movable(sandwich1)", "movable(apple1)", "movable(banana1)", "movable(orange1)",
                       "movable(peach1)", "movable(plate1)", "movable(book1)", "movable(pillow1)",
                       "needs_support(pottedplant1)", "needs_support(broom1)", "needs_support(mop1)",
                       "needs_support(sandwich1)", "needs_support(apple1)", "needs_support(banana1)",
                       "needs_support(orange1)", "needs_support(peach1)", "needs_support(plate1)",
                       "needs_support(book1)", "needs_support(pillow1)"],
     "goal_state": ["on(book1,table1)", "on(plate1,table1)", "on(mop1,table1)"], "max_turns": 50, "optimal_turns": 12,
     "optimal_solution": [["go", "livingroom1"], ["put", "book1", "table1"], ["go", "kitchen1"], ["take", "plate1"],
                          ["go", "livingroom1"], ["put", "plate1", "table1"], ["go", "hallway1"],
                          ["go", "broomcloset1"], ["take", "mop1"], ["go", "hallway1"], ["go", "livingroom1"],
                          ["put", "mop1", "table1"]],
     "optimal_commands": ["go living room", "put book on table", "go kitchen", "take plate", "go living room",
                          "put plate on table", "go hallway", "go broom closet", "take mop", "go hallway",
                          "go living room", "put mop on table"], "action_definitions": ["basic_actions_v2.json"],
     "room_definitions": ["home_rooms.json"], "entity_definitions": ["home_entities.json"],
                           "domain_definitions":["home_domain.json"]}
    # initialize test interpreter:
    test_interpreter = AdventureIFInterpreter(game_instance_exmpl)
    # run optimal solution:
    test_interpreter.execute_optimal_solution()
