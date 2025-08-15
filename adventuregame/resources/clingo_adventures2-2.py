"""
Clingo-based adventure generation and optimal solving.
Generates ASP logic program encoding strings which are then passed to the Clingo solver. Clingo outputs are filtered
and limited to yield intermediate adventure parts like room layout in reasonable running time. Goals are generated using
only Python code in this version ('home deliver' task type only in v1).
Please refer to the ASP documentation for more information on the encodings: https://potassco.org/clingo/
Adventure type definition file containing task goal settings etc is
adventuregame/resources/definitions/adventure_types.json
"""

from typing import List, Tuple, Union, Optional
import json
from itertools import permutations
from datetime import datetime
import os

import numpy as np
from clingo.control import Control
import lark
from lark import Lark, Transformer
from nltk import elementtree_indent
from pydantic_core.core_schema import filter_dict_schema

from adventuregame.adv_util import fact_str_to_tuple, fact_tuple_to_str
from adventuregame.resources.new_word_generation.new_word_definitions import create_new_words_definitions_set, replace_new_words_definitions_set
from adventuregame.resources.pddl_util import PDDLDomainTransformer, PDDLActionTransformer
from adventuregame.resources.pddl_to_asp import augment_action_defs_with_asp


def convert_action_to_tuple(action: str) -> Tuple:
    action_splice = action[9:-1]
    action_split = action_splice.split(",")
    action_split[0] = int(action_split[0])
    action_tuple = tuple(action_split)
    return action_tuple


class ClingoAdventureGenerator(object):
    """
    Generates full raw adventures (initial state and goals), solves each to check optimal number of turns.
    """
    def __init__(self, adventure_type: str = "home_deliver_two", rng_seed: int = 42):
        self.adv_type: str = adventure_type
        # load adventure type definition:
        with open(f"definitions{os.sep}adventure_types.json", 'r', encoding='utf-8') as adventure_types_file:
            adventure_type_definitions = json.load(adventure_types_file)
            self.adv_type_def = adventure_type_definitions[self.adv_type]

        # TODO: overhaul adventure type definitions and usage
        #   - mutable fact types from domain
        #   - new-word experiment parameters

        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(seed=self.rng_seed)

        self._initialize_pddl_definition_parsing()

        if self.adv_type_def['use_premade_definitions']:
            self._load_premade_room_definitions()
            self._load_premade_entity_definitions()
            self._load_premade_action_definitions()
            self._load_premade_domain_definition()
            if 'event_definitions' in self.adv_type_def:
                self._load_premade_event_definitions()
        else:
            if self.adv_type_def['definition_method'] == "create":
                self.new_word_iterate_idx = 0
                self._create_assign_new_word_definitions()
            if self.adv_type_def['definition_method'] == "replace":
                self.new_word_iterate_idx = 0
                self._replace_assign_new_word_definitions()

        clingo_template_file = "clingo_templates.json"
        if 'clingo_templates' in self.adv_type_def:
            clingo_template_file = self.adv_type_def['clingo_templates']

        # load clingo ASP templates:
        with open(clingo_template_file, 'r', encoding='utf-8') as templates_file:
            self.clingo_templates = json.load(templates_file)

    def _initialize_pddl_definition_parsing(self):
        with open("pddl_actions.lark", 'r', encoding='utf-8') as pddl_actions_lark_file:
            action_def_grammar = pddl_actions_lark_file.read()
        self.action_def_parser = Lark(action_def_grammar, start="action")
        self.action_def_transformer = PDDLActionTransformer()

        with open("pddl_domain.lark", 'r', encoding='utf-8') as pddl_domain_lark_file:
            domain_def_grammar = pddl_domain_lark_file.read()
        self.domain_def_parser = Lark(domain_def_grammar, start="define")
        self.domain_def_transformer = PDDLDomainTransformer()

    def _load_premade_room_definitions(self):
        # load room type definitions:
        room_definitions: list = list()
        for room_def_source in self.adv_type_def["room_definitions"]:
            with open(f"definitions/{room_def_source}", 'r', encoding='utf-8') as rooms_file:
                room_definitions += json.load(rooms_file)

        self.room_definitions = dict()
        for type_def in room_definitions:
            type_def_dict = dict()
            for type_key, type_value in type_def.items():
                if not type_key == 'type_name':
                    type_def_dict[type_key] = type_value
            self.room_definitions[type_def['type_name']] = type_def_dict

    def _load_premade_entity_definitions(self):
        # load entity type definitions:
        entity_definitions: list = list()
        for entity_def_source in self.adv_type_def["entity_definitions"]:
            with open(f"definitions/{entity_def_source}", 'r', encoding='utf-8') as entities_file:
                entity_definitions += json.load(entities_file)

        self.entity_definitions = dict()
        for type_def in entity_definitions:
            type_def_dict = dict()
            for type_key, type_value in type_def.items():
                if not type_key == 'type_name':
                    type_def_dict[type_key] = type_value
            self.entity_definitions[type_def['type_name']] = type_def_dict

    def _load_premade_action_definitions(self):
        # load action type definitions:
        action_definitions: list = list()
        for action_def_source in self.adv_type_def["action_definitions"]:
            with open(f"definitions/{action_def_source}", 'r', encoding='utf-8') as actions_file:
                action_definitions += json.load(actions_file)

        # testing asp generation:
        # action_definitions = augment_action_defs_with_asp(action_definitions)
        # -> needs more sophisticated PDDL to ASP to work on v2 home delivery...

        if not hasattr(self, 'action_definitions'):
            self.action_definitions = dict()
        for type_def in action_definitions:
            type_def_dict = dict()
            for type_key, type_value in type_def.items():
                if not type_key == 'type_name':
                    type_def_dict[type_key] = type_value
            self.action_definitions[type_def['type_name']] = type_def_dict

    def _load_premade_event_definitions(self):
        # load event type definitions:
        event_definitions: list = list()
        for event_def_source in self.adv_type_def["event_definitions"]:
            with open(f"definitions/{event_def_source}", 'r', encoding='utf-8') as events_file:
                event_definitions += json.load(events_file)

        # testing asp generation:
        # action_definitions = augment_action_defs_with_asp(action_definitions)
        # -> needs more sophisticated PDDL to ASP to work on v2 home delivery...

        if not hasattr(self, 'event_definitions'):
            self.event_definitions = dict()
        for type_def in event_definitions:
            type_def_dict = dict()
            for type_key, type_value in type_def.items():
                if not type_key == 'type_name':
                    type_def_dict[type_key] = type_value
            self.event_definitions[type_def['type_name']] = type_def_dict

    def _load_premade_domain_definition(self):
        # load domain definition:
        # domain_definitions: list = list()
        # for domain_def_source in self.adv_type_def["domain_definitions"]:
        #    with open(f"definitions/{domain_def_source}", 'r', encoding='utf-8') as domain_file:
        #        domain_definitions += json.load(domain_file)

        with open(f"definitions/{self.adv_type_def['domain_definitions'][0]}", 'r', encoding='utf-8') as domain_file:
            domain_definition = json.load(domain_file)

        # currently hardcoded to only use the first listed domain
        # print(domain_definition)
        domain_definition_pddl = domain_definition['pddl_domain']
        parsed_domain_definition_pddl = self.domain_def_parser.parse(domain_definition_pddl)
        self.domain_def = self.domain_def_transformer.transform(parsed_domain_definition_pddl)

    def _create_assign_new_word_definitions(self):
        # create new-words definitions:
        new_rooms, new_entities, new_actions, new_domain, trait_dict, last_new_word_idx = create_new_words_definitions_set(self.new_word_iterate_idx, seed=self.rng_seed)

        print("self.new_word_iterate_idx before new assigned:", self.new_word_iterate_idx)

        self.new_word_iterate_idx = last_new_word_idx

        print("self.new_word_iterate_idx after new assigned:", self.new_word_iterate_idx)

        # assign mutabilities and traits:
        self.interaction_traits = trait_dict

        # keep track of new-words database iteration:
        self.new_word_iterate_idx = last_new_word_idx
        # assign room definitions:
        self.room_definitions = dict()
        for type_def in new_rooms:
            type_def_dict = dict()
            for type_key, type_value in type_def.items():
                if not type_key == 'type_name':
                    type_def_dict[type_key] = type_value
            self.room_definitions[type_def['type_name']] = type_def_dict
        # assign entity definitions:
        self.entity_definitions = dict()
        for type_def in new_entities:
            type_def_dict = dict()
            for type_key, type_value in type_def.items():
                if not type_key == 'type_name':
                    type_def_dict[type_key] = type_value
            self.entity_definitions[type_def['type_name']] = type_def_dict
        # add ASP from PDDL to created action definitions:
        new_actions = augment_action_defs_with_asp(new_actions, self.interaction_traits)
        # assign action definitions:
        self.action_definitions = dict()
        for type_def in new_actions:
            type_def_dict = dict()
            for type_key, type_value in type_def.items():
                if not type_key == 'type_name':
                    type_def_dict[type_key] = type_value
            self.action_definitions[type_def['type_name']] = type_def_dict

        if 'add_basic_actions' in self.adv_type_def:
            if self.adv_type_def['add_basic_actions']:
                self._load_premade_action_definitions()

        # parse and assign domain definition:
        parsed_domain_definition_pddl = self.domain_def_parser.parse(new_domain)
        self.domain_def = self.domain_def_transformer.transform(parsed_domain_definition_pddl)

    def _replace_assign_new_word_definitions(self):
        # sample seed value for new-word replacement to not replace same definitions every time:
        replace_seed: int = self.rng.integers(9223372036854775808)
        # create new-words definitions:
        new_rooms, new_entities, new_actions, new_domain, \
        trait_dict, replacement_dict, last_new_word_idx = replace_new_words_definitions_set(
            self.new_word_iterate_idx,
            room_replace_n=self.adv_type_def['replace_counts']['rooms'],
            entity_replace_n=self.adv_type_def['replace_counts']['entities'],
            action_replace_n=self.adv_type_def['replace_counts']['actions'],
            seed=replace_seed)

        print("replacement_dict:", replacement_dict)
        self.replacement_dict = replacement_dict

        print("self.new_word_iterate_idx before new assigned:", self.new_word_iterate_idx)

        self.new_word_iterate_idx = last_new_word_idx

        print("self.new_word_iterate_idx after new assigned:", self.new_word_iterate_idx)

        # assign mutabilities and traits:
        self.interaction_traits = trait_dict

        # keep track of new-words database iteration:
        self.new_word_iterate_idx = last_new_word_idx
        # assign room definitions:
        self.room_definitions = dict()
        for type_def in new_rooms:
            type_def_dict = dict()
            for type_key, type_value in type_def.items():
                if not type_key == 'type_name':
                    type_def_dict[type_key] = type_value
            self.room_definitions[type_def['type_name']] = type_def_dict
        # assign entity definitions:
        self.entity_definitions = dict()
        for type_def in new_entities:
            type_def_dict = dict()
            for type_key, type_value in type_def.items():
                if not type_key == 'type_name':
                    type_def_dict[type_key] = type_value
            self.entity_definitions[type_def['type_name']] = type_def_dict

        # print("self.entity_definitions:", self.entity_definitions)

        # add ASP from PDDL to created action definitions:
        # new_actions = augment_action_defs_with_asp(new_actions, self.interaction_traits)

        # assign action definitions:
        self.action_definitions = dict()
        for type_def in new_actions:
            type_def_dict = dict()
            for type_key, type_value in type_def.items():
                if not type_key == 'type_name':
                    type_def_dict[type_key] = type_value
            self.action_definitions[type_def['type_name']] = type_def_dict

        if 'add_basic_actions' in self.adv_type_def:
            if self.adv_type_def['add_basic_actions']:
                self._load_premade_action_definitions()

        # TODO: inventory limit issues; either use no-limit base or adapt; no-limit will probably be quicker...

        # parse and assign domain definition:
        parsed_domain_definition_pddl = self.domain_def_parser.parse(new_domain)
        self.domain_def = self.domain_def_transformer.transform(parsed_domain_definition_pddl)

    def _generate_room_layouts_asp(self):
        """
        Generates an ASP encoding string to generate room layouts based on room definitions.
        Floors with support trait for all rooms included.
        """
        clingo_str = str()

        # generate with one room of each type:
        for room_type_name, room_type_values in self.room_definitions.items():
            # basic atoms:
            room_id = f"{room_type_name}1"  # default to 'kitchen1' etc
            type_atom = f"room({room_id},{room_type_name})."
            # Ex: room(kitchen1,kitchen) = there is a room with internal ID kitchen1 which has the room type kitchen
            clingo_str += "\n" + type_atom

            add_floors = True
            if 'add_floors_to_rooms' in self.adv_type_def['initial_state_config']:
                if not self.adv_type_def['initial_state_config']['add_floors_to_rooms']:
                    add_floors = False

            if add_floors:
                # add floor to room:
                floor_id = f"{room_id}floor1"
                floor_atom = f"type({floor_id},floor)."
                # Ex: type(kitchen1floor,floor) = there is an entity kitchen1floor which has the entity type floor
                clingo_str += "\n" + floor_atom
                # add at() for room floor:
                floor_at = f"at({floor_id},{room_id})."
                # Ex: at(kitchen1floor,kitchen1) = at kitchen1 there is a floor with internal ID kitchen1floor
                clingo_str += "\n" + floor_at
                # add support trait atom for floor:
                floor_support = f"support({floor_id})."
                # Ex: support(kitchen1floor) = the entity with internal ID kitchen1floor can support moveable entities,
                # meaning it can be the second argument of put actions, and there can be on(X,kitchen1floor) facts
                clingo_str += "\n" + floor_support

            # add ceilings to defined room types:
            if 'add_ceiling_rooms' in self.adv_type_def['initial_state_config']:
                # print("add ceilings key in adv_type_def")
                if self.adv_type_def['initial_state_config']['add_ceiling_rooms']:
                    # print("add ceilings value in adv_type_def is truthy")
                    # print(self.adv_type_def['initial_state_config']['add_ceiling_rooms'])
                    if room_type_name in self.adv_type_def['initial_state_config']['add_ceiling_rooms']:
                        # print(f"{room_type_name} in ceiling add list!")
                        # add ceiling to room:
                        ceiling_id = f"{room_id}ceiling1"
                        ceiling_atom = f"type({ceiling_id},ceiling)."
                        clingo_str += "\n" + ceiling_atom
                        # add at() for room floor:
                        ceiling_at = f"at({ceiling_id},{room_id})."
                        clingo_str += "\n" + ceiling_at
                        # add support trait atom for ceiling:
                        ceiling_support = f"support({ceiling_id})."
                        clingo_str += "\n" + ceiling_support

            # add exit rule:
            # room definitions contain a list of possible adjacent rooms
            permitted_exits_list = list()
            for exit_target in room_type_values['exit_targets']:
                exit_target_permit = f"exit(ROOM,TARGET):room(ROOM,{room_type_name}),room(TARGET,{exit_target})"
                permitted_exits_list.append(exit_target_permit)
            permitted_exits = ";".join(permitted_exits_list)
            exit_rule = "1 { $PERMITTEDEXITS$ } $MAXCONNECTIONS$."
            exit_rule = exit_rule.replace("$PERMITTEDEXITS$", permitted_exits)
            exit_rule = exit_rule.replace("$MAXCONNECTIONS$", str(room_type_values['max_connections']))
            clingo_str += "\n" + exit_rule
        # exit pairing rule:
        pair_exits = True  # exits are paired by default
        # unpaired exits for witch house:
        if 'unpaired_exits' in self.adv_type_def['initial_state_config']:
            if self.adv_type_def['initial_state_config']['unpaired_exits']:
                pair_exits = False
        if pair_exits:
            # this makes sure that all passages are usable from both sides
            exit_pairing_rule = "exit(ROOM,TARGET) :- exit(TARGET,ROOM)."
            clingo_str += "\n" + exit_pairing_rule
        # add rule assuring all rooms are reachable from each other:
        reachable_rule = ("reachable(ROOM,TARGET) :- exit(ROOM,TARGET). "
                          # "reachable(ROOM,TARGET) :- reachable(TARGET,ROOM). "
                          "reachable(ROOM,TARGET) :- reachable(ROOM,TARGET1), reachable(TARGET1,TARGET), ROOM != TARGET. "
                          ":- room(ROOM,_), room(TARGET,_), ROOM != TARGET, not reachable(ROOM,TARGET).")
        clingo_str += "\n" + reachable_rule

        return clingo_str

    def _generate_initial_states_asp(self, room_layout_facts: list):
        """
        Generates an ASP encoding string to generate initial world states based on a room layout and entity definitions.
        """
        clingo_str = str()

        # convert/add room layout facts:
        cur_layout = "\n".join([fact + "." for fact in room_layout_facts])
        clingo_str += cur_layout

        # add player type fact:
        player_fact = "type(player1,player)."
        clingo_str += "\n" + player_fact

        # add rule for random player start location:
        player_location_rule = "1 { at(player1,ROOM):room(ROOM,_) } 1."
        # = there can be at() facts for the player for each room and there must be exactly one at() fact for the player
        # witch house constant starting location:
        if 'constant_start_room' in self.adv_type_def['initial_state_config']:
            if self.adv_type_def['initial_state_config']['constant_start_room']:
                player_location_rule = f"at(player1,{self.adv_type_def['initial_state_config']['constant_start_room']})."
        # print("player_location_rule:", player_location_rule)
        clingo_str += "\n" + player_location_rule

        # print(self.entity_definitions)

        for entity_type_name, entity_type_values in self.entity_definitions.items():
            # print(entity_type_name, entity_type_values)
            # do not create type fact for entities to be omitted from initial states:
            if "omit_from_initial_state" in entity_type_values:
                if entity_type_values["omit_from_initial_state"]:
                    continue
            if "standard_locations" in entity_type_values:
                # entity definitions contain a list of rooms the entity type is allowed to be at
                # basic atoms:
                entity_id = f"{entity_type_name}1"  # default to 'apple1' etc
                # create type fact atom for the single entity type instance:
                type_atom = f"type({entity_id},{entity_type_name})."
                # add type atom to asp encoding:
                clingo_str += "\n" + type_atom

                # location rule:
                permitted_location_list = list()

                if entity_type_values["standard_locations"]:
                    for location in entity_type_values['standard_locations']:
                        permitted_location = f"at(ENTITY,ROOM):type(ENTITY,{entity_type_name}),room(ROOM,{location})"
                        permitted_location_list.append(permitted_location)
                    permitted_locations_str = ";".join(permitted_location_list)

                    location_rule = "1 { $PERMITTEDLOCATIONS$ } 1."
                    location_rule = location_rule.replace("$PERMITTEDLOCATIONS$", permitted_locations_str)
                    clingo_str += "\n" + location_rule

                if "traits" in entity_type_values:
                    # add atoms for all traits of this entity type:
                    for trait in entity_type_values['traits']:
                        trait_atom = f"{trait}({entity_id})."
                        clingo_str += "\n" + trait_atom

                    if "needs_support" in entity_type_values['traits'] and entity_type_values["standard_locations"]:
                        # on/in rule:
                        # entities that require being on a support can be on any support in the room they are at
                        support_rule = "1 { on($ENTITY$,SUPPORT):at($ENTITY$,ROOM),at(SUPPORT,ROOM),support(SUPPORT);in($ENTITY$,CONTAINER):at($ENTITY$,ROOM),at(CONTAINER,ROOM),container(CONTAINER) } 1."
                        support_rule = support_rule.replace("$ENTITY$", entity_id)
                        clingo_str += "\n" + support_rule

                    if "openable" in entity_type_values['traits']:
                        closed_atom = f"closed({entity_id})."
                        clingo_str += "\n" + closed_atom

                    if "assign_mutable_states_from_set" in self.adv_type_def['initial_state_config']:
                        if self.adv_type_def['initial_state_config']["assign_mutable_states_from_set"]:
                            # get entity mutability sets:
                            possible_mutable_lists = list()
                            for trait in entity_type_values['traits']:
                                # print(entity_type_values['repr_str'], ":", trait)
                                mutability_mutables = [predicate['predicate_id'] for predicate in self.domain_def['predicates'] if predicate['mutability'] == trait]
                                # print(mutability_mutables)
                                # print()
                                # don't add single mutable fact of irreversible-singular mutability, entity has no initial fact tied to this type of mutability
                                if self.interaction_traits[trait]['mutable_set_type'] == 'singular':
                                    continue
                                possible_mutable_lists.append(mutability_mutables)
                            # assign mutable states from mutability set:
                            for possible_mutable_list in possible_mutable_lists:
                                possible_mutable_asp_facts = [f"{mutable}({entity_id})" for mutable in possible_mutable_list]
                                possible_mutable_asp = "1 { POSSIBLE_MUTABLES } 1."
                                possible_mutable_asp = possible_mutable_asp.replace("POSSIBLE_MUTABLES", ";".join(possible_mutable_asp_facts))
                                clingo_str += "\n" + possible_mutable_asp


                # no adjectives were used in v1/v2.1/v2.2/v2.3 adventures
                if not self.adv_type_def['initial_state_config']["entity_adjectives"] == "none":
                    if "possible_adjs" in entity_type_values:
                        # adjective rule:
                        possible_adj_list = list()
                        for possible_adj in entity_type_values["possible_adjs"]:
                            possible_adj_str = f"adj({entity_id},{possible_adj})"
                            possible_adj_list.append(possible_adj_str)
                        possible_adjs = ";".join(possible_adj_list)
                        if self.adv_type_def['initial_state_config']["entity_adjectives"] == "optional":
                            adj_rule = "0 { $POSSIBLEADJS$ } 1."
                        elif self.adv_type_def['initial_state_config']["entity_adjectives"] == "all":
                            adj_rule = "1 { $POSSIBLEADJS$ } 1."
                        adj_rule = adj_rule.replace("$POSSIBLEADJS$", possible_adjs)
                        clingo_str += "\n" + adj_rule

                        # make sure that same-type entities do not have same adjective:
                        diff_adj_rule = ":- adj(ENTITY1,ADJ), adj(ENTITY2,ADJ), type(ENTITY1,TYPE), type(ENTITY2,TYPE), ENTITY1 != ENTITY2."
                        clingo_str += "\n" + diff_adj_rule

        return clingo_str

    def _generate_goal_facts(self, initial_world_state):
        """
        Generate goal facts based on task type and given adventure initial world state.
        Task types:
            'deliver': Bring takeable objects to support or container; goal facts are of type 'on' or 'in'.
        Only 'deliver' with three objects in v1.
        """
        id_to_type_dict: dict = dict()

        # convert fact strings to tuples:
        initial_facts = [fact_str_to_tuple(fact) for fact in initial_world_state]
        # iterate over initial world state, add fixed basic facts, add turn facts for changeable facts
        for fact in initial_facts:
            if fact[0] == "type":
                # print("goal fact gen current type fact:", fact)
                id_to_type_dict[fact[1]] = {'type': fact[2],
                                            'repr_str': self.entity_definitions[fact[2]]['repr_str']}
                if 'traits' in self.entity_definitions[fact[2]]:
                    id_to_type_dict[fact[1]]['traits'] = self.entity_definitions[fact[2]]['traits']
            if fact[0] == "room":
                id_to_type_dict[fact[1]] = {'type': fact[2],
                                            'repr_str': self.room_definitions[fact[2]]['repr_str']}
                if 'traits' in self.room_definitions[fact[2]]:
                    id_to_type_dict[fact[1]]['traits'] = self.room_definitions[fact[2]]['traits']

        task_config = self.adv_type_def['task_config']
        # print("task_config:", task_config)
        goal_count = self.adv_type_def['goal_count']

        if task_config['task'] == "deliver":
            # get initial in/on of takeables:
            takeables: dict = dict()
            holders: dict = dict()
            for fact in initial_facts:
                if fact[0] == "takeable":
                    if fact[1] not in takeables:
                        takeables[fact[1]] = {'type': id_to_type_dict[fact[1]]['type']}
                    else:
                        takeables[fact[1]]['type'] = id_to_type_dict[fact[1]]['type']
                if fact[0] in ["on", "in"]:
                    if fact[1] not in takeables:
                        takeables[fact[1]] = {'state': fact[0], 'holder': fact[2]}
                    else:
                        takeables[fact[1]]['state'] = fact[0]
                        takeables[fact[1]]['holder'] = fact[2]
                if fact[0] in ["container", "support"]:
                    if fact[1] not in holders:
                        holders[fact[1]] = {'type': id_to_type_dict[fact[1]]['type'], 'holder_type': fact[0]}
                    else:
                        holders[fact[1]]['type'] = id_to_type_dict[fact[1]]['type']
                        holders[fact[1]]['holder_type'] = fact[0]

            if not task_config['deliver_to_floor']:
                bad_holders: list = list()
                for holder, holder_values in holders.items():
                    if holder_values['type'] == "floor":
                        bad_holders.append(holder)
                for bad_holder in bad_holders:
                    del holders[bad_holder]

            possible_destinations: dict = dict()

            for takeable, takeable_values in takeables.items():
                for holder, holder_values in holders.items():
                    if not takeable_values['holder'] == holder:
                        if takeable not in possible_destinations:
                            possible_destinations[takeable] = [holder]
                        else:
                            possible_destinations[takeable].append(holder)

            all_possible_goals: list = list()
            for takeable, destinations in possible_destinations.items():
                for destination in destinations:
                    if holders[destination]['holder_type'] == "container":
                        pred_type = "in"
                    elif holders[destination]['holder_type'] == "support":
                        pred_type = "on"
                    goal_str: str = f"{pred_type}({takeable},{destination})"
                    goal_tuple: tuple = (pred_type, takeable, destination)
                    all_possible_goals.append(goal_tuple)
            goal_permutations = list(permutations(all_possible_goals, goal_count))
            # prevent goal combos with same object at different locations:
            goal_combos = list()
            for goal_combo in goal_permutations:
                duplicate = False
                goal_objects = list()
                goal_strs = list()
                for goal in goal_combo:
                    if goal[1] not in goal_objects:
                        goal_objects.append(goal[1])
                        goal_strs.append(f"{goal[0]}({goal[1]},{goal[2]})")
                    else:
                        duplicate = True
                if not duplicate:
                    goal_combos.append(goal_strs)

        elif task_config['task'] == "new-words_deliver":
            # print("task is new-words_deliver")
            # get initial in/on of takeables:
            takeables: dict = dict()
            holders: dict = dict()
            for fact in initial_facts:
                if fact[0] == "takeable":
                    if fact[1] not in takeables:
                        takeables[fact[1]] = {'type': id_to_type_dict[fact[1]]['type']}
                    else:
                        takeables[fact[1]]['type'] = id_to_type_dict[fact[1]]['type']
                if fact[0] in ["on", "in"]:
                    if fact[1] not in takeables:
                        takeables[fact[1]] = {'state': fact[0], 'holder': fact[2]}
                    else:
                        takeables[fact[1]]['state'] = fact[0]
                        takeables[fact[1]]['holder'] = fact[2]
                if fact[0] in ["container", "support"]:
                    if fact[1] not in holders:
                        holders[fact[1]] = {'type': id_to_type_dict[fact[1]]['type'], 'holder_type': fact[0]}
                    else:
                        holders[fact[1]]['type'] = id_to_type_dict[fact[1]]['type']
                        holders[fact[1]]['holder_type'] = fact[0]

            if not task_config['deliver_to_floor']:
                bad_holders: list = list()
                for holder, holder_values in holders.items():
                    if holder_values['type'] == "floor":
                        bad_holders.append(holder)
                for bad_holder in bad_holders:
                    del holders[bad_holder]

            possible_destinations: dict = dict()

            for takeable, takeable_values in takeables.items():
                for holder, holder_values in holders.items():
                    if not takeable_values['holder'] == holder:
                        if takeable not in possible_destinations:
                            possible_destinations[takeable] = [holder]
                        else:
                            possible_destinations[takeable].append(holder)

            # print("possible_destinations:", possible_destinations)

            all_possible_goals: list = list()
            goal_takeable_count: int = 0
            goal_in_count: int = 0
            goal_on_count: int = 0
            for takeable, destinations in possible_destinations.items():
                # print("takeable:", takeable)
                for destination in destinations:
                    # print("destination:", destination)
                    takeable_replaced: bool = False

                    if takeable[:-1] in self.replacement_dict['entities'].values():
                        # print(f"takeable goal item {takeable} is replaced new-word!")
                        takeable_replaced = True
                        goal_takeable_count += 1

                    if holders[destination]['holder_type'] == "container":
                        pred_type = "in"
                    elif holders[destination]['holder_type'] == "support":
                        pred_type = "on"

                    receptacle_replaced = False
                    if destination[:-1] in self.replacement_dict['entities'].values() and goal_takeable_count > 1:
                        # print(f"destination goal item {destination} is replaced new-word!")

                        # make sure there's at least one 'in' and 'on' goal each if fitting receptacles are assured:
                        # replacing sufficient amount of receptacles is assured by new-word definition creation
                        if self.adv_type_def['replace_counts']['entities'] > 4:
                            # print("more than four entity replacement, splitting targets")
                            if goal_in_count == 0 and pred_type == "in":
                                # print("goal_in_count == 0 and pred_type == 'in'")
                                receptacle_replaced = True
                                goal_in_count += 1
                            if goal_on_count == 0 and pred_type == "on":
                                # print("goal_on_count == 0 and pred_type == 'on'")
                                receptacle_replaced = True
                                goal_on_count += 1
                            if goal_in_count >= 1 and goal_on_count >= 1:
                                # print(f"more than one new-word destination each assigned case: {goal_in_count} 'in', {goal_on_count} 'on'")
                                receptacle_replaced = True
                        else:
                            receptacle_replaced = True

                    goal_str: str = f"{pred_type}({takeable},{destination})"
                    goal_tuple: tuple = (pred_type, takeable, destination)
                    if takeable_replaced or receptacle_replaced:
                        if self.adv_type_def['replace_counts']['entities'] > 4:
                            if goal_in_count >= 1 and goal_on_count >= 1:
                                all_possible_goals.append(goal_tuple)
                                # print(f"added goal tuple: {goal_tuple}")
                        else:
                            all_possible_goals.append(goal_tuple)
                            # print(f"added goal tuple: {goal_tuple}")
                    # all_possible_goals.append(goal_tuple)

                    # print()

            # print("all_possible_goals:", all_possible_goals)

            goal_permutations = list(permutations(all_possible_goals, goal_count))
            # print("goal_permutations:", goal_permutations)

            good_goal_permutations: list = list()
            # make sure there's at least one 'in' and 'on' goal with new-word destination in goal set:
            if self.adv_type_def['replace_counts']['entities'] > 4:
                for goal_combo in goal_permutations:
                    in_new_word = False
                    on_new_word = False
                    for goal in goal_combo:
                        if goal[2][:-1] in self.replacement_dict['entities'].values():
                            if goal[0] == 'in':
                                in_new_word = True
                            if goal[0] == 'on':
                                on_new_word = True
                    if in_new_word and on_new_word:
                        good_goal_permutations.append(goal_combo)
            else:
                good_goal_permutations = goal_permutations

            # print("good_goal_permutations:", good_goal_permutations)

            # prevent goal combos with same object at different locations:
            goal_combos = list()
            for goal_combo in good_goal_permutations:
                duplicate = False
                goal_objects = list()
                goal_strs = list()
                for goal in goal_combo:
                    if goal[1] not in goal_objects:
                        goal_objects.append(goal[1])
                        goal_strs.append(f"{goal[0]}({goal[1]},{goal[2]})")
                    else:
                        duplicate = True
                if not duplicate:
                    goal_combos.append(goal_strs)

        elif task_config['task'] == "new-word_states":
            # TODO?: externalize (more) goal parameters?

            # get mutable predicates from domain:

            # print(self.domain_def['predicates'])

            mutables = [predicate['predicate_id'] for predicate in self.domain_def['predicates']]
            # print(mutables)
            mutabilities = dict()
            for predicate in self.domain_def['predicates']:
                # print("predicate:", predicate)
                if predicate['mutability'] not in mutabilities:
                    mutabilities[predicate['mutability']] = [predicate['predicate_id']]
                else:
                    mutabilities[predicate['mutability']].append(predicate['predicate_id'])
            # print(mutabilities)


            # TODO: convert underscore mutabilities to dashed for surface

            # get initial mutable states of entities:
            mutable_state_entities: dict = dict()
            for fact in initial_facts:
                if fact[0] in mutabilities:
                    # print(fact)
                    if fact[1] not in mutable_state_entities:
                        mutable_state_entities[fact[1]] = {'type': id_to_type_dict[fact[1]]['type'],
                                                           'mutabilities': [fact[0]]}
                    else:
                        mutable_state_entities[fact[1]]['type'] = id_to_type_dict[fact[1]]['type']
                        mutable_state_entities[fact[1]]['mutabilities'].append(fact[0])

            for fact in initial_facts:
                if fact[1] in mutable_state_entities:
                    # print("fact:", fact)
                    mutable_fact_dict = dict()
                    for mutability in mutable_state_entities[fact[1]]['mutabilities']:
                        # print(f"checked mutability of {fact[1]}:", mutability)
                        if fact[0] in mutabilities[mutability]:
                            # print(f"checked {mutability}:", mutabilities[mutability])
                            mutable_fact_dict[mutability] = fact[0]
                    # print(mutable_fact_dict)
                    if 'mutable_facts' not in mutable_state_entities[fact[1]]:
                        mutable_state_entities[fact[1]]['mutable_facts'] = mutable_fact_dict
                    else:
                        mutable_state_entities[fact[1]]['mutable_facts'] = mutable_state_entities[fact[1]]['mutable_facts'] | mutable_fact_dict

            all_possible_goals = list()
            for mutable_state_entity_id, mutable_state_entity_values in mutable_state_entities.items():
                # print("mutable entity:", mutable_state_entity_id, mutable_state_entity_values)
                possible_target_states = list()
                cur_mut_facts = mutable_state_entity_values['mutable_facts']
                for mutability, cur_mut_fact in cur_mut_facts.items():
                    # get set of mutable states this entity does not have initially:
                    possible_target_states += [mutable for mutable in mutabilities[mutability] if not mutable == cur_mut_fact]
                    # print(f"possible target states for {mutability}:", [mutable for mutable in mutabilities[mutability] if not mutable == cur_mut_fact])
                # handle irreversible-singular mutability:
                for mutability in mutable_state_entity_values['mutabilities']:
                    if self.interaction_traits[mutability]['mutable_set_type'] == 'singular':
                        possible_target_states += self.interaction_traits[mutability]['mutable_states']  # must be single
                        # print(f"possible target states for {mutability}:", self.interaction_traits[mutability]['mutable_states'])
                # print(f"all possible target states for {mutable_state_entity_id}:", possible_target_states)
                for possible_target_state in possible_target_states:
                    goal_tuple: tuple = (possible_target_state, mutable_state_entity_id)
                    all_possible_goals.append(goal_tuple)
            # print("all_possible_goals:", all_possible_goals)

            goal_permutations = list(permutations(all_possible_goals, goal_count))
            # prevent goal combos with same object and same mutability:
            goal_combos = list()
            for goal_combo in goal_permutations:
                # print("goal combo:", goal_combo)
                duplicate = False
                goal_mutabilities = list()
                goal_strs = list()
                for goal in goal_combo:
                    # print("goal in goal_combo:", goal)
                    goal_object_mutabilities = mutable_state_entities[goal[1]]['mutabilities']
                    for mutability in goal_object_mutabilities:
                        mutable_states = self.interaction_traits[mutability]['mutable_states']
                        if goal[0] in mutable_states:
                            goal_mutability = (mutability, goal[1])
                            """
                            print("mutability:", self.interaction_traits[mutability])
                            mutability_irreversible = self.interaction_traits[mutability][
                                                          'interaction'] == 'irreversible'
                            print("mutability is irreversible:", mutability_irreversible)
                            """

                    if goal_mutability not in goal_mutabilities:

                        for goal_mut2 in goal_mutabilities:
                            if goal_mutability[0] == goal_mut2[0]:
                                duplicate = True

                        if not duplicate:
                            goal_mutabilities.append(goal_mutability)
                            goal_strs.append(f"{goal[0]}({goal[1]})")
                    else:
                        duplicate = True
                if not duplicate:
                    # print("added goal combo strs:", goal_strs)
                    goal_combos.append(goal_strs)
                    # print("goal_mutabilities:", goal_mutabilities)
                    # print()

        elif task_config['task'] == "potion":
            # potion brewing only requires the finished potion to be in the kitchen
            # goal_combos = [["at(potion1,_)"]]
            goal_combos = [["at(potion1,kitchen1)"]]

        return goal_combos

    def _initialize_adventure_turns_asp(self, initial_world_state, events_list: list = []):
        """
        Set up initial world state and create ASP encoding of mutable facts.
        Turn facts have _t in the fact/atom type, and their first value is the turn at which they are true.
        Mutable facts are defined in the adventure type definition.
        """
        if 'mutable_fact_types' in self.adv_type_def:
            mutable_fact_types: list = self.adv_type_def["mutable_fact_types"]
        elif self.adv_type_def['initial_state_config']['assign_mutable_states_from_set']:
            mutable_fact_types = ["at"]
            for mutability in self.interaction_traits.values():
                # print("mutability:", mutability)
                mutable_fact_types += mutability['mutable_states']
        else:
            raise ValueError("Mutable states are not defined in adventure type definition, can't create solving ASP!")

        clingo_str = str()

        self.id_to_type_dict: dict = dict()

        # convert fact strings to tuples:
        self.initial_facts = [fact_str_to_tuple(fact) for fact in initial_world_state]
        # iterate over initial world state, add fixed basic facts, add turn facts for changeable facts
        for fact in self.initial_facts:
            # set up id_to_type:
            if fact[0] == "type":
                self.id_to_type_dict[fact[1]] = {'type': fact[2],
                                                 'repr_str': self.entity_definitions[fact[2]]['repr_str']}
                if 'traits' in self.entity_definitions[fact[2]]:
                    self.id_to_type_dict[fact[1]]['traits'] = self.entity_definitions[fact[2]]['traits']
            if fact[0] == "room":
                self.id_to_type_dict[fact[1]] = {'type': fact[2],
                                                 'repr_str': self.room_definitions[fact[2]]['repr_str']}
                if 'traits' in self.room_definitions[fact[2]]:
                    self.id_to_type_dict[fact[1]]['traits'] = self.room_definitions[fact[2]]['traits']
            # set up per-turn mutable facts at turn 0:
            if fact[0] in mutable_fact_types:
                # add turn 0 turn fact atom:
                if len(fact) == 3:
                    turn_atom = f"{fact[0]}_t(0,{fact[1]},{fact[2]})."
                    # Ex: in_t(0,apple1,refrigerator1) = the apple is in the refrigerator at turn 0
                    clingo_str += "\n" + turn_atom
                if len(fact) == 2:
                    turn_atom = f"{fact[0]}_t(0,{fact[1]})."
                    # Ex: closed_t(0,refrigerator1) = the refrigerator is closed at turn 0
                    clingo_str += "\n" + turn_atom
            else:
                # add constant fact atom:
                const_atom = f"{fact_tuple_to_str(fact)}."
                # Ex: type(apple1,apple) = the entity with ID apple1 is of type apple (and will always be)
                clingo_str += "\n" + const_atom

        if events_list:
            for event in events_list:
                clingo_str += "\n" + event['asp']

        return clingo_str

    def _solve_optimally_asp(self, initial_world_state, goal_facts: list, return_only_actions: bool = True,
                             events_list: list = []) -> str:
        """
        Generates an optimal solution to an adventure.
        :param initial_world_state: Initial world state fact list.
        :param goal_facts: List of goal facts in string format, ie 'on(sandwich1,table1)'.
        :param return_only_actions: Return only a list of action-at-turn atoms. If False, ALL model atoms are returned.
        :return: ASP adventure solving encoding.
        """
        # get turn limit from adventure type definition:
        turn_limit: int = self.adv_type_def["optimal_solver_turn_limit"]

        clingo_str = str()

        # add turn generation and limit first:
        turns_template: str = self.clingo_templates["turns"]
        turns_clingo = turns_template.replace("$TURNLIMIT$", str(turn_limit))
        clingo_str += "\n" + turns_clingo

        # add initial world state facts:
        if events_list:
            initial_state_clingo = self._initialize_adventure_turns_asp(initial_world_state, events_list=events_list)
        else:
            initial_state_clingo = self._initialize_adventure_turns_asp(initial_world_state)
        clingo_str += "\n" + initial_state_clingo

        # add actions:
        for action_name, action_def in self.action_definitions.items():
            action_asp = action_def['asp']  # action ASP encodings were manually created

            # TODO: create action ASP from PDDL definitions

            clingo_str += "\n" + action_asp

        # add action/turn restraints:
        actions_turns_clingo: str = self.clingo_templates["action_limits"]  # -> only one action per turn
        clingo_str += "\n" + actions_turns_clingo

        # add goals:
        for goal in goal_facts:
            goal_tuple = fact_str_to_tuple(goal)
            if len(goal_tuple) == 2:
                goal_template: str = self.clingo_templates["goal_1"]
                goal_clingo = goal_template.replace("$PREDICATE$", goal_tuple[0])
                goal_clingo = goal_clingo.replace("$THING$", goal_tuple[1])
            if len(goal_tuple) == 3:
                goal_template: str = self.clingo_templates["goal_2"]
                goal_clingo = goal_template.replace("$PREDICATE$", goal_tuple[0])
                goal_clingo = goal_clingo.replace("$THING$", goal_tuple[1])
                goal_clingo = goal_clingo.replace("$TARGET$", goal_tuple[2])
            clingo_str += "\n" + goal_clingo

        # add optimization:
        minimize_clingo = self.clingo_templates["minimize"]  # -> least number of turns is optimal
        clingo_str += "\n" + minimize_clingo

        # add output only actions:
        # this omits all intermediate information and full fact set as only the optimal action sequence is needed
        if return_only_actions:
            only_actions_clingo = self.clingo_templates["return_only_actions"]
            clingo_str += "\n" + only_actions_clingo

        return clingo_str

    def _pre_explore_asp(self, initial_world_state, goal_facts: list, return_only_actions: bool = True,
                         limit_to_goal_object_rooms: bool = False) -> str:
        """
        Generates a pre-exploration action sequence for an adventure.
        This sequence of actions will have the player visit all rooms in the least amount of turns, but this can be
        limited to rooms with task objects.
        Args:
            initial_world_state: Initial world state fact list.
            goal_facts: List of goal facts in string format, ie 'on(sandwich1,table1)'.
            return_only_actions: Return only a list of action-at-turn atoms. If False, ALL model atoms are returned.
            limit_to_goal_object_rooms: If True, only rooms with task objects are visited. Default: False.
        Returns:
            ASP pre-explore encoding.
        """
        # get turn limit from adventure type definition:
        turn_limit: int = self.adv_type_def["optimal_solver_turn_limit"]

        clingo_str = str()

        # add turn generation and limit first:
        turns_template: str = self.clingo_templates["turns"]
        turns_clingo = turns_template.replace("$TURNLIMIT$", str(turn_limit))
        clingo_str += "\n" + turns_clingo

        # add initial world state facts:
        initial_state_clingo = self._initialize_adventure_turns_asp(initial_world_state)
        clingo_str += "\n" + initial_state_clingo

        # add actions:
        for action_name, action_def in self.action_definitions.items():
            action_asp = action_def['asp']  # action ASP encodings were manually created or generated with new-words
            clingo_str += "\n" + action_asp

        # add action/turn restraints:
        actions_turns_clingo: str = self.clingo_templates["action_limits"]  # -> only one action per turn
        clingo_str += "\n" + actions_turns_clingo

        # add rooms visited tracking:
        visited_clingo: str = self.clingo_templates["pre_explore_visited"]  # -> room is visited if player is ever at it
        clingo_str += "\n" + visited_clingo

        if limit_to_goal_object_rooms:
            # add goal object rooms needing to be visited:
            for goal in goal_facts:
                goal_tuple = fact_str_to_tuple(goal)
                goal_template: str = self.clingo_templates["pre_explore_goal"]
                goal_clingo = goal_clingo.replace("$THING$", goal_tuple[1])

                clingo_str += "\n" + goal_clingo
        else:
            # add all rooms visited requirement:
            visited_all_clingo: str = self.clingo_templates["pre_explore_all"]  # -> all rooms need to be visited
            clingo_str += "\n" + visited_all_clingo

        # add optimization:
        minimize_clingo = self.clingo_templates["minimize"]  # -> least number of turns is optimal
        clingo_str += "\n" + minimize_clingo

        # add output only actions:
        # this omits all intermediate information and full fact set as only the optimal action sequence is needed
        if return_only_actions:
            only_actions_clingo = self.clingo_templates["return_only_actions"]
            clingo_str += "\n" + only_actions_clingo

        return clingo_str

    def _convert_adventure_solution(self, adventure_solution: str):
        """
        Convert a raw solution string into list of IF commands and get additional information. Expects only-actions raw
        string.
        Returns both a 'abstract' tuple format of the optimal action sequence and action command strings to pass to the
        IF interpreter directly, as well as the length of the optimal action sequence.
        """
        actions_list: list = adventure_solution.split()
        action_tuples = [convert_action_to_tuple(action) for action in actions_list]
        action_tuples.sort(key=lambda turn: turn[0])

        actions_abstract: list = list()
        action_commands: list = list()
        for action_tuple in action_tuples:
            if len(action_tuple) == 3:
                command: str = f"{action_tuple[1]} {self.id_to_type_dict[action_tuple[2]]['repr_str']}"
                abstract_action = [action_tuple[1], action_tuple[2]]
            if len(action_tuple) == 4:
                if action_tuple[1] == "put":
                    if "support" in self.id_to_type_dict[action_tuple[3]]['traits']:
                        command: str = (f"{action_tuple[1]} {self.id_to_type_dict[action_tuple[2]]['repr_str']} "
                                        f"on {self.id_to_type_dict[action_tuple[3]]['repr_str']}")
                    if "container" in self.id_to_type_dict[action_tuple[3]]['traits']:
                        command: str = (f"{action_tuple[1]} {self.id_to_type_dict[action_tuple[2]]['repr_str']} "
                                        f"in {self.id_to_type_dict[action_tuple[3]]['repr_str']}")
                else:
                    command: str = (f"{action_tuple[1]} {self.id_to_type_dict[action_tuple[2]]['repr_str']} "
                                    f"{self.id_to_type_dict[action_tuple[3]]['repr_str']}")
                abstract_action = [action_tuple[1], action_tuple[2], action_tuple[3]]
            action_commands.append(command)
            actions_abstract.append(abstract_action)

        return actions_abstract, len(action_tuples), action_commands

    def _generate_room_layouts(self, limit: int = 128):
        # init room layout clingo controller:
        room_layout_clingo: Control = Control(["0"])  # ["0"] argument to return all models
        # generate room layout ASP encoding:
        room_layout_asp: str = self._generate_room_layouts_asp()
        # print(room_layout_asp)
        # add room layout ASP encoding to clingo:
        room_layout_clingo.add(room_layout_asp)
        # ground controller:
        room_layout_clingo.ground()
        # solve for all room layouts:
        room_layouts = list()
        layout_count: int = 0
        with room_layout_clingo.solve(yield_=True) as solve:
            for model in solve:
                if layout_count <= limit:
                    room_layouts.append(model.__str__())
                    layout_count += 1
                else:
                    break
        # convert room layout clingo models:
        result_layouts = list()
        for result_layout in room_layouts:
            room_layout_fact_list = result_layout.split()
            # remove 'reachable' helper atoms:
            room_layout_fact_list = [fact for fact in room_layout_fact_list if "reachable" not in fact]
            # print(room_layout_fact_list)
            result_layouts.append(room_layout_fact_list)

        return result_layouts

    def _generate_initial_states(self, result_layouts, initial_states_per_layout):
        initial_states = list()
        # iterate over room layouts:
        for room_layout in result_layouts:
            # init initial state clingo controller:
            initial_states_clingo: Control = Control(["0"])  # ["0"] argument to return all models
            # generate initial state ASP encoding:
            cur_initial_states_asp = self._generate_initial_states_asp(room_layout)
            # print("cur_initial_states_asp:", cur_initial_states_asp)
            # print()
            # add initial state ASP encoding to clingo:
            initial_states_clingo.add(cur_initial_states_asp)
            # ground controller:
            initial_states_clingo.ground()
            # solve for all room layouts:
            initial_states_per_layout_count: int = 0
            with initial_states_clingo.solve(yield_=True) as solve:
                satisfiable = str(solve.get())
                if satisfiable == "SAT":
                    # print("intial state generation clingo solvable")
                    pass
                else:
                    # print("!!! intial state generation clingo NOT solvable !!!")
                    pass
                for model in solve:
                    # print("initial state solve model:\n", model)
                    # print("initial_states_per_layout_count:", initial_states_per_layout_count)
                    # print()
                    if initial_states_per_layout_count <= initial_states_per_layout:
                        initial_states.append(model.__str__().split())
                        initial_states_per_layout_count += 1
                    else:
                        break

        return initial_states

    def generate_adventures(self, initial_states_per_layout: int = 2, initial_state_picking: str = "iterative",
                            initial_state_limit: int = 30,
                            adventures_per_initial_state: int = 1,
                            goal_set_picking: str = "iterative",
                            target_adventure_count: int = 16,
                            save_to_file: bool = True, indent_output_json: bool = True):
        """
        Generate raw adventures based on various parameters. Main purpose of the parameters is to limit the runtime of
        adventure generation - even for simple v1 deliver-three without adjectives the number of possible adventures is
        highly exponential, and exhaustive generation would take a very long time.
        The number of possible room layouts is limited based on the basic/home room definitions, so it is not
        additionally limited here.
        This method uses all ASP encoding strings created by other methods of this class.
        :param initial_states_per_layout: How many initial world states are generated per room layout.
        :param initial_state_limit: The maximum number of initial states to generate. This number should be kept low, as
            it is the main limiter preventing excessive computational resource use.
        :param initial_state_picking: Method to pick from all possible goal states:
            "iterate" - Picks initial states from the first available iteratively until initial_state_limit is reached.
            "random" - Picks random initial states from all available until initial_state_limit is reached.
        :param adventures_per_initial_state: How many adventures to generate for each initial state.
        :param goal_set_picking: Method to pick from all possible goal states:
            "iterate" - Picks goal sets from the first permutation iteratively until goals_per_adventure is met.
            "random" - Picks random goal sets from all permutations until goals_per_adventure is met.
        :param save_to_file: File name for saving generated adventures. If empty string, generated adventures will not
            be saved.
        :param indent_output_json: If True, raw adventures JSON saved will be indented for readability.
        """
        print(self.adv_type)
        print(self.adv_type_def)

        task_config: dict = self.adv_type_def["task_config"]
        min_optimal_turns: int = self.adv_type_def["min_optimal_turns"]
        max_optimal_turns: int = self.adv_type_def["max_optimal_turns"]

        generated_adventures: list = list()
        total_generated_adventure_count = 0

        while total_generated_adventure_count < target_adventure_count:
            # generate new new-word definitions and assign them for each adventure for the new-words_created adv type:
            if self.adv_type == "new-words_created" and total_generated_adventure_count > 0:  # start replacing initial new-words once first init set was used
                print("Assigning new new-words...")
                self._create_assign_new_word_definitions()
            if "new-words_home-delivery" in self.adv_type and total_generated_adventure_count > 0:  # start replacing initial new-words once first init set was used
                print("Assigning new new-words...")
                self._replace_assign_new_word_definitions()

            # ROOM LAYOUTS
            # NOTE: As the number of room layouts is relatively small, generating all to iterate over is viable.
            # print(f"{datetime.now()} Generating room layouts...")
            result_layouts = self._generate_room_layouts()
            print("Room layouts generated.")
            # print("result_layouts:", result_layouts)
            print("result_layouts count:", len(result_layouts))

            # INITIAL STATES
            initial_states = self._generate_initial_states(result_layouts, initial_states_per_layout)
            print("Initial states generated.")
            # print("initial_states:", initial_states)
            # print("initial_states count:", len(initial_states))

            # print(initial_states)
            # print(len(initial_states))
            # print(list(range(initial_state_limit)))

            # get initial states to generate adventures with:
            if initial_state_picking == "iterative":
                if initial_state_limit:
                    print("range(initial_state_limit):", list(range(initial_state_limit)))
                    # print("initial_states:", initial_states)
                    print("length initial_states:", len(initial_states))
                    initial_states_used = [initial_states[idx] for idx in range(initial_state_limit)]
                else:
                    initial_states_used = initial_states
            elif initial_state_picking == "random":
                assert initial_state_limit > 0, (
                    "Random initial state picking without a limit is equivalent to getting all"
                    " iteratively.")
                initial_state_indices = self.rng.choice(len(initial_states), size=initial_state_limit, replace=False,
                                                        shuffle=False)
                initial_states_used = [initial_states[idx] for idx in initial_state_indices]

            # iterate over initial states used:
            for initial_state in initial_states_used:
                print("initial state:", initial_state)

                cur_adventure_count = 0
                keep_generating_adventures = True
                goal_set_idx = 0

                while keep_generating_adventures:
                    # generate goals for current initial state:
                    cur_all_goals = self._generate_goal_facts(initial_state)

                    if not cur_all_goals:
                        print("No goals generated for current initial state.")
                        continue

                    print("Goals generated.")

                    if goal_set_picking == "iterative":
                        goal_set = cur_all_goals[goal_set_idx]
                        goal_set_idx += 1

                    elif goal_set_picking == "random":
                        goal_set = self.rng.choice(cur_all_goals, size=1).tolist()[0]

                    print("Current goal set:", goal_set)

                    # potion brewing events:
                    if self.adv_type_def['task_config']['task'] == 'potion':
                        from potion_adventures import generate_potion_recipe, create_potion_recipe_events
                        potion_recipe = generate_potion_recipe(self.domain_def, self.entity_definitions,
                                                               rng_seed=self.rng.integers(9223372036854775808))
                        # print("potion recipe:\n", potion_recipe)
                        # print("potion recipe text:\n", potion_recipe['text'])
                        potion_events = create_potion_recipe_events(potion_recipe,
                                                                    self.domain_def, self.entity_definitions,
                                                                    rng_seed=self.rng.integers(9223372036854775808))
                        # print("potion events:\n", potion_events)
                        # load event type definitions:
                        event_definitions: list = list()
                        for event_def_source in self.adv_type_def["event_definitions"]:
                            with open(f"definitions/{event_def_source}", 'r', encoding='utf-8') as events_file:
                                event_definitions += json.load(events_file)
                        # add potion step events to event definitions list:
                        event_definitions += potion_events
                        # solve current adventure ASP encoding:
                        solve_asp: str = self._solve_optimally_asp(initial_state, goal_set, events_list=potion_events)
                        # print(solve_asp)
                    else:
                        # solve current adventure ASP encoding:
                        solve_asp: str = self._solve_optimally_asp(initial_state, goal_set)

                    print("solve ASP:\n", solve_asp)
                    # init fresh clingo controller:
                    cur_adv_solve_control: Control = Control(["0"])  # ["0"] argument to return all models
                    # add adventure solving asp encoding:
                    cur_adv_solve_control.add(solve_asp)
                    # ground clingo controller:
                    cur_adv_solve_control.ground()

                    print("Adventure solving grounded.")

                    cur_adv_solutions = list()
                    solvable: bool = False
                    with cur_adv_solve_control.solve(yield_=True) as solve:
                        for model in solve:
                            cur_adv_solutions.append(model.__str__())
                            # print(model)
                        satisfiable = str(solve.get())
                        # print("satisfiable:", satisfiable)
                        if satisfiable == "SAT":
                            solvable = True
                        elif satisfiable == "UNSAT":
                            solvable = False

                    print("Adventure solving performed.")

                    # skip this raw adventure if it is not solvable under the defined constraints:
                    if not solvable:
                        print("Adventure is NOT solvable.")
                        continue

                    print("Adventure is solvable.")
                    print()

                    # last yielded model is optimal solution:
                    cur_optimal_solution = cur_adv_solutions[-1]
                    # convert optimal solution:
                    cur_sol_abstract, optimal_turns, cur_sol_cmds = self._convert_adventure_solution(
                        cur_optimal_solution)
                    # check if optimal turns within bounds:
                    if min_optimal_turns <= optimal_turns <= max_optimal_turns:
                        pre_explore = True
                        if 'no_pre_explore' in task_config:
                            if task_config['no_pre_explore']:
                                pre_explore = False
                        if pre_explore:
                            # pre-explore rooms:
                            # solve current adventure:
                            visit_asp: str = self._pre_explore_asp(initial_state, goal_set)
                            # print("visit ASP:\n", visit_asp)
                            # init fresh clingo controller:
                            visit_solve_control: Control = Control(["0"])  # ["0"] argument to return all models
                            # add adventure solving asp encoding:
                            visit_solve_control.add(visit_asp)
                            # ground clingo controller:
                            visit_solve_control.ground()

                            print("Visiting solving grounded.")

                            visit_solutions = list()
                            solvable: bool = False
                            with visit_solve_control.solve(yield_=True) as solve:
                                for model in solve:
                                    visit_solutions.append(model.__str__())
                                    # print(model)
                                satisfiable = str(solve.get())
                                # print("satisfiable:", satisfiable)
                                if satisfiable == "SAT":
                                    solvable = True
                                elif satisfiable == "UNSAT":
                                    solvable = False

                            print("Visiting solving performed.")

                            # skip this raw adventure if it is not solvable under the defined constraints:
                            if not solvable:
                                print("Visiting is NOT solvable.")
                                continue

                            print("Visiting is solvable.")
                            print()

                            # last yielded model is optimal solution:
                            visit_optimal_solution = visit_solutions[-1]
                            # convert optimal solution:
                            visiting_abstract, visiting_turns, visiting_cmds = self._convert_adventure_solution(
                                visit_optimal_solution)

                            print("visiting_cmds:", visiting_cmds)

                        # get tuple world state:
                        world_state: set = set()
                        for fact in initial_state:
                            world_state.add(fact_str_to_tuple(fact))

                        # get tuple goals:
                        goal_tuples: list = list()
                        for goal in goal_set:
                            goal_tuples.append(fact_str_to_tuple(goal))

                        if task_config['task'] == 'deliver':
                            goal_strings: list = list()
                            for goal_tuple in goal_tuples:
                                # get string representations of delivery item and target:
                                item_type: str = str()
                                item_adjs: list = list()
                                target_type: str = str()
                                target_adjs: list = list()
                                for fact in world_state:
                                    if fact[0] == "type":
                                        if goal_tuple[1] == fact[1]:
                                            item_type = self.entity_definitions[fact[2]]['repr_str']
                                        if goal_tuple[2] == fact[1]:
                                            target_type = self.entity_definitions[fact[2]]['repr_str']
                                    if fact[0] == "adj":
                                        if goal_tuple[1] == fact[1]:
                                            item_adjs.append(fact[2])
                                        if goal_tuple[2] == fact[1]:
                                            target_adjs.append(fact[2])
                                item_adjs_str: str = " ".join(item_adjs)
                                if item_adjs:
                                    item_str: str = f"{item_adjs_str} {item_type}"
                                else:
                                    item_str: str = f"{item_type}"
                                target_adjs_str: str = " ".join(target_adjs)
                                if target_adjs:
                                    target_str: str = f"{target_adjs_str} {target_type}"
                                else:
                                    target_str: str = f"{target_type}"
                                goal_str: str = f"the {item_str} {goal_tuple[0]} the {target_str}"
                                goal_strings.append(goal_str)

                            if len(goal_strings) == 1:
                                goal_desc: str = f"Put {goal_strings[0]}."
                            if len(goal_strings) == 2:
                                goal_desc: str = f"Put {goal_strings[0]} and {goal_strings[1]}."
                            if len(goal_strings) >= 3:
                                goal_listing_str: str = ", ".join(goal_strings[:-1])
                                goal_desc: str = f"Put {goal_listing_str} and {goal_strings[-1]}."

                            # full raw adventure data:
                            viable_adventure = {
                                'adventure_type': self.adv_type,
                                'goal': goal_desc, 'initial_state': initial_state, 'goal_state': goal_set,
                                'optimal_turns': optimal_turns,
                                'optimal_solution': cur_sol_abstract, 'optimal_commands': cur_sol_cmds,
                                'visiting_turns': visiting_turns,
                                'visiting_solution': visiting_abstract, 'visiting_commands': visiting_cmds,
                                'action_definitions': self.adv_type_def['action_definitions'],
                                'room_definitions': self.adv_type_def['room_definitions'],
                                'entity_definitions': self.adv_type_def['entity_definitions'],
                                'bench_turn_limit': self.adv_type_def['bench_turn_limit']
                            }

                        if task_config['task'] == 'new-words_deliver':
                            goal_strings: list = list()
                            for goal_tuple in goal_tuples:
                                # get string representations of delivery item and target:
                                item_type: str = str()
                                item_adjs: list = list()
                                target_type: str = str()
                                target_adjs: list = list()
                                for fact in world_state:
                                    if fact[0] == "type":
                                        if goal_tuple[1] == fact[1]:
                                            item_type = self.entity_definitions[fact[2]]['repr_str']
                                        if goal_tuple[2] == fact[1]:
                                            target_type = self.entity_definitions[fact[2]]['repr_str']
                                    if fact[0] == "adj":
                                        if goal_tuple[1] == fact[1]:
                                            item_adjs.append(fact[2])
                                        if goal_tuple[2] == fact[1]:
                                            target_adjs.append(fact[2])
                                item_adjs_str: str = " ".join(item_adjs)
                                if item_adjs:
                                    item_str: str = f"{item_adjs_str} {item_type}"
                                else:
                                    item_str: str = f"{item_type}"
                                target_adjs_str: str = " ".join(target_adjs)
                                if target_adjs:
                                    target_str: str = f"{target_adjs_str} {target_type}"
                                else:
                                    target_str: str = f"{target_type}"
                                goal_str: str = f"the {item_str} {goal_tuple[0]} the {target_str}"
                                goal_strings.append(goal_str)

                            if len(goal_strings) == 1:
                                goal_desc: str = f"Put {goal_strings[0]}."
                            if len(goal_strings) == 2:
                                goal_desc: str = f"Put {goal_strings[0]} and {goal_strings[1]}."
                            if len(goal_strings) >= 3:
                                goal_listing_str: str = ", ".join(goal_strings[:-1])
                                goal_desc: str = f"Put {goal_listing_str} and {goal_strings[-1]}."

                            # convert new-word definitions to default format and store in adventure:
                            final_action_definitions = list()
                            for action_def_type, action_def_content in self.action_definitions.items():
                                # print(action_def_type, action_def_content)
                                final_action_def = action_def_content
                                action_def_content['type_name'] = action_def_type
                                final_action_definitions.append(final_action_def)
                            final_room_definitions = list()
                            for room_def_type, room_def_content in self.room_definitions.items():
                                final_room_def = room_def_content
                                room_def_content['type_name'] = room_def_type
                                final_room_definitions.append(final_room_def)
                            final_entity_definitions = list()
                            for entity_def_type, entity_def_content in self.entity_definitions.items():
                                final_entity_def = entity_def_content
                                entity_def_content['type_name'] = entity_def_type
                                final_entity_definitions.append(final_entity_def)

                            # TODO: replace new-word actions in optimal_commands

                            # full raw adventure data:
                            viable_adventure = {
                                'adventure_type': self.adv_type,
                                'goal': goal_desc, 'initial_state': initial_state, 'goal_state': goal_set,
                                'optimal_turns': optimal_turns,
                                'optimal_solution': cur_sol_abstract, 'optimal_commands': cur_sol_cmds,
                                'replacement_dict': self.replacement_dict,
                                'action_definitions': final_action_definitions,
                                'room_definitions': final_room_definitions,
                                'entity_definitions': final_entity_definitions,
                                'domain_definitions': [self.domain_def],
                                'prompt_template_set': self.adv_type_def['prompt_template_set'],
                                'bench_turn_limit': self.adv_type_def['bench_turn_limit']
                            }

                        if task_config['task'] == 'new-word_states':
                            goal_strings: list = list()
                            for goal_tuple in goal_tuples:
                                # get string representations of delivery item and target:
                                item_type: str = str()
                                item_adjs: list = list()
                                for fact in world_state:
                                    if fact[0] == "type":
                                        if goal_tuple[1] == fact[1]:
                                            item_type = self.entity_definitions[fact[2]]['repr_str']
                                    if fact[0] == "adj":
                                        if goal_tuple[1] == fact[1]:
                                            item_adjs.append(fact[2])
                                item_adjs_str: str = " ".join(item_adjs)
                                if item_adjs:
                                    item_str: str = f"{item_adjs_str} {item_type}"
                                else:
                                    item_str: str = f"{item_type}"
                                goal_str: str = f"the {item_str} {goal_tuple[0]}"
                                goal_strings.append(goal_str)

                            # shuffle goal strings to mitigate first goal and new-word verb matching:
                            # print("goal strings before shuffling:", goal_strings)
                            goal_strings_remap = np.arange(len(goal_strings))
                            self.rng.shuffle(goal_strings_remap)
                            goal_strings = [goal_strings[remap_idx] for remap_idx in goal_strings_remap]
                            # print("goal strings after shuffling:", goal_strings)

                            if len(goal_strings) == 1:
                                goal_desc: str = f"Make {goal_strings[0]}."
                            if len(goal_strings) == 2:
                                goal_desc: str = f"Make {goal_strings[0]} and {goal_strings[1]}."
                            if len(goal_strings) >= 3:
                                goal_listing_str: str = ", ".join(goal_strings[:-1])
                                goal_desc: str = f"Make {goal_listing_str} and {goal_strings[-1]}."

                            # convert new-word definitions to default format and store in adventure:
                            final_action_definitions = list()
                            for action_def_type, action_def_content in self.action_definitions.items():
                                # print(action_def_type, action_def_content)
                                final_action_def = action_def_content
                                action_def_content['type_name'] = action_def_type
                                final_action_definitions.append(final_action_def)
                            final_room_definitions = list()
                            for room_def_type, room_def_content in self.room_definitions.items():
                                final_room_def = room_def_content
                                room_def_content['type_name'] = room_def_type
                                final_room_definitions.append(final_room_def)
                            final_entity_definitions = list()
                            for entity_def_type, entity_def_content in self.entity_definitions.items():
                                final_entity_def = entity_def_content
                                entity_def_content['type_name'] = entity_def_type
                                final_entity_definitions.append(final_entity_def)

                            viable_adventure = {
                                'adventure_type': self.adv_type,
                                'goal': goal_desc, 'initial_state': initial_state, 'goal_state': goal_set,
                                'optimal_turns': optimal_turns,
                                'optimal_solution': cur_sol_abstract, 'optimal_commands': cur_sol_cmds,
                                'action_definitions': final_action_definitions,
                                'room_definitions': final_room_definitions,
                                'entity_definitions': final_entity_definitions,
                                'domain_definitions': [self.domain_def],
                                'bench_turn_limit': self.adv_type_def['bench_turn_limit'],
                                'prompt_template_set': self.adv_type_def['prompt_template_set']
                            }

                        if task_config['task'] == 'potion':
                            goal_desc = "Brew the potion."
                            # add recipe text fact:
                            initial_state.append(f"text(potionrecipe1,{potion_recipe['text']})")
                            # full raw adventure data:
                            viable_adventure = {
                                'adventure_type': self.adv_type,
                                'goal': goal_desc, 'initial_state': initial_state, 'goal_state': goal_set,
                                'optimal_turns': optimal_turns,
                                'optimal_solution': cur_sol_abstract, 'optimal_commands': cur_sol_cmds,
                                'action_definitions': self.adv_type_def['action_definitions'],
                                'room_definitions': self.adv_type_def['room_definitions'],
                                'entity_definitions': self.adv_type_def['entity_definitions'],
                                'domain_definitions': [self.domain_def],
                                'event_definitions': event_definitions,
                                'bench_turn_limit': self.adv_type_def['bench_turn_limit'],
                                'prompt_template_set': self.adv_type_def['prompt_template_set']
                            }

                        print("viable_adventure:", viable_adventure)

                        generated_adventures.append(viable_adventure)
                        cur_adventure_count += 1
                        total_generated_adventure_count += 1

                        if adventures_per_initial_state and cur_adventure_count == adventures_per_initial_state:
                            keep_generating_adventures = False
                    else:  # optimal turns not within bounds, discard this raw adventure
                        continue

        # adventures difficulty from adventure type definition:
        if 'difficulty' in self.adv_type_def:
            dict_by_difficulty = {self.adv_type_def['difficulty']: generated_adventures}
        else:
            dict_by_difficulty = {"undefined": generated_adventures}

        if save_to_file:
            with open(f"generated_{self.adv_type}_adventures.json", 'w', encoding='utf-8') as out_adv_file:
                if indent_output_json:
                    out_adv_file.write(json.dumps(dict_by_difficulty, indent=2))
                else:
                    out_adv_file.write(json.dumps(dict_by_difficulty))

        return dict_by_difficulty

    def generate_from_initial_goals(self, initial_state: list, goal_set: list,
                                    save_to_file: bool = True, indent_output_json: bool = True):
        """
        Generate optimal solution and create complete adventure from (manually created) initial world state and goal
        set.
        :param initial_state: List of clingo-style string facts of initial world state.
        :param goal_set: List of clingo-style string facts of goal states.
        """
        task_config: dict = self.adv_type_def["task_config"]
        min_optimal_turns: int = self.adv_type_def["min_optimal_turns"]
        max_optimal_turns: int = self.adv_type_def["max_optimal_turns"]

        # solve current adventure:
        solve_asp: str = self._solve_optimally_asp(initial_state, goal_set)
        # init fresh clingo controller:
        cur_adv_solve_control: Control = Control(["0"])  # ["0"] argument to return all models
        # add adventure solving asp encoding:
        cur_adv_solve_control.add(solve_asp)
        # ground clingo controller:
        cur_adv_solve_control.ground()

        cur_adv_solutions = list()
        solvable: bool = False
        with cur_adv_solve_control.solve(yield_=True) as solve:
            for model in solve:
                cur_adv_solutions.append(model.__str__())
            satisfiable = str(solve.get())
            if satisfiable == "SAT":
                solvable = True
            elif satisfiable == "UNSAT":
                solvable = False
        # skip this raw adventure if it is not solvable under the defined constraints:
        if not solvable:
            print("Adventure is not solvable!")
            return
        # last yielded model is optimal solution:
        cur_optimal_solution = cur_adv_solutions[-1]
        # convert optimal solution:
        cur_sol_abstract, optimal_turns, cur_sol_cmds = self._convert_adventure_solution(cur_optimal_solution)
        # check if optimal turns within bounds:
        if min_optimal_turns <= optimal_turns <= max_optimal_turns:
            # pre-explore rooms:
            # solve current adventure pre-exploration:
            visit_asp: str = self._pre_explore_asp(initial_state, goal_set)
            # print("visit ASP:\n", visit_asp)
            # init fresh clingo controller:
            visit_solve_control: Control = Control(["0"])  # ["0"] argument to return all models
            # add adventure solving asp encoding:
            visit_solve_control.add(visit_asp)
            # ground clingo controller:
            visit_solve_control.ground()

            print("Visiting solving grounded.")

            visit_solutions = list()
            solvable: bool = False
            with visit_solve_control.solve(yield_=True) as solve:
                for model in solve:
                    visit_solutions.append(model.__str__())
                    # print(model)
                satisfiable = str(solve.get())
                # print("satisfiable:", satisfiable)
                if satisfiable == "SAT":
                    solvable = True
                elif satisfiable == "UNSAT":
                    solvable = False

            print("Visiting solving performed.")

            # skip this raw adventure if it is not solvable under the defined constraints:
            if not solvable:
                print("Visiting is NOT solvable.")
                return

            print("Visiting is solvable.")
            print()

            # last yielded model is optimal solution:
            visit_optimal_solution = visit_solutions[-1]
            # convert optimal solution:
            visiting_abstract, visiting_turns, visiting_cmds = self._convert_adventure_solution(
                visit_optimal_solution)

            print("visiting_cmds:", visiting_cmds)

            # get tuple world state:
            world_state: set = set()
            for fact in initial_state:
                world_state.add(fact_str_to_tuple(fact))

            # get tuple goals:
            goal_tuples: list = list()
            for goal in goal_set:
                goal_tuples.append(fact_str_to_tuple(goal))

            if task_config['task'] == 'deliver':
                goal_strings: list = list()
                for goal_tuple in goal_tuples:
                    # get string representations of delivery item and target:
                    item_type: str = str()
                    item_adjs: list = list()
                    target_type: str = str()
                    target_adjs: list = list()
                    for fact in world_state:
                        if fact[0] == "type":
                            if goal_tuple[1] == fact[1]:
                                item_type = self.entity_definitions[fact[2]]['repr_str']
                            if goal_tuple[2] == fact[1]:
                                target_type = self.entity_definitions[fact[2]]['repr_str']
                        if fact[0] == "adj":
                            if goal_tuple[1] == fact[1]:
                                item_adjs.append(fact[2])
                            if goal_tuple[2] == fact[1]:
                                target_adjs.append(fact[2])
                    item_adjs_str: str = " ".join(item_adjs)
                    if item_adjs:
                        item_str: str = f"{item_adjs_str} {item_type}"
                    else:
                        item_str: str = f"{item_type}"
                    target_adjs_str: str = " ".join(target_adjs)
                    if target_adjs:
                        target_str: str = f"{target_adjs_str} {target_type}"
                    else:
                        target_str: str = f"{target_type}"
                    goal_str: str = f"the {item_str} {goal_tuple[0]} the {target_str}"
                    goal_strings.append(goal_str)

                if len(goal_strings) == 1:
                    goal_desc: str = f"Put {goal_strings[0]}."
                if len(goal_strings) == 2:
                    goal_desc: str = f"Put {goal_strings[0]} and {goal_strings[1]}."
                if len(goal_strings) >= 3:
                    goal_listing_str: str = ", ".join(goal_strings[:-1])
                    goal_desc: str = f"Put {goal_listing_str} and {goal_strings[-1]}."

            viable_adventure = {
                'adventure_type': self.adv_type,
                'goal': goal_desc, 'initial_state': initial_state, 'goal_state': goal_set,
                'optimal_turns': optimal_turns,
                'optimal_solution': cur_sol_abstract, 'optimal_commands': cur_sol_cmds,
                'visiting_turns': visiting_turns,
                'visiting_solution': visiting_abstract, 'visiting_commands': visiting_cmds,
                'action_definitions': self.adv_type_def['action_definitions'],
                'room_definitions': self.adv_type_def['room_definitions'],
                'entity_definitions': self.adv_type_def['entity_definitions'],
                'bench_turn_limit': self.adv_type_def['bench_turn_limit']
            }

            if save_to_file:
                timestamp: str = datetime.now().strftime("%Y%m%d-%H%M%S")
                with open(f"adventure_{timestamp}.json", 'w', encoding='utf-8') as out_adv_file:
                    if indent_output_json:
                        out_adv_file.write(json.dumps(viable_adventure, indent=2))
                    else:
                        out_adv_file.write(json.dumps(viable_adventure))

            return viable_adventure

        else:
            print(f"Optimal solution length of {optimal_turns} is outside of bounds.")
            return

    def generate_from_initial_goals_file(self, source_file_path: str):
        """
        Generate adventure from initial state and goals stored as file.
        """
        with open(source_file_path, 'r', encoding='utf-8') as source_file:
            source = json.load(source_file)

        initial_state = source['initial_state']
        goal_state = source['goal_state']

        self.generate_from_initial_goals(initial_state, goal_state)

    def augment_raw_adventures_pre_explore(self, source_file_path: str):
        """
        Add pre-explore sequence to raw adventures in a file.
        """
        # load raw adventures file:
        with open(source_file_path, 'r', encoding='utf-8') as source_file:
            raw_adventures = json.load(source_file)

        difficulties = list(raw_adventures.keys())

        for difficulty in difficulties:
            dif_raws = raw_adventures[difficulty]
            for raw_idx, raw_adv in enumerate(dif_raws):
                print(f"Augmenting {difficulty} adventure number {raw_idx}...")

                cur_initial_state = raw_adv['initial_state']
                print(cur_initial_state)
                cur_goal_set = raw_adv['goal_state']
                print(cur_goal_set)

                # pre-explore rooms:
                # solve current adventure pre-exploration:
                visit_asp: str = self._pre_explore_asp(cur_initial_state, cur_goal_set)
                # print("visit ASP:\n", visit_asp)
                # init fresh clingo controller:
                visit_solve_control: Control = Control(["0"])  # ["0"] argument to return all models
                # add adventure solving asp encoding:
                visit_solve_control.add(visit_asp)
                # ground clingo controller:
                visit_solve_control.ground()

                print("Visiting solving grounded.")

                visit_solutions = list()
                solvable: bool = False
                with visit_solve_control.solve(yield_=True) as solve:
                    for model in solve:
                        visit_solutions.append(model.__str__())
                        # print(model)
                    satisfiable = str(solve.get())
                    # print("satisfiable:", satisfiable)
                    if satisfiable == "SAT":
                        solvable = True
                    elif satisfiable == "UNSAT":
                        solvable = False

                print("Visiting solving performed.")

                # skip this raw adventure if it is not solvable under the defined constraints:
                if not solvable:
                    print("Visiting is NOT solvable.")
                    return

                print("Visiting is solvable.")
                print()

                # last yielded model is optimal solution:
                visit_optimal_solution = visit_solutions[-1]
                # convert optimal solution:
                visiting_abstract, visiting_turns, visiting_cmds = self._convert_adventure_solution(
                    visit_optimal_solution)

                # print("visiting_cmds:", visiting_cmds)

                raw_adv['visiting_turns'] = visiting_turns
                raw_adv['visiting_solution'] = visiting_abstract
                raw_adv['visiting_commands'] = visiting_cmds

                # print(raw_adv)

                # break
            # break

        # save augmented raw adventures:
        with open(f"{source_file_path[:-5]}_a.json", 'w', encoding='utf-8') as out_raw_adv_file:
            out_raw_adv_file.write(json.dumps(raw_adventures, indent=2))




if __name__ == "__main__":
    # init generator:
    # adventure_generator = ClingoAdventureGenerator(adventure_type="home_deliver_three")
    # adventure_generator = ClingoAdventureGenerator(adventure_type="home_deliver_three")
    # adventure_generator = ClingoAdventureGenerator(adventure_type="new-words_created")
    # adventure_generator = ClingoAdventureGenerator(adventure_type="new-words_home-delivery_easy")
    # adventure_generator = ClingoAdventureGenerator(adventure_type="new-words_home-delivery_medium")
    adventure_generator = ClingoAdventureGenerator(adventure_type="potion_brewing")

    # print(adventure_generator.domain_def)
    # print(adventure_generator.entity_definitions)

    # adventure_generator._generate_room_layouts(limit=1)

    # generate adventure including metadata from manually edited source:
    # adventure_generator.generate_from_initial_goals_file("adv_source.json")

    # adventure_generator.generate_adventures(initial_state_limit=1, initial_states_per_layout=1, goal_set_picking="random")
    adventure_generator.generate_adventures(initial_state_limit=1, initial_states_per_layout=1, goal_set_picking="random")

    # adventure_generator.generate_adventures(initial_state_limit=1, initial_states_per_layout=1, save_to_file=False)

    # adventure_generator.augment_raw_adventures_pre_explore("curated_home_deliver_three_adventures_v2_2.json")