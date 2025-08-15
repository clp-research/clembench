"""Functions to create AdventureGame definitions (actions, domains, entities, rooms) with generated new-words."""
import json
from copy import deepcopy
import numpy as np
import re

from adventuregame.resources.new_word_generation.new_word_util import read_new_words_file

# ROOMS
"""
New-word room types:
- replace existing room type name and surface form (repr_str)
- new, arbitrary types

Possible entities in room is part of entity def.
"""

def new_word_rooms_replace(source_definition_file_path: str, num_replace: int = 0, last_new_words_idx: int = 0,
                           seed: int = 42):
    """Replace room representation strings of an existing rooms definition with new words.
    This leaves other values intact, only changing the surface form the rooms are referred to as in the IF feedback.
    Args:
        source_definition_file_path: Path to the source definition file.
        num_replace: How many of the room definitions in the source definition file will have their representation
            replaced by new words. If this value is 0 (default), all room types will have their surface string replaced.
        last_new_words_idx: New word source index of next new word to use when iterating.
        seed: Seed number for the RNG.
    """
    # init RNG:
    rng = np.random.default_rng(seed)

    # load new words from file:
    # new_words_source = read_new_words_file("new_words.tsv")
    new_words_source = read_new_words_file("new_word_generation/new_words.tsv")
    new_word_idx = last_new_words_idx

    # load source room definitions:
    with open(source_definition_file_path, 'r', encoding='utf-8') as source_definition_file:
        source_definitions = json.load(source_definition_file)

    if num_replace > 0:
        # get random list of room def indices to replace:
        def_idx_to_replace = rng.choice(range(len(source_definitions)), size=num_replace, replace=False).tolist()
        # print("room def_idx_to_replace:", def_idx_to_replace)
    else:
        def_idx_to_replace = list(range(len(source_definitions)))

    replacement_dict = dict()
    new_room_definitions = deepcopy(source_definitions)
    for def_idx in def_idx_to_replace:
        cur_def = new_room_definitions[def_idx]
        old_repr_str = cur_def['repr_str']
        cur_def['repr_str'] = new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['NN']

        old_type_name = cur_def['type_name']
        cur_def['type_name'] = new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['NN']

        new_word_idx += 1
        # replacement_dict[old_repr_str] = cur_def['repr_str']
        replacement_dict[old_type_name] = cur_def['type_name']

    return new_room_definitions, new_word_idx, replacement_dict


def new_word_rooms_create(num_rooms_created: int = 4,
                          min_connections: int = 1, max_connections: int = 4, max_exit_targets: int = 4,
                          last_new_words_idx: int = 0, seed: int = 42):
        """Create rooms definition with new words.
        Args:
            num_rooms_created: Number of new-word room definitions to create.
            min_connections: Minimum number of room connections. Default: 1
            max_connections: Maximum number of room connections. Default: 4 (Beware of non-euclidian passages!)
            max_exit_targets: Maximum number of exit target room types.
            last_new_words_idx: New-word source index of next new word to use when iterating.
            seed: Seed number for the RNG.
        """
        # init RNG:
        rng = np.random.default_rng(seed)
        # load new words from file:
        # new_words_source = read_new_words_file("new_words.tsv")
        new_words_source = read_new_words_file("new_word_generation/new_words.tsv")
        new_word_idx = last_new_words_idx

        new_room_definitions = list()
        new_room_type_names = list()
        # create specified number of new word room definitions:
        for def_idx in range(num_rooms_created):
            new_room_type_dict = dict()
            new_room_type_name = new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['NN']
            new_word_idx += 1
            new_room_type_dict['type_name'] = new_room_type_name
            new_room_type_dict['repr_str'] = new_room_type_name
            new_room_type_dict['exit_targets'] = []  # left empty here due to incomplete info about all room types
            new_room_type_dict['max_connections'] = int(rng.integers(min_connections, max_connections))
            new_room_definitions.append(new_room_type_dict)
            new_room_type_names.append(new_room_type_name)
        # add exit targets after all new room types are defined:
        for new_room_type_def in new_room_definitions:
            # new room type name list without current new room type name:
            new_room_type_names_culled = deepcopy(new_room_type_names)
            new_room_type_names_culled.remove(new_room_type_def['type_name'])
            # random other-room-type exit targets:
            new_exit_targets = rng.choice(new_room_type_names_culled,
                                          size=rng.integers(1, max_exit_targets),
                                          replace=False)
            new_room_type_def['exit_targets'] = new_exit_targets.tolist()

        return new_room_definitions, new_word_idx


# ENTITIES
"""
New-word entity types:
- replace existing entity type name and surface form (repr_str)
- new, arbitrary types

Mutable states:
Binary: Get two new-words, use first to make entity trait (suffixing with _able), link to action to switch between new-words states.
Trinary?
"""

def new_word_entities_replace(source_definition_file_path: str, num_replace: int = 0, last_new_words_idx: int = 0,
                              seed: int = 42):
    """Replace entities representation strings of an existing entities definition with new words.
    This leaves other values intact, only changing the surface form the entities are referred to as in the IF feedback.
    Args:
        source_definition_file_path: Path to the source definition file.
        num_replace: How many of the entity definitions in the source definition file will have their representation
            replaced by new words. If this value is 0 (default), all entity types will have their surface string
            replaced.
        last_new_words_idx: New word source index of next new word to use when iterating.
        seed: Seed number for the RNG.
    """
    # init RNG:
    rng = np.random.default_rng(seed)

    # load new words from file:
    # new_words_source = read_new_words_file("new_words.tsv")
    new_words_source = read_new_words_file("new_word_generation/new_words.tsv")
    new_word_idx = last_new_words_idx

    # load source room definitions:
    with open(source_definition_file_path, 'r', encoding='utf-8') as source_definition_file:
        source_definitions = json.load(source_definition_file)

    if num_replace > 0:
        # print(f"entity num_replace: {num_replace}")
        # print(source_definitions[2:])
        takeables: list = [entity_idx for entity_idx, entity_def in enumerate(source_definitions)
                           if "takeable" in entity_def['traits']]
        # print("takeables:", takeables)
        # print(source_definitions[takeables[0]])

        supports: list = [entity_idx for entity_idx, entity_def in enumerate(source_definitions)
                             if "support" in entity_def['traits']
                             and entity_def['type_name'] not in ['floor']]
        # print("supports:", supports)
        containers: list = [entity_idx for entity_idx, entity_def in enumerate(source_definitions)
                             if "container" in entity_def['traits']
                             and entity_def['type_name'] not in ['inventory']]
        # print("containers:", containers)
        receptacles: list = supports + containers

        # print("receptacles:", receptacles)
        # print(source_definitions[receptacles[0]])
        if num_replace == 1:
            # replace one takeable:
            def_idx_to_replace: list = rng.choice(takeables, 1).tolist()
            # print("def_idx_to_replace with takeables:", def_idx_to_replace)
        if num_replace == 2:
            # replace one takeable and one receptacle:
            def_idx_to_replace: list = rng.choice(takeables, 1).tolist()
            # print("def_idx_to_replace with takeables:", def_idx_to_replace)
            def_idx_to_replace.append(rng.choice(receptacles))
            # print("def_idx_to_replace with receptacles:", def_idx_to_replace)
        elif num_replace == 3:
            # replace two takeables and one receptacle:
            def_idx_to_replace: list = rng.choice(takeables, 2, replace=False).tolist()
            # print("def_idx_to_replace with takeables:", def_idx_to_replace)
            def_idx_to_replace.append(rng.choice(receptacles))
            # print("def_idx_to_replace with receptacles:", def_idx_to_replace)
        elif num_replace > 3:
            # replace at least one container and one support receptacle, and at least two takeables:
            def_idx_to_replace: list = rng.choice(takeables, 2, replace=False).tolist()
            # print("def_idx_to_replace with takeables:", def_idx_to_replace)
            def_idx_to_replace.append(rng.choice(containers))
            # print("def_idx_to_replace with container:", def_idx_to_replace)
            def_idx_to_replace.append(rng.choice(supports))
            # print("def_idx_to_replace with support:", def_idx_to_replace)
            if num_replace > 4:
                # replace random remaining entities up to target new-word entity number:
                used_idx_set = set(def_idx_to_replace)
                # print("used_idx_set:", used_idx_set)
                all_idx_set = set(range(3, len(source_definitions)))
                # print("all_idx_set:", all_idx_set)
                remaining_idx_set = all_idx_set - used_idx_set
                remaining_idx = list(remaining_idx_set)
                # print("remaining_idx:", remaining_idx)
                def_idx_to_replace += rng.choice(remaining_idx, num_replace-4, replace=False).tolist()
                # print("def_idx_to_replace with remaining:", def_idx_to_replace)

        # get random list of entity def indices to replace:
        # range offset by three to prevent replacement of player, floor and inventory entities
        # def_idx_to_replace = rng.choice(range(2, len(source_definitions)), size=num_replace).tolist()
        # print("entity def_idx_to_replace:", def_idx_to_replace)
    else:
        def_idx_to_replace = list(range(2, len(source_definitions)))

    replacement_dict = dict()
    new_entity_definitions = deepcopy(source_definitions)
    for def_idx in def_idx_to_replace:
        cur_def = new_entity_definitions[def_idx]
        old_repr_str = cur_def['repr_str']
        cur_def['repr_str'] = new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['NN']

        old_type_name = cur_def['type_name']
        cur_def['type_name'] = new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['NN']

        new_word_idx += 1
        replacement_dict[old_type_name] = cur_def['type_name']
        # replacement_dict[old_repr_str] = cur_def['repr_str']

    # TODO?: new-word mutability traits and/or mutable states?

    return new_entity_definitions, new_word_idx, replacement_dict


def new_word_entities_create(room_definitions: list, num_entities_created: int = 10,
                             min_std_locations: int = 1, max_std_locations: int = 3,
                             add_traits: bool = False, premade_traits: list = [], limited_trait_pool: int = 0,
                             min_traits: int = 0, max_traits: int = 3,
                             add_adjectives: bool = False, premade_adjectives: list = [], limited_adjective_pool: int = 0,
                             min_adjectives: int = 0, max_adjectives: int = 3,
                             add_core_entities: bool = True,
                             last_new_words_idx: int = 0, seed: int = 42):
    """Create entity definitions with new words.
    Args:
        room_definitions: Room definitions list to use for entity standard locations.
        num_entities_created: Number of new-word entity definitions to create.
        min_std_locations: Minimum number of standard locations sampled from room definitions.
        max_std_locations: Maximum number of standard locations sampled from room definitions.
        add_traits: If True, new-word traits are assigned to created entities.
        limited_trait_pool: Use a pool of new-word traits of the given size for all created entity definitions. 0 will
            assign different new-word traits to each created entity definition.
        min_traits: Minimum number of traits for all created entities.
        max_traits: Maximum number of traits for all created entities.
        add_adjectives: If True, possible new-word adjectives are assigned to created entities.
        limited_adjective_pool: Use a pool of possible new-word adjectives of the given size for all created entity
            definitions. 0 will assign different new-word traits to each created entity definition.
        min_adjectives: Minimum number of possible adjectives for all created entities.
        max_adjectives: Maximum number of possible adjectives for all created entities.
        add_core_entities: If True (default), core 'player', 'inventory' and 'floor' entity definitions are added.
        last_new_words_idx: New-word source index of next new word to use when iterating.
        seed: Seed number for the RNG.
    Returns:
        Tuple of:
        new_entity_definitions: List of new-word entity definitions.
        new_word_idx: Next unused new-word index.
        trait_pool: List of trait words used (new-words or passed premade traits).
        adjective_pool: List of adjective words used (new-words or passed premade adjectives).
    """
    # init RNG:
    rng = np.random.default_rng(seed)
    # load new words from file:
    new_words_source = read_new_words_file("new_word_generation/new_words.tsv")
    # new_words_source = read_new_words_file("new_words.tsv")
    new_word_idx = last_new_words_idx

    # traits pool:
    trait_pool = list()
    if add_traits and not premade_traits and limited_trait_pool:
        for new_word in range(limited_trait_pool):
            # trait_pool.append(new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['JJ'])
            trait_pool.append(f"{new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['JJ']}_able")
            new_word_idx += 1

    # adjectives pool:
    adjective_pool = list()
    if add_adjectives and not premade_adjectives and limited_adjective_pool:
        for new_word in range(limited_adjective_pool):
            adjective_pool.append(new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['JJ'])
            new_word_idx += 1

    new_entity_definitions = list()
    new_entity_type_names = list()
    # create specified number of new word room definitions:
    for def_idx in range(num_entities_created):
        new_entity_type_dict = dict()
        new_entity_type_name = new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['NN']
        new_word_idx += 1
        new_entity_type_dict['type_name'] = new_entity_type_name
        new_entity_type_dict['repr_str'] = new_entity_type_name
        # traits:
        if add_traits:
            new_entity_type_dict['traits'] = list()
            if limited_trait_pool == 0 and not premade_traits:
                for add_trait_idx in range(min_traits, rng.integers(min_traits, max_traits)):
                    taken_new_word = f"{new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['JJ']}_able"
                    new_word_idx += 1
                    new_entity_type_dict['traits'].append(taken_new_word)
                    trait_pool.append(taken_new_word)
            else:
                if premade_traits:
                    trait_pool = premade_traits
                new_entity_type_dict['traits']+= rng.choice(trait_pool,
                                                                 size=rng.integers(min_traits, max_traits),
                                                                 replace=False).tolist()
        # adjectives:
        if add_adjectives:
            new_entity_type_dict['possible_adjs'] = list()
            if limited_adjective_pool == 0 and not premade_adjectives:
                for add_adj_idx in range(min_adjectives, rng.integers(min_adjectives, max_adjectives)):
                    taken_new_word = new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['JJ']
                    new_word_idx += 1
                    new_entity_type_dict['possible_adjs'].append(taken_new_word)
                    adjective_pool.append(taken_new_word)
            else:
                if premade_adjectives:
                    adjective_pool = premade_adjectives
                new_entity_type_dict['possible_adjs'] += rng.choice(adjective_pool,
                                                                 size=rng.integers(min_adjectives, max_adjectives),
                                                                 replace=False).tolist()
        # standard locations:
        new_entity_type_dict['standard_locations'] = rng.choice([room_def['type_name'] for room_def in room_definitions],
                                                                 size=rng.integers(min_std_locations, max_std_locations),
                                                                 replace=False).tolist()

        new_entity_definitions.append(new_entity_type_dict)
        new_entity_type_names.append(new_entity_type_name)

    if add_core_entities:
        # default player entity type:
        player_def = {"type_name": "player", "repr_str": "you", "traits": [], "hidden": True}
        new_entity_definitions.append(player_def)

        # default inventory entity type:
        inventory_def = {"type_name": "inventory", "repr_str": "", "traits": [], "hidden": True}
        new_entity_definitions.append(inventory_def)

        # default floor entity type:
        floor_def = {"type_name": "floor", "repr_str": "", "traits": [], "hidden": True}
        new_entity_definitions.append(floor_def)


    return new_entity_definitions, new_word_idx, trait_pool, adjective_pool


# ACTIONS
"""
New-word action types:
- replace existing action surface form
- change applicable entity's new-word mutable state
    - requires connected mutability trait and transition trajectories
- ???

Explanations (to be put into initial prompts)
- "X is like Y existing action"
- circumscription: Handwritten for the small amount of existing default actions
"""

def new_word_actions_replace(source_definition_file_path: str, num_replace: int = 0, is_like_explanations: bool = False,
                             last_new_words_idx: int = 0, seed: int = 42):
    """Replace action definition strings of an existing actions definition with new words.
    This leaves other values intact, only changing the surface form of the actions. The key/id of the action will not
    change, only surface grammar is adapted.
    Args:
        source_definition_file_path: Path to the source definition file.
        num_replace: How many of the action definitions in the source definition file will have their representation
            replaced by new words. If this value is 0 (default), all action types will have surface strings replaced.
        is_like_explanations: Instead of replacing the verb in the original explanation, the explanation will be 'To
            NEW-WORD is like to ORIGINAL.'
        last_new_words_idx: New word source index of next new word to use when iterating.
        seed: Seed number for the RNG.
    """
    # init RNG:
    rng = np.random.default_rng(seed)

    # load new words from file:
    # new_words_source = read_new_words_file("new_words.tsv")
    new_words_source = read_new_words_file("new_word_generation/new_words.tsv")
    new_word_idx = last_new_words_idx

    # load source action definitions:
    with open(source_definition_file_path, 'r', encoding='utf-8') as source_definition_file:
        source_definitions = json.load(source_definition_file)

    if num_replace > 0:
        # get random list of action def indices to replace:
        def_idx_to_replace = rng.choice(range(len(source_definitions)), size=num_replace, replace=False).tolist()
    else:
        def_idx_to_replace = list(range(len(source_definitions)))

    replacement_dict = dict()
    new_action_definitions = deepcopy(source_definitions)
    for def_idx in def_idx_to_replace:
        cur_def = new_action_definitions[def_idx]
        # get new verb:
        new_action_verb = new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['VB']
        new_word_idx += 1
        # replace surface verb portion of lark grammar snippet:
        old_lark = cur_def['lark']
        lark_verb_regex = r"\.1: (.+) WS"  # this assumes action lark grammar snippets with separated verb surface forms
        verb_forms = re.search(lark_verb_regex, old_lark)
        new_lark = old_lark.replace(verb_forms.group(1), f'"{new_action_verb}"')
        cur_def['lark'] = new_lark
        # store new word surface verb form for explanation filling:
        cur_def['new_word'] = new_action_verb
        # explanation replacement:
        if is_like_explanations:
            cur_def['explanation'] = f"To {new_action_verb} is like to {cur_def['type_name']}."
        else:
            cur_def['explanation'] = cur_def['explanation'].replace("VERB", new_action_verb)
        # record replacement:
        replacement_dict[cur_def['type_name']] = new_action_verb

    return new_action_definitions, new_word_idx, replacement_dict


def new_word_actions_create(entity_definitions: list, num_actions_created: int = 6, trait_pool: list = [],
                            allowed_mutable_state_interactions: tuple = ("irreversible", "binary", "trinary"),
                            last_new_words_idx: int = 0, seed: int = 42):
    """Create action definitions with new words.
    Args:
        entity_definitions: Entity definitions list to use for entity mutable states and traits.
        num_actions_created: Number of new-word action definitions to create.
        trait_pool: List of mutable state applicability traits strings. To limit mutable state applicability traits
            used for new-word actions to a specific list, if not given, mutable state applicability traits will be taken
            from entity definitions.
        allowed_mutable_state_interactions: Tuple of string keywords determining which types of mutable state
            interactions created new word actions can result in. Actions fitting the interaction sets are created in
            order of the list. The types are:
                - irreversible: This allows an action to be created that changes the mutable state tied to a mutable
                    state applicability trait only once. Mutable state space can be singular (single mutable state is
                    not applied to entity until action is performed with it) or paired (one of two applicable mutable
                    states is initially applied to entity and when action is performed with it, the other mutable state
                    is applied).
                - binary: This allows action pairs to be created that switch between two mutable states tied to a mutable
                    state applicability trait. Mutable state space is paired (one of two applicable mutable states
                    initially applies to entity and when actions are performed with it, the other mutable state is
                    applied). Example: OPEN and CLOSE actions, with mutable state applicability trait 'openable'
                    switching entities between 'opened' and 'closed' respectively.
                - trinary: This allows action triplets to be created that switch between three mutable states tied to a
                    mutable state applicability trait. Mutable state space is chained (one of three applicable mutable
                    states initially applies to entity and when actions are performed with it, the next mutable state is
                    applied if allowed by chain order). Example: mutable state applicability trait = 'reheatable',
                    'frozen' + THAW -> 'unfrozen' + COOK -> 'edible' + FREEZE -> 'frozen'.
        last_new_words_idx: New-word source index of next new word to use when iterating.
        seed: Seed number for the RNG.
    Returns:
        Tuple of:
        new_action_definitions: List of new-word action definitions.
        new_word_idx: Next unused new-word index.
        trait_pool: List of trait words used (new-words or passed premade traits).
    """
    # init RNG:
    rng = np.random.default_rng(seed)
    # load new words from file:
    new_words_source = read_new_words_file("new_word_generation/new_words.tsv")
    # new_words_source = read_new_words_file("new_words.tsv")
    new_word_idx = last_new_words_idx

    # print("entities:", entity_definitions)

    # get mutable state applicability traits and create mutable state sets:
    mutable_state_sets = dict()
    if not trait_pool:
        traits = list()
        for entity_def in entity_definitions:
            if 'traits' in entity_def:
                for trait in entity_def['traits']:
                    if trait not in traits:
                        traits.append(trait)
    else:
        traits = trait_pool
    # print(traits)
    trait_dict = dict()
    mutable_state_interaction_idx = 0
    for trait in traits:
        cur_trait_list = list()
        cur_trait_dict = dict()
        if allowed_mutable_state_interactions[mutable_state_interaction_idx] == "irreversible":
            mutable_set_type = rng.choice(["singular", "paired"])
            # always use mutability trait word as mutable state:
            cur_trait_list.append(trait.replace("_able", ""))
            if mutable_set_type == "paired":
                cur_trait_list.append(new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['JJ'])
                new_word_idx += 1
                # reverse list to have other new word as initial mutable state:
                cur_trait_list.reverse()
            cur_trait_dict['interaction'] = "irreversible"
            cur_trait_dict['mutable_states'] = cur_trait_list
            cur_trait_dict['mutable_set_type'] = str(mutable_set_type)
        elif allowed_mutable_state_interactions[mutable_state_interaction_idx] == "binary":
            cur_trait_list.append(trait.replace("_able", ""))
            cur_trait_list.append(new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['JJ'])
            new_word_idx += 1
            cur_trait_dict['interaction'] = "binary"
            cur_trait_dict['mutable_states'] = cur_trait_list
            cur_trait_dict['mutable_set_type'] = "paired"
        elif allowed_mutable_state_interactions[mutable_state_interaction_idx] == "trinary":
            cur_trait_list.append(trait.replace("_able", ""))
            cur_trait_list.append(new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['JJ'])
            new_word_idx += 1
            cur_trait_list.append(new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['JJ'])
            new_word_idx += 1
            cur_trait_dict['interaction'] = "trinary"
            cur_trait_dict['mutable_states'] = cur_trait_list
            cur_trait_dict['mutable_set_type'] = "chained"
            # TODO?: arbitrary switching in addition to chaining?

        # loop through mutable state interaction type tuple:
        if mutable_state_interaction_idx < len(allowed_mutable_state_interactions)-1:
            mutable_state_interaction_idx += 1
        else:
            mutable_state_interaction_idx = 0

        trait_dict[trait] = cur_trait_dict

    # print(trait_dict)

    # TODO?: single-action pair/chain progression?

    new_word_actions_definitions = list()
    created_actions_count = 0
    # iterate through trait dict and create actions resulting in transitions between mutable states:
    for trait, trait_features in trait_dict.items():
        if trait_features['interaction'] == "irreversible" and created_actions_count < num_actions_created:
            # create single action type
            new_action = dict()

            new_word = new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['VB']
            new_word_idx += 1

            action_tag = new_word.upper()

            new_action['type_name'] = new_word
            # lark grammar snippet:
            lark_string = f"{new_word}: {action_tag} thing\n{action_tag}.1: \"{new_word}\" WS"
            # print(lark_string)
            new_action['lark'] = lark_string

            # PDDL
            # expose values to allow feedback creation?

            # parameters:
            # NB: new word mutable state is always first parameters item
            pddl_parameters = f":parameters (?e - {trait} ?r - room ?p - player)"

            # precondition:
            if trait_features['mutable_set_type'] == "singular":
                pddl_precondition = f":precondition (and\n        (at ?p ?r)\n        (at ?e ?r)\n)"
            elif trait_features['mutable_set_type'] == "paired":
                # NB: new word mutable state is always third precondition item
                pddl_precondition = f":precondition (and\n        (at ?p ?r)\n        (at ?e ?r)\n        ({trait_features['mutable_states'][0]} ?e)\n        )"

            # effect:
            if trait_features['mutable_set_type'] == "singular":
                # just add mutable state
                pddl_effect = f":effect (and\n        ({trait_features['mutable_states'][0]} ?e)\n    )"
            elif trait_features['mutable_set_type'] == "paired":
                # add second mutable state, remove first mutable state
                pddl_effect = f":effect (and\n        ({trait_features['mutable_states'][1]} ?e)\n        (not ({trait_features['mutable_states'][0]} ?e))\n    )"

            # full PDDL action string:
            pddl_action = f"(:action {action_tag}\n    {pddl_parameters}\n    {pddl_precondition}\n    {pddl_effect}\n)"
            # print(pddl_action)
            new_action['pddl'] = pddl_action

            # PDDL parameter mapping:
            new_action['pddl_parameter_mapping'] = {
                "?e": ["arg1"],
                "?r": ["current_player_room"],
                "?p": ["player"]
            }

            # FAILURE FEEDBACK
            # parameters failures:
            trait_dashed = trait.replace("_", "-")
            fail_parameters_trait_type = "The {{ e }} is not MUTABILITY_TRAIT.".replace("MUTABILITY_TRAIT", trait_dashed)
            fail_parameters = [
                [fail_parameters_trait_type, "domain_trait_type_mismatch"],
                ["{{ r }} is not a room. (This should not occur.)", "domain_type_discrepancy"],
                ["{{ p }} is not a player. (This should not occur.)", "domain_type_discrepancy"]
            ]

            # precondition failures:
            if trait_features['mutable_set_type'] == "singular":
                # since there is no mutable state precondition just the base accessibility needs feedback
                fail_precondition = [
                    ["You are not where you are! (This should not occur.)", "world_state_discrepancy"],
                    ["You can't see a {{ e }} here.", "entity_not_accessible"]
                ]
            elif trait_features['mutable_set_type'] == "paired":
                fail_precondition_entity_state = "The {{ e }} is not MUTABLE_STATE.".replace("MUTABLE_STATE", trait_features['mutable_states'][0])
                fail_precondition = [
                    ["You are not where you are! (This should not occur.)", "world_state_discrepancy"],
                    ["You can't see a {{ e }} here.", "entity_not_accessible"],
                    [fail_precondition_entity_state, "entity_state_mismatch"]
                ]

            # full failure feedback:
            new_action['failure_feedback'] = {
                'parameters': fail_parameters,
                'precondition': fail_precondition
            }

            # SUCCESS FEEDBACK
            if trait_features['mutable_set_type'] == "singular":
                success_feedback = "The {{ e }} is now MUTABLE_STATE_POST.".replace("MUTABLE_STATE_POST", trait_features['mutable_states'][0])
            elif trait_features['mutable_set_type'] == "paired":
                success_feedback = "The {{ e }} is now MUTABLE_STATE_POST.".replace("MUTABLE_STATE_POST",
                                                                                trait_features['mutable_states'][1])
            new_action['success_feedback'] = success_feedback

            # Note: ASP skipped for now, hopefully PDDL->ASP conversion is in the cards to automate this

            # EPISTEMIC/PRAGMATIC
            new_action['epistemic'] = False
            new_action['pragmatic'] = True

            # EXPLANATION
            # generic and containing full mutable state and trait info
            if trait_features['mutable_set_type'] == "singular":
                explanation_text = f"To {new_word} is to make something {trait} {trait_features['mutable_states'][0]}."
            elif trait_features['mutable_set_type'] == "paired":
                explanation_text = f"To {new_word} is to make something {trait} and {trait_features['mutable_states'][0]} be {trait_features['mutable_states'][1]}."
            new_action['explanation'] = explanation_text

            # FINISH ACTION
            new_word_actions_definitions.append(new_action)
            created_actions_count += 1

        if trait_features['interaction'] == "binary" and created_actions_count < num_actions_created:
            # create two action types
            # FIRST ACTION TYPE: A->B
            new_action = dict()

            new_word = new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['VB']
            new_word_idx += 1

            action_tag = new_word.upper()

            new_action['type_name'] = new_word
            # lark grammar snippet:
            lark_string = f"{new_word}: {action_tag} thing\n{action_tag}.1: \"{new_word}\" WS"
            # print(lark_string)
            new_action['lark'] = lark_string

            # PDDL
            # expose values to allow feedback creation?

            # parameters:
            # NB: new word mutable state is always first parameters item
            pddl_parameters = f":parameters (?e - {trait} ?r - room ?p - player)"
            # precondition:
            # NB: new word mutable state is always third precondition item
            pddl_precondition = f":precondition (and\n        (at ?p ?r)\n        (at ?e ?r)\n        ({trait_features['mutable_states'][0]} ?e)\n        )"
            # effect:
            # add second mutable state, remove first mutable state
            pddl_effect = f":effect (and\n        ({trait_features['mutable_states'][1]} ?e)\n        (not ({trait_features['mutable_states'][0]} ?e))\n    )"
            # full PDDL action string:
            pddl_action = f"(:action {action_tag}\n    {pddl_parameters}\n    {pddl_precondition}\n    {pddl_effect}\n)"
            # print(pddl_action)
            new_action['pddl'] = pddl_action

            # PDDL parameter mapping:
            new_action['pddl_parameter_mapping'] = {
                "?e": ["arg1"],
                "?r": ["current_player_room"],
                "?p": ["player"]
            }

            # FAILURE FEEDBACK
            # parameters failures:
            trait_dashed = trait.replace("_", "-")
            fail_parameters_trait_type = "The {{ e }} is not MUTABILITY_TRAIT.".replace("MUTABILITY_TRAIT", trait_dashed)
            fail_parameters = [
                [fail_parameters_trait_type, "domain_trait_type_mismatch"],
                ["{{ r }} is not a room. (This should not occur.)", "domain_type_discrepancy"],
                ["{{ p }} is not a player. (This should not occur.)", "domain_type_discrepancy"]
            ]

            # precondition failures:
            fail_precondition_entity_state = "The {{ e }} is not MUTABLE_STATE.".replace("MUTABLE_STATE", trait_features['mutable_states'][0])
            fail_precondition = [
                ["You are not where you are! (This should not occur.)", "world_state_discrepancy"],
                ["You can't see a {{ e }} here.", "entity_not_accessible"],
                [fail_precondition_entity_state, "entity_state_mismatch"]
            ]

            # full failure feedback:
            new_action['failure_feedback'] = {
                'parameters': fail_parameters,
                'precondition': fail_precondition
            }

            # SUCCESS FEEDBACK
            success_feedback = "The {{ e }} is now MUTABLE_STATE_POST.".replace("MUTABLE_STATE_POST", trait_features['mutable_states'][1])
            new_action['success_feedback'] = success_feedback

            # Note: ASP skipped for now, hopefully PDDL->ASP conversion is in the cards to automate this

            # EPISTEMIC/PRAGMATIC
            new_action['epistemic'] = False
            new_action['pragmatic'] = True

            # EXPLANATION
            # generic and containing full mutable state and trait info
            explanation_text = f"To {new_word} is to make something {trait} and {trait_features['mutable_states'][0]} be {trait_features['mutable_states'][1]}."
            new_action['explanation'] = explanation_text

            # FINISH ACTION
            new_word_actions_definitions.append(new_action)
            created_actions_count += 1

            # SECOND ACTION TYPE: B->A
            new_action = dict()

            new_word = new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['VB']
            new_word_idx += 1

            action_tag = new_word.upper()

            new_action['type_name'] = new_word
            # lark grammar snippet:
            lark_string = f"{new_word}: {action_tag} thing\n{action_tag}.1: \"{new_word}\" WS"
            # print(lark_string)
            new_action['lark'] = lark_string

            # PDDL
            # expose values to allow feedback creation?

            # parameters:
            # NB: new word mutable state is always first parameters item
            pddl_parameters = f":parameters (?e - {trait} ?r - room ?p - player)"
            # precondition:
            # NB: new word mutable state is always third precondition item
            pddl_precondition = f":precondition (and\n        (at ?p ?r)\n        (at ?e ?r)\n        ({trait_features['mutable_states'][1]} ?e)\n        )"
            # effect:
            # add second mutable state, remove first mutable state
            pddl_effect = f":effect (and\n        ({trait_features['mutable_states'][0]} ?e)\n        (not ({trait_features['mutable_states'][1]} ?e))\n    )"
            # full PDDL action string:
            pddl_action = f"(:action {action_tag}\n    {pddl_parameters}\n    {pddl_precondition}\n    {pddl_effect}\n)"
            # print(pddl_action)
            new_action['pddl'] = pddl_action

            # PDDL parameter mapping:
            new_action['pddl_parameter_mapping'] = {
                "?e": ["arg1"],
                "?r": ["current_player_room"],
                "?p": ["player"]
            }

            # FAILURE FEEDBACK
            # parameters failures:
            trait_dashed = trait.replace("_", "-")
            fail_parameters_trait_type = "The {{ e }} is not MUTABILITY_TRAIT.".replace("MUTABILITY_TRAIT", trait_dashed)
            fail_parameters = [
                [fail_parameters_trait_type, "domain_trait_type_mismatch"],
                ["{{ r }} is not a room. (This should not occur.)", "domain_type_discrepancy"],
                ["{{ p }} is not a player. (This should not occur.)", "domain_type_discrepancy"]
            ]

            # precondition failures:
            fail_precondition_entity_state = "The {{ e }} is not MUTABLE_STATE.".replace("MUTABLE_STATE",
                                                                                         trait_features[
                                                                                             'mutable_states'][1])
            fail_precondition = [
                ["You are not where you are! (This should not occur.)", "world_state_discrepancy"],
                ["You can't see a {{ e }} here.", "entity_not_accessible"],
                [fail_precondition_entity_state, "entity_state_mismatch"]
            ]

            # full failure feedback:
            new_action['failure_feedback'] = {
                'parameters': fail_parameters,
                'precondition': fail_precondition
            }

            # SUCCESS FEEDBACK
            success_feedback = "The {{ e }} is now MUTABLE_STATE_POST.".replace("MUTABLE_STATE_POST",
                                                                                trait_features['mutable_states'][0])
            new_action['success_feedback'] = success_feedback

            # Note: ASP skipped for now, hopefully PDDL->ASP conversion is in the cards to automate this

            # EPISTEMIC/PRAGMATIC
            new_action['epistemic'] = False
            new_action['pragmatic'] = True

            # EXPLANATION
            # generic and containing full mutable state and trait info
            explanation_text = f"To {new_word} is to make something {trait} and {trait_features['mutable_states'][1]} be {trait_features['mutable_states'][0]}."
            new_action['explanation'] = explanation_text

            # FINISH ACTION
            new_word_actions_definitions.append(new_action)
            created_actions_count += 1

        if trait_features['interaction'] == "trinary" and created_actions_count < num_actions_created:
            # create three action types
            # FIRST ACTION TYPE: A->B
            new_action = dict()

            new_word = new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['VB']
            new_word_idx += 1

            action_tag = new_word.upper()

            new_action['type_name'] = new_word
            # lark grammar snippet:
            lark_string = f"{new_word}: {action_tag} thing\n{action_tag}.1: \"{new_word}\" WS"
            # print(lark_string)
            new_action['lark'] = lark_string

            # PDDL
            # expose values to allow feedback creation?

            # parameters:
            # NB: new word mutable state is always first parameters item
            pddl_parameters = f":parameters (?e - {trait} ?r - room ?p - player)"
            # precondition:
            # NB: new word mutable state is always third precondition item
            pddl_precondition = f":precondition (and\n        (at ?p ?r)\n        (at ?e ?r)\n        ({trait_features['mutable_states'][0]} ?e)\n        )"
            # effect:
            # add second mutable state, remove first mutable state
            pddl_effect = f":effect (and\n        ({trait_features['mutable_states'][1]} ?e)\n        (not ({trait_features['mutable_states'][0]} ?e))\n    )"
            # full PDDL action string:
            pddl_action = f"(:action {action_tag}\n    {pddl_parameters}\n    {pddl_precondition}\n    {pddl_effect}\n)"
            # print(pddl_action)
            new_action['pddl'] = pddl_action

            # PDDL parameter mapping:
            new_action['pddl_parameter_mapping'] = {
                "?e": ["arg1"],
                "?r": ["current_player_room"],
                "?p": ["player"]
            }

            # FAILURE FEEDBACK
            # parameters failures:
            trait_dashed = trait.replace("_", "-")
            fail_parameters_trait_type = "The {{ e }} is not MUTABILITY_TRAIT.".replace("MUTABILITY_TRAIT", trait_dashed)
            fail_parameters = [
                [fail_parameters_trait_type, "domain_trait_type_mismatch"],
                ["{{ r }} is not a room. (This should not occur.)", "domain_type_discrepancy"],
                ["{{ p }} is not a player. (This should not occur.)", "domain_type_discrepancy"]
            ]

            # precondition failures:
            fail_precondition_entity_state = "The {{ e }} is not MUTABLE_STATE.".replace("MUTABLE_STATE",
                                                                                         trait_features[
                                                                                             'mutable_states'][0])
            fail_precondition = [
                ["You are not where you are! (This should not occur.)", "world_state_discrepancy"],
                ["You can't see a {{ e }} here.", "entity_not_accessible"],
                [fail_precondition_entity_state, "entity_state_mismatch"]
            ]

            # full failure feedback:
            new_action['failure_feedback'] = {
                'parameters': fail_parameters,
                'precondition': fail_precondition
            }

            # SUCCESS FEEDBACK
            success_feedback = "The {{ e }} is now MUTABLE_STATE_POST.".replace("MUTABLE_STATE_POST",
                                                                                trait_features['mutable_states'][1])
            new_action['success_feedback'] = success_feedback

            # Note: ASP skipped for now, hopefully PDDL->ASP conversion is in the cards to automate this

            # EPISTEMIC/PRAGMATIC
            new_action['epistemic'] = False
            new_action['pragmatic'] = True

            # EXPLANATION
            # generic and containing full mutable state and trait info
            explanation_text = f"To {new_word} is to make something {trait} and {trait_features['mutable_states'][0]} be {trait_features['mutable_states'][1]}."
            new_action['explanation'] = explanation_text

            # FINISH ACTION
            new_word_actions_definitions.append(new_action)
            created_actions_count += 1

            # SECOND ACTION TYPE: B->C
            new_action = dict()

            new_word = new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['VB']
            new_word_idx += 1

            action_tag = new_word.upper()

            new_action['type_name'] = new_word
            # lark grammar snippet:
            lark_string = f"{new_word}: {action_tag} thing\n{action_tag}.1: \"{new_word}\" WS"
            # print(lark_string)
            new_action['lark'] = lark_string

            # PDDL
            # expose values to allow feedback creation?

            # parameters:
            # NB: new word mutable state is always first parameters item
            pddl_parameters = f":parameters (?e - {trait} ?r - room ?p - player)"
            # precondition:
            # NB: new word mutable state is always third precondition item
            pddl_precondition = f":precondition (and\n        (at ?p ?r)\n        (at ?e ?r)\n        ({trait_features['mutable_states'][1]} ?e)\n        )"
            # effect:
            # add second mutable state, remove first mutable state
            pddl_effect = f":effect (and\n        ({trait_features['mutable_states'][2]} ?e)\n        (not ({trait_features['mutable_states'][1]} ?e))\n    )"
            # full PDDL action string:
            pddl_action = f"(:action {action_tag}\n    {pddl_parameters}\n    {pddl_precondition}\n    {pddl_effect}\n)"
            # print(pddl_action)
            new_action['pddl'] = pddl_action

            # PDDL parameter mapping:
            new_action['pddl_parameter_mapping'] = {
                "?e": ["arg1"],
                "?r": ["current_player_room"],
                "?p": ["player"]
            }

            # FAILURE FEEDBACK
            # parameters failures:
            trait_dashed = trait.replace("_", "-")
            fail_parameters_trait_type = "The {{ e }} is not MUTABILITY_TRAIT.".replace("MUTABILITY_TRAIT", trait_dashed)
            fail_parameters = [
                [fail_parameters_trait_type, "domain_trait_type_mismatch"],
                ["{{ r }} is not a room. (This should not occur.)", "domain_type_discrepancy"],
                ["{{ p }} is not a player. (This should not occur.)", "domain_type_discrepancy"]
            ]

            # precondition failures:
            fail_precondition_entity_state = "The {{ e }} is not MUTABLE_STATE.".replace("MUTABLE_STATE",
                                                                                         trait_features[
                                                                                             'mutable_states'][1])
            fail_precondition = [
                ["You are not where you are! (This should not occur.)", "world_state_discrepancy"],
                ["You can't see a {{ e }} here.", "entity_not_accessible"],
                [fail_precondition_entity_state, "entity_state_mismatch"]
            ]

            # full failure feedback:
            new_action['failure_feedback'] = {
                'parameters': fail_parameters,
                'precondition': fail_precondition
            }

            # SUCCESS FEEDBACK
            success_feedback = "The {{ e }} is now MUTABLE_STATE_POST.".replace("MUTABLE_STATE_POST",
                                                                                trait_features['mutable_states'][2])
            new_action['success_feedback'] = success_feedback

            # Note: ASP skipped for now, hopefully PDDL->ASP conversion is in the cards to automate this

            # EPISTEMIC/PRAGMATIC
            new_action['epistemic'] = False
            new_action['pragmatic'] = True

            # EXPLANATION
            # generic and containing full mutable state and trait info
            explanation_text = f"To {new_word} is to make something {trait} and {trait_features['mutable_states'][1]} be {trait_features['mutable_states'][2]}."
            new_action['explanation'] = explanation_text

            # FINISH ACTION
            new_word_actions_definitions.append(new_action)
            created_actions_count += 1

            # THIRD ACTION TYPE: C->A
            new_action = dict()

            new_word = new_words_source[list(new_words_source.keys())[new_word_idx]]['pos']['VB']
            new_word_idx += 1

            action_tag = new_word.upper()

            new_action['type_name'] = new_word
            # lark grammar snippet:
            lark_string = f"{new_word}: {action_tag} thing\n{action_tag}.1: \"{new_word}\" WS"
            # print(lark_string)
            new_action['lark'] = lark_string

            # PDDL
            # expose values to allow feedback creation?

            # parameters:
            # NB: new word mutable state is always first parameters item
            pddl_parameters = f":parameters (?e - {trait} ?r - room ?p - player)"
            # precondition:
            # NB: new word mutable state is always third precondition item
            pddl_precondition = f":precondition (and\n        (at ?p ?r)\n        (at ?e ?r)\n        ({trait_features['mutable_states'][2]} ?e)\n        )"
            # effect:
            # add second mutable state, remove first mutable state
            pddl_effect = f":effect (and\n        ({trait_features['mutable_states'][0]} ?e)\n        (not ({trait_features['mutable_states'][2]} ?e))\n    )"
            # full PDDL action string:
            pddl_action = f"(:action {action_tag}\n    {pddl_parameters}\n    {pddl_precondition}\n    {pddl_effect}\n)"
            # print(pddl_action)
            new_action['pddl'] = pddl_action

            # PDDL parameter mapping:
            new_action['pddl_parameter_mapping'] = {
                "?e": ["arg1"],
                "?r": ["current_player_room"],
                "?p": ["player"]
            }

            # FAILURE FEEDBACK
            # parameters failures:
            trait_dashed = trait.replace("_", "-")
            fail_parameters_trait_type = "The {{ e }} is not MUTABILITY_TRAIT.".replace("MUTABILITY_TRAIT", trait_dashed)
            fail_parameters = [
                [fail_parameters_trait_type, "domain_trait_type_mismatch"],
                ["{{ r }} is not a room. (This should not occur.)", "domain_type_discrepancy"],
                ["{{ p }} is not a player. (This should not occur.)", "domain_type_discrepancy"]
            ]

            # precondition failures:
            fail_precondition_entity_state = "The {{ e }} is not MUTABLE_STATE.".replace("MUTABLE_STATE",
                                                                                         trait_features[
                                                                                             'mutable_states'][2])
            fail_precondition = [
                ["You are not where you are! (This should not occur.)", "world_state_discrepancy"],
                ["You can't see a {{ e }} here.", "entity_not_accessible"],
                [fail_precondition_entity_state, "entity_state_mismatch"]
            ]

            # full failure feedback:
            new_action['failure_feedback'] = {
                'parameters': fail_parameters,
                'precondition': fail_precondition
            }

            # SUCCESS FEEDBACK
            success_feedback = "The {{ e }} is now MUTABLE_STATE_POST.".replace("MUTABLE_STATE_POST",
                                                                                trait_features['mutable_states'][0])
            new_action['success_feedback'] = success_feedback

            # Note: ASP skipped for now, hopefully PDDL->ASP conversion is in the cards to automate this

            # EPISTEMIC/PRAGMATIC
            new_action['epistemic'] = False
            new_action['pragmatic'] = True

            # EXPLANATION
            # generic and containing full mutable state and trait info
            explanation_text = f"To {new_word} is to make something {trait} and {trait_features['mutable_states'][2]} be {trait_features['mutable_states'][0]}."
            new_action['explanation'] = explanation_text

            # FINISH ACTION
            new_word_actions_definitions.append(new_action)
            created_actions_count += 1

    return new_word_actions_definitions, trait_dict, new_word_idx


# DOMAINS
# best after other def types, as it relies on their contents existing in adventure/domain
# not covering functions, basic facts are enough for planned experiments

def process_to_pddl_domain(domain_name: str, room_definitions: list, entity_definitions: list, trait_dict: dict):
    """Get domain definition from room and entity definitions.
    Covers only type definition for now, functions not required for v2.2 experiments.
    Args:
        domain_name: Name of the resulting domain. (For PDDL conformation, but might get used for experiments/instances)
        room_definitions: List of room type definitions.
        entity_definitions: List of entity type definitions.
        trait_dict: Dict of mutability traits and their associated mutable predicates.
    """
    # ROOMS
    room_types = list()
    for room_type in room_definitions:
        room_types.append(room_type['type_name'])
    room_line = f"        {' '.join(room_types)} - room\n"

    # ENTITIES
    entity_types = list()
    for entity_type in entity_definitions:
        entity_types.append(entity_type['type_name'])
    entity_line = f"        player inventory floor {' '.join(entity_types)} - entity\n"

    # TRAITS
    # get all traits:
    traits = list()
    for entity_def in entity_definitions:
        if 'traits' in entity_def:
            for trait in entity_def['traits']:
                if trait not in traits:
                    traits.append(trait)

    trait_lines = list()
    for trait in traits:
        entities_with_trait = list()
        for entity_def in entity_definitions:
            if trait in entity_def['traits']:
                entities_with_trait.append(entity_def['type_name'])
        trait_line = f"        {' '.join(entities_with_trait)} - {trait}\n"
        trait_lines.append(trait_line)

    # PREDICATES
    # print(trait_dict)
    predicate_lines = list()
    for mutability_name, mutability_values in trait_dict.items():
        for mutable in mutability_values['mutable_states']:
            predicate_line = f"        ({mutable} ?e - {mutability_name})\n"
            predicate_lines.append(predicate_line)

    # COMBINED DOMAIN DEFINITION
    full_domain = (f"(define\n"
                   f"    (domain {domain_name})\n"
                   f"    (:types\n{room_line}{entity_line}{''.join(trait_lines)}        )\n"
                   f"    (:predicates\n{''.join(predicate_lines)}        )\n"
                   f"    )")

    return full_domain


# COMBINED CREATION FUNCTIONS

def create_new_words_definitions_set(initial_new_word_idx: int = 0, seed: int = 42, verbose: bool = False):
    """Create an entire set of new-word definitions.
    Returns:
        Tuple of: New room definitions, entity definitions, action definitions and domain definition.
    """
    new_room_definitions, last_new_word_idx = new_word_rooms_create(last_new_words_idx=initial_new_word_idx, seed=seed)
    new_entity_definitions, last_new_word_idx, trait_pool, adjective_pool = new_word_entities_create(new_room_definitions,
                                                                                                     add_traits=True, limited_trait_pool=3, min_traits=1,
                                                                                   last_new_words_idx=last_new_word_idx, seed=seed)
    new_action_definitions, trait_dict, last_new_word_idx = new_word_actions_create(new_entity_definitions,
                                                                        last_new_words_idx=last_new_word_idx, seed=seed)

    # print("trait_dict:", trait_dict)

    if verbose:
        num_defs_created = len(new_room_definitions) + len(new_entity_definitions) + len(new_action_definitions)
        mutabilities_created = [mutability_name for mutability_name in trait_dict]
        mutables_created = list()
        for mutability_values in trait_dict.values():
            for mutable in mutability_values['mutable_states']:
                mutables_created.append(mutable)

        print(f"{last_new_word_idx+1} new-words used to create {num_defs_created} definitions "
              f"({len(new_room_definitions)} room types, {len(new_entity_definitions)} entity types, "
              f"{len(new_action_definitions)} action types) with {len(mutables_created)} mutable predicates under "
              f"{len(mutabilities_created)} mutability traits.")
        print(f"Mutable predicates: {mutables_created}; mutability traits: {mutabilities_created}")

    new_domain_definition = process_to_pddl_domain("new_words", new_room_definitions, new_entity_definitions, trait_dict)

    return new_room_definitions, new_entity_definitions, new_action_definitions, new_domain_definition, trait_dict, last_new_word_idx


def replace_new_words_definitions_set(initial_new_word_idx: int = 0, seed: int = 42, verbose: bool = False,
                                      # room_definition_source: str = "../definitions/home_rooms.json",
                                      room_definition_source: str = "definitions/home_rooms.json",
                                      room_replace_n: int = 1,
                                      # entity_definition_source: str = "../definitions/home_entities.json",
                                      entity_definition_source: str = "definitions/home_entities_v2.json",
                                      entity_replace_n: int = 3,
                                      # action_definition_source: str = "../definitions/basic_actions_v2_2_replace.json",
                                      action_definition_source: str = "definitions/basic_actions_v2_2_replace.json",
                                      action_replace_n: int = 1,
                                      ):
    """Replace a number of definitions with new-word definitions.
    Returns:
        Tuple of: New room definitions, entity definitions, action definitions and domain definition.
    """
    new_room_definitions, last_new_word_idx, rooms_replaced = new_word_rooms_replace(
        source_definition_file_path=room_definition_source, num_replace=room_replace_n,
        last_new_words_idx=initial_new_word_idx, seed=seed)

    new_entity_definitions, last_new_word_idx, entities_replaced = new_word_entities_replace(
        source_definition_file_path=entity_definition_source, num_replace=entity_replace_n,
        last_new_words_idx=last_new_word_idx, seed=seed)

    new_action_definitions, last_new_word_idx, actions_replaced = new_word_actions_replace(
        source_definition_file_path=action_definition_source, num_replace=action_replace_n,
        last_new_words_idx=last_new_word_idx, seed=seed)

    # TODO: randomize replaced room/entities/action properly

    # preliminary hardcode trait dict for home delivery:
    trait_dict = {'openable': {'interaction': 'binary',
                               'mutable_states': ['open', 'closed'],
                               'mutable_set_type': 'paired'}}
    # print("trait_dict:", trait_dict)

    for room_def in new_room_definitions:
        # print(room_def['standard_content'])
        for entity_idx, std_entity in enumerate(room_def['standard_content']):
            if std_entity in entities_replaced:
                room_def['standard_content'][entity_idx] = entities_replaced[std_entity]
        for exit_idx, std_exit in enumerate(room_def['exit_targets']):
            if std_exit in rooms_replaced:
                room_def['exit_targets'][exit_idx] = rooms_replaced[std_exit]

    for entity_def in new_entity_definitions:
        if 'standard_locations' in entity_def:
            for room_idx, std_room in enumerate(entity_def['standard_locations']):
                if std_room in rooms_replaced:
                    entity_def['standard_locations'][room_idx] = rooms_replaced[std_room]

    replacement_dict = {
        "rooms": rooms_replaced,
        "entities": entities_replaced,
        "actions": actions_replaced
    }

    if verbose:
        print(f"{last_new_word_idx+1} new-words used to replace definitions "
              f"({len(new_room_definitions)} room types, {len(new_entity_definitions)} entity types, "
              f"{len(new_action_definitions)} action types.")
        print("Rooms replaced:", rooms_replaced)
        print("Entities replaced:", entities_replaced)
        print("Actions replaced:", actions_replaced)

    new_domain_definition = process_to_pddl_domain("partial_new_words", new_room_definitions, new_entity_definitions, trait_dict)

    return new_room_definitions, new_entity_definitions, new_action_definitions, new_domain_definition, trait_dict, replacement_dict, last_new_word_idx


if __name__ == "__main__":
    """
    new_word_rooms, replacement_dict = new_word_rooms_replace("../definitions/home_rooms.json", 2)
    print(new_word_rooms)
    print(replacement_dict)
    """

    # new_word_actions, new_word_idx, replacement_dict = new_word_actions_replace("../definitions/basic_actions_v2-2.json")

    # create set of new word rooms, entities and actions:
    # new_rooms, new_entities, new_actions, new_domain, trait_dict, last_new_word_idx = create_new_words_definitions_set(verbose=True)
    # replace home_delivery defs:
    new_rooms, new_entities, new_actions, new_domain, trait_dict, replacement_dict, last_new_word_idx = replace_new_words_definitions_set(verbose=True)
    print(new_rooms)
    print(new_domain)
    print(trait_dict)
    """
    # save created definitions to JSON:
    with open("new_rooms_test.json", 'w', encoding='utf-8') as rooms_out_file:
        json.dump(new_rooms, rooms_out_file, indent=2)
    with open("new_entities_test.json", 'w', encoding='utf-8') as entities_out_file:
        json.dump(new_entities, entities_out_file, indent=2)
    with open("new_actions_test.json", 'w', encoding='utf-8') as actions_out_file:
        json.dump(new_actions, actions_out_file, indent=2)
    with open("new_domain_test.json", 'w', encoding='utf-8') as domain_out_file:
        json.dump({'pddl_domain': new_domain}, domain_out_file, indent=2)
    """