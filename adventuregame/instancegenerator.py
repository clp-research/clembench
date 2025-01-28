"""
Generate instances for adventuregame.

Creates files in ./in
"""
import os

from tqdm import tqdm

import clemcore
from clemcore.clemgame import GameInstanceGenerator

import logging

logger = logging.getLogger(__name__)


class AdventureGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(os.path.abspath(__file__)))

    def on_generate(self, raw_adventures_file: str):
        """Generate both basic and planning variant instances from raw adventures.
        Args:
            raw_adventures_file: File name of the JSON file containing raw adventures data.
        """
        # load generated home_deliver_two adventures:
        adventures = self.load_json(f"resources/{raw_adventures_file}")

        # get difficulties:
        difficulties = list(adventures.keys())

        # get adventure type from first raw adventure:
        adventure_type = adventures[difficulties[0]][0]['adventure_type']

        for difficulty in difficulties:
            # BASIC

            # create an experiment:
            basic_experiment = self.add_experiment(f"{adventure_type}_basic_{difficulty}")

            # Load the prepared initial prompt
            # basic_prompt = self.load_template("resources/initial_prompts/basic_prompt")
            basic_prompt = self.load_template("resources/initial_prompts/basic_prompt_done")
            # TODO?: externalize which prompt templates to use into adventure_type/experiment_type definition(s)?

            for adventure_id in tqdm(range(len(adventures[difficulty]))):
                goal_str = adventures[difficulty][adventure_id]['goal']

                initial_state = adventures[difficulty][adventure_id]['initial_state']
                goal_state = adventures[difficulty][adventure_id]['goal_state']

                # Replace the goal in the templated initial prompt
                instance_prompt = basic_prompt.replace("$GOAL$", goal_str)
                # instance_prompt = instance_prompt.replace("$FIRST_ROOM$", first_room_str)

                # Create a game instance
                game_instance = self.add_game_instance(basic_experiment, adventure_id)
                game_instance["variant"] = "basic"  # game parameters
                game_instance["prompt"] = instance_prompt  # game parameters
                # game_instance["goal_str"] = goal_str  # game parameters
                # game_instance["first_room_str"] = first_room_str  # game parameters
                game_instance["initial_state"] = initial_state  # game parameters
                game_instance["goal_state"] = goal_state  # game parameters
                game_instance["max_turns"] = adventures[difficulty][adventure_id]['bench_turn_limit']  # game parameters
                game_instance["optimal_turns"] = adventures[difficulty][adventure_id]['optimal_turns']  # game parameters
                game_instance["optimal_solution"] = adventures[difficulty][adventure_id]['optimal_solution']  # game parameters
                game_instance["optimal_commands"] = adventures[difficulty][adventure_id]['optimal_commands']  # game parameters
                game_instance["action_definitions"] = adventures[difficulty][adventure_id]['action_definitions']  # game parameters
                game_instance["room_definitions"] = adventures[difficulty][adventure_id]['room_definitions']  # game parameters
                game_instance["entity_definitions"] = adventures[difficulty][adventure_id]['entity_definitions']  # game parameters
                game_instance["domain_definitions"] = adventures[difficulty][adventure_id]['domain_definitions']  # game parameters

            # PLANNING

            # create an experiment:
            planning_experiment = self.add_experiment(f"{adventure_type}_planning_{difficulty}")

            # Load the prepared initial prompt
            # planning_prompt = self.load_template("resources/initial_prompts/plan_prompt")
            planning_prompt = self.load_template("resources/initial_prompts/plan_prompt_done")

            for adventure_id in tqdm(range(len(adventures[difficulty]))):
                goal_str = adventures[difficulty][adventure_id]['goal']
                # first_room_str = adventures[adventure_id]['first_room']

                initial_state = adventures[difficulty][adventure_id]['initial_state']
                goal_state = adventures[difficulty][adventure_id]['goal_state']

                # Replace the goal in the templated initial prompt
                instance_prompt = planning_prompt.replace("$GOAL$", goal_str)
                # instance_prompt = instance_prompt.replace("$FIRST_ROOM$", first_room_str)

                # Create a game instance
                game_instance = self.add_game_instance(planning_experiment, adventure_id)
                game_instance["variant"] = "plan"  # game parameters
                game_instance["prompt"] = instance_prompt  # game parameters
                # game_instance["goal_str"] = goal_str  # game parameters
                # game_instance["first_room_str"] = first_room_str  # game parameters
                game_instance["initial_state"] = initial_state  # game parameters
                game_instance["goal_state"] = goal_state  # game parameters
                game_instance["max_turns"] = adventures[difficulty][adventure_id]['bench_turn_limit']  # game parameters
                game_instance["optimal_turns"] = adventures[difficulty][adventure_id]['optimal_turns']  # game parameters
                game_instance["optimal_solution"] = adventures[difficulty][adventure_id]['optimal_solution']  # game parameters
                game_instance["optimal_commands"] = adventures[difficulty][adventure_id]['optimal_commands']  # game parameters
                game_instance["action_definitions"] = adventures[difficulty][adventure_id]['action_definitions']  # game parameters
                game_instance["room_definitions"] = adventures[difficulty][adventure_id]['room_definitions']  # game parameters
                game_instance["entity_definitions"] = adventures[difficulty][adventure_id]['entity_definitions']  # game parameters
                game_instance["domain_definitions"] = adventures[difficulty][adventure_id]['domain_definitions']  # game parameters

            # BASIC INVENTORY LIMIT

            # create an experiment:
            basic_invlimit_experiment = self.add_experiment(f"{adventure_type}_basic_{difficulty}_invlimittwo")

            # Load the prepared initial prompt
            basic_invlimit_prompt = self.load_template("resources/initial_prompts/basic_prompt_done_invlimittwo")

            for adventure_id in tqdm(range(len(adventures[difficulty]))):
                goal_str = adventures[difficulty][adventure_id]['goal']

                initial_state = adventures[difficulty][adventure_id]['initial_state']
                goal_state = adventures[difficulty][adventure_id]['goal_state']

                # Replace the goal in the templated initial prompt
                instance_prompt = basic_invlimit_prompt.replace("$GOAL$", goal_str)
                # instance_prompt = instance_prompt.replace("$FIRST_ROOM$", first_room_str)

                # Create a game instance
                game_instance = self.add_game_instance(basic_invlimit_experiment, adventure_id)
                game_instance["variant"] = "basic"  # game parameters
                game_instance["prompt"] = instance_prompt  # game parameters
                # game_instance["goal_str"] = goal_str  # game parameters
                # game_instance["first_room_str"] = first_room_str  # game parameters
                game_instance["initial_state"] = initial_state  # game parameters
                game_instance["goal_state"] = goal_state  # game parameters
                game_instance["max_turns"] = adventures[difficulty][adventure_id]['bench_turn_limit']  # game parameters
                game_instance["optimal_turns"] = adventures[difficulty][adventure_id]['optimal_turns']  # game parameters
                game_instance["optimal_solution"] = adventures[difficulty][adventure_id]['optimal_solution']  # game parameters
                game_instance["optimal_commands"] = adventures[difficulty][adventure_id]['optimal_commands']  # game parameters

                game_instance["action_definitions"] = ["basic_actions_v2_invlimit.json"]  # game parameters

                game_instance["room_definitions"] = adventures[difficulty][adventure_id]['room_definitions']  # game parameters
                game_instance["entity_definitions"] = adventures[difficulty][adventure_id]['entity_definitions']  # game parameters

                game_instance["domain_definitions"] = ["home_domain_invlimit.json"]  # game parameters

            # PLANNING INVENTORY LIMIT

            # create an experiment:
            planning_invlimit_experiment = self.add_experiment(f"{adventure_type}_planning_{difficulty}_invlimittwo")

            # Load the prepared initial prompt
            planning_invlimit_prompt = self.load_template("resources/initial_prompts/plan_prompt_done_invlimittwo")

            for adventure_id in tqdm(range(len(adventures[difficulty]))):
                goal_str = adventures[difficulty][adventure_id]['goal']

                initial_state = adventures[difficulty][adventure_id]['initial_state']
                goal_state = adventures[difficulty][adventure_id]['goal_state']

                # Replace the goal in the templated initial prompt
                instance_prompt = planning_invlimit_prompt.replace("$GOAL$", goal_str)
                # instance_prompt = instance_prompt.replace("$FIRST_ROOM$", first_room_str)

                # Create a game instance
                game_instance = self.add_game_instance(planning_invlimit_experiment, adventure_id)
                game_instance["variant"] = "plan"  # game parameters
                game_instance["prompt"] = instance_prompt  # game parameters
                # game_instance["goal_str"] = goal_str  # game parameters
                # game_instance["first_room_str"] = first_room_str  # game parameters
                game_instance["initial_state"] = initial_state  # game parameters
                game_instance["goal_state"] = goal_state  # game parameters
                game_instance["max_turns"] = adventures[difficulty][adventure_id]['bench_turn_limit']  # game parameters
                game_instance["optimal_turns"] = adventures[difficulty][adventure_id][
                    'optimal_turns']  # game parameters
                game_instance["optimal_solution"] = adventures[difficulty][adventure_id][
                    'optimal_solution']  # game parameters
                game_instance["optimal_commands"] = adventures[difficulty][adventure_id][
                    'optimal_commands']  # game parameters

                game_instance["action_definitions"] = ["basic_actions_v2_invlimit.json"]  # game parameters

                game_instance["room_definitions"] = adventures[difficulty][adventure_id][
                    'room_definitions']  # game parameters
                game_instance["entity_definitions"] = adventures[difficulty][adventure_id][
                    'entity_definitions']  # game parameters

                game_instance["domain_definitions"] = ["home_domain_invlimit.json"]  # game parameters


if __name__ == '__main__':
    # The resulting instances.json is automatically saved to the "in" directory of the game folder
    AdventureGameInstanceGenerator().generate(raw_adventures_file="curated_home_deliver_three_adventures_v2")
