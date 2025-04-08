import os

from typing import List, Dict

from clemcore import backends
from clemcore.backends import Model
from clemcore.utils import file_utils
import clemcore.clemgame.metrics as metrics
from clemcore.clemgame import GameSpec, GameMaster, GameBenchmark, GameScorer, DialogueGameMaster, Player

import logging

import numpy as np

from if_wrapper import AdventureIFInterpreter

logger = logging.getLogger(__name__)


class Adventurer(Player):

    def __init__(self, model: backends.Model):
        super().__init__(model)

    def _custom_response(self, context: Dict) -> str:
        return "Go"


class AdventureGameMaster(DialogueGameMaster):
    """
    DialogueGameMaster subclass for AdventureGame.
    Runs the benchmark by prompting the model and passing model outputs to the IF interpreter.
    Handles prompted format adherence checks and creates episode records.
    """

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)
        self.game_path = game_path
        self.success = True
        self.invalid_format: str = ""  # to track responses with invalid format
        self.finished: bool = False  # game finished successfully
        self.model_done = False  # model used DONE action to end game

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance  # fetch game parameters here
        # check game variant; 'basic' or 'planning':
        self.if_variant = self.game_instance['variant']
        # initialize IF interpreter:
        self.if_interpreter = AdventureIFInterpreter(self.game_path, self.game_instance)
        # create clem player:
        self.player = Adventurer(self.player_models[0])
        # Add the players: these will be logged to the records interactions.json
        # Note: During game play the players will be called in the order added here
        self.add_player(self.player)
        # keep history of plans:
        if self.if_variant == 'plan':
            self.plan_history: list = list()
            self.plan_success_ratio_history: list = list()  # for 'bad' plan scoring
        # get goal data set from game instance:
        self.goals_required = set(self.game_instance['goal_state'])
        self.goals_required_cnt = len(self.goals_required)
        # initially empty set of achieved goals:
        self.goals_achieved = set()
        # get and record adventure information:
        adventure_info: dict = {"variant": self.game_instance['variant'], "max_turns": self.game_instance['max_turns'],
                                "optimal_turns": self.game_instance['optimal_turns'],
                                "goal_count": self.goals_required_cnt}
        self.log_key("adventure_info", adventure_info)

    def _on_before_game(self):
        # get initial room description from IF interpreter:
        initial_room_desc = self.if_interpreter.get_full_room_desc()
        # combine prompt with initial room description as first message:
        first_message = self.game_instance["prompt"] + initial_room_desc
        # add the initial prompts to the message history:
        self.set_context_for(self.player, first_message)

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        self.log_to_self("metadata", f"Round: {self.current_round}")
        # logger.info(f"Player response:\n{utterance}")
        # check player response:
        if player == self.player:
            # check rule: response must start with IF >
            if not utterance.startswith(">"):
                self.success = False
                # hallucinated finish heuristic:
                hallucinated_finish_strs = ["complete", "finish", "done", "successfully"]
                for hallucinated_finish_str in hallucinated_finish_strs:
                    if hallucinated_finish_str in utterance:
                        self.log_to_self("hallucinated_finish", utterance)
                        break
                self.invalid_format = "command_tag_missing"
                return False
            if self.if_variant == 'plan':
                # check rule: response must contain 'Next actions:' on its own line
                # if utterance is DONE action, don't fail
                if "\nNext actions:" not in utterance and "done" not in utterance:
                    self.success = False
                    self.invalid_format = "next_actions_missing"
                    return False
        return True

    def _parse_response(self, player: Player, utterance: str) -> str:
        """
        Decide if a response utterance should be modified. If not simply return the utterance.
        When a modified utterance and a true value is returned, then a 'parse' event is logged.

        For the planning variant, this extracts plans and adds them to the plan history for later processing.

        :param player: Clem player that produced the response.
        :param utterance: The response string to be potentially modified.
        :return: The (modified) response and if to log the parse action (default: True)
        """
        # logger.info(f"AdventureGameMaster._on_parse_response() input utterance: {utterance}")
        if self.if_variant == 'plan':
            # do not split for next actions plan if action is 'done'
            if utterance == "> done":
                return utterance
            # split the response to extract only the planned actions:
            split_response = utterance.split("\nNext actions:")
            if len(split_response) >= 2:
                new_plan = utterance.split("\nNext actions:")[1]
                # split by comma and strip to get assumed individual action commands:
                plan_sequence = [command.strip() for command in new_plan.split(",")]
                # add new plan sequence to plan history:
                self.plan_history.append(plan_sequence)
                # record the new plan for processing:
                self.log_to_self(f"turn_plan", plan_sequence)
        return utterance

    def _does_game_proceed(self) -> bool:
        """
        Checks if game proceeds.
        Game does NOT proceed due to: Invalid output format, reaching the turn limit or model performing DONE action.
        Achieving all goal states is recorded here, but does not end the episode.
        """
        # record invalid format failures:
        if self.invalid_format:
            self.log_to_self("invalid_format", self.invalid_format)
            return False
        # check if all goal states have been achieved:
        if self.goals_achieved == self.goals_required:
            self.finished = True
            self.log_to_self("adventure_finished", list(self.goals_achieved))  # can be JSON'd; for easier eval
            # return False  # do not stop game when all goal states have been achieved
        # stop game when turn limit is reached:
        if self.current_round >= self.game_instance['max_turns']:
            self.log_to_self("turn_limit_reached",
                             f"Turn limit {self.game_instance['max_turns']} reached, end episode.")
            return False
        # stop game when model used DONE action:
        if self.model_done:
            self.log_to_self("model_done",
                             f"Model produced DONE action at turn {self.current_round}, end episode.")
            return False
        # otherwise keep playing:
        return True

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        # IF INTERACTION
        # logger.info(f"Raw last message:\n{last_action}")
        # strip player action to IF input; only first line action command is used:
        if_input: str = parsed_response[1:].split("\n")[0].strip()
        logger.info(f"Stripped IF input: {if_input}")

        # count achieved goals:
        prior_goal_count = len(self.goals_achieved)

        # send to IF interpreter to process action:
        goals_achieved, if_response, action_info = self.if_interpreter.process_action(if_input)
        # IF interpreter returns: set of achieved goal states in string form,
        # textual feedback response, failure/action info dict
        logger.info(f"IF response: {if_response}")

        if 'fail_type' in action_info:
            # record failure dict for scoring:
            self.log_to_self("action_fail", action_info)  # can be JSON'd; for easier eval
        else:
            self.log_to_self("action_info", action_info)

        # catch DONE action to end game after this turn:
        if 'done_action' in action_info:
            logger.info(f"model_done: {action_info['done_action']}")
            # self.log_to_self("model_done", if_input)
            self.model_done = True

        # if 'exploration_info' in action_info:
        #    self.log_to_self("exploration_info", action_info['exploration_info'])

        # handle goals:
        self.goals_achieved = goals_achieved
        # count goals achieved this turn:
        post_goal_count = len(self.goals_achieved)
        # calculate turn goal score; can be negative if a goal is 'unachieved':
        turn_score = post_goal_count - prior_goal_count
        # combine goal info into dict:
        goal_status = {"goal_states_achieved": list(self.goals_achieved), "turn_goal_score": turn_score}
        # record goal status dict for scoring:
        self.log_to_self("goal_status", goal_status)  # can be JSON'd; for easier eval

        if self.if_variant == 'plan':
            # current plan viability:
            # get latest/current plan from plan history:
            cur_plan: list = self.plan_history[-1]
            self.log_to_self("current_plan", f"{str(cur_plan)}")
            # get length of plan:
            cur_plan_command_count: int = len(cur_plan)
            self.log_to_self("plan_length", cur_plan_command_count)
            # pass plan to IF interpreter for execution:
            cur_plan_results: list = self.if_interpreter.execute_plan_sequence(cur_plan)
            self.log_to_self("plan_results", cur_plan_results)
            # plan result sequences cut off after the first failed plan action
            # so the sequence at this point only contains one failed action
            # or successful actions followed by a single failed action
            # get successful planned actions:
            cur_plan_successes: list = list()
            for plan_result in cur_plan_results:
                # plan_result[2] is action_info dict, if it does not contain fail_type key, the action succeeded
                if 'fail_type' not in plan_result[2]:
                    cur_plan_successes.append(plan_result)
            # calculate the ratio of successful planned actions:
            cur_plan_success_ratio: float = len(cur_plan_successes) / cur_plan_command_count
            self.log_to_self("plan_command_success_ratio", cur_plan_success_ratio)
            # append success ratio to history for 'bad' plan scoring:
            self.plan_success_ratio_history.append(cur_plan_success_ratio)
            # plan following:
            if len(self.plan_history) >= 2:
                prior_plan: list = self.plan_history[-2]
                first_prior_plan_command: str = prior_plan[0]
                plan_followed: int = 0
                # check if this turn's action matches the next action planned in the turn before:
                if first_prior_plan_command == if_input:
                    plan_followed = 1
                else:
                    plan_followed = 0
                # since plan scoring is intended to check for plan adaptation, only two-turn plan execution is
                # covered; longer planned sequences and their execution would require this to be a lot more
                # elaborate and recursive than this
                self.log_to_self("plan_followed", plan_followed)  # can be JSON'd; for easier eval
        # add IF response to dialog:
        self.set_context_for(self.player, if_response)

    def _on_after_game(self):
        # record final results once game episode has ended:
        game_result = {"goal_states_achieved": list(self.goals_achieved), "game_successfully_finished": self.finished}
        self.log_to_self("game_result", game_result)


class AdventureGameScorer(GameScorer):
    """
    GameScorer subclass for AdventureGame.
    Reads episode records, counts failures, calculates scores and stores the results in score files.
    """

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        """
        Episode level scores.
        Writes to score file in the episode directory.
        :param episode_interactions: Dict containing episode records for entire episode and turns.
        """
        # get adventure/episode-level info:
        adventure_info: dict = episode_interactions['adventure_info']
        turn_scores = []
        # IF interpreter interaction fail phases/types; first two must be 'parsing' and 'resolution' phases:
        fail_types = ['parsing', 'resolution', 'lark_exception', 'malformed_command', 'undefined_action_verb',
                      'undefined_action', 'undefined_repr_str', 'manipulating_room', 'undefined_argument_type',
                      'taking_from_inventory', 'other_room_argument',
                      'domain_trait_type_mismatch', 'domain_type_discrepancy',
                      'world_state_discrepancy', 'entity_not_accessible', 'entity_state_mismatch',
                      'entity_trait_mismatch', 'entity_already_inventory', 'going_to_current_room', 'no_exit_to',
                      'inventory_limit_exceeded']
        turn_fails = []  # list eventually containing failure counts for each turn
        turn_hallucinations = []  # list eventually containing hallucinated finish counts for each turn
        turn_explorations = []

        invalid_format: str = ""  # there can be only one invalid format or none, missing > or missing plan
        turn_limit_loss: bool = False
        successfully_finished = False
        final_goals_achieved: list = list()
        # planning variant:
        plan_types = ["plan_followed", "plan_command_success_ratio", "bad_plan_followed"]
        plan_records = []  # list eventually containing plans for all turns
        # iterate over turns:
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            turn_score = {"request_count": 1, "goal_score": 0}  # only one request per turn; no re-prompting
            turn_fail = {fail_type: 0 for fail_type in fail_types}  # start with zero failures
            plan_record = {plan_type: 0 for plan_type in plan_types}  # start with zero plan values
            hallucination = 0
            turn_exploration = dict()
            # iterate over individual record entries for turn:
            for event in turn:  # 'event' following clembench nomenclature, not connected to IF events
                action = event["action"]  # 'action' following clembench nomenclature, not connected to IF actions
                # check for format failures:
                if action["type"] == "invalid_format":
                    invalid_format = action['content']

                # check for adventure finish:
                if action["type"] == "adventure_finished":
                    successfully_finished = True

                # check for hallucinated finishes:
                if action["type"] == "hallucinated_finish":
                    hallucination = 1

                # handle DONE as hallucinated finish if the adventure is not finished:
                if action["type"] == "action_info" and action['content']['action_type'] == "done":
                    if not successfully_finished:
                        hallucination = 1

                # check for IF interaction failures:
                if action["type"] == "action_fail":
                    # check for unlisted fail type:
                    if action['content']['fail_type'] not in fail_types:
                        logger.info(f"Unlisted fail type: {action['content']['fail_type']}")
                    # record IF interaction fail phase:
                    turn_fail[action['content']['phase']] = 1
                    # record IF interaction fail type:
                    turn_fail[action['content']['fail_type']] = 1

                # get exploration values:
                if action["type"] == "action_info" or action["type"] == "action_fail":
                    exploration_info = action['content']['exploration_info']
                    logger.info(f"exploration_info: {exploration_info}")
                    if exploration_info['action_epistemic']:
                        turn_exploration['epistemic_action'] = 1
                    else:
                        turn_exploration['epistemic_action'] = 0
                    if exploration_info['action_pragmatic']:
                        turn_exploration['pragmatic_action'] = 1
                    else:
                        turn_exploration['pragmatic_action'] = 0
                    turn_exploration['effective_epistemic_gain_amount'] = exploration_info[
                        'effective_epistemic_gain_amount']
                    turn_exploration['known_entities_ratio'] = exploration_info['known_entities_ratio']
                    turn_exploration['visited_rooms_ratio'] = exploration_info['visited_rooms_ratio']
                    turn_exploration['known_goal_entities_ratio'] = exploration_info['known_goal_entities_ratio']

                # get plan values:
                if action["type"] in plan_types:
                    plan_record[action["type"]] = action["content"]
                # check for turn limit episode end:
                if action["type"] == "turn_limit_reached":
                    turn_limit_loss = True
                    # with DONE ending now being mandatory, episode is effectively lost without DONE before turn limit
                    # even if all goal states have been achieved:
                    successfully_finished = False
                # get goal values:
                if action["type"] == "goal_status":
                    turn_score["goal_score"] = action['content']['turn_goal_score']
                # get final game values (last turn):
                if action["type"] == "game_result":
                    successfully_finished = action['content']['game_successfully_finished']
                    final_goals_achieved = action['content']['goal_states_achieved']
            # check for format following, set turn violated/parsed values:
            if invalid_format:
                turn_score["violated_request_count"] = 1
                turn_score["parsed_request_count"] = 0
            else:
                turn_score["violated_request_count"] = 0
                turn_score["parsed_request_count"] = 1
            # record standard turn-level request scores:
            self.log_turn_score(turn_idx, metrics.METRIC_REQUEST_COUNT, turn_score["request_count"])
            self.log_turn_score(turn_idx, metrics.METRIC_REQUEST_COUNT_PARSED, turn_score["parsed_request_count"])
            self.log_turn_score(turn_idx, metrics.METRIC_REQUEST_COUNT_VIOLATED, turn_score["violated_request_count"])
            # record invalid format type turn values:
            if invalid_format == "command_tag_missing":
                self.log_turn_score(turn_idx, 'command_tag_missing', 1)
                self.log_turn_score(turn_idx, 'next_actions_missing', 0)
            elif invalid_format == "next_actions_missing":
                self.log_turn_score(turn_idx, 'command_tag_missing', 0)
                self.log_turn_score(turn_idx, 'next_actions_missing', 1)
            else:
                self.log_turn_score(turn_idx, 'command_tag_missing', 0)
                self.log_turn_score(turn_idx, 'next_actions_missing', 0)
            # record hallucinated finish:
            self.log_turn_score(turn_idx, 'hallucination', hallucination)
            # record IF interaction fail values by phase:
            self.log_turn_score(turn_idx, 'action_parsing_fail', turn_fail["parsing"])
            self.log_turn_score(turn_idx, 'action_resolution_fail', turn_fail["resolution"])
            # record fine-grained IF interaction fail values:
            for fail_type in fail_types[2:]:
                self.log_turn_score(turn_idx, fail_type, turn_fail[fail_type])
            # record turn-level goal score:
            self.log_turn_score(turn_idx, 'goal_score', turn_score["goal_score"])

            # exploration:
            if turn_exploration:
                self.log_turn_score(turn_idx, 'epistemic_action', turn_exploration['epistemic_action'])
                self.log_turn_score(turn_idx, 'pragmatic_action', turn_exploration['pragmatic_action'])
                self.log_turn_score(turn_idx, 'effective_epistemic_gain_amount',
                                    turn_exploration['effective_epistemic_gain_amount'])
                self.log_turn_score(turn_idx, 'known_entities_ratio', turn_exploration['known_entities_ratio'])
                self.log_turn_score(turn_idx, 'visited_rooms_ratio', turn_exploration['visited_rooms_ratio'])
                self.log_turn_score(turn_idx, 'known_goal_entities_ratio',
                                    turn_exploration['known_goal_entities_ratio'])

            # append turn values to episode-level lists:
            turn_scores.append(turn_score)
            turn_fails.append(turn_fail)
            turn_hallucinations.append(hallucination)
            turn_explorations.append(turn_exploration)

            # record planning values:
            for plan_type in plan_types:
                self.log_turn_score(turn_idx, plan_type, plan_record[plan_type])
            # BAD PLAN FOLLOWING
            if turn_idx >= 1:
                followed_bad_plan: int = 0
                # check if prior turn plan is viable at all and was followed:
                if plan_records[-1]["plan_command_success_ratio"] == 0.0 and plan_record["plan_followed"]:
                    followed_bad_plan = 1
                # record 'bad' plan following value:
                plan_record["bad_plan_followed"] = followed_bad_plan
            # append planning turn values to episode-level list:
            plan_records.append(plan_record)

        # sum up and record standard episode-level request scores:
        violated_request_count = sum([turn["violated_request_count"] for turn in turn_scores])
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)
        parsed_request_count = sum([turn["parsed_request_count"] for turn in turn_scores])
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_PARSED, parsed_request_count)
        request_count = sum([turn["request_count"] for turn in turn_scores])
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT, request_count)
        self.log_episode_score(metrics.METRIC_REQUEST_SUCCESS, parsed_request_count / request_count)

        # sum up and record episode-level action hallucination values:
        hallucination_count = sum(turn_hallucinations)
        self.log_episode_score('hallucination_count', hallucination_count)

        # sum up and record episode-level action fail scores:
        action_parsing_fail_count = sum([turn["parsing"] for turn in turn_fails])
        self.log_episode_score('action_parsing_fail', action_parsing_fail_count)
        action_resolution_fail_count = sum([turn["resolution"] for turn in turn_fails])
        self.log_episode_score('action_resolution_fail', action_resolution_fail_count)
        for fail_type in fail_types[2:]:
            type_fail_count = sum([turn[fail_type] for turn in turn_fails])
            self.log_episode_score(fail_type, type_fail_count)
        fail_sum = action_parsing_fail_count + action_resolution_fail_count
        sucessful_actions = parsed_request_count - fail_sum
        self.log_episode_score('successful_actions', sucessful_actions)

        # record turn limit exceeding loss:
        if turn_limit_loss:
            self.log_episode_score("turn_limit_loss", 1)
        else:
            self.log_episode_score("turn_limit_loss", 0)

        # SPEED
        # NOTE: Speed metrics were not informative in v1, and are now inaccurate, specially with inventory limit
        # NOTE: Instance generation and optimal solving are yet to be updated to take v2 changes into account
        # turn count for metrics based on it:
        turn_count: int = len(turn_scores)
        # get optimal turns for this episode:
        optimal_turns: int = adventure_info['optimal_turns']
        # 'on par' score; how far off the episode is from the optimal number of turns:
        turns_over_par: int = turn_count - optimal_turns
        if successfully_finished:
            self.log_episode_score("turns_over_par", turns_over_par)
        else:
            self.log_episode_score("turns_over_par", np.nan)
        # range of possible number of turns:
        turn_range = adventure_info['max_turns'] - adventure_info['optimal_turns']
        # ratio of turns taken / possible turn range:
        turn_ratio = 1 - (turns_over_par / turn_range)
        if successfully_finished:
            self.log_episode_score("turn_ratio", turn_ratio)
        else:
            self.log_episode_score("turn_ratio", np.nan)
        # finishing speed rating:
        finish_speed_rating = 1 - turn_ratio
        if successfully_finished:
            self.log_episode_score("finish_speed", finish_speed_rating)
        else:
            self.log_episode_score("finish_speed", np.nan)

        # MAIN SCORE
        # count goals achieved:
        final_goal_score = len(final_goals_achieved)
        # ratio of goals achieved to total number of goals:
        goal_count: int = adventure_info['goal_count']
        achieved_ratio = final_goal_score / goal_count
        # record achieved goal ratio:
        self.log_episode_score("achieved_goal_ratio", achieved_ratio)
        # combine goals/turns into overall rating; scrapped due to badly representing performance:
        partial_success_rating = achieved_ratio
        # scale full rating to 0-100:
        partial_success_rating = partial_success_rating * 100

        # invalid format or turn limit aborted:
        if invalid_format or turn_limit_loss:
            self.log_episode_score(metrics.METRIC_ABORTED, 1)
            self.log_episode_score(metrics.METRIC_SUCCESS, 0)
            self.log_episode_score(metrics.METRIC_LOSE, 0)
            # when game is aborted, BENCH_SCORE must be NaN to appease Pandas:
            self.log_episode_score(metrics.BENCH_SCORE, np.nan)
        else:
            self.log_episode_score(metrics.METRIC_ABORTED, 0)
            # log successful/failed play:
            if successfully_finished:
                self.log_episode_score(metrics.METRIC_SUCCESS, 1)
                self.log_episode_score(metrics.METRIC_LOSE, 0)
            else:
                self.log_episode_score(metrics.METRIC_SUCCESS, 0)
                self.log_episode_score(metrics.METRIC_LOSE, 1)
            # log full rating as main score:
            self.log_episode_score(metrics.BENCH_SCORE, partial_success_rating)

        # planning episode-level:
        # plan following:
        plan_followed_count = sum([turn["plan_followed"] for turn in plan_records[1:]])  # start at turn 2
        plan_followed_ratio = plan_followed_count / turn_count
        self.log_episode_score('plan_followed_ratio', plan_followed_ratio)
        # plan viability:
        plan_viability_sum = sum([turn["plan_command_success_ratio"] for turn in plan_records])
        plan_average_viability_ratio = plan_viability_sum / turn_count
        self.log_episode_score('plan_average_viability_ratio', plan_average_viability_ratio)
        # bad plan following:
        bad_plan_followed_sum = sum([turn["bad_plan_followed"] for turn in plan_records])
        bad_plan_followed_ratio = bad_plan_followed_sum / turn_count
        self.log_episode_score('bad_plan_follow_ratio', bad_plan_followed_ratio)
        bad_plan_dismiss_ratio = 1 - bad_plan_followed_ratio
        self.log_episode_score('bad_plan_dismiss_ratio', bad_plan_dismiss_ratio)


class AdventureGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def get_description(self):
        return "Interactive Fiction clemgame"

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return AdventureGameMaster(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return AdventureGameScorer(self.game_name, experiment, game_instance)


def main():
    game_path = os.path.dirname(os.path.abspath(__file__))
    experiments = file_utils.load_json("in/instances.json", game_path)
