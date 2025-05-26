import random
from typing import List, Dict

import numpy as np
import logging
import os

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, Player
from clemcore.clemgame import metrics
from clemcore.clemgame import DialogueGameMaster, GameBenchmark, GameScorer

import re

logger = logging.getLogger(__name__)


class InstructionFollower(Player):

    def _custom_response(self, context):
        answer = random.choice(["first", "second", "third"])
        added = random.choice(["", ".", " grid"])
        return f"Answer: {answer}{added}"


class InstructionGiver(Player):

    def _custom_response(self, context):
        return "Expression: The one that looks like the target."


class ReferenceGame:

    def __init__(self, game_instance: Dict):
        self.lang = game_instance['lang']
        self.p1_mode = game_instance['p1_mode']
        self.p2_mode = game_instance['p2_mode']
        self.game_id = game_instance['game_id']
        self.player_1_prompt_header = game_instance['player_1_prompt_header']
        self.player_2_prompt_header = game_instance['player_2_prompt_header']
        self.target_grid_name = game_instance['target_grid_name']

        self.player_1_response_pattern = r'{}'.format(game_instance['player_1_response_pattern'])
        self.player_2_response_pattern = r'{}'.format(game_instance['player_2_response_pattern'])

        self.player_1_target_grid = game_instance['player_1_target_grid']
        self.player_1_second_grid = game_instance['player_1_second_grid']
        self.player_1_third_grid = game_instance['player_1_third_grid']

        self.player_2_first_grid = game_instance['player_2_first_grid']
        self.player_2_second_grid = game_instance['player_2_second_grid']
        self.player_2_third_grid = game_instance['player_2_third_grid']

        self.terminate = False


class ReferenceGameMaster(DialogueGameMaster):

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)

    def _on_setup(self, **game_instance):
        self.game = ReferenceGame(game_instance)
        self.instruction_giver = InstructionGiver(self.player_models[0],
                                                  name="Player 1",
                                                  game_role="Instruction Giver",
                                                  game_recorder=self.game_recorder)
        self.instruction_follower = InstructionFollower(self.player_models[1],
                                                        name="Player 2",
                                                        game_role="Instruction Follower",
                                                        game_recorder=self.game_recorder)
        p1_initial_prompt = self.game.player_1_prompt_header
        self.add_player(self.instruction_giver, initial_context=p1_initial_prompt)
        self.add_player(self.instruction_follower)

    # def play(self) -> None:
    #     self.turn()

    def _validate_player_response(self, player, response):
        """
        Decide if a player response matches the valid response patterns.
        An invalid response breaks the game rules and ends the game.

        Args:
            player: The player that gave the response.
            response: The response of the current player.
        Returns:
            True, if the response is fine. Otherwise, False.
        """
        if player == self.instruction_giver:
            # Player 1 response validation
            p1_match = re.compile(self.game.player_1_response_pattern, re.IGNORECASE).match(response)
            if p1_match:
                if self.game.p1_mode == "liberal" or (self.game.p1_mode == "strict" and p1_match.group('remainder') == ""):
                    # in liberal mode, we don't care how much more the model generated
                    # in strict mode, the model should not produce more than one paragraph
                    return True
            self.game.terminate = True
            self.log_to_self("invalid format", "Invalid generated expression")
            return False
        elif player == self.instruction_follower:
            # Game only has one round, so we terminate regardless of the response
            self.game.terminate = True
            # Player 2 response validation
            p2_match = re.compile(self.game.player_2_response_pattern, re.IGNORECASE).match(response)
            if p2_match:
                if self.game.p2_mode == "liberal" or (self.game.p2_mode == "strict" and p2_match.group('remainder') == ""):
                    # in liberal mode, we don't care how much more the model generated (like "grid" or punctuation)
                    # in strict mode, the model should only produce the label
                    return True
            self.log_to_self("invalid format", "Invalid generated expression")
            return False
        
    def _parse_response(self, player, response):
        """ Takes a valid player response and parses it.

        Args:
            player: The Player instance that produced the response.
            response: The response of the current player.
        Returns:
            The parsed response
        """
        if player == self.instruction_giver:
            # Player 1 response parsing
            p1_match = re.compile(self.game.player_1_response_pattern, re.IGNORECASE).match(response)
            if p1_match:
                return response
        elif player == self.instruction_follower:
            # Player 2 response parsing
            player_2_pattern = re.compile(self.game.player_2_response_pattern, re.IGNORECASE)
            p2_match = re.match(player_2_pattern, response)
            if p2_match:
                return p2_match.group('response').lower()  # return the label only
        return None
    
    def _on_valid_player_response(self, player: Player, parsed_response: str) -> None:
        """Method executed after a player response has been parsed and validated.
        This method is used to set the context for the other player.

        Args:
            player: The Player instance that produced the response (or has been modified by the GM).
            parsed_response: The parsed and valid response of the current player.
        """
        if player == self.instruction_giver:
            self.log_to_self('parse', parsed_response)
            # Game only has one round, so we can use the initial prompt header here
            p2_prompt = self.game.player_2_prompt_header.replace('TARGET_EXPRESSION', parsed_response)
            self.set_context_for(self.instruction_follower, p2_prompt)
        else:
            if parsed_response in self.game.target_grid_name:
                self.log_to_self('parse_correct', parsed_response)
            else:
                self.log_to_self('parse_wrong', parsed_response)

    def _does_game_proceed(self):
        if self.game.terminate:
            return False
        return True

class ReferenceGameScorer(GameScorer):

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        """
        Compute and log scores for one episode of referencegame.
        Args:
            episode_interactions: The interactions of the episode, as a dictionary.
        """

        # For referencegame, there is just one turn (one exchange of p1-p2 is logged as one turn)
        turn = episode_interactions["turns"][0]
        turn_index = 0

        aborted = False

        turn_request_count = 0
        turn_parsed_request_count = 0
        episode_request_count = 0
        episode_parsed_request_count = 0

        success = 0

        # evaluate Player 1
        turn_request_count += 1
        episode_request_count += 1
        # check if the Player 1 message followed the rule
        # (true if third interaction (GM to GM) has type "parse")
        if turn[2]['action']['type'] == "parse":
            turn_parsed_request_count += 1
            episode_parsed_request_count += 1

            # log the Player 1 - message length
            p1_expression = turn[2]['action']['content']
            expression_length = len(p1_expression)
            self.log_turn_score(turn_index, 'Generated Expression Length', expression_length)
            # as there is just one turn, this is the same as episode scores
            self.log_episode_score('Generated Expression Length', expression_length)

            # log the Player 1 - number of tokens in the generated expression
            number_of_tokens = len(p1_expression.split(' '))
            self.log_turn_score(turn_index, 'Generated Expression Number of Tokens', number_of_tokens)
            # as there is just one turn, this is the same as episode scores
            self.log_episode_score('Generated Expression Number of Tokens', number_of_tokens)

            # evaluate Player 2 (only if Player 1's response was valid)
            turn_request_count += 1
            episode_request_count += 1
            # check if the Player 2 message matched the rule
            # (true if sixth interaction (GM to GM) has type "parse")

            if turn[5]['action']['type'].startswith("parse"):
                turn_parsed_request_count += 1
                episode_parsed_request_count += 1

                if "correct" in turn[5]['action']['type']:
                    success = 1

                self.log_episode_score('Aborted at Player 1', 0)
                self.log_episode_score('Aborted at Player 2', 0)
            else:
                self.log_episode_score('Aborted at Player 1', 0)
                self.log_episode_score('Aborted at Player 2', 1)
                aborted = True
        else:
            self.log_episode_score('Aborted at Player 1', 1)
            self.log_episode_score('Aborted at Player 2', 0)
            self.log_turn_score(turn_index, 'Generated Expression Length', np.nan)
            self.log_episode_score('Generated Expression Length', np.nan)
            self.log_turn_score(turn_index, 'Generated Expression Number of Tokens', np.nan)
            self.log_episode_score('Generated Expression Number of Tokens', np.nan)
            aborted = True

        # log the turn request count, parsed & violated request counts
        self.log_turn_score(turn_index, metrics.METRIC_REQUEST_COUNT, turn_request_count)
        self.log_turn_score(turn_index, metrics.METRIC_REQUEST_COUNT_PARSED, turn_parsed_request_count)
        self.log_turn_score(turn_index, metrics.METRIC_REQUEST_COUNT_VIOLATED,
                            turn_request_count - turn_parsed_request_count)
        self.log_turn_score(turn_index, metrics.METRIC_SUCCESS, success)

        # log the episode request count, parsed & violated request counts
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT, episode_request_count)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_PARSED, episode_parsed_request_count)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_VIOLATED,
                               episode_request_count - episode_parsed_request_count)

        self.log_episode_score(metrics.METRIC_ABORTED, int(aborted))
        success = success if not aborted else np.nan
        self.log_episode_score(metrics.METRIC_SUCCESS, success)
        loose = 1 - success if not aborted else np.nan
        self.log_episode_score(metrics.METRIC_LOSE, loose)

        bench_score = success * 100 if not aborted else np.nan
        self.log_episode_score(metrics.BENCH_SCORE, bench_score)

        request_success_ratio = round(episode_parsed_request_count / float(episode_request_count), 4)
        self.log_episode_score(metrics.METRIC_REQUEST_SUCCESS, request_success_ratio)


class ReferenceGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> DialogueGameMaster:
        return ReferenceGameMaster(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return ReferenceGameScorer(self.game_name, experiment, game_instance)


def main():
    # select one instance
    game_path = os.path.dirname(os.path.abspath(__file__))
    from clemcore.utils import file_utils
    experiments = file_utils.load_json("in/instances.json", game_path)
    instance = experiments["experiments"][0]["game_instances"][0]
    master = ReferenceGameMaster(instance, ["gpt-3.5-turbo", "gpt-3.5-turbo"])
    master.setup(**instance)
    master.play()


if __name__ == '__main__':
    main()
