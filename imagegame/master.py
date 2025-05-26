from typing import List, Dict
import logging

from clemcore.backends import Model
from clemcore.clemgame import GameMaster, DialogueGameMaster, GameBenchmark, GameScorer, metrics, Player
from evaluator import evaluate, calculate_flipped_pixels

import re
import math

logger = logging.getLogger(__name__)


class InstructionFollower(Player):

    def _custom_response(self, context):
        return "▢ P O T ▢\n▢ S ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ D A M ▢"


class InstructionGiver(Player):

    def _custom_response(self, context):
        return "Command: Put X in all cells"


class ImageGame:

    def __init__(self, game_instance: Dict):
        self.game_id = game_instance['game_id']
        self.player_1_prompt_header = game_instance['player_1_prompt_header']
        self.player_2_prompt_header = game_instance['player_2_prompt_header']
        self.player_1_question = game_instance['player_1_question']
        self.target_grid = game_instance['target_grid']
        self.grid_dimension = game_instance['grid_dimension']
        self.number_of_letters = game_instance['number_of_letters']
        self.fill_row = game_instance['fill_row']
        self.fill_column = game_instance['fill_column']
        self.player_1_response_pattern = r'{}'.format(game_instance['player_1_response_pattern'])
        self.player_1_terminate_pattern = r'{}'.format(game_instance['player_1_terminate_pattern'])
        self.player_2_response_pattern = r'{}'.format(game_instance['player_2_response_pattern'])
        self.context_for_player: Dict[str, Dict] = {}
        self.current_turn = 0
        self.max_rounds = self.grid_dimension * self.grid_dimension * 2
        self.terminate = False

class ImageGameMaster(DialogueGameMaster):

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)

    def _on_setup(self, **game_instance):
        self.request_count = 0
        self.parsed_request_count = 0
        self.violated_request_count = 0
        self.aborted_ratio = 0

        self.game_instance = game_instance
        self.game = ImageGame(self.game_instance)
        self.instruction_giver = InstructionGiver(self.player_models[0],
                                                  name="Player 1",
                                                  game_role="Instruction Giver",
                                                  game_recorder=self.game_recorder)
        self.instruction_follower = InstructionFollower(self.player_models[1],
                                                        name="Player 2",
                                                        game_role="Instruction Follower",
                                                        game_recorder=self.game_recorder)
        p1_initial_prompt = self.game.player_1_prompt_header + '\n' + self.game.target_grid + '\n' + self.game.player_1_question
        self.add_player(self.instruction_giver, initial_context=p1_initial_prompt)
        self.add_player(self.instruction_follower, initial_prompt=self.game.player_2_prompt_header)

    def _validate_player_response(self, player: Player, response: str) -> bool:
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
            match = re.compile(self.game.player_1_terminate_pattern, re.IGNORECASE).match(response)
            if match:
                return True
            else:
                match = re.compile(self.game.player_1_response_pattern, re.IGNORECASE).match(response)
                if match:
                    return True
                else:
                    self.game.terminate = True
                    self.log_to_self("invalid format", "Invalid instruction format")
                    return False
        else:
            match = re.compile(self.game.player_2_response_pattern).match(response)
            if match:
                return True
            else:
                self.game.terminate = True
                self.log_to_self("invalid format", "Invalid grid format")
                return False

    def _parse_response(self, player: Player, response: str) -> str:
        """ Takes a valid player response and parses it.

        Args:
            player: The Player instance that produced the response.
            response: The response of the current player.
        Returns:
            The parsed response
        """
        if player == self.instruction_giver:
            match = re.compile(self.game.player_1_terminate_pattern, re.IGNORECASE).match(response)
            if match:
                self.game.terminate = True
                self.log_to_self("found terminate pattern", response)
                return None
            # check if the Player 1 message follows the rule => start with "Instruction:"
            match = re.compile(self.game.player_1_response_pattern, re.IGNORECASE).match(response)
            if match:
                if '\n' in response:
                    parsed_instruction = response.split('\n')[0]
                else:
                    parsed_instruction = response
                self.log_to_self("found instruction", parsed_instruction)
                return parsed_instruction
        elif player == self.instruction_follower:
            match = re.compile(self.game.player_2_response_pattern).match(response)
            if match:
                self.log_to_self("found grid", response)
                return response
        return None
            
    def _on_valid_player_response(self, player: Player, parsed_response: str) -> None:
        """Method executed after a player response has been parsed and validated.
        This method is used to set the context for the other player.

        Args:
            player: The Player instance that produced the response (or has been modified by the GM).
            parsed_response: The parsed and valid response of the current player.
        """
        if player == self.instruction_giver:
            self.set_context_for(self.instruction_follower, parsed_response)
        else:
            self.set_context_for(self.instruction_giver, self.game.player_1_question)

    def _does_game_proceed(self) -> bool:
        """Check if game should proceed.
        This method is called after each turn to check if the game should continue or stop.
        It returns False if `self.game.terminate` is True or if the maximum number of rounds has been reached.
        
        Returns:
            A bool, True if game continues, False if game should stop.
        """
        if self.game.terminate:
            return False
        if self.current_round >= self.game.max_rounds:
            self.log_to_self("game end", "turn limit reached")
            return False
        return True

class ImageGameScorer(GameScorer):

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.target_grid = game_instance["target_grid"]
        self.player1_response_pattern = r'{}'.format(game_instance["player_1_response_pattern"])
        self.player2_response_pattern = r'{}'.format(game_instance["player_2_response_pattern"])
        self.player1_terminate_pattern = r'{}'.format(game_instance["player_1_terminate_pattern"])
                    
    def compute_scores(self, episode_interactions: Dict) -> None:

        precision, recall, f1 = 0, 0, 0

        previous_turn_grid = '▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢'
        flipped_count_sum = 0
        expression_length_sum = 0
        expression_number_of_tokens = 0
        current_turn_grid = ''

        episode_request_count = 0
        episode_parsed_request_count = 0
        episode_violated_request_count = 0

        aborted = False
        terminated = False
        number_of_turns = 0

        # loop over each turn and calculate the metrics for both Player 1 and 2.

        for t_index, turn in enumerate(episode_interactions["turns"]):

            turn_request_count = 0
            turn_parsed_request_count = 0
            turn_violated_request_count = 0
            player_1_message, player_2_message = None, None
            
            for event in turn:
                invalid_response = False
                action = event['action']
                if action['type'] == 'found terminate pattern':
                    terminated = True
                    break
                elif action['type'] == 'invalid format':
                    invalid_response = True
                elif action['type'] == 'found instruction':
                    player_1_message = action['content']
                elif action['type'] == 'found grid':
                    player_2_message = action['content']
                turn_request_count += 1
                episode_request_count += 1
                if invalid_response:
                    turn_violated_request_count += 1
                    episode_violated_request_count += 1
                    aborted = True
                    break
                else:
                    turn_parsed_request_count += 1
                    episode_parsed_request_count += 1
            
            if aborted or terminated:
                break

            try:
                precision, recall, f1 = evaluate(self.target_grid, player_2_message)
            except:
                pass

            # number of turns other
            number_of_turns += 1

            # Player 1 - message length
            expression_length = len(player_1_message.replace('Instruction:', '').strip())
            self.log_turn_score(t_index, 'Generated Expression Length', expression_length)
            expression_length_sum += expression_length

            # Player 1 - number of tokens in the generated expression
            number_of_tokens = len(player_1_message.replace('Instruction:', '').strip().split(' '))
            self.log_turn_score(t_index, 'Generated Expression Number of Tokens', number_of_tokens)
            expression_number_of_tokens += number_of_tokens

            self.log_turn_score(t_index, 'Precision', precision)
            self.log_turn_score(t_index, 'Recall', recall)
            self.log_turn_score(t_index, 'F1', f1)

            # calculate flipped pixel counts
            flipped_count = 0
            try:
                current_turn_grid = player_2_message
                flipped_count = calculate_flipped_pixels(previous_turn_grid, current_turn_grid)
            except:
                pass

            flipped_count_sum += flipped_count
            previous_turn_grid = current_turn_grid
            self.log_turn_score(t_index, 'Changed Cell Count', flipped_count)

            # request count, parsed & violated request counts
            self.log_turn_score(t_index, metrics.METRIC_REQUEST_COUNT,
                                turn_request_count)
            self.log_turn_score(t_index, metrics.METRIC_REQUEST_COUNT_PARSED,
                                turn_parsed_request_count)
            self.log_turn_score(t_index, metrics.METRIC_REQUEST_COUNT_VIOLATED,
                                turn_violated_request_count)

        # Episode level logging
        if aborted:
            # if aborted give NaN value to all metrics
            self.log_episode_score('Precision', math.nan)
            self.log_episode_score('Recall', math.nan)
            self.log_episode_score('F1', math.nan)
            self.log_episode_score(metrics.BENCH_SCORE, math.nan)

            # average of flipped pixel counts
            self.log_episode_score('Average Changed Cell Count', math.nan)

            # average of expression length
            self.log_episode_score('Average Generated Instruction Length', math.nan)

            # average of number of tokens in generated expression
            self.log_episode_score('Average Generated Expression Number of Tokens', math.nan)

            # the last turn scores are also the scores for the episode
            self.log_episode_score(metrics.METRIC_SUCCESS, 0)

            # lose ratio
            self.log_episode_score(metrics.METRIC_LOSE, 0)

            # aborted ratio
            self.log_episode_score(metrics.METRIC_ABORTED, 1)
        else:
            # the last turn scores are also the scores for the episode
            self.log_episode_score('Precision', precision)
            self.log_episode_score('Recall', recall)
            self.log_episode_score('F1', f1)
            self.log_episode_score(metrics.BENCH_SCORE, f1)

            # average of flipped pixel counts
            flipped_count_sum = round(flipped_count_sum / float(number_of_turns), 4)
            self.log_episode_score('Average Changed Cell Count', flipped_count_sum)

            # average of expression length
            expression_length_sum = round(expression_length_sum / float(number_of_turns), 4)
            self.log_episode_score('Average Generated Instruction Length', expression_length_sum)

            # average of number of tokens in generated expression
            expression_number_of_tokens = round(expression_number_of_tokens / float(number_of_turns), 4)
            self.log_episode_score('Average Generated Expression Number of Tokens', expression_number_of_tokens)

            # the last turn scores are also the scores for the episode
            self.log_episode_score(metrics.METRIC_SUCCESS, 1 if f1 >= 99 else 0)

            # lose ratio
            self.log_episode_score(metrics.METRIC_LOSE, 0 if f1 >= 99 else 1)

            # aborted ratio
            self.log_episode_score(metrics.METRIC_ABORTED, 0)

        # request count, parsed & violated request counts
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT, episode_request_count)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_VIOLATED, episode_violated_request_count)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_PARSED, episode_parsed_request_count)

        # request success ratio
        if episode_request_count == 0:
            self.log_episode_score(metrics.METRIC_REQUEST_SUCCESS, 0)
        else:
            request_success_ratio = round(episode_parsed_request_count / float(episode_request_count), 4)
            self.log_episode_score(metrics.METRIC_REQUEST_SUCCESS, request_success_ratio)


class ImageGameBenchmark(GameBenchmark):

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return ImageGameMaster(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return ImageGameScorer(self.game_name, experiment, game_instance)
