from typing import List, Dict
import logging

from clemcore.backends import Model
from clemcore.clemgame import GameMaster, GameBenchmark, GameScorer, metrics, Player
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
        self.max_turns = self.grid_dimension * self.grid_dimension
        self.terminate = False


class ImageGameMaster(GameMaster):

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)
        self.experiment = experiment
        self.request_count = 0
        self.parsed_request_count = 0
        self.violated_request_count = 0
        self.aborted_ratio = 0
        self.turn_request_stats = {}

    def set_context_for(self, player, content):
        self.game.context_for_player[player.name] = dict(role="user", content=content)

    def get_context_for(self, player):
        return self.game.context_for_player[player.name]

    def setup(self, **game_instance):
        self.game_instance = game_instance
        self.game = ImageGame(self.game_instance)
        self.instruction_giver = InstructionGiver(self.player_models[0],
                                                  name="Player 1 (InstructionGiver)",
                                                  game_recorder=self.game_recorder)
        self.instruction_follower = InstructionFollower(self.player_models[1],
                                                        name="Player 2 (InstructionFollower)",
                                                        game_recorder=self.game_recorder)
        self.log_players({
            "GM": "Game master for imagegame",
            "Player_1": self.player_models[0].get_name(),
            "Player_2": self.player_models[1].get_name()}
        )

    def proceeds(self) -> bool:
        if self.game.terminate:
            return False
        if self.game.current_turn >= self.game.max_turns:
            self.log_to_self("game end", "turn limit reached")
            return False
        return True

    def play(self) -> None:
        while self.proceeds():
            self.turn()

    def turn(self):
        # instruction giving - A side
        self.turn_request_stats[self.game.current_turn] = {'request_count': 0, 'parsed_count': 0, 'violated_count': 0}

        context = dict(role="user", content="")
        if self.game.current_turn == 0:  # add prompt
            context["content"] = self.game.player_1_prompt_header + '\n' + self.game.target_grid + '\n'
        context["content"] += self.game.player_1_question
        player_1_response_text = self.instruction_giver(context)

        self.request_count += 1
        self.turn_request_stats[self.game.current_turn]['request_count'] += 1

        # check if it reached the end on 1 side
        match = re.compile(self.game.player_1_terminate_pattern, re.IGNORECASE).match(player_1_response_text)
        if match:
            self.parsed_request_count += 1
            self.turn_request_stats[self.game.current_turn]['parsed_count'] += 1
            self.game.terminate = True
            self.log_to_self("found terminate pattern", player_1_response_text)
            return
        else:
            # continue if the Player didn't say -> Instruction: DONE
            # check if Player 1 message follows the rule => start with "Instruction:"
            player_1_message_matched = re.compile(self.game.player_1_response_pattern, re.IGNORECASE).match(
                player_1_response_text)
            if player_1_message_matched:
                if '\n' in player_1_response_text:
                    parsed_instruction = player_1_response_text.split('\n')[0]
                else:
                    parsed_instruction = player_1_response_text

                self.parsed_request_count += 1
                self.turn_request_stats[self.game.current_turn]['parsed_count'] += 1
                action = {'type': 'parse', 'content': parsed_instruction,
                          'original_content': player_1_response_text}
                self.log_event(from_="GM", to="GM", action=action)
                player_1_response_text = parsed_instruction
            else:
                # log invalid format for the Player 1
                action = {'type': 'invalid format', 'content': 'Invalid instruction format',
                          'original_content': player_1_response_text}
                self.log_event(from_="GM", to="GM", action=action)
                self.turn_request_stats[self.game.current_turn]['violated_count'] += 1

                self.violated_request_count += 1
                self.aborted_ratio += 1
                # terminate the game play when the message doesn't follow the rule
                self.game.terminate = True
                return

            # instruction following - 2 side
            if self.game.current_turn == 0:
                self.set_context_for(self.instruction_follower,
                                     self.game.player_2_prompt_header + '\n' + player_1_response_text)
            else:
                self.set_context_for(self.instruction_follower, player_1_response_text)

            context = self.get_context_for(self.instruction_follower)
            player_2_response_text = self.instruction_follower(context)

            # increase the request count
            self.turn_request_stats[self.game.current_turn]['request_count'] += 1
            self.request_count += 1

            # check if Player 2 message has the required format: grid
            match = re.compile(self.game.player_2_response_pattern).match(player_2_response_text)
            if match:
                self.parsed_request_count += 1
                self.turn_request_stats[self.game.current_turn]['parsed_count'] += 1

                action = {'type': 'parse', 'content': player_2_response_text,
                          'original_content': player_2_response_text}
                self.log_event(from_="GM", to="GM", action=action)
            else:
                # log the invalid format
                action = {'type': 'invalid format', 'content': 'Invalid grid format',
                          'original_content': player_2_response_text}
                self.log_event(from_="GM", to="GM", action=action)
                self.turn_request_stats[self.game.current_turn]['violated_count'] += 1
                self.violated_request_count += 1
                self.aborted_ratio += 1
                # terminate the game play when the message doesn't follow the rule
                self.game.terminate = True
                return

        self.game.current_turn += 1


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
        number_of_turns = 0

        # loop over each turn and calculate the metrics for both Player 1 and 2.

        for t_index, turn in enumerate(episode_interactions["turns"]):

            turn_request_count = 0
            turn_parsed_request_count = 0
            turn_violated_request_count = 0

            # Player 1 message
            player_1_message = turn[1]['action']['content']

            # Player generates "DONE"
            match = re.compile(self.player1_terminate_pattern, re.IGNORECASE).match(player_1_message)
            if match:
                break

            turn_request_count += 1
            episode_request_count += 1

            # check the Player 1 message if it matches the rule
            player_1_message_matched = re.compile(self.player1_response_pattern, re.IGNORECASE).match(player_1_message)
            if player_1_message_matched:
                if '\n' in player_1_message:
                    parsed_instruction = player_1_message.split('\n')[0]
                    player_1_message = parsed_instruction

                turn_parsed_request_count += 1
                episode_parsed_request_count += 1
            else:
                turn_violated_request_count += 1
                episode_violated_request_count += 1
                aborted = True
                # do not continue processing the rest of the turn when the game is aborted
                break

            # check if the turn includes the Player 2 message
            # in case the turn doesn't include an item and index position 4, it means the game has been aborted
            if len(turn) < 4:
                aborted = True
                break

            # Player 2 message
            player_2_message = turn[4]['action']['content']
            turn_request_count += 1
            episode_request_count += 1

            # check Player 2 message if it matches the instruction => grid
            match = re.compile(self.player2_response_pattern).match(player_2_message)
            if match:
                turn_parsed_request_count += 1
                episode_parsed_request_count += 1
            else:
                turn_violated_request_count += 1
                episode_violated_request_count += 1
                aborted = True
                break

            # calculate player-specific and turn-specific metrics

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
