import random
from typing import List, Dict
import numpy as np
import re
import logging

from clemcore.backends import Model
from clemcore.utils import file_utils
from clemcore.clemgame import metrics, Player
from clemcore.clemgame import GameMaster, GameBenchmark, GameScorer, GameSpec

logger = logging.getLogger(__name__)


class InstructionFollower(Player):

    def _custom_response(self, context):
        answer = random.choice(["first", "second", "third"])
        return f"Answer: {answer}"


class InstructionGiver(Player):

    def _custom_response(self, context):
        return "Expression: The one that looks like the target."


class MultimodalReferenceGame:

    def __init__(self, game_instance: Dict):
        self.game_id = game_instance['game_id']
        self.player_1_prompt_header = game_instance['player_1_prompt_header']
        self.player_2_prompt_header = game_instance['player_2_prompt_header']
        self.target_image_name = game_instance['target_image_name']

        self.player_1_response_pattern = r'{}'.format(game_instance['player_1_response_pattern'])
        self.player_2_response_pattern = r'{}'.format(game_instance['player_2_response_pattern'])

        self.player_1_first_image = game_instance['player_1_first_image']
        self.player_1_second_image = game_instance['player_1_second_image']
        self.player_1_third_image = game_instance['player_1_third_image']

        self.player_2_first_image = game_instance['player_2_first_image']
        self.player_2_second_image = game_instance['player_2_second_image']
        self.player_2_third_image = game_instance['player_2_third_image']


class MultimodalReferenceGameMaster(GameMaster):

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)
        self.experiment = experiment
        self.game = None

    def setup(self, **game_instance):
        self.game = MultimodalReferenceGame(game_instance)
        self.instruction_giver = InstructionGiver(self.player_models[0],
                                                  name="Player 1 (InstructionGiver)",
                                                  game_recorder=self.game_recorder)
        self.instruction_follower = InstructionFollower(self.player_models[1],
                                                        name="Player 2 (InstructionFollower)",
                                                        game_recorder=self.game_recorder)
        self.log_players({
            "GM": "Game master for multimodal referencegame",
            "Player_1": self.player_models[0].get_name(),
            "Player_2": self.player_models[1].get_name()}
        )

    def play(self) -> None:
        self.turn()

    def turn(self):
        # generate referring expression - Player 1 side
        context = dict(role="user",
                       content=self.game.player_1_prompt_header,
                       image=[self.game.player_1_first_image,
                              self.game.player_1_second_image,
                              self.game.player_1_third_image])
        player_1_response_text = self.instruction_giver(context)

        player_1_pattern = re.compile(self.game.player_1_response_pattern, re.IGNORECASE)
        p1_match = re.match(player_1_pattern, player_1_response_text.strip())
        if p1_match and p1_match.group('remainder') == "":
            action = {'type': 'parse', 'content': player_1_response_text,
                      'expression': p1_match.group('content')}
            self.log_event(from_="GM", to="GM", action=action)
        else:
            # if the Player 1 message don't match the rule => start with "Expression: " and contains only one paragraph
            # log the message and abort the game
            action = {'type': 'invalid format', 'content': 'Invalid generated expression',
                      'original_content': player_1_response_text}
            self.log_event(from_="GM", to="GM", action=action)
            return

        # guess the grid - Player 2 side
        context = dict(role="user",
                       content=self.game.player_2_prompt_header.replace('TARGET_EXPRESSION', player_1_response_text),
                       image=[self.game.player_2_first_image,
                              self.game.player_2_second_image,
                              self.game.player_2_third_image])

        player_2_response_text = self.instruction_follower(context)

        # check if the Player 2 message matches the rule => start with "Answer: " and generate only the label
        player_2_pattern = re.compile(self.game.player_2_response_pattern, re.IGNORECASE)
        p2_match = re.match(player_2_pattern, player_2_response_text)
        if p2_match and p2_match.group('remainder') == "":
            action = {'type': 'parse', 'content': player_2_response_text,
                      'answer': p2_match.group('content')}
            self.log_event(from_="GM", to="GM", action=action)

            action = {'type': 'expected answer', 'content': self.game.target_image_name,
                      'answer': self.game.target_image_name}
            self.log_event(from_="GM", to="GM", action=action)
        else:
            # abort the game if the output doesn't match the rule
            action = {'type': 'invalid format', 'content': 'Invalid generated choice',
                      'original_content': player_2_response_text}
            self.log_event(from_="GM", to="GM", action=action)


class MultimodalReferenceGameScorer(GameScorer):

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.target_grid_name = game_instance["target_image_name"]
        self.player_2_response_pattern = game_instance["player_2_response_pattern"]

    def compute_scores(self, episode_interactions: Dict) -> None:
        '''
        Compute and log scores for one episode of referencegame.
        :param episode_interactions: the game episode interactions log
        '''

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
            p1_expression = turn[2]['action']['expression']
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

            # allow for more liberal player 2 parsing by rematching original response with more liberal regex
            # TODO: move to game master for future runs
            p2_match = False
            if turn[5]['action']['type'] == "invalid format":
                player_2_pattern = re.compile(self.player_2_response_pattern, re.IGNORECASE)
                p2_match = re.match(player_2_pattern, turn[5]['action']['original_content'])

            if turn[5]['action']['type'] == "parse" or p2_match:
                turn_parsed_request_count += 1
                episode_parsed_request_count += 1
                # check if the target grid number matches the output from Player 2
                player_2_answer = ""
                if p2_match:
                    player_2_answer = p2_match.group('content')
                elif turn[5]['action']['type'] == "parse":
                    player_2_answer = turn[5]['action']['answer']

                if player_2_answer.lower() in self.target_grid_name:
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
        self.log_episode_score(metrics.METRIC_SUCCESS, success)
        self.log_episode_score(metrics.METRIC_LOSE, 1 - success)
        self.log_episode_score(metrics.METRIC_ABORTED, int(aborted))

        bench_score = success * 100 if not aborted else np.nan
        self.log_episode_score(metrics.BENCH_SCORE, bench_score)

        request_success_ratio = round(episode_parsed_request_count / float(episode_request_count), 4)
        self.log_episode_score(metrics.METRIC_REQUEST_SUCCESS, request_success_ratio)


class MultimodalReferenceGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return MultimodalReferenceGameMaster(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return MultimodalReferenceGameScorer(self.game_name, experiment, game_instance)


def main():
    # select one instance
    experiments = file_utils.load_json("in/instances.json", "referencegame")
    instance = experiments["experiments"][0]["game_instances"][0]
    master = MultimodalReferenceGameMaster(instance, ["gpt-3.5-turbo", "gpt-3.5-turbo"])
    master.setup(**instance)
    master.play()


if __name__ == '__main__':
    main()
