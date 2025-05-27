from dataclasses import dataclass
from typing import List, Dict
import logging
import re
import math

from clemcore.backends import Model
from clemcore.clemgame import GameMaster, DialogueGameMaster, GameBenchmark, GameScorer, metrics, Player
from clemcore.clemgame.master import ParseError, RuleViolationError, GameError
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE
from evaluator import evaluate, calculate_flipped_pixels

logger = logging.getLogger(__name__)

INSTRUCTION_PREFIX = "Instruction:"


class InstructionFollower(Player):

    def _custom_response(self, context):
        return "▢ P O T ▢\n▢ S ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ D A M ▢"


class InstructionGiver(Player):

    def _custom_response(self, context):
        return "Instruction: Put X in all cells"


@dataclass
class GameState:
    game_id: str
    player_1_prompt_header: str
    player_2_prompt_header: str
    player_1_question: str
    target_grid: str
    grid_dimension: int
    number_of_letters: int
    fill_row: int
    fill_column: int
    player_1_response_pattern: str
    player_1_terminate_pattern: str
    player_2_response_pattern: str
    max_rounds: int
    success: bool = False
    failure: bool = False
    aborted: bool = False
    terminated: bool = False
    last_instruction: str = None
    last_grid: str = None

    def __post_init__(self):
        # Convert patterns to compiled regex objects for better performance
        self.player_1_response_regex = re.compile(self.player_1_response_pattern, re.IGNORECASE)
        self.player_1_terminate_regex = re.compile(self.player_1_terminate_pattern, re.IGNORECASE)
        self.player_2_response_regex = re.compile(self.player_2_response_pattern)


class ImageGameMaster(DialogueGameMaster):

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)

    def _on_setup(self, **game_instance):
        # Initialize game state
        max_rounds = game_instance['grid_dimension'] * game_instance['grid_dimension'] * 2

        self.state = GameState(
            game_id=game_instance['game_id'],
            player_1_prompt_header=game_instance['player_1_prompt_header'],
            player_2_prompt_header=game_instance['player_2_prompt_header'],
            player_1_question=game_instance['player_1_question'],
            target_grid=game_instance['target_grid'],
            grid_dimension=game_instance['grid_dimension'],
            number_of_letters=game_instance['number_of_letters'],
            fill_row=game_instance['fill_row'],
            fill_column=game_instance['fill_column'],
            player_1_response_pattern=r'{}'.format(game_instance['player_1_response_pattern']),
            player_1_terminate_pattern=r'{}'.format(game_instance['player_1_terminate_pattern']),
            player_2_response_pattern=r'{}'.format(game_instance['player_2_response_pattern']),
            max_rounds=max_rounds
        )

        # Initialize players
        self.instruction_giver = InstructionGiver(
            self.player_models[0],
            name="Player 1",
            game_role="Instruction Giver",
            game_recorder=self.game_recorder
        )
        self.instruction_follower = InstructionFollower(
            self.player_models[1],
            name="Player 2",
            game_role="Instruction Follower",
            game_recorder=self.game_recorder
        )

        # Set up initial prompts
        p1_initial_prompt = (self.state.player_1_prompt_header + '\n' +
                             self.state.target_grid + '\n' +
                             self.state.player_1_question)

        self.add_player(self.instruction_giver, initial_context=p1_initial_prompt)
        self.add_player(self.instruction_follower, initial_prompt=self.state.player_2_prompt_header)

    def _does_game_proceed(self):
        return not (self.state.aborted or self.state.failure or
                    self.state.success or self.state.terminated)

    def _parse_response(self, player: Player, response: str) -> str:
        if player == self.instruction_giver:
            # Check for termination pattern first
            if self.state.player_1_terminate_regex.match(response):
                self.log_to_self("found terminate pattern", response)
                self.state.terminated = True
                return None

            # Check for instruction pattern
            if not self.state.player_1_response_regex.match(response):
                raise ParseError("Invalid instruction format", response)

            # Parse instruction content
            if '\n' in response:
                parsed_instruction = response.split('\n')[0]
            else:
                parsed_instruction = response

            self.log_to_self("found instruction", parsed_instruction)
            return parsed_instruction.replace(INSTRUCTION_PREFIX, "").strip()

        elif player == self.instruction_follower:
            if not self.state.player_2_response_regex.match(response):
                raise ParseError("Invalid grid format", response)

            self.log_to_self("found grid", response)
            return response

        raise ParseError(f"Unknown player: {player}", response)

    def _on_parse_error(self, error: ParseError):
        self.log_to_self("invalid format", error.reason)
        self.state.aborted = True

    def _advance_game(self, player: Player, parsed_response: str):
        if player == self.instruction_giver:
            # No additional rule validation needed for instruction giver
            self.state.last_instruction = parsed_response
            # Send full instruction with prefix to follower
            self.set_context_for(self.instruction_follower, f"{INSTRUCTION_PREFIX} {parsed_response}")

        elif player == self.instruction_follower:
            # No additional rule validation needed for instruction follower
            self.state.last_grid = parsed_response
            self.set_context_for(self.instruction_giver, self.state.player_1_question)

            # Check for success condition (could be enhanced with specific criteria)
            try:
                precision, recall, f1 = evaluate(self.state.target_grid, parsed_response)
                if f1 >= 99:  # Success threshold from v2 scorer
                    self.log_to_self("target achieved", f"F1 score: {f1}")
                    self.state.success = True
            except Exception as e:
                self.log_to_self("evaluation error", str(e))

            # Check for max rounds
            if self.current_round >= self.state.max_rounds - 1:  # zero-based
                if not self.state.success:
                    raise RuleViolationError(f"max rounds ({self.state.max_rounds}) reached")

    def _on_game_error(self, error: GameError):
        self.log_to_self(error.reason, "failed game")
        self.state.failure = True

    def compute_turn_score(self):
        if self.state.last_grid:
            try:
                precision, recall, f1 = evaluate(self.state.target_grid, self.state.last_grid)
                return f1
            except:
                return 0
        return 0

    def compute_episode_score(self):
        if self.state.success:
            try:
                precision, recall, f1 = evaluate(self.state.target_grid, self.state.last_grid)
                return f1
            except:
                return 0
        return 0

    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, int(self.state.aborted))
        self.log_key(METRIC_LOSE, int(self.state.failure or (not self.state.success and self.state.terminated)))
        self.log_key(METRIC_SUCCESS, int(self.state.success))


class ImageGameScorer(GameScorer):

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.target_grid = game_instance["target_grid"]
        self.player1_response_pattern = r'{}'.format(game_instance["player_1_response_pattern"])
        self.player2_response_pattern = r'{}'.format(game_instance["player_2_response_pattern"])
        self.player1_terminate_pattern = r'{}'.format(game_instance["player_1_terminate_pattern"])

    def compute_round_score(self, round_idx, round_events: List[Dict]) -> None:
        precision, recall, f1 = 0, 0, 0
        player_1_message, player_2_message = None, None

        for event in round_events:
            action = event['action']
            if action['type'] == 'found instruction':
                player_1_message = action['content']
            elif action['type'] == 'found grid':
                player_2_message = action['content']

        if player_2_message:
            try:
                precision, recall, f1 = evaluate(self.target_grid, player_2_message)
            except:
                pass

        self.log_round_score(round_idx, 'Precision', precision)
        self.log_round_score(round_idx, 'Recall', recall)
        self.log_round_score(round_idx, 'F1', f1)

        if player_1_message:
            # Message length metrics
            expression_length = len(player_1_message.replace('Instruction:', '').strip())
            number_of_tokens = len(player_1_message.replace('Instruction:', '').strip().split(' '))

            self.log_round_score(round_idx, 'Generated Expression Length', expression_length)
            self.log_round_score(round_idx, 'Generated Expression Number of Tokens', number_of_tokens)

    def compute_episode_scores(self, interactions: Dict):
        turns = interactions.get("turns", [])

        if interactions.get(METRIC_ABORTED, False):
            # Set NaN for all metrics if aborted
            self.log_episode_score('Precision', math.nan)
            self.log_episode_score('Recall', math.nan)
            self.log_episode_score('F1', math.nan)
            self.log_episode_score(BENCH_SCORE, math.nan)
            self.log_episode_score('Average Changed Cell Count', math.nan)
            self.log_episode_score('Average Generated Instruction Length', math.nan)
            self.log_episode_score('Average Generated Expression Number of Tokens', math.nan)
        else:
            # Calculate final metrics from last valid turn
            final_precision, final_recall, final_f1 = 0, 0, 0
            previous_turn_grid = '▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢'

            flipped_count_sum = 0
            expression_length_sum = 0
            expression_token_sum = 0
            valid_turns = 0

            for turn_events in turns:
                player_1_message, player_2_message = None, None

                for event in turn_events:
                    action = event['action']
                    if action['type'] == 'found instruction':
                        player_1_message = action['content']
                    elif action['type'] == 'found grid':
                        player_2_message = action['content']

                if player_2_message:
                    try:
                        final_precision, final_recall, final_f1 = evaluate(self.target_grid, player_2_message)

                        # Calculate flipped pixels
                        flipped_count = calculate_flipped_pixels(previous_turn_grid, player_2_message)
                        flipped_count_sum += flipped_count
                        previous_turn_grid = player_2_message

                        valid_turns += 1
                    except:
                        pass

                if player_1_message:
                    expression_length = len(player_1_message.replace('Instruction:', '').strip())
                    expression_tokens = len(player_1_message.replace('Instruction:', '').strip().split(' '))
                    expression_length_sum += expression_length
                    expression_token_sum += expression_tokens

            # Log final episode scores
            self.log_episode_score('Precision', final_precision)
            self.log_episode_score('Recall', final_recall)
            self.log_episode_score('F1', final_f1)
            self.log_episode_score(BENCH_SCORE, final_f1)

            # Calculate averages
            if valid_turns > 0:
                avg_flipped = round(flipped_count_sum / float(valid_turns), 4)
                avg_length = round(expression_length_sum / float(valid_turns), 4)
                avg_tokens = round(expression_token_sum / float(valid_turns), 4)
            else:
                avg_flipped = avg_length = avg_tokens = 0

            self.log_episode_score('Average Changed Cell Count', avg_flipped)
            self.log_episode_score('Average Generated Instruction Length', avg_length)
            self.log_episode_score('Average Generated Expression Number of Tokens', avg_tokens)


class ImageGameBenchmark(GameBenchmark):

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return ImageGameMaster(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return ImageGameScorer(self.game_name, experiment, game_instance)