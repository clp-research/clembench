import re
import os
import logging
from typing import Dict, List, Tuple
import numpy as np
from string import Template
import time
import random
import abc

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, GameBenchmark, GameMaster, DialogueGameMaster, GameScorer, Player, ParseError, GameError, RuleViolationError
from clemcore.clemgame.events import GameEventSource
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE
from resources.game_state.utils import GameObject, png_to_base64
from resources.game_state.game_state import PicState, GridState, HybridState, SemanticGridState
from resources.metrics import MetricPreparer, MetricCalculator, END_DISTANCE_SUM, EXPECTED_DISTANCE_SUM, MOVES, INIT_STATES, END_STATES, ingredients_registry, VALID_MOVES, INVALID_MOVES, PARSE_ERRORS

# Stores the game state class for each modality
STATE_DICT = {
    "text": GridState,
    "image": PicState,
    "hybrid": HybridState, 
    "semantic_text": SemanticGridState
}

# These additional metrics provide information about which part of the game a model struggles with
# Parse Error: model struggles to follow the response format
# Invalid Move: Format is correct, but the move is invalid according to the game rules. This hints at problems with spatial understanding of the grid.
# Valid Move: Valid moves made by the model
MESSAGE_STATS = {
    "invalid_move": INVALID_MOVES,
    "valid_move": VALID_MOVES,
    "parse_error": PARSE_ERRORS
}

logger = logging.getLogger(__name__)

class Cleaner(Player):
    def __init__(self, model: Model):
        logger.debug(f"Initializing {self.__class__.__name__}")
        super().__init__(model, forget_extras=["image"])
        self._custom_responses = self._prepare_custom_responses()
        self.game_state = None  # This will be set by the game master
        self._relay_message = ""
        self._relay_images = []
        self.finished = False # Used to store whether the player already suggested finishing the game

    @abc.abstractmethod
    def _prepare_custom_responses(self):
        """
        Prepare custom responses for the player. Differs per modality.
        """
        pass
    
    def _custom_response(self, messages):
        response = self._custom_responses[np.random.randint(0, len(self._custom_responses))]
        return response
    
    @abc.abstractmethod
    def store_relay_message(self, message: str, images: List[str] = None):
        """
        Store the relay message to add it to the next message.
        For image and hybrid, images also have to be taken into account.
        """
        pass

    def prepare_context(self, context: Dict) -> Dict:
        """
        Prepare the context used for perceive_context.
        """
        if self._relay_message:
            context['content'] = self._relay_message + context['content']
            self._relay_message = ""
        return context

    def perceive_context(self, context, *, log_event=True, memorize=True):
        context = self.prepare_context(context)
        return super().perceive_context(context, log_event=log_event, memorize=memorize)

class GridCleaner(Cleaner):
    def _prepare_custom_responses(self):
        return [
            "MOVE: C, (1, 1)",
            "MOVE: W, (3, 2)",
            "SAY: Move C to (1, 1).",
            "SAY: Ok, let's start the game."
            "MOVE: C, (2, 1)\nSAY: I did it! C is now in the top-left corner."
            ]
    
    def store_relay_message(self, message: str, images: List[str] = None):
        """
        Store the relay message to add it to the next message.
        Ignore images, as GridState does not have images.
        """
        self._relay_message = message
    
class PicCleaner(Cleaner):
    def _prepare_custom_responses(self):
        return [
            "MOVE: C, (100, 100)",
            "SAY: Move C to the top left corner.",
            "SAY: Ok, let's start the game."
            ]
    
    def store_relay_message(self, message: str, images: List[str] = None):
        """
        Store the relay message to add it to the next message.
        Images are only logged and not added to the next message.
        """
        self._relay_message = message
        self._relay_images = images

    def prepare_context(self, context):
        context = super().prepare_context(context)
        if self._relay_images:
            context['image'] = self._relay_images
            self._relay_images = []
        return context
    
class HybridCleaner(Cleaner):
    def _prepare_custom_responses(self):
        return [
            "MOVE: C, (1, 1)",
            "SAY: Move C to (1, 1).",
            "SAY: Ok, let's start the game."
            ]
    
    def store_relay_message(self, message: str, images: List[str] = None):
        """
        Store the relay message to add it to the next message.
        """
        self._relay_message = message
        self._relay_images = images

    def prepare_context(self, context):
        context = super().prepare_context(context)
        if self._relay_images:
            context['image'] = self._relay_images
            self._relay_images = []
        return context

def log_images(game_event_source: GameEventSource, images: list[str], player: Player=None):
    """
    Logs images outside of normal messages
    images: list of image paths
    player: if None, assume the game_event_source is the player
    """
    if not player:
        player = game_event_source
    assert isinstance(player, Player), "player must be an instance of Player"
    for image in images:
        if image.startswith('clean_up/'):
            image = image[len('clean_up/'):]
        action = {
                    'type': 'send message',  
                    'content': 'logged image',
                    'image': [image]
                }
        game_event_source.log_event(from_='GM', to=player.name, action=action)

PLAYER_DICT = {
    "text": GridCleaner,
    "image": PicCleaner,
    "hybrid": HybridCleaner, 
    "semantic_text": GridCleaner
}

class CleanUpMaster(DialogueGameMaster):
    def __init__(self, game_spec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance
        self.modality = game_instance['modality']

        self.intermittent_prompts = game_instance['intermittent_prompts']
        self.parse_errors = game_instance['parse_errors']

        # Compile all regex patterns used in the game instance
        self.say_pattern = re.compile(self.game_instance['say_pattern'], re.DOTALL)
        self.move_pattern = re.compile(self.game_instance['move_pattern'], re.DOTALL)
        self.restricted_patterns = []
        for pattern in self.game_instance.get('restricted_patterns', []):
            self.restricted_patterns.append(re.compile(pattern, re.DOTALL))

        self.img_prefixes = []  # List of image prefixes, one per player
        self.player_1 = PLAYER_DICT[self.modality](self.player_models[0])
        self.player_2 = PLAYER_DICT[self.modality](self.player_models[1])
        self.add_player(self.player_1, objects=self.game_instance['objects_1'])
        self.add_player(self.player_2, objects=self.game_instance['objects_2'])
        if self.modality in ['text', 'semantic_text']:
            self.initial_board = "```\nPlayer 1:\n" + str(self.player_1.game_state) + "\n\nPlayer 2:\n" + str(self.player_2.game_state) + "\n```"

        self.initial_distance = self.player_1.game_state.distance_sum(self.player_2.game_state)

        self.success = False    # True if game finished regularly
        self.terminate = False  # True if game game should end
        self.aborted = False    # True if game is aborted due to a rule violation or parse error
        self.penalties = 0      # Number of collectively accumulated penalties
        self.max_penalties = self.game_instance['max_penalties']    # For strict mode, max_penalties is 0
        self.pass_turn = True
        self.max_rounds = self.game_instance['max_rounds']

        self.metric_preparer = MetricPreparer(self, self.player_1, self.player_2)
        self.message_stats = {v: 0 for v in MESSAGE_STATS.values()}

    def add_player(self, player: Player, objects: List[GameObject] = None):
        """
        Add a player to the game. Needs to be implemented per modality.
        """
        super().add_player(player)
        # Initialize img_prefix, consisting of experiment name, game instance ID, and player ID
        id = len(self.players_by_names)  # Player IDs start from 1
        img_prefix = f"{self.experiment['name']}_{self.game_instance['game_id']}_player{id}_{player._model.name}"
        self.img_prefixes.append(img_prefix)
        player.game_state = STATE_DICT[self.modality](background=self.game_instance['background'], move_messages=self.game_instance['move_messages'], objects=objects, img_prefix=img_prefix)
    
    def _other_player(self) -> Player:
        """
        Returns the player who will be next.
        """
        other_player_idx = (self._current_player_idx + 1) % len(self.players_by_names)
        return self.get_players()[other_player_idx]
    
    def _on_before_game(self):
        """
        Set the initial context for the first player.
        """
        images = self.player_1.game_state.draw()  # returns None for GridState
        if images:
            self.set_context_for(self.player_1, self.game_instance['p1_initial_prompt'], image=images)
        else:
            self.set_context_for(self.player_1, self.game_instance['p1_initial_prompt'])

    def _check_head_tail(self, match: re.Match) -> bool:
        """
        Check if the head and tail of the match are empty.
        """
        # if not self.game_instance['lenient']:
        head = match.group('head')
        tail = match.group('tail')
        if head != '' and tail != '':
            self.log_to_self('parse_error', f"Invalid format: head and tail are not empty\nhead: '{head}'\ntail: '{tail}'")
            self.message_stats[MESSAGE_STATS['parse_error']] += 1
            raise ParseError(reason=self.parse_errors["head_tail"], response=match.group(0))
        elif head != '':
            self.log_to_self('parse_error', f"Invalid format: head is not empty: '{head}'")
            self.message_stats[MESSAGE_STATS['parse_error']] += 1
            raise ParseError(reason=self.parse_errors["head"], response=match.group(0))
        elif tail != '':
            self.log_to_self('parse_error', f"Invalid format: tail is not empty: '{tail}")
            self.message_stats[MESSAGE_STATS['parse_error']] += 1
            raise ParseError(reason=self.parse_errors["tail"], response=match.group(0))

    def _parse_response(self, player: Player, response: str) -> str:
        logger.debug(f"{player.name}: {response}")
        self.log_to_self('player_response', response)
        # We just remove backticks
        response = response.replace('`', '').strip()
        move_matches = list(self.move_pattern.finditer(response))
        say_matches = list(self.say_pattern.finditer(response))
        if len(move_matches) + len(say_matches) > 1:
            self.log_to_self('parse_error', f"Invalid response format: {response}")
            logger.debug(f"Response '{response}' contains several commands.")
            self.message_stats[MESSAGE_STATS['parse_error']] += 1
            raise ParseError(reason=self.parse_errors["several_commands"], response=response)
        move_match = move_matches[0] if move_matches else None
        say_match = say_matches[0] if say_matches else None
        if player == self.player_1 and self.current_round == 0 and not say_match:
            # In this case, the command needs to be a message
            self.log_to_self('parse_error', f"Invalid response: {response}")
            logger.debug(f"Response '{response}' is not a valid message, first command must be a message.")
            self.message_stats[MESSAGE_STATS['parse_error']] += 1
            raise ParseError(reason=self.parse_errors["invalid_start"], response=response)
        if move_match:
            self._check_head_tail(move_match)
            return response
        if say_match:
            self._check_head_tail(say_match)
            if self.game_instance['terminate_question'] in say_match.group('message'):
                player.finished = True
            if self.game_instance['terminate_answer'] in say_match.group('message') and self._other_player().finished:
                self.success = True
                self.terminate = True
                self.log_to_self('success', 'true')
            for restricted_pattern in self.restricted_patterns:
                restricted_match = restricted_pattern.search(say_match.group('message'))
                if restricted_match:
                    self.pass_turn = False
                    self.log_to_self('rule_violation', f"Response violates restriction: {restricted_pattern}")
                    self.message_stats[MESSAGE_STATS['parse_error']] += 1
                    raise ParseError(reason=self.parse_errors["restriction"], response=response)
            return response
        else:
            self.log_to_self('parse_error', f"Invalid response format")
            self.message_stats[MESSAGE_STATS['parse_error']] += 1
            raise ParseError(reason=self.parse_errors["invalid_format"], response=response)

    def _on_parse_error(self, error: GameError):
        self.pass_turn = False
        self.penalties += 1
        logger.debug(f"Parse error: {error}")
        message = self._reprompt_message(error.reason)
        self.set_context_for(self._current_player, message)

    def _reprompt_message(self, reason) -> str:
        message = Template(self.intermittent_prompts['invalid_response']).substitute(reason=reason)
        message += '\n' + self._penalty_counter_message() + self.intermittent_prompts['penalty_reprompt']
        return message

    def _should_pass_turn(self) -> bool:
        """
        Check if the player should pass their turn.
        """
        # When working with inference providers, uncomment the following line to not exceed free inference budget too quickly.
        # time.sleep(random.uniform(1, 2))
        return self.pass_turn

    def _start_next_round(self) -> bool:
        """
        :return: True, when it's the first player's turn to start a new round
        """
        if self.pass_turn:
            return self._current_player_idx == 0     
        else:
            return False

    def _advance_game(self, player: Player, parsed_response: str):
        if not parsed_response:
            raise RuleViolationError
        match = self.move_pattern.match(parsed_response)
        if match:
            obj = match.group('obj')
            x = match.group('x')
            y = match.group('y')
            success, message, images = player.game_state.move_abs(obj, x, y)
            self.pass_turn = success
            if success:
                logger.debug(f"{player.name} moved {obj} to ({x}, {y}) successfully.")
                self.metric_preparer.add_move((player.name, obj))
                # log the move message to the player and add it to the message history (without response)
                self.log_to_self('valid_move', f'{obj} moved to ({x}, {y})')
                self.message_stats[MESSAGE_STATS['valid_move']] += 1
                player.store_relay_message(message, images=images)
                # turn is passed to the other player
                next_player_prompt = self._new_turn_prompt(self.intermittent_prompts["new_turn_move"])
                self.set_context_for(self._other_player(), next_player_prompt)
            if not success:
                logger.debug(f"{player.name} failed to move {obj} to ({x}, {y}): {message}")
                # Player is reprompted with a penalty, their turn continues. 
                self.penalties += 1
                message = message + "\n" + Template(self.intermittent_prompts['penalty_counter']).substitute(penalty=self.penalties) + self.intermittent_prompts['penalty_reprompt']
                self.log_to_self('invalid_move', message)
                self.message_stats[MESSAGE_STATS['invalid_move']] += 1
                self.set_context_for(player, message)
                raise RuleViolationError(f"Invalid move: {message}")
        else:
            match = self.say_pattern.match(parsed_response)
            if match:
                message = match.group('message')
                self.pass_turn = True
                player.store_relay_message(Template(self.intermittent_prompts['message_relay']).substitute(message=message))
                if player == self.player_1 and self.current_round == 0:
                    p2_initial_prompt = Template(self.game_instance['p2_initial_prompt']).substitute(
                        start_message=message
                    )
                    images = self.player_2.game_state.draw() # returns None for text-based versions
                    if images:
                        self.set_context_for(self.player_2, p2_initial_prompt, image=images)
                    else:
                        self.set_context_for(self.player_2, p2_initial_prompt)
                else:
                    next_player_prompt = self._new_turn_prompt(Template(self.intermittent_prompts['new_turn']).substitute(turn_message=message))
                    self.set_context_for(self._other_player(), next_player_prompt)

    def _new_turn_prompt(self, content: str) -> str:
        """
        Adds round counter and penalty counter before `content` and command prompt after it.
        :param content: The content to add the round and penalty counters to.
        :return: The new prompt with round and penalty counters.
        """
        content = self._counter_messages() + content
        content += self.intermittent_prompts["command_prompt"]
        return content

    def _counter_messages(self) -> str:
        """
        Returns a message with the current turn count and penalty count.
        """
        return self._round_counter_message() + self._penalty_counter_message()
    
    def _round_counter_message(self) -> str:
        """
        Returns a message with the current turn count.
        """
        return Template(self.intermittent_prompts['round_counter']).substitute(
            round=self.current_round + 1
        )
            
    def _penalty_counter_message(self) -> str:
        """
        Returns a message with the current penalty count.
        """
        return Template(self.intermittent_prompts['penalty_counter']).substitute(
            penalty=self.penalties
        )

    def _does_game_proceed(self):
        """
        Check if the game should continue.
        """
        if self.penalties > self.max_penalties:
            self.log_to_self('end', 'Maximum number of penalties exceeded')
            self.aborted = True
            return False
        if self.terminate:
            return False
        if self.current_round >= self.max_rounds:  # Arbitrary limit for rounds
            logger.info("Maximum number of rounds reached, ending game.")
            self.log_to_self('end', 'Maximum number of rounds reached')
            # Reaching the maximum number of rounds is considered a success
            self.success = True
            return False
        return True

    def compute_turn_score(self):
        return 1 if self.success else 0

    def compute_episode_score(self):
        if self.success:
            return 100 / (self.current_round + 1)  # zero-based
        return 0
    
    def _after_game_logs(self):
        if self.modality == 'image':
            for player in self.get_players():
                move_image = player.game_state.draw_moves()
                if move_image:
                    log_images(self, move_image, player)

    def _on_after_game(self):
        self._after_game_logs()
        self.log_key("markdown", True)
        ingredients = self.metric_preparer.compute_ingredients()
        ingredients_string = ""
        for key, val in ingredients.items():
            # log all the necessary metrics to `interaction.json`
            self.log_key(key, val)
            # not display some of the ingredients in transcript
            if key not in [INIT_STATES, END_STATES]:
                if type(val) is list:
                    continue
                else:
                    ingredients_string += f"* {key}: {float(val):.2f}\n"

        lose = not self.success
        if self.success:
            # If the game is terminated successfully, we check whether 
            # the end distance is greater than the expected distance
            lose = ingredients[END_DISTANCE_SUM] > ingredients[EXPECTED_DISTANCE_SUM]

        self.log_key(METRIC_ABORTED, int(self.aborted))
        self.log_key(METRIC_LOSE, int(lose))
        self.log_key(METRIC_SUCCESS, int(self.success))
        if self.modality in ['text', 'semantic_text']:
            self.log_to_self("initial_state", "Initial states:\n" + self.initial_board)
            self.log_to_self("end_state", "End states:\n```\nPlayer 1:\n" + str(self.player_1.game_state) + "\n\nPlayer 2:\n" + str(self.player_2.game_state) + "\n```")

        self.log_to_self('game_finished', f"* success: {self.success}\n* lose: {lose}\n* aborted: {self.aborted}\n-------\n{ingredients_string}") 

        for key, val in self.message_stats.items():
            self.log_key(key, val)           
        
        # ----------------------------------------------------------
        # dev: also compute sub-metrics and bench score to show on transcript
        # metrics_calculator = MetricCalculator(ingredients)
        # sub_metrics, bench_score, temp_log  = metrics_calculator.compute_metrics()

        # bench_score_string = f"* {BENCH_SCORE}: {float(bench_score):.2f}\n"

        # sub_metrics_string = ""
        # for key, val in sub_metrics.items(): 
        #     sub_metrics_string += f"* {key}: {float(val):.2f}\n"

        # temp_log_string = ""
        # for key, val in temp_log.items(): 
        #     if type(val) is list or type(val) is dict:
        #         continue
        #     else:
        #         temp_log_string += f"* {key}: {float(val):.2f}\n"

        # self.log_to_self('dev:game_finished', f"{bench_score_string}\n-------\n{sub_metrics_string}\n-------\n{temp_log_string}")
        # print(f"\n\n{bench_score_string}\n-------\n{sub_metrics_string}\n-------\n{temp_log_string}")
        # ----------------------------------------------------------


class CleanUpScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def score_turns(self, episode_interactions: Dict) -> None:
        """ Turn-level scores """
        for turn_idx in range(len(episode_interactions)):
            for event in episode_interactions[turn_idx]:
                if event['type'] == 'player_response':
                    self.log_turn_score(turn_idx, 'response_received', 1)

    def compute_episode_scores(self, episode_interactions: Dict) -> float:
        """ Compute the episode score based on the ingredients logged in interactions """
        # reconstruct ingredients from episode_interactions
        ingredients = {}
        for key in ingredients_registry:
            if key not in episode_interactions:
                logger.warning(f"Missing Key: Key {key} should be in episode interactions. ")            
            ingredients[key] = episode_interactions[key]
        
        metrics_calculator = MetricCalculator(ingredients)
        sub_metrics, bench_score, temp_log = metrics_calculator.compute_metrics()        

        # log sub-metrics
        for key in sub_metrics:
            self.log_episode_score(key, sub_metrics[key])

        for key in temp_log:
            self.log_episode_score(key, temp_log[key])

        # log the bench score
        if episode_interactions[METRIC_SUCCESS]:
            # the case when game is LOSE is taken care of by MetricCalculator
            self.log_episode_score(BENCH_SCORE, bench_score) 
        else:
            logger.debug(f'aborted, logging Main Score as np.nan')
            self.log_episode_score(BENCH_SCORE, np.nan)

class CleanUpBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return CleanUpMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return CleanUpScorer(self.game_name, experiment, game_instance)

