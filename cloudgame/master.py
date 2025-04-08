# TODO add to _validate_player_response: do not automatically return True (important for when not mock)
# TODO add played or aborted metric to compute_scores (see prev. todo)
import os
import random
from typing import List, Dict
import logging
import numpy as np

import clemcore.clemgame.metrics as ms
from clemcore.clemgame import GameBenchmark, DialogueGameMaster, GameScorer, GameSpec, GameRecorder
from clemcore.clemgame import Player

from clemcore.backends import Model, CustomResponseModel
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_COUNT_VIOLATED

logger = logging.getLogger(__name__)


class Speaker(Player):
    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, context) -> str:
        """Return yes or no randomly."""
        k = random.randint(0, 1)
        if k == 0:
            return "No"
        else:
            return "Yes"


class Judge(Player):

    def __init__(self):
        super().__init__(CustomResponseModel())

    def _custom_response(self, context):
        return "That seems right."


class Cloudgame(DialogueGameMaster):
    """Implement mechanisms for playing Cloudgame."""

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)
        # fetch experiment parameters here
        self.game_path = game_path
        self.max_words = 2
        self.allowed_words = ["yes", "no"]
        self.success = True
        self.aborted: bool = False

        self.experiment = experiment['name']
        self.model_a = player_models[0]

    def _on_setup(self, **game_instance):

        """" sets the information you specify in instances.json """

        self.game_instance = game_instance
        self.image = os.path.join(self.game_path, game_instance["image"])
        self.initial_prompt = game_instance["prompt"]

        self.speaker = Speaker(self.model_a)
        self.judge = Judge(self)

        self.add_player(self.speaker)
        self.add_player(self.judge)

    def _does_game_proceed(self):
        if not self.aborted and self.current_round <= 1:
            return True
        return False

    def _on_before_round(self):
        if self.current_round == 0:
            self.set_context_for(self.speaker, self.initial_prompt, image=self.image)
            self.set_context_for(self.judge, "Do you think this is correct?")
        if self.current_round == 1:
            self.set_context_for(self.speaker,
                                 'Are there any chickens in the picture? Answer with only "Yes" or "No".')
            self.set_context_for(self.judge, "Do you think this is correct?")

    def _validate_player_response(self, player: Player, answer: str) -> bool:
        """Check if the utterance conforms to rules (cloudgame specific)."""

        # Remove \n from answer
        answer = answer.replace("\n", "")
        # there should never be a chicken in a picture

        if player == self.speaker:
            true_answer = self.experiment
            split_answer = answer.strip(" .").split(" ")
            # only one word allowed
            if len(split_answer) != 1:
                self.success = False
                self.aborted = True
                self.log_to_self("Invalid word count", "Game aborted.")
                return False

            # only yes or no allowed
            if answer.lower().strip(" .") not in self.allowed_words:
                self.success = False
                self.aborted = True
                self.log_to_self("Invalid words", "Game aborted.")
                return False
            # is answer correct?
            elif answer.lower() != true_answer:
                self.success = False

            self.log_to_self("Valid format", "Continue")

        return True

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        # if player == self.speaker:
        #     self.add_user_message(self.judge, utterance)
        if player == self.judge:
            self.set_context_for(self.speaker, parsed_response)

    def _on_after_round(self):
        self.log_to_self(type_="judgement", value=self.success)
        if self.aborted:
            self.log_to_self(type_="aborted", value=self.aborted)


class CloudgameScorer(GameScorer):

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:

        all_turn_scores = []
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            # player_1_message = turn[1]['action']['content']
            score = 0
            turn_score_dict = {"request_count": 0, "violated_request_count": 0, "parsed_request_count": 0}
            aborted = False

            for event in turn:
                action = event["action"]

                if action["type"] == "get message":
                    turn_score_dict["request_count"] += 1
                if action["type"] == "Valid format":
                    turn_score_dict["parsed_request_count"] += 1
                if action["type"] == "Invalid word count":
                    turn_score_dict["violated_request_count"] += 1
                    aborted = True
                if action["type"] == "Invalid words":
                    turn_score_dict["violated_request_count"] = 1
                    aborted = True
                if action["type"] == "judgement":
                    score = action["content"]

            # log turn request scores
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_VIOLATED, turn_score_dict["violated_request_count"])
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_PARSED, turn_score_dict["parsed_request_count"])
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT, turn_score_dict["request_count"])

            all_turn_scores.append(turn_score_dict)

        violated_request_count = sum([turn["violated_request_count"] for turn in all_turn_scores])
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)
        parsed_request_count = sum([turn["parsed_request_count"] for turn in all_turn_scores])
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, parsed_request_count)
        request_count = sum([turn["request_count"] for turn in all_turn_scores])
        self.log_episode_score(ms.METRIC_REQUEST_COUNT, request_count)

        if aborted:
            self.log_episode_score(METRIC_ABORTED, 1)
            self.log_episode_score(METRIC_SUCCESS, 0)
            self.log_episode_score(METRIC_LOSE, 0)
            # Game-specific metrics
            self.log_episode_score(BENCH_SCORE, np.nan)
        else:
            self.log_episode_score(METRIC_ABORTED, 0)
            self.log_episode_score(METRIC_SUCCESS, 1 if score else 0)
            self.log_episode_score(METRIC_LOSE, 0 if score else 1)
            self.log_episode_score(BENCH_SCORE, 100)


class CloudgameBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    # copy this, replacing the name of the game master in the return statement
    def create_game_master(self,
                           experiment: Dict,
                           player_models: List[Model]
                           ) -> DialogueGameMaster:
        return Cloudgame(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return CloudgameScorer(self.game_name, experiment, game_instance)
