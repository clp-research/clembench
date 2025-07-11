import logging
from typing import List, Dict, Tuple
import numpy as np

from clemcore.backends import Model
from clemcore.clemgame import Player, GameMaster, GameBenchmark, GameSpec
from clemcore.clemgame.legacy.master import DialogueGameMaster
from clemcore.clemgame.legacy.scorer import GameScorer
from clemcore.clemgame import metrics as ms

logger = logging.getLogger(__name__)


class MatchItPlayer(Player):
    def __init__(self, backend: Model, role: str):
        super().__init__(backend)
        self.role: str = role
        self.description: str = ""
        self.question: str = ""
        self.answer: str = ""
        self.decision: str = ""
        self.response_counter: int = 0  # count turns per round
        self.had_success: bool = False

    def reset(self):
        self.description = ""
        self.question = ""
        self.answer = ""
        self.decision = ""
        self.response_counter = 0

    def _custom_response(self, context) -> str:
        last_message = context["content"]

        if "collaborative" in last_message:
            return f"DESCRIPTION: from Player {self.role}"
        elif "ask" in last_message:
            return f"QUESTION: from Player {self.role}"
        elif "QUESTION" in last_message:
            return f"ANSWER: from Player {self.role}"
        elif "decision" in last_message:
            return "DECISION: Same grid."
        else:
            return "ANSWER: How did we land here? This is the else in the mock answers."


class MatchItAscii(DialogueGameMaster):
    """ A game where both players must decide whether looking at the same or a different grid.
    The round pattern is as follows:
    ------ Round 0
    DESC A <- PROMPT A
    DESC B <- PROMPT B
    QEST B <- DESC A
    ------ Round 1
    ANSW A <- DESC B + QUEST B
    QEST A
    ANSW B <- ANSW A + QUEST A
    QEST B
    ------ Round 2
    ANSW A <- ANSW B + QUEST B
    QEST A
    ANSW B <- ANSW A + QUEST A
    QEST B
    ------ Round 3
    ANSW A <- ANSW B + QUEST B
    QEST A
    ANSW B <- ANSW A + QUEST A
    DECI B
    ------ Round 4
    DECI A <- ANSW B
    """

    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)

        self.experiment: str = experiment["name"]
        self.flags: dict[str, str] = experiment["flags"]

        self.initial_prompt: str = experiment["initial_prompt"]
        self.desc_intro: str = experiment["desc_intro"]  # "This is my"
        self.q_reprompt: str = experiment["q_reprompt"]  # "Reprompt: Now ask a question, starting with \"QUESTION: \""
        self.d_reprompt: str = experiment["d_reprompt"]  # "Make a decision."
        self.a_request: str = experiment["a_request"]  # "Start your answer with ANSWER:"

        self.solution: str = experiment["solution"]
        self.wrong_solution: str = experiment["wrong_solution"]

        self.final_decision: bool = False
        self.success_a: bool = True
        self.success_b: bool = True
        self.aborted: bool = False

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance
        self.grid_a: str = game_instance["grid_a"]
        self.grid_b: str = game_instance["grid_b"]
        self.prompt_a = self.initial_prompt.replace("$GRID$", self.grid_a)
        self.prompt_b = self.initial_prompt.replace("$GRID$", self.grid_b)

        self.decision_turn: int = game_instance["decision_turn"]

        self.player_a: MatchItPlayer = MatchItPlayer(self.player_models[0], "A")
        self.player_b: MatchItPlayer = MatchItPlayer(self.player_models[1], "B")

        self.add_player(self.player_a, initial_context=self.prompt_a)
        self.add_player(self.player_b, initial_context=self.prompt_b)

    @property
    def current_player(self) -> MatchItPlayer:
        return self._current_player

    def _on_after_game(self):
        print("###### GAME END")

    def _on_before_round(self):
        print("----- NEXT ROUND")
        self.player_a.reset()
        self.player_b.reset()

    def _does_game_proceed(self) -> bool:
        if self.aborted:
            self.log_to_self("Game over", "Aborted")
            return False
        if self.final_decision:
            return False
        return True

    def check_flag(self, first_word: str, flag: str) -> bool:
        if first_word == flag:
            self.log_to_self("valid format", "continue")
            return True
        self.log_to_self("invalid format", f"abort, first word: {first_word}, but expected {flag}")
        self.aborted = True
        return False

    def _validate_player_response(self, player: MatchItPlayer, utterance: str) -> bool:
        player.response_counter += 1

        if not utterance.strip():  # check for empty message
            self.log_to_self("invalid content", "abort, empty message")
            self.aborted = True
            return False

        # filter to be sure that there are no empty strings
        utt_parts = list(filter(None, utterance.strip().split("\n")))
        first_word = utt_parts[0].split(" ")[0]

        # first round
        if self.current_round == 0:
            if self.current_player.response_counter == 1:  # first response should be a description
                return self.check_flag(first_word, self.flags["description"])
            return self.check_flag(first_word, self.flags["question"])  # only relevant for player B

        # decision round
        if self.current_round == self.decision_turn and player == self.player_b:
            if self.player_b.response_counter == 1:
                return self.check_flag(first_word, self.flags["answer"])
            if self.player_b.response_counter == 2:
                if not self.check_flag(first_word, self.flags["decision"]):
                    return False
                if utterance.lower().strip(".\n") == (self.flags["decision"] + " " + self.solution).lower():
                    player.success = True
                    self.log_to_self(f"Decision Player {player.role}", "success")
                    return True
                elif utterance.lower().strip(".\n") == (self.flags["decision"] + " " + self.wrong_solution).lower():
                    player.success = False
                    self.log_to_self(f"Decision Player {player.role}", "loss")
                    return True
                else:
                    self.log_to_self("invalid content", "abort, wrong message content.")
                    self.aborted = True
                    return False

        # last turn, only for Player A's decision
        if self.current_round == self.decision_turn + 1:
            self.final_decision = True
            if not self.check_flag(first_word, self.flags["decision"]):
                return False
            if utterance.lower().strip(".\n") == (self.flags["decision"] + " " + self.solution).lower():
                player.success = True
                self.log_to_self(f"Decision Player {player.role}", "success")
                return True
            elif utterance.lower().strip(".\n") == (self.flags["decision"] + " " + self.wrong_solution).lower():
                player.success = False
                self.log_to_self(f"Decision Player {player.role}", "loss")
                return True
            else:
                self.log_to_self("invalid content", "abort, wrong message content.")
                self.aborted = True
                return False

        # all other turns
        if self.current_player.response_counter == 1:
            return self.check_flag(first_word, self.flags["answer"])
        else:  # second response should be a question
            return self.check_flag(first_word, self.flags["question"])

    def _parse_response(self, player: MatchItPlayer, utterance: str) -> str:
        utterance = utterance.strip()
        if utterance.startswith(self.flags["description"]):
            player.description = utterance
        elif utterance.startswith(self.flags["question"]):
            player.question = utterance
        elif utterance.startswith(self.flags["answer"]):
            player.answer = utterance
        elif utterance.startswith(self.flags["decision"]):
            player.decision = utterance
        return utterance

    def _on_valid_player_response(self, player: MatchItPlayer, utterance: str):
        if self.current_round == 0:  # special handling for first round
            if player == self.player_b:
                if player.response_counter == 1:  # after the description, ask for a question
                    content = (self.desc_intro + self.player_a.description + "\n"
                               + self.q_reprompt)
                    self.set_context_for(self.player_b, content)
                    return
                if player.response_counter == 2:  # after the question, set context for A using player B's question
                    content = (self.desc_intro + self.player_b.description + "\n"
                               + self.player_b.question + self.a_request)
                    self.set_context_for(self.player_a, content)
                    return

        if self.current_round == self.decision_turn:  # special handling for decision round
            if player == self.player_b:
                if self.player_b.response_counter == 1:  # after last answer, ask for a decision
                    self.set_context_for(self.player_b, self.d_reprompt)
                    return
                if self.player_b.response_counter == 2:  # give last answer of B to player A and ask for decision
                    self.set_context_for(self.player_a, self.player_b.answer + "\n" + self.d_reprompt)
                    return

        # all other turns
        if player.response_counter == 1:  # after answer, ask for a question
            self.set_context_for(player, self.q_reprompt)
        if player.response_counter == 2:  # after question, ask the other for an answer
            other_player = self.player_a if player == self.player_b else self.player_b
            self.set_context_for(other_player, player.answer + "\n" + player.question + self.a_request)

    def _start_next_round(self):
        next_round = self.player_b.response_counter == 2
        return next_round

    def _should_pass_turn(self):
        # special case: pass turn after initial description of player a
        if self.current_round == 0 and self.current_player == self.player_a:
            print(self.current_player.role, "passes turn")
            return True
        # pass turn if single player has given two responses, e.g., question and description
        if self.current_player.response_counter == 2:
            print(self.current_player.role, "passes turn")
            return True
        print(self.current_player.role, "keeps turn")
        return False


class MatchItScorer(GameScorer):

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:

        all_turn_scores = []
        success_a = False
        success_b = False
        aborted = False
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            turn_score_dict = {"request_count": 0, "violated_request_count": 0, "parsed_request_count": 0}

            for event in turn:
                action = event["action"]
                # parsed requests
                if action["type"] == "invalid format":
                    turn_score_dict["violated_request_count"] += 1
                    turn_score_dict["request_count"] += 1
                    first_word = action["content"].split(" ")[-1]
                    with open("first_words.txt", "a") as myfile:
                        myfile.write(first_word + "\n")
                elif action["type"] == "invalid content":
                    turn_score_dict["violated_request_count"] += 1
                    turn_score_dict["request_count"] += 1
                elif action["type"] == "valid format":
                    turn_score_dict["parsed_request_count"] += 1
                    turn_score_dict["request_count"] += 1
                elif action["content"].startswith("Abort"):
                    aborted = True
                # decision success
                elif action["type"] == "Decision Player A":
                    if action["content"] == "success":
                        success_a = True
                elif action["type"] == "Decision Player B":
                    if action["content"] == "success":
                        success_b = True

            # log turn request scores   
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT_VIOLATED, turn_score_dict["violated_request_count"])
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT_PARSED, turn_score_dict["parsed_request_count"])
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT, turn_score_dict["request_count"])
            # calculate episode scores from turn scores
            all_turn_scores.append(turn_score_dict)

            violated_request_count = sum([turn["violated_request_count"] for turn in all_turn_scores])
            self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)
            parsed_request_count = sum([turn["parsed_request_count"] for turn in all_turn_scores])
            self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, parsed_request_count)
            request_count = sum([turn["request_count"] for turn in all_turn_scores])
            self.log_episode_score(ms.METRIC_REQUEST_COUNT, request_count)
            # log episode "success" scores
            if aborted:
                self.log_episode_score(ms.METRIC_ABORTED, 1)
                self.log_episode_score(ms.METRIC_SUCCESS, 0)
                self.log_episode_score(ms.METRIC_LOSE, 0)
                # Game-specific metrics
                self.log_episode_score(ms.BENCH_SCORE, np.nan)  # metric not applicable
                self.log_episode_score("Player Score", np.nan)
            else:
                # two wrong decisions:
                if not success_a and not success_b:
                    self.log_episode_score(ms.METRIC_ABORTED, 0)
                    self.log_episode_score(ms.METRIC_SUCCESS, 0)
                    self.log_episode_score(ms.METRIC_LOSE, 1)
                    # Game-specific metrics
                    self.log_episode_score(ms.BENCH_SCORE, 0)
                    self.log_episode_score("Player Score", 0)
                # only one decided correctly
                elif success_a != success_b:
                    self.log_episode_score(ms.METRIC_ABORTED, 0)
                    self.log_episode_score(ms.METRIC_SUCCESS, 0)
                    self.log_episode_score(ms.METRIC_LOSE, 1)
                    # Game-specific metrics
                    self.log_episode_score(ms.BENCH_SCORE, 0)  # current decision, may change (before: 50)
                    self.log_episode_score("Player Score", 50)

                else:  # = success_a and success_b:
                    self.log_episode_score(ms.METRIC_ABORTED, 0)
                    self.log_episode_score(ms.METRIC_SUCCESS, 1)
                    self.log_episode_score(ms.METRIC_LOSE, 0)
                    # Game-specific metrics
                    self.log_episode_score(ms.BENCH_SCORE, 100)
                    self.log_episode_score("Player Score", 100)


class MatchItBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return MatchItAscii(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return MatchItScorer(self.game_name, experiment, game_instance)
