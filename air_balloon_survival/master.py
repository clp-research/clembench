from typing import Dict, List, Any
import logging
import numpy as np
import re
import ast

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, GameMaster, DialogueGameMaster, GameScorer, GameBenchmark, Player, GameError, ParseError
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, METRIC_REQUEST_COUNT
from clemcore.clemgame.metrics import METRIC_REQUEST_COUNT_VIOLATED, BENCH_SCORE

logger = logging.getLogger(__name__)


class Negotiator(Player):
    def __init__(self,
                 model: Model,
                 name: str,
                 preference_scale: Dict[str,int|float],
                 mock_response: str | None = None
                 ):
        super().__init__(model)

        #self.name = name # this gives a default "Player 1" or "Player 2" which creates an issue with our naming <player_1>

        self.patience = None
        self.preference_scale = preference_scale
        # a list which will track all active proposals for the player
        self.active_proposals: list = []

        # The assumption here is always that mock plays against mock
        # otherwise it cannot be guaranteed that the game is played
        self.mock_response = mock_response

        self.internal_name = name #set up variable for violated request tracking

    def _custom_response(self, messages):
        # proposal = self._custom_responses.pop(0)
        # return proposal
        return self.mock_response


class HotAirBalloon(DialogueGameMaster):
    """
    Template class for game master.
    """
    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)

    def _on_setup(self, **game_instance):

        self.game_instance = game_instance

        self.items = game_instance["item_weights"].keys()

        self.player_1 = Negotiator(self.player_models[0],
                                   "player_1",
                                   game_instance["player1_preferences"],
                                   game_instance["mock_response_p1"])
        self.player_2 = Negotiator(self.player_models[1],
                                   "player_2",
                                   game_instance["player2_preferences"],
                                   game_instance["mock_response_p2"])

        # add prompts as contexts to the LLM instances
        self.add_player(self.player_1, initial_context=game_instance['player1_initial_prompt'])
        self.add_player(self.player_2, initial_prompt=game_instance['player2_initial_prompt'])

        # a dictionary which will track all active proposals for a given player
        self.active_proposals: dict = {'player_1': [], 'player_2': []}

        # initialize turn variable
        self.current_turn: int = 0
        # initialize api request variable
        self.request_count: int = 0
        # attribute indicating number of correctly parsed requests
        self.parsed_request_count: int = 0
        # initialize variable counting subsequent turns with no proposals
        self.successive_empty_turns: int = 0
        # initialize variables counting requests that violated the rules
        self.violated_request_count: int = 0
        # initialize a counter that tracks violated request count per player
        self.violated_requests_per_player = {'player_1': 0, 'player_2': 0}

        # attributes indicating whether the game
        self.aborted: bool = False
        self.lose: bool = False
        self.deal_finalized: bool = False
        self.final_deal = None

        self.agree_tag: str = rf"{game_instance['agree_tag']}"
        self.refusal_tag: str = rf"{game_instance['refusal_tag']}"
        self.proposal_tag: str = rf"{game_instance['proposal_tag']}"
        self.argument_tag: str = rf"{game_instance['argument_tag']}"

        self.strategic_reasoning_tag: str = rf"{game_instance['strategic_reasoning_tag']}"

        self.empty_turn_limit = game_instance["empty_turn_limit"]
        self.player_1.patience = game_instance["patience"]
        self.player_2.patience = game_instance["patience"]

        # initialize calculated benchmarking values
        self.max_u1 = self.game_instance["max_u1"]
        self.max_u2 = self.game_instance["max_u2"]
        self.max_total_utility = self.game_instance["max_total_utility"]
        self.max_harmonic = self.game_instance["max_harmonic"]
        self.pareto_optima_sets = {frozenset(opt) for opt in self.game_instance["pareto_optima_sets"]}



    def _parse_response(self, player: Player, response: str) -> bool | dict[str, list[Any]]:
        '''
        logs any tagged event the player introduces. ParseError is raised in case an illegal move has been made
        '''

        # increase the number of API requests:
        self.request_count += 1
        self.current_turn += 1
        
        # this variable will be changed if strategic reasoning tag is used
        chat_response = response

        # All valid tags a player can use.
        # The finalized dictionary will also be the parsed response being handed to advance game to check for rule violations
        tags_dict = {self.agree_tag: [],
                    self.refusal_tag: [],
                    self.proposal_tag: [],
                    }
        
        # tailor parse error prompts to the type of mistake
        if self.game_instance["strategic_reasoning"]:
            tags_dict[self.strategic_reasoning_tag] = []
            # check if no strategic reasoning tag at the very beginning of response
            if not re.match(self.strategic_reasoning_tag, response, re.DOTALL):
                return self.reprompt(player, ParseError, self.game_instance["no_sr_prompt"])
        if self.game_instance['require_argument']:
            # check if response contains argument format
            if not re.search(self.argument_tag, response, re.DOTALL):
                return self.reprompt(player, ParseError, self.game_instance["no_argument_prompt"])
            tags_dict[self.argument_tag] = []

        remaining_text = response
        for tag_pattern in tags_dict.keys():
            remaining_text = re.sub(tag_pattern, "", remaining_text, flags=re.DOTALL)

        # check if some part of the response was untagged
        if remaining_text.strip():
            return self.reprompt(player, ParseError, self.game_instance["untagged_sequence_prompt"])
            
        # iterate over all possible tags and check for a match
        for tag in tags_dict:

            # find all matches for a given tag
            matches = re.findall(tag, response)

            # iterate over all matches (multiple e.g. when several proposals were made)
            for match in matches:
                # extract the set following the tag
                
                # remove tag and space to extract string representation of a python set
                set_str = match.split(":", 1)[1].strip()
                # case where tag is "STRATEGIC REASONING"
                if tag == self.strategic_reasoning_tag:

                    # log strategic reasoning
                    tags_dict[tag] = set_str.translate(str.maketrans('', '', '{""}'))

                    # remove strategic reasoning from the message being sent to the other player
                    chat_response = re.sub(re.compile(self.strategic_reasoning_tag, re.DOTALL),
                                           "",
                                           response).strip()
                    # check if there is anything in the chat response (would be only whitespace if the player only used strategic reasoning)
                    if not re.search(r'\S', chat_response):
                        return self.reprompt(player, ParseError, self.game_instance["only_sr_prompt"])

                # case where tag is ARGUMENT
                if tag == self.argument_tag:
                    tags_dict[tag] = set_str

                # we need to parse the set if we are dealing with PROPOSAL, REFUSE or AGREE
                if tag in {self.proposal_tag, self.refusal_tag, self.agree_tag}:
                    
                    # set_str = set_str.replace('"', "'") # convert to valid Python literal
                    try:
                        parsed_set = ast.literal_eval(set_str)
                    # the object is not expressed as a valid python set
                    except:
                        raise self.reprompt(player, ParseError, self.game_instance["invalid_python_set_error"])
                    
                    # append the mentioned set to the parsed response dictionary under the respective tag
                    tags_dict[tag].append(parsed_set)

        # check if an arguemtn has been made
        if self.game_instance["require_argument"] and not tags_dict[self.argument_tag] and not tags_dict[self.agree_tag]:
            return self.reprompt(player, ParseError, self.game_instance["parse_error_prompt"])
        
        # set filtered reponse as new context for players
        if player == self.player_1:
            self.set_context_for(self.player_2, chat_response)
        if player == self.player_2:
            self.set_context_for(self.player_1, chat_response)

        # if we reached this return statement we have not met any Parsing exceptions
        self.parsed_request_count += 1
        return tags_dict

    def _on_parse_error(self, error: GameError):
        """Abort the game due to failed parsing."""
        # set the game to be aborted:
        self.aborted = True
        # increment violated request count because of parsing error
        self.violated_request_count += 1
        self.violated_requests_per_player[self._current_player.internal_name] += 1
        # log the abortion event:
        action = {'type': 'missing tag', 'content': 'abort'}
        self.log_event(from_='GM', to='GM', action=action)

    def _on_game_error(self, error: GameError):
        """Lose the game due to violated rules."""
        self.lose = True
        # log the fact that the game is now lost:
        action = {'type': 'rule violation',
                  'content': error.reason}
        self.log_event(from_='GM', to='GM', action=action)

    def _advance_game(self, player: Player, parsed_response: str):
        
        # if true, means a parsing error occured and _advance_game is redundant
        if parsed_response == "retry":
            return
        
        # store other player in a variable
        other_player = next(p for p in self.get_players() if p != player)
        
        # check for error in propsal
        for proposal in parsed_response[self.proposal_tag]:
            # the player mentioned objects which are not in the game
            if not proposal.issubset(self.items):
                self.reprompt(player, GameError, self.game_instance["proposal_error_prompt"])
                return

        # check for error in refusal
        for refusal in parsed_response[self.refusal_tag]:
            if not refusal in other_player.active_proposals:
               self.reprompt(player, GameError, self.game_instance["refuse_error_prompt"])
               return

        # check for error in agreement
        # check if agreement was valid, i.e. in response to an active proposal
        if parsed_response[self.agree_tag]:
            agreement = parsed_response[self.agree_tag]
            if len(agreement) > 1:
                self.reprompt(player, GameError, self.game_instance["agreement_ambiguous_prompt"])
            if not agreement[0] in other_player.active_proposals:
                self.reprompt(player, GameError, self.game_instance["agreement_non_active_prompt"])
            
            # we return None here to make sure both mistakes can be caught
            if len(agreement) > 1 or not agreement[0] in other_player.active_proposals:
                return

        if not parsed_response[self.proposal_tag]:
            self.successive_empty_turns += 1
        else:
            self.successive_empty_turns = 0 #removed double <self.self>

        if self.successive_empty_turns > self.empty_turn_limit: #max_turn_limit not used anywhere else
            raise GameError

        else:
            for proposal in parsed_response[self.proposal_tag]:
                # the player mentioned objects which are not in the game
                if not proposal.issubset(self.items):
                    self.reprompt(player, GameError, self.game_instance["proposal_error_prompt"])
                    return
                else:
                    self.current_player.active_proposals.append(proposal)

            # check if refusal was valid, i.e. in response to an active proposal
            for refusal in parsed_response[self.refusal_tag]:
                if not refusal in other_player.active_proposals:
                    raise GameError
                else:
                    other_player.active_proposals.remove(refusal)

            # check if agreement was valid, i.e. in response to an active proposal
            if parsed_response[self.agree_tag]:
                agreement = parsed_response[self.agree_tag]
                if not agreement[0] in other_player.active_proposals or len(agreement) > 1:
                    raise GameError
                else:
                    self.deal_finalized = True
                    self.final_deal = {item for item in agreement[0]}

    def reprompt(self, player: Player, error: Exception, prompt: str) -> bool:

        # decrease patience by 1
        player.patience -= 1

        # if patience ran out throw the specified error
        if player.patience < 0:
            raise error
        
        # set specified context and return retry indicating that the player has been reprompted and needs to make another try
        else:
            self.set_context_for(player, prompt)
            # already move to the next player, so that player will be set as current player in play
            self._current_player = self._next_player()
            return "retry"


    def _does_game_proceed(self) -> bool:
        """Check if game should proceed."""
        return (not self.deal_finalized
                and not self.aborted
                and not self.lose)

    
    def compute_turn_score(self):
        """Calculate turn-level quality score. Percentage of how many of the requests were valid."""
        if self.request_count == 0:
            return float('NaN')
        return (self.request_count - self.violated_request_count / self.request_count)
    
    def compute_episode_score(self):
        if self.deal_finalized and not self.aborted and not self.lose:
            if self.current_turn > 0:
                return 100.0 / self.current_turn
            else:
                return 100.0
        elif self.lose:
            return 0.0
        elif self.aborted:
            return float('NaN')
        else:
            return 0.0
        
    def is_final_deal_pareto_optimum(self):
        """Cross checks the final deal with the pre-calculated possible pareto-optima sets within game instnace.
        Returns True/False
        """
        if not self.deal_finalized:
            return False
        return frozenset(self.final_deal) in self.pareto_optima_sets

    def compute_utility_score(self):
        """Computes the utility score of the final deal"""
        if not self.deal_finalized or not self.final_deal:
            # if the deal was not finalized or no final deal was made, set scores to None
            self.player1_score = None
            self.player2_score = None
            
        else:
            # compute the score based on the utility scale of both players
            self.player1_score = sum(self.player_1.preference_scale[item] for item in self.final_deal)
            self.player2_score = sum(self.player_2.preference_scale[item] for item in self.final_deal)

        return self.player1_score, self.player2_score
    
    def normalize_scores(self):
        if self.player1_score is None or self.player2_score is None:
            self.normalized_u1 = float('NaN')
            self.normalized_u2 = float('NaN')
            self.harmonic_mean = float('NaN') #fairness metric using normalized utilities
            self.max_harmonic = float('NaN')
            self.normalized_harmonic_mean = float('NaN') #efficiency-fairness trade-off score. main metric
            return
        
        #1)Normalized utility scores per player
        self.normalized_u1 = (self.player1_score / self.max_u1 * 100) if self.max_u1 > 0 else float('NaN')
        self.normalized_u2 = (self.player2_score / self.max_u2 * 100) if self.max_u2 > 0 else float('NaN')

        #2)Harmonic mean of normalized scores
        if self.normalized_u1 + self.normalized_u2 > 0:
            self.harmonic_mean = 2 * (self.normalized_u1 * self.normalized_u2) / (self.normalized_u1 + self.normalized_u2)
        else:
            self.harmonic_mean = float('NaN')

        #3)Normalized Collective Harmonic Mean - divide 2 by the optimal solution to normalize the score.
        if self.max_total_utility > 0:
            self.normalized_harmonic_mean = self.harmonic_mean / self.max_harmonic * 100
        else:
            self.normalized_harmonic_mean = float('NaN')

        # overwrite if utility distributions are the same
        if self.game_instance['negotiation_mode'] == 'equal':
            self.normalized_harmonic_mean = self.normalized_u1

        # if the players make a deal which has too much weight they get scored 0
        if self.final_deal is not None:
            self.weight_of_deal = sum([self.game_instance['item_weights'][item] for item in self.final_deal])
        else:
            self.weight_of_deal = 0
        if self.weight_of_deal > self.game_instance['max_weight']:
            self.normalized_u1 = 0
            self.normalized_u2 = 0
            self.player1_score = 0
            self.player2_score = 0
            self.normalized_harmonic_mean = 0

            

    def make_json_serializable(self, obj):
        if isinstance(obj, (set, frozenset)): # converts set/frozenset to list
            return list(obj)
        elif isinstance(obj, dict): #converts dict recursively to every value
            return {k: self.make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list): #converts list recursively to each element
            return [self.make_json_serializable(i) for i in obj]
        else:
            return obj

    def _on_after_game(self) -> None:
        """Log variables needed for scoring."""

        if self.final_deal is not None:
            self.weight_of_deal = sum([self.game_instance['item_weights'][item] for item in self.final_deal])
        else:
            self.weight_of_deal = None

        # log a message informing that the game was successfully played:
        if not self.aborted and not self.lose:
            action = {'type': 'info', 'content': 'game successful'}
            self.log_event(from_='GM', to='GM', action=action)
        # log a final message saying that the game did come to an end:
        action = {'type': 'info', 'content': 'end game'}
        self.log_event(from_='GM', to='GM', action=action)

        # Convert final deal and proposal history into JSON-safe format
        final_deal_serializable = self.make_json_serializable(self.final_deal)

        self.log_key('final deal', final_deal_serializable)
        
        # Track and calculate player's unique violated request count
        total_violations = self.violated_request_count
        if total_violations > 0:
            percent_player1 = (self.violated_requests_per_player['player_1'] / total_violations) * 100
            percent_player2 = (self.violated_requests_per_player['player_2'] / total_violations) * 100
        else:
            percent_player1 = percent_player2 = 0.0

        self.compute_utility_score()
        self.normalize_scores()

        self.log_key("Total weight deal", self.weight_of_deal)
        self.log_key("Max harmonic mean", self.max_harmonic)
        self.log_key("Max weight allowed", self.game_instance['max_weight'])    
        self.log_key("Is Pareto Optimum Deal", self.is_final_deal_pareto_optimum()) #boolean output
        self.log_key("Pareto Optima Count", len(self.pareto_optima_sets))

        self.log_key("Normalized Utility Player 1", self.normalized_u1)
        self.log_key("Normalized Utility Player 2", self.normalized_u2)
        self.log_key("Normalized harmonic mean", self.normalized_harmonic_mean) #Fairness & efficiency of final deal
        
        # Log the clemcore metrics:
        self.log_key(METRIC_ABORTED, int(self.aborted))
        self.log_key(METRIC_LOSE, int(self.lose))
        self.log_key(METRIC_SUCCESS, int(self.deal_finalized))
        self.log_key(METRIC_REQUEST_COUNT, int(self.request_count))
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, int(self.violated_request_count))
        # Custom metrics
        self.log_key("Played turns", self.current_turn)
        self.log_key("Percent Violations Player 1", percent_player1)
        self.log_key("Percent Violations Player 2", percent_player2)
        # Add global values for aggregation
        self.log_key("Violations Player 1", self.violated_requests_per_player['player_1'])
        self.log_key("Violations Player 2", self.violated_requests_per_player['player_2'])
        self.log_key("Total Violations", total_violations)

class SomeGameScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def score_turns(self, episode_interactions: Dict) -> None:
        
        for turn_idx, turn in enumerate(episode_interactions.get("turns", [])):
            valid_turn = True
            for event in turn:
                if event.get("action", {}).get("type") in {"missing tag", "rule violation"}:
                    valid_turn = False
                    break
        
        self.log_turn_score(turn_idx, "Valid Turn", 1 if valid_turn else 0)

    def compute_episode_scores(self, episode_interactions: Dict):

        # if negotiation mode is 'equal' then Normalized Utility Player 1/2 are the same
        # and hence the same as the harmonic mean
        if self.game_instance['negotiation_mode'] == 'equal':
            score = episode_interactions['Normalized Utility Player 1']
        else:
            score = episode_interactions['Normalized harmonic mean']
        norm_u1 = episode_interactions["Normalized Utility Player 1"]
        norm_u2 = episode_interactions["Normalized Utility Player 2"]
        
        # Secondary metrics
        violations_p1 = episode_interactions["Violations Player 1"]
        violations_p2 = episode_interactions["Violations Player 2"]
        total_violations = episode_interactions["Total Violations"]

        # Logging
        self.log_episode_score("Violations Player 1", violations_p1)
        self.log_episode_score("Violations Player 2", violations_p2)
        self.log_episode_score("Total Violations", total_violations)

        # Log them for aggregation
        self.log_episode_score("Normalized Utility Player 1", norm_u1)
        self.log_episode_score("Normalized Utility Player 2", norm_u2)
        self.log_episode_score("Harmonic Mean Score", score)

        if episode_interactions[METRIC_SUCCESS]:
            self.log_episode_score(BENCH_SCORE, score)
        elif episode_interactions[METRIC_LOSE]:
            self.log_episode_score(BENCH_SCORE, 0)
        elif episode_interactions[METRIC_ABORTED]:
            self.log_episode_score(BENCH_SCORE, np.nan)
        else:
            raise ValueError("Missing outcome value in interactions.json")

class SomeGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment, player_models) -> GameMaster:
        return HotAirBalloon(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return SomeGameScorer(self.game_name, experiment, game_instance)