import os
import ast
import re
from pathlib import Path
import copy
from copy import deepcopy
from typing import List, Dict, Tuple
import numpy as np
import json
from html import escape
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, METRIC_SUCCESS, BENCH_SCORE
from clemcore.clemgame import DialogueGameMaster, GameBenchmark, GameScorer, GameSpec, Player, GameError, ParseError
from clemcore.backends import Model


from players import InstructionGiver, InstructionFollower
from utils.prepareasciirep import PrepareASCIIRep
from utils.preparellmsandbox import PrepareLLMSandBox


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CCBTSCollabMaster(DialogueGameMaster):
    def __init__(
        self,
        game_spec: GameSpec,
        experiment: Dict,
        player_models: List[str]
    ):
        super().__init__(game_spec, experiment, player_models)
        # save experiment and player attributes that will be necessary later
        logger.info(f"Initializing ImageCCBTSMaster: model_a = {player_models[0]} ")
        self.model_a = player_models[0]
        self.model_b = player_models[1] if len(player_models) > 1 else player_models[0]

    def _on_setup(self, **game_instance) -> None:
        """Setup the episode (mandatory)."""

        #logging.disable(logging.CRITICAL)
        self.instancedata = game_instance["data"]
        self.game_id = game_instance["game_id"]

        data = self.instancedata
        self.use_dspy_collab = data["use_dspy_collab"]
        self.use_dspy_collab_history = data["use_dspy_collab_history"]
        self.use_dspy_collab_retry = data["use_dspy_collab_retry"]
        self.num_dspy_collab_retry = data["num_dspy_collab_retry"]
        self.num_collab_retry = data["num_collab_retry"]
        self.use_sandbox_llm = data["use_sandbox_llm"]
        self.use_error_feedback = data["use_error_feedback"]
        self.num_collab_optim_turns = data["num_collab_optim_turns"]
        self.n_turns = data["n_turns"]


        self.prompts_dict = data["prompts_dict"]
        self.prompt_player_a_base = self.prompts_dict["prompt_a"]
        self.prompt_player_b_base = self.prompts_dict["prompt_b"]
        self.prompt_player_b_optim = self.prompts_dict["prompt_b_optim"]
        self.turn_prompt_a = self.prompts_dict.get("turn_prompt_a", "")
        self.turn_prompt_a_newboard = self.prompts_dict.get("turn_prompt_a_newboard", "")        
        self.turn_prompt_b = self.prompts_dict.get("turn_prompt_b", "")
        self.turn_prompt_b_optim = self.prompts_dict.get("turn_prompt_b_optim", "")

        self.current_collab_retry = 0


        # initialise game variables:
        self.current_turn: int = 0

        self.log_key('n_turns', self.n_turns)
        self.turn_scores = [0] * (self.n_turns)

        # initialise attributes that will be used for the evaluation scores
        self.aborted: bool = False
        self.lose: bool = False
        self.success: bool = False
        self.complete_turns: int = 0
        self.reconstruct_success: bool = False
        self.used_clarification: bool = False
        self.num_clarifications: int = 0
        self.used_acknowledgement: bool = False
        self.num_acknowledgements: int = 0


        self.used_clear: bool = False
        self.num_clears: int = 0
        self.used_undo: bool = False
        self.num_undos: int = 0
        self.used_remove: bool = False
        self.num_removes: int = 0
        self.used_move: bool = False
        self.num_moves: int = 0

        self.current_reconst_retry: int = 0
        self.used_retry_reconst: bool = False
        self.current_collab_optim: int = 0
        self.optim_progress: bool = False
        self.regboard_probe: bool = False

        # initialise common metrics:
        self.request_count: int = 0
        self.parsed_request_count: int = 0
        self.violated_request_count: int = 0 

        self.set_pass_turn = True

        self.prepare_ascii_rep = PrepareASCIIRep()
        if self.use_sandbox_llm:
            self.prepare_sandbox = PrepareLLMSandBox(config=data["sandbox_llm"])

        self.simpleboards = data["simple_boards"]
        self.regularboards = data["regular_boards"]
        self.instance_metadata = data["metadata"]
        self.use_regular_boards = data["use_regular"]
        self.use_simple_reuse = data["use_simple_reuse"]
        self.use_regular_challenging = data["use_regular_challenging"]

        self.boardgen = self.board_generator(self.simpleboards, self.regularboards)
        self.board_info = None
        self.gtcode = None
        self.gt_usage = None

        # instantiate players:
        self.player_a = InstructionGiver(self.model_a, "A", "user_simulator", self.use_dspy_collab, self.use_dspy_collab_history)
        self.player_a_type = self.player_a.get_player_type()
        # add players, including assigning their initial prompts:
        if self.player_a_type == "human":
            self.prompt_player_a = self.prompt_player_a_human
        elif self.player_a_type == "programmatic":
            self.prompt_player_a = self.prompt_player_a_base
        else:
            if self.use_dspy_collab:
                self.prompt_player_a = self.player_a.get_player_prompt()
            else:
                self.prompt_player_a = self.prompt_player_a_base
        self.add_player(self.player_a, initial_context=self.prompt_player_a)

        # add player B, the one that follows the instructions:
        self.player_b = InstructionFollower(self.model_b, "B", "cobot", self.use_dspy_collab, self.use_dspy_collab_history)
        self.player_b_type = self.player_b.get_player_type()
        if self.player_b_type == "human":
            self.prompt_player_b = self.prompt_player_b_human
        elif self.player_b_type == "programmatic":
            self.prompt_player_b = self.prompt_player_b_base
        else:
            if self.use_dspy_collab:
                self.prompt_player_b = self.player_b.get_player_prompt()
            else:
                self.prompt_player_b = self.prompt_player_b_base
        self.add_player(self.player_b, initial_prompt=self.prompt_player_b)


        self.gamedata = {
            "genresponse": None,
            "n_turns": self.n_turns,
            "play_turns": None,
            "loss_reason": None,
            "reconstruction_status": self.reconstruct_success,
            "used_clarification": self.used_clarification,
            "num_clarifications": self.num_clarifications,
            "used_clear": self.used_clear,
            "num_clears": self.num_clears,
            "used_undo": self.used_undo,
            "num_undos": self.num_undos,
            "used_remove": self.used_remove,
            "num_removes": self.num_removes,
            "used_move": self.used_move,
            "num_moves": self.num_moves,
            "use_dspy_collab": self.use_dspy_collab,
            "use_dspy_collab_history": self.use_dspy_collab_history,
            "max_collab_retry": self.num_collab_retry,
            "total_reconst_retry": self.current_reconst_retry,
            "used_retry_reconst": self.used_retry_reconst,
            "inst_code_pairs": None,
        }
        self.genresponse = []#{"instructions": [], "code": []}
        self.genboard = None
        self.gen_board_cells = None
        self.avail_skills = {}
        self.reuse_skills = []
        self.current_variant = None



    def _prepapre_boardinfo(self, variant: str) -> None:
        try: 
            self.board_info = next(self.boardgen)
        except StopIteration:
            self.board_info = None

        if self.board_info is None:
            logger.info("No more boards available from the generator.")
            return False

        if 'rows' in self.board_info and 'cols' in self.board_info:
            self.board_info["size"] = {"rows": self.board_info['rows'], "cols": self.board_info['cols']}
        else:
            self.board_info["size"] = {"rows": 8, "cols": 8}

        self.genboard = None
        self.reuse_skills = []
        
        if variant == "simple":
            self.gtcode = {"function": self.board_info["code"]["single_turn"]["function"],
                             "usage": self.board_info["code"]["single_turn"]["usage"]}
            self.gt_usage = self.board_info["code"]["single_turn"]["usage"]

            ascii_rep_board, board_rep = self.prepare_ascii_rep.get_ascii_representation(self.gtcode, self.board_info["size"])
            self.player_a_goal = ascii_rep_board #Layer-wise representation        
            self.board_info["locations"] = {"row": self.board_info["x"][0]+1, "col": self.board_info["y"][0]+1}    

        elif variant == "regular":
            self.gtcode = {"function": self.board_info["code"]["function"],
                             "usage": self.board_info["code"]["output"]}
            self.gt_usage = self.board_info["code"]["output"] #This would be some loop, not really usage
            self.board_info["locations"] = None

            self.regboard_probe = True

            repeat_locations = self.board_info["repeat_locations"]
            target_board_rep, target_board, target_board_cells, target_board_cells_repeat = self.prepare_ascii_rep.get_ascii_representation_rb(self.board_info["size"], self.gtcode, self.board_info["combo_name"], self.board_info["colors"], repeat_locations)

            self.player_a_goal = target_board_rep
        self.board_info["funcusage"] = self.gt_usage
        return True


    def _on_before_game(self) -> None:
        """Initialise the dialogue history (firstlast specific)."""

        self.current_variant = "simple"
        self._prepapre_boardinfo("simple")

        p1_data = f"grid_size: 8x8\nskill name: {self.board_info['combo_name']}\ncolors: {self.board_info['colors']}\nlocation: {self.board_info['locations']}\ntarget_grid:{self.player_a_goal}\ndifference_grid: None\nclarification: None\nacknowledgement: None"
        if self.player_a_type == "human" and self.use_diff_human_prompts:
            #Add GT image filename in html format
            safe_text = escape(self.prompt_player_a+"\n"+str(p1_data)+"\n\nReference Images are given below:\n\n")
            data_uri_shapes = self.shapes_references_base64
            data_uri_1 = f"data:image/png;base64,{self.gt_image_base64}"
            data_uri_2 = f"data:image/png;base64,{self.empty_player_board_base64}"
            #p1_messages = f"""<div>{safe_text}</div><div style="display:flex; gap:8px; align-items:center;"><img src="{data_uri_1}" #width="200" height="200" /> <img src="{data_uri_2}" width="200" height="200" /></div> """
            p1_messages = f"""<div>{safe_text}</div><img src="{data_uri_shapes}" width="400" height="60"/><div style="display:flex; gap:8px; align-items:flex-start;"><figure style="text-align:center;"><figcaption style="font-size:14px; margin-bottom:4px;">Goal Grid</figcaption><img src="{data_uri_1}" width="350" height="350" /></figure><figure style="text-align:center;"><figcaption style="font-size:14px; margin-bottom:4px;">Current Player Grid (Empty)</figcaption><img src="{data_uri_2}" width="350" height="350" /></figure></div>"""
        else:
            p1_messages = self.prompt_player_a+"\n"+str(p1_data)



        if self.use_dspy_collab:
            action = {'type': 'send message', 'content': p1_messages}
            self.log_event(from_='GM', to='Player 1', action=action)


        self.set_context_for(self.player_a, p1_messages)
        gt_cells, _ = self.prepare_ascii_rep.get_ascii_representation(self.gtcode, self.board_info["size"])
        logger.info(f"Ground truth cells:\n{gt_cells}")
        self.gt_occupied_cells = self.prepare_ascii_rep.get_occupied_cells(self.gtcode, self.board_info["size"])
        logger.info(f"Ground truth occupied cells:\n{self.gt_occupied_cells}")

    def _prepare_next_board_after_optim(self):
        logger.info("Preparing next board after optimization attempts.")
        self.current_variant = "regular"
        boardstatus = self._prepapre_boardinfo("regular")

        if not boardstatus:
            return "COMPLETED", None

        p1_data = f"grid_size: 8x8\nskill_required: {self.board_info['combo_name']}\ncolors: {self.board_info['colors']}\nlocation: {self.board_info['locations']}\ntarget_grid:{self.player_a_goal}\ndifference_grid: None\nclarification: None\nacknowledgement: None"


        p1_prompt = self.turn_prompt_a_newboard + "\n" + str(p1_data)
        return "SUCCESS", p1_prompt


    def board_generator(self, simple_boards, regular_boards):
        for outer_key, inner_dict in simple_boards.items():
            yield from inner_dict 

        for outer_key, inner_dict in regular_boards.items():
            yield from inner_dict 


        #yield from simple_boards
        #yield from regular_boards


    def _set_pass_turn(self, player: Player, pass_turn) -> None:
        #Set the turn to be passed for a player (game specific).
        logger.info(f"Setting pass turn for player: {player}")
        self.set_pass_turn = pass_turn

    def _should_pass_turn(self):
        #Currently not checking for any condition to pass the turn
        logger.info(f"Checking if turn should be passed: {self.set_pass_turn}")
        return self.set_pass_turn   

    def _extract_reuse_func_data(self, response: str):
        pattern = re.compile(
            r"""
            ^\s*
            (?P<func_name>\w+)                  # function name
            \s*\(
            \s*board\s*,                        # first arg: board
            \s*(?:colors\s*=\s*)?               # optional 'colors='
            (?P<func_colors>                    # list OR tuple of colors
                \[[^\]]*\]                      #   [...] form
                |                               #   OR
                \([^)]*\)                       #   (...) form
            )
            \s*,\s*
            (?:
                # --- case 1: keyword x-like then keyword y-like ---
                (?:
                    (?:x|r|row)\s*=\s*(?P<x_kw1>\d+)
                    \s*,\s*
                    (?:y|c|col)\s*=\s*(?P<y_kw1>\d+)
                )
                |
                # --- case 2: keyword y-like then keyword x-like ---
                (?:
                    (?:y|c|col)\s*=\s*(?P<y_kw2>\d+)
                    \s*,\s*
                    (?:x|r|row)\s*=\s*(?P<x_kw2>\d+)
                )
                |
                # --- case 3: pure positional ints ---
                (?P<x_pos>\d+)\s*,\s*(?P<y_pos>\d+)
            )
            \s*\)
            \s*$
            """,
            re.VERBOSE,
        )     

        response_list = response.splitlines()
        logger.info(f"Extracting reuse function data from response lines: {response_list}")

        func_name_list = []
        func_colors_list = []
        x_list = []
        y_list = []

        for resp in response_list:
            m = pattern.search(resp)
            if not m:
                logger.info(f"No match found in response line: {resp}")
                return None, None, None, None

            g = m.groupdict()

            func_name   = g["func_name"]
            func_colors = ast.literal_eval(g["func_colors"])

            # normalize x
            x = g["x_kw1"] or g["x_kw2"] or g["x_pos"]
            # normalize y
            y = g["y_kw1"] or g["y_kw2"] or g["y_pos"]

            func_name_list.append(func_name)
            func_colors_list.append(func_colors)
            x_list.append(x)
            y_list.append(y)
        return func_name_list, func_colors_list, x_list, y_list

    def _compare_reuse_function_data(self, func_name_list, func_colors_list, func_x_list, func_y_list):

        for func_name, func_colors, func_x, func_y in zip(func_name_list, func_colors_list, func_x_list, func_y_list):
            expected_combo_name = self.board_info["combo_name"]
            if expected_combo_name != func_name:
                error = f"Function name '{func_name}' does not match expected combo name '{expected_combo_name}'."
                return None, error
            
            expected_colors = self.board_info["colors"]
            current_colors = func_colors
            if expected_colors != current_colors:
                error = f"Function colors '{current_colors}' do not match expected colors '{expected_colors}'."
                return None, error
            
            expected_location = self.board_info["repeat_locations"]
            current_x = int(func_x)
            current_y = int(func_y)
            if [current_x, current_y] not in expected_location:
                error = f"Function location '[{current_x}, {current_y}]' is not in expected locations '{expected_location}'."
                return None, error
            
            self.reuse_skills.append({"name": func_name, "colors": current_colors, "x": current_x, "y": current_y})
        return True, None
        


    def _does_game_proceed(self) -> None:
        """Check if the game loop should continue (game specific)."""
        logger.info(f"Inside _does_game_proceed: Current Round: {self.current_round}, aborted: {self.aborted}, lose: {self.lose}, success: {self.success}")
        # Determine if the game should proceed. This is also called once initially.
        #if self.current_round < self.n_turns and not self.aborted and not self.lose and not self.success:
        if self.current_round < self.n_turns and not self.aborted and not self.lose and not self.success:        
            logger.info(f"Game continues: {self.current_round} < {self.n_turns}")
            return True

        if self.success:
            action_type = "info"
            action_content = "The game is successful;"

        else:
            if not self.aborted:# and self.current_round == self.n_turns:
                self.lose = True

        logger.info(f"Game status: {self.success}, {self.lose}, {self.aborted}")

        if self.lose:
            action_type = "info"
            action_content = "Maximum turns reached; lost game"
        elif self.aborted:
            action_type = "invalid format"
            action_content = "The game has been aborted."

        self.log_to_self(action_type, action_content)
        #self._log_game_end()
        return False   

    def _advance_game(self, player: Player, parsed_response: Dict):
        """Advance the game with the parsed response."""
        logger.info(f"Advancing game with parsed response: player = {player}")
        if player == self.player_a:
            # The validitiy of the generated code can be checked during scoring.
            # If there are no issues in format, the game is considered successful.
            self._set_pass_turn(self.player_a, True)
            if parsed_response.get("status") == "success":
                self.success = True

            elif parsed_response.get("status") == "failure":
                action_type = "info"
                action_content = "Failure in user simulator response."

                self.log_to_self(action_type, action_content)                
                self.aborted = True
                # set the reason for the loss:
                self.gamedata["loss_reason"] = parsed_response.get("error", "Unknown error")

            elif parsed_response.get("status") == "next_rb_board":
                logger.info("Preparing next regular board after reuse attempts.")
                self._set_pass_turn(self.player_a, False)


        elif player == self.player_b:
            if parsed_response["status"] == "failure":
                # Time to reprobe the player with error feedback
                logger.info(f"Player B response validation failed with error: {parsed_response['error']}")
                # No need to check if retry is allowed here; it is handled in _set_violated_req_count() which triggers parseerror
                if self.optim_progress:
                    if self.current_collab_optim < self.num_collab_optim_turns:
                        self._set_pass_turn(self.player_b, False)
                    else:
                        # Optimization attempts exhausted
                        # Go ahead with processing next board
                        self._set_pass_turn(self.player_b, True)
                        # increment current turn:
                        self.current_turn += 1
                        # increment complete turns:
                        self.complete_turns += 1
                        #self.gencode = parsed_response
                        # set the current turn's score to 1:
                        self.turn_scores[self.current_turn-1] = 1
                else:
                    self._set_pass_turn(self.player_b, False)
            else:
                self._set_pass_turn(self.player_b, True)
                self.correct_response = True
                # increment current turn:
                self.current_turn += 1
                # increment complete turns:
                self.complete_turns += 1
                #self.gencode = parsed_response
                # set the current turn's score to 1:
                self.turn_scores[self.current_turn-1] = 1


    def _model_response_cleanup(self, response: str) -> str:
        clean_response = re.sub(r'```json(.*?)```', r'\1', response, flags=re.DOTALL).strip()
        clean_response = re.sub(r'```(.*?)```', r'\1', clean_response, flags=re.DOTALL).strip()
        clean_response = re.sub(r'```', '', clean_response).strip()
        #Remove [[ ## instruction ## ]] and [[ ## completed ## ]] from the response
        clean_response = re.sub(r'\[\[\s*##\s*instruction\s*##\s*\]\]', '', clean_response, flags=re.IGNORECASE).strip()
        clean_response = re.sub(r'\[\[\s*##\s*player_response\s*##\s*\]\]', '', clean_response, flags=re.IGNORECASE).strip()
        clean_response = re.sub(r'\[\[\s*##\s*optimized_function\s*##\s*\]\]', '', clean_response, flags=re.IGNORECASE).strip()        
        clean_response = re.sub(r'\[\[\s*##\s*completed\s*##\s*\]\]', '', clean_response, flags=re.IGNORECASE).strip()
        #Remove brackets like[[ or ]]
        clean_response = re.sub(r'(?m)^\s*\[\[\s*$','', clean_response).strip()
        clean_response = re.sub(r'(?m)^\s*\]\]\s*$','', clean_response).strip()
        return clean_response


    def _set_parsed_req_count(self) -> None:
        # increase the counter of requests that conform to form rules
        self.parsed_request_count += 1  

        # log the event that the string was valid (no strange characters)
        action = {'type': 'valid response', 'content': 'response conforms to rules'}
        self.log_event(from_='GM', to='GM', action=action)


    def _on_after_game(self) -> None:
        """Executed once at the end, after exiting the play loop."""
        #Do the game validation
        # log a final message saying that the game did come to an end:
        #action = {'type': 'info', 'content': 'end game'}
        #self.log_event(from_='GM', to='GM', action=action)
        self.gamedata["genresponse"] = self.genresponse
        #TODO: Check what to log here
        #self.gamedata["genboard"] = self.genboard
        self.gamedata["play_turns"] = self.current_round
        if self.use_sandbox_llm:
            self.prepare_sandbox.close()
        #self._save_instruction_code_pairs()
        self._log_game_end()  


    def _prepare_playerb_turn_response(self, gen_image_base64, difference_grid, clarification, reconstruction_status, cfq):

        if cfq:
            clarification_text = clarification
            ack_text = "None"
        else:
            clarification_text = "None"
            ack_text = clarification


        if clarification:
            if self.player_a_type == "human":
                clarification_text = f'<span style="color: red; font-size: 20px;">{clarification}</span>'


        clarification_header = "Clarification"
        ack_header = "Acknowledgement"
        p2_data = f"\ntarget_grid:{self.player_a_goal}\nDifference_grid: {difference_grid}\n{clarification_header}: {clarification_text}\n{ack_header}: {ack_text}\nAre the target grid and player grid equal?\nReconstruction Status: {reconstruction_status}"

        if self.player_a_type == "human":
            #Add GT image filename in html format
            if clarification:
                safe_text = p2_data+"\n\nReference Images are given below:\n\n"                
            else:
                safe_text = escape(str(p2_data)+"\n\nReference Images are given below:\n\n")
            data_uri_shapes = self.shapes_references_base64
            data_uri_1 = f"data:image/png;base64,{self.gt_image_base64}"
            if gen_image_base64:
                data_uri_2 = f"data:image/png;base64,{gen_image_base64}"
            else:
                data_uri_2 = f"data:image/png;base64,{self.empty_player_board_base64}"
            #p1_messages = f"""<div>{safe_text}</div><div style="display:flex; gap:8px; align-items:center;"><img src="{data_uri_1}" #width="200" height="200" /> <img src="{data_uri_2}" width="200" height="200" /></div> """
            p2_message = f"""<div>{safe_text}</div><img src="{data_uri_shapes}" width="400" height="60"/><div style="display:flex; gap:8px; align-items:flex-start;"><figure style="text-align:center;"><figcaption style="font-size:14px; margin-bottom:4px;">Goal Grid</figcaption><img src="{data_uri_1}" width="350" height="350" /></figure><figure style="text-align:center;"><figcaption style="font-size:14px; margin-bottom:4px;">Current Player Grid (Empty)</figcaption><img src="{data_uri_2}" width="350" height="350" /></figure></div>"""
        else:
            p2_message = p2_data

        return p2_message
    
    def _get_playerb_grid(self, variant):
        if self.genboard is None:
            return "None"

        #_, gen_occupied_cells = self.prepare_ascii_rep.get_ascii_representation_from_board_layers(self.genboard, self.board_info["size"])        
        if variant == "simple":
            _, gen_occupied_cells = self.prepare_ascii_rep.get_ascii_representation_from_board_layers(self.genboard, self.board_info["size"])
            diff_grid = self.prepare_ascii_rep.get_layer_representation_diff(self.gt_occupied_cells, gen_occupied_cells)

        else:
            func_name = self.reuse_skills[-1]['name'] if self.reuse_skills else ""
            func_colors = self.reuse_skills[-1]['colors'] if self.reuse_skills else []
            repeat_locations = [[skill['x'], skill['y']] for skill in self.reuse_skills]

            logger.info(f"Generating occupied repeats for difference grid calculation.repeat_locations: {repeat_locations}")
            logger.info(f"GT Repeat locations: {self.board_info['repeat_locations']}")
            #gen_occupied_repeats = self.prepare_ascii_rep._list_occupied_cells_with_repeats(func_name, func_colors, repeat_locations)

            diff_grid = self.prepare_ascii_rep.get_layer_representation_diff_rb(self.board_info['repeat_locations'], repeat_locations)
        return diff_grid    
    
    def _prepare_playerb_clarification_response(self, variant: str, details: str, cfq: bool) -> str:
        gen_image_base64 = None#self.genresponse[-1]["gen_image_base64"] if self.genresponse and len(self.genresponse) > 0 and "gen_image_base64" in self.genresponse[-1] else None
        diff_grid = self._get_playerb_grid(variant)
        grid_match = self._validate_game(self.genboard)
        reconstruction_complete = "True" if grid_match["status"] == "success" else "False"

        return self._prepare_playerb_turn_response(gen_image_base64, diff_grid, details, reconstruction_complete, cfq)
    
    def _prepare_playerb_code_response(self, variant: str, details: str) -> str:
        logger.info(f"Preparing player B code response for execution: {details}")
        if self.genboard is None:
            board_gen_call = None
        else:
            board_gen_call = copy.deepcopy(self.genboard)
        try:
            #board_gen_call = self.prepare_ascii_rep.execute_generated_response_skill(
            #    details, self.board_info["size"], board_gen_call, self.gtcode["function"]
            #)
            if not self.use_sandbox_llm:
                if variant == "simple":
                    board_gen_call, error, code_stats = self.prepare_ascii_rep.execute_generated_response(details, self.board_info["size"], board_gen_call)
                else:
                    board_gen_call, error, code_stats = self.prepare_ascii_rep.execute_generated_response_skill(details, self.board_info["size"], board_gen_call, self.gtcode["function"])

            else:
                logger.info("Calling sandbox to run the generated code.")
                result = self.prepare_sandbox.run_code(details, board_gen_call, self.board_info["size"]["rows"], self.board_info["size"]["cols"])
                logger.info(f"Generated board state after executing response: {result['error']}")

                error = result["error"]# result["stderror"]
                code_stats = result["code_stats"]
                board_gen_call = result["board"]                

            if error:
                logger.error("Error executing generated response.")
                return None, error

            if code_stats:
                self.gamedata["used_move"] = True if code_stats.get("move", 0) else False
                self.gamedata["num_moves"] += code_stats.get("move", 0)
                self.gamedata["used_remove"] = True if code_stats.get("remove", 0) else False
                self.gamedata["num_removes"] += code_stats.get("remove", 0)
                self.gamedata["used_undo"] = True if code_stats.get("undo", 0) else False
                self.gamedata["num_undos"] += code_stats.get("undo", 0)
                self.gamedata["used_clear"] = True if code_stats.get("clear", 0) else False
                self.gamedata["num_clears"] += code_stats.get("clear", 0)


            if variant == "regular":
                func_name_list, func_colors_list, func_x_list, func_y_list = self._extract_reuse_func_data(details)
                status, error = self._compare_reuse_function_data(func_name_list, func_colors_list, func_x_list, func_y_list)
                if not status:
                    logger.error(f"Error in reuse function data extraction: {error}")
                    return None, error
                


            self.genboard = copy.deepcopy(board_gen_call)

            if variant == "simple":
                logger.info("Calling get_ascii_representation to generate ASCII representation")
                gen_ascii_rep, gen_occupied_cells = self.prepare_ascii_rep.get_ascii_representation_from_board_layers(board_gen_call, self.board_info["size"])

                logger.info(f"Generated ASCII representation:\n{gen_ascii_rep}")
                if gen_ascii_rep is None:
                    if gen_occupied_cells is None:
                        logger.error("Error generating ASCII representation from generated response.")
                        return None, "Error generating ASCII representation from generated response."
            
                #self.prepare_ascii_rep.set_occupied_cells(gen_occupied_cells)
                #gen_board_image_filename = f"turn_{self.current_turn+1}_playerb_board.png"
                #gen_image_base64 = None#self.prepare_ascii_rep.get_image_gen_board(self.genboard, gen_board_image_filename)
            else:
                gen_ascii_rep, _, gen_occupied_cells, gen_occupied_repeats = self.prepare_ascii_rep.get_ascii_representation_from_combo_names(board_gen_call, self.board_info["size"], self.reuse_skills, self.gtcode["function"])

            if self.regboard_probe:
                self.genresponse[-1]["regular_rep"] = gen_ascii_rep
            else:
                self.genresponse[-1]["ascii_rep"] = gen_ascii_rep
            self.genresponse[-1]["occupied_cells"] = gen_occupied_cells
            self.genresponse[-1]["code_stats"] = code_stats
            self.genresponse[-1]["gen_image_base64"] = None#gen_image_base64
            logger.info(f"Generated occupied cells:\n{gen_occupied_cells}")
            logger.info(f"Ground truth occupied cells:\n{self.gt_occupied_cells}")
            logger.info("Calling diff grid")

            if variant == "simple":
                diff_grid = self.prepare_ascii_rep.get_layer_representation_diff(self.gt_occupied_cells, gen_occupied_cells)
            else:
                repeat_locations = [[skill['x'], skill['y']] for skill in self.reuse_skills]
                diff_grid = self.prepare_ascii_rep.get_layer_representation_diff_rb(self.board_info['repeat_locations'], repeat_locations)
            logger.info(f"Difference grid representation:\n{diff_grid}")

            grid_equality = self._validate_game(self.genboard)
            reconstruction_complete = "True" if grid_equality["status"] == "success" else "False"

            p2_response = self._prepare_playerb_turn_response(self.genresponse[-1]["gen_image_base64"], diff_grid, None, reconstruction_complete,False)

            return p2_response, None

        except Exception as e:
            logger.error(f"Error executing generated response: {e}")
            return None, "Error executing generated response from player B."


    def _validate_playerb_response(self, model_response: str) -> bool:
        parse_b, error = None, None
        logger.info(f"Validating response from player B: {model_response}")
        if self.optim_progress:
            #Optimization response is only string, not a json
            parse_b = {"status": "code", "details": model_response}
            return parse_b, error

        try:
            parse_b = json.loads(model_response)
            if "status" not in parse_b or "details" not in parse_b:
                error = f"Missing 'status' or 'details' in response ({parse_b}) from player B."
            
            if parse_b["status"] not in ["code", "clarification", "acknowledgement"]:
                logger.error(f"Invalid status in response from player B: {parse_b['status']}")
                error = f"Invalid status: {parse_b['status']} (not 'code' or 'clarification' or 'acknowledgement') in response from player B."

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error while parsing response from player B: {e}")
            error = e
       
        return parse_b, error

    def _handle_playera_response(self, response: str) -> str:
        return self._model_response_cleanup(response)

    def _handle_playerb_response(self, response: str) -> Tuple[Dict, str]:
        playerb_response = self._model_response_cleanup(response)
        #logger.debug(f"Cleaned response from player B: {type(model_response)}\n{model_response}")            

        return self._validate_playerb_response(playerb_response)     

    def _prepare_instruction_code_pairs(self):
        if self.genresponse is None or len(self.genresponse) == 0:
            logger.info("No generated responses to prepare instruction-code pairs.")
            return None

        inst_code_pairs = []
        for turn in self.genresponse:
            if "instruction" in turn and "response" in turn and "details" in turn["response"]:
                inst_code_pairs.append({"instruction": turn["instruction"], "status": turn["response"]["status"], "details": turn["response"]["details"]})

        logger.info(f"Instruction Code Pairs are: {inst_code_pairs}")
        if len(inst_code_pairs) == 0:
            logger.info("No valid instruction-code pairs found.")
            return None

        return inst_code_pairs

    def _prepare_function_signature_usage(self) -> str:
        return f"Function signature: def {self.board_info['combo_name']}(board, colors, x, y)\nFunction Usage: {self.board_info['funcusage']}"


    def _prepare_optimizer(self):
        logger.info("Starting code optimization phase.")
        self.current_collab_optim = 0

        self.targetboard_cells = self.prepare_ascii_rep.get_occupied_cells_from_board(self.genboard)

        inst_code_pairs = self._prepare_instruction_code_pairs()
        if inst_code_pairs is None:
            logger.info("No instruction-code pairs to save.")
            return

        self.gamedata["inst_code_pairs"] = inst_code_pairs
        self.optim_progress = True

        p2_prompt = self.prompt_player_b_optim + self._prepare_function_signature_usage() + "\n" + f"Instruction-Code Snippets: {json.dumps(inst_code_pairs)}\nError Feedback: None"

        action = {'type': 'info',
                    'content': "Starting code optimization phase."}
        self.log_event(from_='GM', to='GM', action=action)        

        return p2_prompt


    def _on_parse_error(self, error: ParseError):
        """Abort the game due to failed parsing."""
        logger.error(f"Parse error: {error}")
        # set the game to be aborted:
        self.aborted = True
        # increase the counter of requests that violate the move format rule:
        self.violated_request_count += 1
        # log the abortion event:
        action = {'type': 'invalid format', 'content': 'abort'}
        self.log_event(from_='GM', to='GM', action=action)              
        
    def _on_game_error(self, error: GameError):
        """Lose the game due to violated rules."""
        self.lose = True
        # log the fact that the game is now lost:
        action = {'type': 'rule violation',
                  'content': error.reason}
        self.log_event(from_='GM', to='GM', action=action)    

    def _set_violated_req_count(self, error) -> None:
        # increase the counter of requests that violate the move format rule
        self.violated_request_count += 1

        # log the event that the string was invalid (strange characters)
        # We are logging the error in ParseError, so no need to log it here
        #action = {'type': 'invalid format', 'content': f'response does not conform to rules. {error}'}
        #self.log_event(from_='GM', to='GM', action=action)

        retry = False
        if error:
            if self.optim_progress:
                if self.current_collab_optim < self.num_collab_optim_turns:
                    logger.info(f"Response did not conform to rules during optimization. Reprobing the player for optimization. Current optimization turn: {self.current_collab_optim+1}")
                    self.log_event(from_='GM', to='GM', action={'type': 'info', 'content': 'Response did not conform to rules during optimization. Reprobing the player for optimization.'})
                    retry = True
            else:
                if self.num_collab_retry and self.current_collab_retry < self.num_collab_retry:
                    logger.info(f"Response did not conform to rules. Reprobing the player. Current retry: {self.current_collab_retry+1}")
                    self.log_event(from_='GM', to='GM', action={'type': 'info', 'content': 'Response did not conform to rules. Reprobing the player.'})
                    self.current_collab_retry += 1
                    self.used_retry_collab = True
                    self.gamedata["used_retry_collab"] = self.used_retry_collab
                    retry = True
                else:
                    logger.error(f"Response did not conform to rules. Reprobing tries exceeded.")
                    self.log_event(from_='GM', to='GM', action={'type': 'info', 'content': 'Response did not conform to rules. Reprobing tries exceeded.'})
                    # Not checking for optimization flow here as it is handled separately
        if not retry:
            raise ParseError(error)    


    def _check_function_def(self, optim_func: str):
        """Check if the optimized function has the correct definition."""
        pattern = rf'def\s+{self.board_info["combo_name"]}\s*\(\s*board\s*,\s*colors\s*,\s*x\s*,\s*y\s*\)\s*:'
        if not re.search(pattern, optim_func):
            optim_func_error = f"Optimized function does not have the correct definition. Expected 'def {self.board_info['combo_name']}(board, colors, x, y):'"
            return False, optim_func_error
        return True, None


    def _validate_optimized_code(self, optim_func):
        correct_def, optim_error = self._check_function_def(optim_func)
        if not correct_def:
            logger.error("Optimized function definition is incorrect. Retry")
            optim_error = f"Error feedback:\n{optim_error}"
            return False, optim_error

        usage = self.board_info['funcusage']
        optim_error = None
        logger.info(f"Validating optimized code with usage:\n{usage}")
        optim_board, optim_error = self.prepare_ascii_rep.execute_optimized_response(self.board_info["size"], None, optim_func, usage)
        if optim_board is None:
            logger.error("Error executing optimized code. No board generated.")
            optim_error = f"Error feedback:\n{optim_error}"
            return False, optim_error
        optim_cells = self.prepare_ascii_rep.get_ascii_representation_from_board_forvalidation(optim_board, self.board_info["size"])
        #gt_cells = self.prepare_ascii_rep.get_ascii_representation_forvalidation(self.gtcode, self.board_info["size"])
        logger.info(f"Optimized board cells:\n{type(optim_cells)}, {optim_cells}")
        logger.info(f"Target board cells:\n{type(self.targetboard_cells)}, {self.targetboard_cells}")
        if optim_cells == self.targetboard_cells:
            logger.info("The optimized code generates the correct board.")
            return True, None
        else:
            logger.error("The optimized code does not generate the correct board.")
            diff_grid = self.prepare_ascii_rep.get_layer_representation_diff_for_optimization(self.targetboard_cells, optim_cells)
            return False, diff_grid


    def _prepare_function_header(self, func_name: str) -> str:
        return f"# Optimized function for object {func_name}\n" + f"# This function uses the following shapes:\n# {self.board_info['shapes']}\n\n"
    
    def _next_version_file(self, directory, base_name, ext=".json"):
        directory = Path(directory)
        # combo_name_v<number>_inst_code_pairs_v<number>.json
        pattern = re.compile(
            re.escape(base_name) + r"_v(\d+)" + re.escape(ext) + r"$"
        )

        versions = []
        for path in directory.glob(f"{base_name}_v*{ext}"):
            match = pattern.match(path.name)
            if match:
                versions.append(int(match.group(1)))

        next_version = (max(versions) if versions else 0) + 1
        return directory / f"{base_name}_v{next_version}{ext}"    

    def _save_optimized_function(self, func_name: str, func_code: str, func_usage: str):
        func_header = self._prepare_function_header(func_name)
        self.gamedata["optimized_function"] = func_code
        self.gamedata["optimized_function_header"] = func_header
        self.gamedata["func_usage"] = func_usage
        self.gamedata["optimized_function_signature"] = f"def {func_name}(board, colors, x, y):"
        """
        reuse_data = self._prepare_data_for_reuse()
        if reuse_data:
            reuse_data["optimized_function"] = func_code
            reuse_data["optimized_function_header"] = func_header

        reuse_data["func_usage"] = func_usage
        reuse_data["optimized_function_signature"] = f"def {func_name}(board, colors, x, y):"
        """

        reuse_data = {  "board_info": self.board_info,
                        "optimized_function": func_code,
                        "optimized_function_header": func_header,
                    }

        """Save the optimized function to a file."""
        os.makedirs("optimized_functions", exist_ok=True)
        #filename = f"combo_name_{self.combo_name}_optimized_v1.json"
        filename = f"combo_name_{self.board_info['combo_name']}_optimized"
        use_filename = self._next_version_file("optimized_functions", filename)
        with open(use_filename, "w") as f:
            #json.dump(inst_code_pairs, f, indent=4)
            json.dump(reuse_data, f, indent=4)


        combo_name = self.board_info['combo_name']
        if combo_name not in self.avail_skills:
            self.avail_skills[combo_name] = []
        
        usage_template = f"{combo_name}(board: np.ndarray, colors: list, x: int, y: int)"
        self.avail_skills[combo_name].append({"filename": use_filename, "usage_template": usage_template, "header": func_header, "skill_usage": func_usage})
        
        action = {'type': 'info',
                    'content': f"Saved the optimized function to file: {use_filename}"}
        self.log_event(from_='GM', to='GM', action=action)

        """
        with open(filename, "w") as f:
            f.write(f"# Optimized function for object {func_name}\n")
            f.write(f"# This function uses the following shapes:\n# {self.board_info['shapes']}\n\n")
            #f.write(f"# Import put(), move() etc functions as follows:\n#from coco import (\n#\tinit_board,\n#\tput,\n#\tmove,\n#\tremove,\n#\tclear,\n#\tundo\n#)\n\n")
            f.write(func_code + "\n\n")
            #Commenting function usage because that uses a specific colors and location values
            #f.write(f"# Function Usage:\n# create an empty board with dimensions as per the game\n# board=init_board(max_rows, max_columns)\n# {func_usage}\n")
        """    



    def _prepare_optimizer_reprobe_response(self, optim_error):
        logger.info(f"Preparing reprobe response for Player A. Current turn: {self.current_turn}, Error:\n{optim_error}")

        input_data = "Optimized function did not reconstruct the target grid correctly. Retry\n"+ self._prepare_function_signature_usage() + "\n"


        turn_prompt_co = input_data + "\n" + f"Target Grid:\n{self.player_a_goal}\n{optim_error}"
        if not self.use_dspy_collab:
            add_anchors = "[[ ## prompt ## ]]"
            p2_data = add_anchors + "\n" + turn_prompt_co + "\n\n" + "Do not generate any other reasoning traces as it will fail code execution\n\n" + "Respond with the corresponding output fields, starting with the field `[[ ## optimized_function ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."
        else:
            p2_data = turn_prompt_co

        return p2_data


    def _process_optimizer_response(self, parse_b: Dict) -> Dict:
        #parse_b is a dictionary with status and details keys
        # execute the optimized function code and check if it matches the ground truth
        optimized_function = parse_b["details"]
        #logger.info(f"Processing optimization response from player B: {parse_b}")
        optimization_result = {"status": "failure", "details": None, "error": None}
        optim_func_status, optim_error = self._validate_optimized_code(optimized_function)
        if optim_func_status:
            action = {'type': 'info',
                      'content': "Optimized function validated successfully."}
            self.log_event(from_='GM', to='GM', action=action)

            optimization_result["status"] = "success"
            optimization_result["details"] = optimized_function
            #self.gamedata["optimization_status"] = {"status": optimization_result["status"], "details": optimization_result["details"], "error": None}
            self._save_optimized_function(self.board_info["combo_name"], optimized_function, self.board_info["funcusage"])
            p2_prompt = None
        else:
            self._set_violated_req_count(optim_error)
            optimization_result["error"] = optim_error
            #TODO: May need to add the turn prompt based on the value of current turn
            p2_prompt = self._prepare_optimizer_reprobe_response(optim_error)

            if self.use_dspy_collab:
                action = {'type': 'send message', 'content': p2_prompt}
                self.log_event(from_='GM', to='Player 2', action=action)

            #self.current_optim_retry += 1
            self.used_retry_optim = True
            self.gamedata["used_retry_optim"] = self.used_retry_optim  

        return optimization_result, p2_prompt

                             

    def _parse_response(self, player: Player, response: str) -> str:
        # increase the number of API requests:
        self.request_count += 1
        logger.debug(f"Current turn: {self.current_turn}, Received response from player: {player}:{type(response)}")
        parse_a = {"status": "failure", "details": None, "error": None}
        if player == self.player_a:
            user_instruction = self._handle_playera_response(response)
            logger.info(f"Parsed response from player A: {user_instruction}")

            action = {'type': 'parsed response',
                    'content': f"User instruction:\n{user_instruction}"}
            self.log_event(from_='GM', to='GM', action=action)

            if user_instruction is None:
                error = "Response from player A does not conform to required format."
                self._set_violated_req_count(error)
                parse_a["error"] = error
                return parse_a

            action = {'type': 'get message',
                      'content': user_instruction}
            #self.log_event(from_='Player 1', to='GM', action=action)
            self._set_parsed_req_count()            


            if user_instruction.upper() == "DONE":
                logger.info(f"Current turn: {self.current_turn}, Received 'DONE' from player A, validating the game.")
                parse_a = self._validate_game(self.genboard)
                logger.info(f"After validation, parse_a:{parse_a}")
                if parse_a["status"] == "success":
                    # Go ahead with code optimization
                    # TODO: Need to set the flag for the board : There are m simple boards and n regular boards
                    self.reconstruct_success = True
                    self.gamedata["reconstruction_status"] = self.reconstruct_success
                    if self.current_variant == "simple":
                        parse_a["status"] = "on-going"
                        p2_prompt = self._prepare_optimizer()
                        self.set_context_for(self.player_b, p2_prompt)
                    else:
                        #if there are more boards, should go to next board directly
                        status, p1_prompt = self._prepare_next_board_after_optim()
                        if status == "COMPLETED":
                            #self.game_over = True
                            action = {'type': 'info',
                                    'content': "All boards completed. Ending the game."}
                            self.log_event(from_='GM', to='GM', action=action)  
                            parse_a["status"] = "success"
                            parse_a["details"] = "All boards completed. Ending the game."
                            parse_a["error"] = None
                        else:
                            parse_a["status"] = "next_rb_board"
                            parse_a["details"] = "Moving to next board."
                            parse_a["error"] = None
                            self.set_context_for(self.player_a, p1_prompt)


                    # TODO: Need to trigger the next board reconstruction here
                else:
                    # This means the game is lost and the game master will handle it in advance_game
                    pass
            elif user_instruction == "SKILL_UNKNOWN":
                logger.info(f"Current turn: {self.current_turn}, Received 'SKILL_UNKNOWN' from player A.")
                parse_a["status"] = "failure"
                parse_a["details"] = None
                parse_a["error"] = "The requested skill is unknown or not available for reuse."
            else:
                #TODO: May need to add the turn prompt based on the value of current turn
                p2_prompt = self._prepare_playera_turn_response(user_instruction)

                if self.use_dspy_collab:
                    action = {'type': 'send message', 'content': p2_prompt}
                    self.log_event(from_='GM', to='Player 2', action=action)

                logger.info(f"Setting context for player B with parsed response from player A")
                self.set_context_for(self.player_b, p2_prompt)
                self.genresponse.append({"current_turn": self.current_turn, "instruction": user_instruction})
                parse_a["status"] = "on-going"
                parse_a["details"] = user_instruction
            return parse_a

        elif player == self.player_b:
            parse_b, error = self._handle_playerb_response(response)
            logger.info(f"Parsed response from player B: parse_b: {parse_b}, error : {error}")

            if error:
                action = {'type': 'error while parsing response',
                        'content': f"Error: {error}"}
                self.log_event(from_='GM', to='GM', action=action)
                parse_b = {"status": "failure", "details": None, "error": error}
                error_prompt = f"Error during parsing of the generated response\n{error}\nPlease follow the response format strictly." 

                if self.use_dspy_collab:
                    action = {'type': 'send message',
                            'content': error_prompt}
                    self.log_event(from_='GM', to='Player 2', action=action)  

                self.set_context_for(self.player_b, error_prompt)
                self._set_violated_req_count(error)
            else:
                if self.optim_progress:
                    self._set_parsed_req_count()                    
                    self.current_collab_optim += 1
                    optim_status, p2_prompt = self._process_optimizer_response(parse_b)
                    if optim_status["status"] == "failure":
                        if self.current_collab_optim < self.num_collab_optim_turns:
                            logger.info(f"Optimization not successful yet. Reprobing player B for optimization. Current optimization turn: {self.current_collab_optim}")

                            self.set_context_for(self.player_b, p2_prompt)
                        else:
                            logger.info(f"Optimization attempts exhausted. Moving to next board.")
                            # Go ahead with processing next board
                            status, p1_prompt = self._prepare_next_board_after_optim()
                            if status == "COMPLETED":
                                #self.game_over = True
                                logger.info("Received COMPLETED during OPTIM_PROGRESS. Check the flow")
                                action = {'type': 'info',
                                        'content': "All boards completed. Ending the game."}
                                self.log_event(from_='GM', to='GM', action=action)  
                                optim_status["status"] = "completed"
                                optim_status["details"] = "All boards completed. Ending the game."
                                optim_status["error"] = None
                            else:
                                self.set_context_for(self.player_a, p1_prompt)
                                optim_status["status"] = "success"
                                optim_status["details"] = "Optimization attempts exhausted. Moving to next board."
                                optim_status["error"] = None
                    else:
                        #Optimization is successful
                        self.optim_progress = False
                        self.current_collab_optim = 0
                        logger.info(f"Optimization successful. Moving to next board.")
                        status, p1_prompt = self._prepare_next_board_after_optim()
                        if status == "COMPLETED":
                            #self.game_over = True
                            logger.info("Received COMPLETED")
                            action = {'type': 'info',
                                    'content': "All boards completed. Ending the game."}
                            self.log_event(from_='GM', to='GM', action=action)  
                            optim_status["status"] = "completed"
                            optim_status["details"] = "All boards completed. Ending the game."
                            optim_status["error"] = None
                        self.set_context_for(self.player_a, p1_prompt)
                    return optim_status

                #variant = "simple" if not self.regboard_probe else "regular"
                self.genresponse[-1]["response"] = parse_b
                if parse_b["status"] == "clarification":
                    self.used_clarification = True
                    self.num_clarifications += 1
                    self.gamedata["used_clarification"] = self.used_clarification
                    self.gamedata["num_clarifications"] = self.num_clarifications
                    response_b = self._prepare_playerb_clarification_response(self.current_variant, parse_b["details"], True)
                    parsed_response = f"Clarification question:\n{parse_b['details']}"

                elif parse_b["status"] == "acknowledgement":
                    self.used_acknowledgement = True
                    self.num_acknowledgements += 1
                    self.gamedata["used_acknowledgement"] = self.used_acknowledgement
                    self.gamedata["num_acknowledgements"] = self.num_acknowledgements
                    response_b = self._prepare_playerb_clarification_response(self.current_variant, parse_b["details"], False)
                    parsed_response = f"Clarification question:\n{parse_b['details']}"                    

                else:
                    # Temporary code to simulate execution error
                    #parse_b['details'] = "put(board, 'bridge-h', 'red', 0,7)"
                    parsed_response = f"Code to execute:\n{parse_b['details']}"
                    response_b, error = self._prepare_playerb_code_response(self.current_variant, parse_b["details"])

                action = {'type': 'parsed response',
                        'content': parsed_response}
                self.log_event(from_='GM', to='GM', action=action)

                if error:
                    action = {'type': 'error while executing parsed response',
                            'content': error}
                    self.log_event(from_='GM', to='GM', action=action) 
                    error_prompt = f"Error during execution of the code:\n{error}\nPlease correct the code to fix the error."                       

                    if self.use_dspy_collab:
                        action = {'type': 'send message',
                                'content': error_prompt}
                        self.log_event(from_='GM', to='Player 2', action=action)  

                    parse_b = {"status": "failure", "details": None, "error": error}
                    self.set_context_for(self.player_b, error_prompt)
                    self._set_violated_req_count(error)
                else:
                    self._set_parsed_req_count()
                    self.current_reconst_retry = 0

                    p1_prompt = self.turn_prompt_a+"\n"+response_b
                    if self.use_dspy_collab:
                        action = {'type': 'send message',
                                'content': p1_prompt}
                        self.log_event(from_='GM', to='Player 1', action=action)  

                    logger.info(f"Setting context for player A with parsed response from player B")
                    self.set_context_for(self.player_a, p1_prompt)
            return parse_b        
        
    def _get_current_filled_grid(self, regboard_probe: bool) -> str:
        if not self.genresponse:
            return "None"

        if not regboard_probe:
            if "ascii_rep" in self.genresponse[-1]:
                use_current_grid = self.genresponse[-1]['ascii_rep']
            else:
                turn_number = -2
                while abs(turn_number) <= len(self.genresponse) and "ascii_rep" not in self.genresponse[turn_number]:
                    turn_number -= 1

                if abs(turn_number) > len(self.genresponse):
                    use_current_grid = "None"
                else:
                    use_current_grid = self.genresponse[turn_number]['ascii_rep']
        else:
            if "regular_rep" in self.genresponse[-1]:
                use_current_grid = self.genresponse[-1]['regular_rep']
            else:
                turn_number = -2
                while abs(turn_number) <= len(self.genresponse) and "regular_rep" not in self.genresponse[turn_number]:
                    turn_number -= 1

                if abs(turn_number) > len(self.genresponse):
                    use_current_grid = "None"
                else:
                    use_current_grid = self.genresponse[turn_number]['regular_rep']

        return use_current_grid         

    def _get_available_skills(self):
        
        if not self.avail_skills:
            return None
        skills_info = ""
        for combo_name, skills in self.avail_skills.items():
            skills_info += f"To place an object: {combo_name}, please use this function:\n"
            #usage is similar for all skills for a combo_name
            skills_info += f"{skills[-1]['usage_template']}\n\n"#{skills[-1]['header']}\n"
        return skills_info

    def _prepare_playera_turn_response(self, user_instruction):
        if self.current_turn == 0:
            p1_data = self.prompt_player_b
            use_current_grid = "None"
        else:
            p1_data = self.turn_prompt_b
            use_current_grid = self._get_current_filled_grid(self.regboard_probe)

            if self.regboard_probe:
                p1_data += f"\n\nAvailable Skills:\n{self._get_available_skills()}"


        p1_data += f"\n\nUser Instruction:\n{user_instruction}" + f"\n\nCurrent Grid:\n{use_current_grid}"

        return p1_data            

    def _validate_game(self, genboard) -> Dict:
        """Run the game validation process."""
        logger.info("Starting game validation process")

        result = {"status": "failure", "details": None, "error": None}

        gt_cells = self.prepare_ascii_rep.get_ascii_representation_forvalidation(self.gtcode, self.board_info["size"])
        gen_cells = self.prepare_ascii_rep.get_ascii_representation_from_board_forvalidation(genboard, self.board_info["size"])
        self.gen_board_cells = gen_cells

        logger.info(f"Ground truth cells: {gt_cells}")
        logger.info(f"Generated cells: {gen_cells}")
        if gt_cells is None or gen_cells is None:
            logger.error("One of the representations is None, cannot proceed with validation.")
            result["error"] = "One of the representations is None."

        if gt_cells == gen_cells:
            logger.info("The generated board matches the ground truth.")
            result["status"] = "success"

        else:
            logger.error("The generated board does not match the ground truth.")
            #TODO: May need to log the details of the mismatch
            result["status"] = "failure"
            result["error"] = "The generated board does not match the ground truth."

        return result


    def _log_game_end(self) -> None:
        """Aux to log variables needed for scoring (firstlast specific)"""
        self.log_key("Played turns", self.current_turn)
        self.log_key("Complete turns", self.complete_turns)
        self.log_key('Turn scores', self.turn_scores)        
        self.log_key(METRIC_ABORTED, self.aborted)
        self.log_key(METRIC_LOSE, self.lose)
        self.log_key(METRIC_SUCCESS, self.success)        
        self.log_key(METRIC_REQUEST_COUNT, self.request_count)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.parsed_request_count)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_count)

        self.log_key("Evaluation", self.gamedata)
        logger.info("Game ended. Logged game data.")  

    def compute_turn_score(self):
        return self.turn_scores[self.current_turn-1]

    def compute_episode_score(self):
        """
        Calculate a score for the episode based on successful turns and target number of turns.
        Returns:
            Episode score value in range 0-100.
        """
        turn_score_sum = sum(self.turn_scores)
        success_ratio = turn_score_sum / self.n_turns
        return success_ratio * 100       



class CCBTSCollabScorer(GameScorer):
    """Scorer for the firstlast game."""
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        #self.codevalidator = CodeValidator()        

    def score_turns(self, episode_interactions: Dict) -> None:
        """Calculate and log turn-level scores."""
        played_turns = episode_interactions['Played turns']
        turn_scores = episode_interactions['Turn scores']
        for turn in range(0, played_turns):
            self.log_round_score(turn, "turn score", turn_scores[turn])

    def log_main_score(self, episode_interactions: Dict):
        complete_turns = episode_interactions['Complete turns']
        n_turns = episode_interactions['n_turns']
        aborted = int(episode_interactions[METRIC_ABORTED])
        success = int(episode_interactions[METRIC_SUCCESS])
        # IMPORTANT: aborted episodes MUST have a bench score of NaN!
        bench_score = 1 if success else 0 if not aborted else np.nan
        self.log_episode_score(BENCH_SCORE, bench_score)

    """
    def game_specific_score(self, episode_interactions: Dict) -> None:
        # check the validity of the code:
        board_size = episode_interactions["Evaluation"]["boardinfo"]["size"]
        variant = episode_interactions["Evaluation"]["boardinfo"]["variant"]
        #gtcode = episode_interactions["Evaluation"]["gtcode"]
        #gencode = episode_interactions["Evaluation"]["gencode"]
        # Compute all three metrics: EM, CB and ES
        board_details = {"rows": board_size["rows"], "cols": board_size["cols"], "variant": variant}
        if episode_interactions["Evaluation"]["reconstruction_status"] is not None:
            reconst_success = episode_interactions["Evaluation"]["reconstruction_status"]
        else:
            reconst_success = False
        logger.info(f"Reconstruction success status: {reconst_success}")
        self.log_episode_score("reconstruction_success", 1 if reconst_success is True else 0)
    """

    def compute_scores(self, episode_interactions: Dict) -> None:
        # Log turn-level scores
        self.score_turns(episode_interactions)
        # Log main score
        self.log_main_score(episode_interactions)
        # Log game-specific scores
        #self.game_specific_score(episode_interactions)

     
class CCBTSCollabBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    # copy this, replacing the name of the game master in the return statement
    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> DialogueGameMaster:
        return CCBTSCollabMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return CCBTSCollabScorer(self.game_name, experiment, game_instance)