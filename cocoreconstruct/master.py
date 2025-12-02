import os
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

class CCBTSReconstMaster(DialogueGameMaster):
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
        self.use_dspy_reconst = data["use_dspy_reconst"]
        self.use_dspy_history = data["use_dspy_history"]
        self.use_retry_for_reconstruction = data["use_reconst_retry"]
        self.use_diff_human_prompts = data["use_diff_human_prompts"]
        self.n_turns = data["n_turns"]
        self.num_reconst_retry = data["num_reconst_retry"]
        self.board_info = data["boardinfo"]
        self.gtcode = data["boardinfo"]["gtcode"]   
        self.gtimagefile = data["boardinfo"]["gtimage_filename"]    
        self.gt_image_base64 = None#data["boardinfo"]["gt_image_base64"]
        self.empty_player_board_base64 = data["boardinfo"]["empty_player_board_base64"]
        self.shapes_references_base64 = data["boardinfo"]["shapes_references_base64"]
        self.prompts_dict = data["prompts_dict"]
        self.prompt_player_a_base = self.prompts_dict["prompt_a"]
        self.prompt_player_a_human = self.prompts_dict["prompt_a_human"]        
        self.prompt_player_b_base = self.prompts_dict["prompt_b"]
        self.prompt_player_b_human = self.prompts_dict["prompt_b_human"]
        self.turn_prompt_a = self.prompts_dict.get("turn_prompt_a", "")
        self.turn_prompt_a_human = self.prompts_dict.get("turn_prompt_a_human", "")
        self.turn_prompt_b = self.prompts_dict.get("turn_prompt_b", "")
        self.turn_prompt_b_human = self.prompts_dict.get("turn_prompt_b_human", "")
        self.player_a_goal = data["ascii_rep"]
        self.use_sandbox_llm = data["use_sandbox_llm"]


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

        # initialise common metrics:
        self.request_count: int = 0
        self.parsed_request_count: int = 0
        self.violated_request_count: int = 0 

        self.set_pass_turn = True

        self.prepare_ascii_rep = PrepareASCIIRep()
        if self.use_sandbox_llm:
            self.prepare_sandbox = PrepareLLMSandBox(config=data["sandbox_llm"])



        logger.info(f"GT Code:\n{self.gtcode}")
        self.gt_image_base64 = None#self.prepare_ascii_rep.get_image_base64_from_filepath(self.gtimagefile)

        # instantiate players:
        self.player_a = InstructionGiver(self.model_a, "A", self.board_info["variant"], self.use_dspy_reconst, self.use_dspy_history)
        self.player_a_type = self.player_a.get_player_type()
        # add players, including assigning their initial prompts:
        if self.player_a_type == "human":
            self.prompt_player_a = self.prompt_player_a_human
        elif self.player_a_type == "programmatic":
            self.prompt_player_a = self.prompt_player_a_base
        else:
            if self.use_dspy_reconst:
                self.prompt_player_a = self.player_a.get_player_prompt()
            else:
                self.prompt_player_a = self.prompt_player_a_base
        self.add_player(self.player_a, initial_context=self.prompt_player_a)

        # add player B, the one that follows the instructions:
        self.player_b = InstructionFollower(self.model_b, "B", self.board_info["variant"], self.use_dspy_reconst, self.use_dspy_history)
        self.player_b_type = self.player_b.get_player_type()
        if self.player_b_type == "human":
            self.prompt_player_b = self.prompt_player_b_human
        elif self.player_b_type == "programmatic":
            self.prompt_player_b = self.prompt_player_b_base
        else:
            if self.use_dspy_reconst:
                self.prompt_player_b = self.player_b.get_player_prompt()
            else:
                self.prompt_player_b = self.prompt_player_b_base
        self.add_player(self.player_b, initial_prompt=self.prompt_player_b)


        self.gamedata = {
            "boardinfo": self.board_info,
            "gtcode": self.gtcode,
            "gtimage_filename": self.gtimagefile,
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
            "use_dspy_reconst": self.use_dspy_reconst,
            "use_dspy_history": self.use_dspy_history,
            "max_reconst_retry": self.num_reconst_retry,
            "total_reconst_retry": self.current_reconst_retry,
            "used_retry_reconst": self.used_retry_reconst,
            "inst_code_pairs": None,
        }
        self.genresponse = []#{"instructions": [], "code": []}
        self.genboard = None
        self.gen_board_cells = None


    def _on_before_game(self) -> None:
        """Initialise the dialogue history (firstlast specific)."""
        p1_data = f"grid_size: 8x8\nskill name: {self.board_info['combo_name']}\ncolors: {self.board_info['colors']}\nlocation: {self.board_info['locations']}\ntarget_grid:{self.player_a_goal}\ndifference_grid: None\nclarification: None"
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

        if self.use_dspy_reconst:
            action = {'type': 'send message', 'content': p1_messages}
            self.log_event(from_='GM', to='Player 1', action=action)


        self.set_context_for(self.player_a, p1_messages)
        gt_cells, _ = self.prepare_ascii_rep.get_ascii_representation(self.gtcode, self.board_info["size"])
        logger.info(f"Ground truth cells:\n{gt_cells}")
        self.gt_occupied_cells = self.prepare_ascii_rep.get_occupied_cells(self.gtcode, self.board_info["size"])
        logger.info(f"Ground truth occupied cells:\n{self.gt_occupied_cells}")

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
        self._save_instruction_code_pairs()
        self._log_game_end()  


    def _set_pass_turn(self, player: Player, pass_turn) -> None:
        #Set the turn to be passed for a player (game specific).
        logger.info(f"Setting pass turn for player: {player}")
        self.set_pass_turn = pass_turn

    def _should_pass_turn(self):
        #Currently not checking for any condition to pass the turn
        logger.info(f"Checking if turn should be passed: {self.set_pass_turn}")
        return self.set_pass_turn       

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
            action_content = "The game is successful; target board is reconstructed successfully."

        else:
            if not self.aborted:# and self.current_round == self.n_turns:
                self.lose = True

        logger.info(f"Game status: {self.success}, {self.lose}, {self.aborted}")

        if self.lose:
            action_type = "info"
            action_content = "Maximum turns reached; lost game"
        elif self.aborted:
            action_type = "invalid format"
            action_content = "The game has been aborted due to an invalid input."

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
                action_content = "The player grid did not match with the target grid."

                self.log_to_self(action_type, action_content)                
                self.lose = True
                # set the reason for the loss:
                self.gamedata["loss_reason"] = parsed_response.get("error", "Unknown error")


        elif player == self.player_b:
            if parsed_response["status"] == "failure":
                # Time to reprobe the player with error feedback
                logger.info(f"Player B response validation failed with error: {parsed_response['error']}")
                # No need to check if retry is allowed here; it is handled in _set_violated_req_count() which triggers parseerror
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


            if user_instruction == "DONE":
                logger.info(f"Current turn: {self.current_turn}, Received 'DONE' from player A, validating the game.")
                parse_a = self._validate_game(self.genboard)
                logger.info(f"After validation, parse_a:{parse_a}")
                if parse_a["status"] == "success":
                    # Go ahead with code optimization
                    self.reconstruct_success = True
                    self.gamedata["reconstruction_status"] = self.reconstruct_success
                else:
                    # This means the game is lost and the game master will handle it in advance_game
                    pass
            else:
                #TODO: May need to add the turn prompt based on the value of current turn
                p2_prompt = self._prepare_playera_turn_response(user_instruction)

                if self.use_dspy_reconst:
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

                if self.use_dspy_reconst:
                    action = {'type': 'send message',
                            'content': error_prompt}
                    self.log_event(from_='GM', to='Player 2', action=action)  

                self.set_context_for(self.player_b, error_prompt)
                self._set_violated_req_count(error)
            else:
                self.genresponse[-1]["response"] = parse_b
                if parse_b["status"] == "clarification":
                    self.used_clarification = True
                    self.num_clarifications += 1
                    self.gamedata["used_clarification"] = self.used_clarification
                    self.gamedata["num_clarifications"] = self.num_clarifications
                    response_b = self._prepare_playerb_clarification_response(parse_b["details"])
                    parsed_response = f"Clarification question:\n{parse_b['details']}"

                else:
                    # Temporary code to simulate execution error
                    #parse_b['details'] = "put(board, 'bridge-h', 'red', 0,7)"
                    parsed_response = f"Code to execute:\n{parse_b['details']}"
                    response_b, error = self._prepare_playerb_code_response(parse_b["details"])

                action = {'type': 'parsed response',
                        'content': parsed_response}
                self.log_event(from_='GM', to='GM', action=action)

                if error:
                    action = {'type': 'error while executing parsed response',
                            'content': error}
                    self.log_event(from_='GM', to='GM', action=action) 
                    error_prompt = f"Error during execution of the code:\n{error}\nPlease correct the code to fix the error."                       

                    if self.use_dspy_reconst:
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
                    if self.use_dspy_reconst:
                        action = {'type': 'send message',
                                'content': p1_prompt}
                        self.log_event(from_='GM', to='Player 1', action=action)  

                    logger.info(f"Setting context for player A with parsed response from player B")
                    self.set_context_for(self.player_a, p1_prompt)
            return parse_b
    
    def _model_response_cleanup(self, response: str) -> str:
        clean_response = re.sub(r'```json(.*?)```', r'\1', response, flags=re.DOTALL).strip()
        clean_response = re.sub(r'```(.*?)```', r'\1', clean_response, flags=re.DOTALL).strip()
        clean_response = re.sub(r'```', '', clean_response).strip()
        #Remove [[ ## instruction ## ]] and [[ ## completed ## ]] from the response
        clean_response = re.sub(r'\[\[\s*##\s*instruction\s*##\s*\]\]', '', clean_response, flags=re.IGNORECASE).strip()
        clean_response = re.sub(r'\[\[\s*##\s*player_response\s*##\s*\]\]', '', clean_response, flags=re.IGNORECASE).strip()
        clean_response = re.sub(r'\[\[\s*##\s*completed\s*##\s*\]\]', '', clean_response, flags=re.IGNORECASE).strip()
        #Remove brackets like[[ or ]]
        clean_response = re.sub(r'(?m)^\s*\[\[\s*$','', clean_response).strip()
        clean_response = re.sub(r'(?m)^\s*\]\]\s*$','', clean_response).strip()
        return clean_response

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
    
    def _prepare_data_for_optimizer(self):

        if self.gt_occupied_cells:
            groundtruth_board_rep = self.prepare_ascii_rep.get_layer_representation_for_optimization(self.gt_occupied_cells, True)

        generated_board = None
        generated_board_rep = None
        gen_occupied_cells = None        
        if self.genboard is not None:
            generated_board_rep, gen_occupied_cells = self.prepare_ascii_rep.get_layer_representation_for_optimization(self.genboard)
            generated_board = None#copy.deepcopy(self.genboard)
            #generated_board = generated_board.tolist()

        optimizer_data = {"board_info": self.board_info,
                          "use_dspy_reconst": self.use_dspy_reconst,
                          "use_dspy_history": self.use_dspy_history,
                          "use_reconst_retry": self.use_retry_for_reconstruction,
                          "num_reconst_retry": self.num_reconst_retry,
                          "use_diff_human_prompts": self.use_diff_human_prompts,
                          "reconstruction_status": self.success,
                          "n_turns": self.n_turns,
                          "played_turns": self.current_round,
                          "level_rep_detailed": self.player_a_goal,
                          "groundtruth_board_rep": groundtruth_board_rep,
                          "generated_board_rep": generated_board_rep,
                          "generated_board": generated_board,
                          "gen_occupied_cells": gen_occupied_cells,
                          "reconstruction_status": self.success,
                          "reconstruction_aborted": self.aborted,
                         }
        return optimizer_data


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


    def _save_instruction_code_pairs(self) -> None:
        inst_code_pairs = self._prepare_instruction_code_pairs()
        if inst_code_pairs is None:
            logger.info("No instruction-code pairs to save.")
            return

        self.gamedata["inst_code_pairs"] = inst_code_pairs
        optimizer_data = self._prepare_data_for_optimizer()
        if optimizer_data:
            optimizer_data["inst_code_pairs"] = inst_code_pairs


        # Save to a json file
        os.makedirs("reconstruct-data-pairs", exist_ok=True)
        filename = f"combo_name_{self.board_info['combo_name']}_inst_code_pairs"
        use_filename = self._next_version_file("reconstruct-data-pairs", filename)
        with open(use_filename, "w") as f:
            #json.dump(inst_code_pairs, f, indent=4)
            json.dump(optimizer_data, f, indent=4)
    
    def _get_playerb_grid(self):
        if self.genboard is None:
            return "None"

        _, gen_occupied_cells = self.prepare_ascii_rep.get_ascii_representation_from_board_layers(self.genboard, self.board_info["size"])        
        diff_grid = self.prepare_ascii_rep.get_layer_representation_diff(self.gt_occupied_cells, gen_occupied_cells)
        return diff_grid


    def _get_current_filled_grid(self):
        if not self.genresponse:
            return "None"

        elif "ascii_rep" in self.genresponse[-1]:
            use_current_grid = self.genresponse[-1]['ascii_rep']
        else:
            turn_number = -2
            while abs(turn_number) <= len(self.genresponse) and "ascii_rep" not in self.genresponse[turn_number]:
                turn_number -= 1

            if abs(turn_number) > len(self.genresponse):
                use_current_grid = "None"
            else:
                use_current_grid = self.genresponse[turn_number]['ascii_rep']
        return use_current_grid      
    
    def _prepare_playera_turn_response(self, user_instruction):
        if self.current_turn == 0:
            p1_data = self.prompt_player_b
            use_current_grid = "None"
        else:
            p1_data = self.turn_prompt_b
            use_current_grid = self._get_current_filled_grid()            


        p1_data += f"\n\nUser Instruction:\n{user_instruction}" + f"\n\nCurrent Grid:\n{use_current_grid}"

        return p1_data

    def _prepare_playerb_turn_response(self, gen_image_base64, difference_grid, clarification, reconstruction_status):

        clarification_text = clarification
        if clarification:
            if self.player_a_type == "human":
                clarification_text = f'<span style="color: red; font-size: 20px;">{clarification}</span>'

        p2_data = f"\ntarget_grid:{self.player_a_goal}\nDifference_grid: {difference_grid}\nClarification: {clarification_text}\nAre the target grid and player grid equal?\nReconstruction Status: {reconstruction_status}"

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
    
    def _prepare_playerb_clarification_response(self, details: str) -> str:
        gen_image_base64 = None#self.genresponse[-1]["gen_image_base64"] if self.genresponse and len(self.genresponse) > 0 and "gen_image_base64" in self.genresponse[-1] else None
        diff_grid = self._get_playerb_grid()
        grid_match = self._validate_game(self.genboard)
        reconstruction_complete = "True" if grid_match["status"] == "success" else "False"

        return self._prepare_playerb_turn_response(gen_image_base64, diff_grid, details, reconstruction_complete)
    
    def _prepare_playerb_code_response(self, details: str) -> str:
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
                board_gen_call, error, code_stats = self.prepare_ascii_rep.execute_generated_response(details, self.board_info["size"], board_gen_call)

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


            self.genboard = copy.deepcopy(board_gen_call)
            logger.info("Calling get_ascii_representation to generate ASCII representation")
            gen_ascii_rep, gen_occupied_cells = self.prepare_ascii_rep.get_ascii_representation_from_board_layers(board_gen_call, self.board_info["size"])

            logger.info(f"Generated ASCII representation:\n{gen_ascii_rep}")
            if gen_ascii_rep is None:
                if gen_occupied_cells is None:
                    logger.error("Error generating ASCII representation from generated response.")
                    return None, "Error generating ASCII representation from generated response."
            
            self.prepare_ascii_rep.set_occupied_cells(gen_occupied_cells)
            gen_board_image_filename = f"turn_{self.current_turn+1}_playerb_board.png"
            gen_image_base64 = None#self.prepare_ascii_rep.get_image_gen_board(self.genboard, gen_board_image_filename)

            self.genresponse[-1]["ascii_rep"] = gen_ascii_rep
            self.genresponse[-1]["occupied_cells"] = gen_occupied_cells
            self.genresponse[-1]["code_stats"] = code_stats
            self.genresponse[-1]["gen_image_base64"] = gen_image_base64
            logger.info(f"Generated occupied cells:\n{gen_occupied_cells}")
            logger.info(f"Ground truth occupied cells:\n{self.gt_occupied_cells}")
            logger.info("Calling diff grid")
            diff_grid = self.prepare_ascii_rep.get_layer_representation_diff(self.gt_occupied_cells, gen_occupied_cells)
            logger.info(f"Difference grid representation:\n{diff_grid}")

            grid_equality = self._validate_game(self.genboard)
            reconstruction_complete = "True" if grid_equality["status"] == "success" else "False"

            p2_response = self._prepare_playerb_turn_response(self.genresponse[-1]["gen_image_base64"], diff_grid, None, reconstruction_complete)

            return p2_response, None

        except Exception as e:
            logger.error(f"Error executing generated response: {e}")
            return None, "Error executing generated response from player B."


    def _validate_playerb_response(self, model_response: str) -> bool:
        parse_b, error = None, None
        logger.info(f"Validating response from player B: {model_response}")
        try:
            parse_b = json.loads(model_response)
            if "status" not in parse_b or "details" not in parse_b:
                error = f"Missing 'status' or 'details' in response ({parse_b}) from player B."
            
            if parse_b["status"] not in ["code", "clarification"]:
                logger.error(f"Invalid status in response from player B: {parse_b['status']}")
                error = f"Invalid status: {parse_b['status']} (not 'code' or 'clarification') in response from player B."

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

    def _set_parsed_req_count(self) -> None:
        # increase the counter of requests that conform to form rules
        self.parsed_request_count += 1  

        # log the event that the string was valid (no strange characters)
        action = {'type': 'valid response', 'content': 'response conforms to rules'}
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
            if self.use_retry_for_reconstruction and self.current_reconst_retry < self.num_reconst_retry:
                logger.info(f"Response did not conform to rules. Reprobing the player. Current retry: {self.current_reconst_retry+1}")
                self.log_event(from_='GM', to='GM', action={'type': 'info', 'content': 'Response did not conform to rules. Reprobing the player.'})
                self.current_reconst_retry += 1
                self.used_retry_reconst = True
                self.gamedata["used_retry_reconst"] = self.used_retry_reconst
                retry = True
            else:
                logger.error(f"Response did not conform to rules. Reprobing tries exceeded.")
                self.log_event(from_='GM', to='GM', action={'type': 'info', 'content': 'Response did not conform to rules. Reprobing tries exceeded.'})
                # Not checking for optimization flow here as it is handled separately
        if not retry:
            raise ParseError(error)

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



class CCBTSReconstScorer(GameScorer):
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

     
class CCBTSReconstBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    # copy this, replacing the name of the game master in the return statement
    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> DialogueGameMaster:
        return CCBTSReconstMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return CCBTSReconstScorer(self.game_name, experiment, game_instance)      