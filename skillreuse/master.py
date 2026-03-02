import os
import re
import ast
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



import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SkillReuseMaster(DialogueGameMaster):
    def __init__(
        self,
        game_spec: GameSpec,
        experiment: Dict,
        player_models: List[str]
    ):
        super().__init__(game_spec, experiment, player_models)
        # save experiment and player attributes that will be necessary later
        logger.info(f"Initializing SkillReuseMaster: model_a = {player_models[0]} ")
        self.model_a = player_models[0]
        self.model_b = player_models[1] if len(player_models) > 1 else player_models[0]

    def _on_setup(self, **game_instance) -> None:
        """Setup the episode (mandatory)."""

        #logging.disable(logging.CRITICAL)
        self.instancedata = game_instance["data"]
        self.game_id = game_instance["game_id"]

        data = self.instancedata
        self.boards = data["boards"]
        self.reuse_data = data["reuse_data"]
        self.gt_occupied_cells = data["reuse_data"]["target_board_cells"]
        self.optimfunc_occupied_cells = None
        self.goal_layerwise_rep = data["reuse_data"]["layerwiserep"]
        self.prompts_dict = data["prompts_dict"]
        self.shapes_references_base64 = data["shapes_references_base64"]        

        self.prompt_a = {"reuse": self.prompts_dict["prompt_a_reuse"]}
        self.turn_prompt_a = {"reuse": self.prompts_dict["turn_prompt_a_reuse"]}
        self.prompt_b = {"reuse": self.prompts_dict["prompt_b_reuse"]}
        self.turn_prompt_b = {"reuse": self.prompts_dict["turn_prompt_b_reuse"]}

        self.use_error_feedback = data["use_error_feedback"]
        self.use_images_in_human_prompts = data["use_images_in_human_prompts"]
        self.use_skills = data["use_skills"]
        self.use_oracle_code = data["use_oracle_code"]
        self.use_sandbox_llm = data["use_sandbox_llm"]
        self.num_retry = data["num_retry"]
        self.existing_skills_filename = data["existing_skills_filename"]
        self.avail_skills = data["existing_skills"]
        self.avail_skills_code = data["skills_code"]
        self.skillandtargetcellsdiscrep = data["skillandtargetcellsdiscrep"]   

        # n_turns is multiplied by 3 because each task has n_turns and there are 3 tasks (reconst, reuse, repeat) + x turns for optim
        self.n_turns = data["n_turns"]#3 * data["n_turns"] + (self.num_optim_turns if self.use_optimizer else 0)
        self.config_max_turns = data["n_turns"]
        self.max_task_turns: int = 0
        # initialise game variables:
        self.current_turn: int = 0
        self.current_retry: int = 0

        self.log_key('n_turns', self.n_turns)
        self.turn_scores = [0] * (self.n_turns)

        # initialise attributes that will be used for the evaluation scores
        self.aborted: bool = False
        self.lose: bool = False
        self.success: bool = False
        self.complete_turns: int = 0

        self.reuse_success: bool = False
        self.play_turns_reuse: int = 0
        self.play_turns_total: int = 0

        
        self.reuse_details = {"loss_reason": None, "play_reuse_turns": 0, "num_clarifications": 0,
                               "used_clarification": False, "num_clears": 0, "used_clear": False, "num_undos": 0,
                               "used_undo": False, "num_removes": 0, "used_remove": False, "num_moves": 0, "used_move": False,}
        

        # initialise common metrics:
        self.request_count: int = 0
        self.parsed_request_count: int = 0
        self.violated_request_count: int = 0 

        self.set_pass_turn = True
        self.prepare_ascii_rep = PrepareASCIIRep()

        self.variant = None
        self.current_task = None
        self.current_task_turns = 0
        self.player_grid_match_status = False
        self.player_grid = None
        self.player_occupied_cells = None
        self.gt_image_base64 = None
        self.empty_board_base64 = None
        self.gen_image_base64 = None
        self.difference_grid = None
        self.used_oracle_code_as_skill_not_available = False


                

        # instantiate players:
        self.player_a = InstructionGiver(self.model_a, "A", )
        self.player_a_type = self.player_a.get_player_type()
        self.add_player(self.player_a, initial_context=self.prompt_a["reuse"])

        self.player_b = InstructionFollower(self.model_b, "B")
        self.player_b_type = self.player_b.get_player_type()
        self.add_player(self.player_b, initial_prompt=self.prompt_b["reuse"])

        self.genresponse = {"reuse": []}
        self.genboard = {"reuse": None}
        self.gen_board_cells = {"reuse_cells": None}

        self.gamedata = {
            "boardinfo": self.boards,
            "reuse_input_data": self.reuse_data,
            "reuse_genresponse": self.reuse_details,

            "n_turns": self.n_turns,
            "play_turns": None,

            "n_turns_reuse": self.reuse_data["max_task_turns"],

            "play_turns_reuse": self.play_turns_reuse,
            "play_turns_total": self.play_turns_total,

            "use_skills": self.use_skills,
            "existing_skills_filename": self.existing_skills_filename,
            "existing_skills": self.avail_skills,
            "skills_code": self.avail_skills_code,
            "use_oracle_code": self.use_oracle_code,
            "used_oracle_code_as_skill_not_available": self.used_oracle_code_as_skill_not_available,
            "skillandtargetcellsdiscrep": self.skillandtargetcellsdiscrep,

            "reuse_success": self.reuse_success,
            "loss_reason": None,
        }      

        self.boardgen = self.board_generator(self.boards)

    def board_generator(self, boards: Dict):
        for variant, board in boards.items():
            yield  variant, board

        #yield from simple_boards
        #yield from regular_boards


    def _prepare_boardinfo(self, goal: str=None) -> Tuple[bool, str]:
        try: 
            variant, self.board_info = next(self.boardgen)
        except StopIteration:
            variant = None
            self.board_info = None


        if self.board_info is None:
            logger.info("No more boards available from the generator.")
            return False, variant

        if 'rows' in self.board_info and 'cols' in self.board_info:
            self.board_info["size"] = {"rows": self.board_info['rows'], "cols": self.board_info['cols']}
        else:
            self.board_info["size"] = {"rows": 8, "cols": 8}

        logger.info(f"Preparing board info for variant: {variant} with skill: {self.board_info['combo_name']}")

        self.genboard[variant] = None
        self.reuse_skills = []
        self.player_grid_match_status = False
        self.player_grid = None
        self.player_occupied_cells = None
        self.player_a_goal = goal
        self.gtcode = None
        self.gt_usage = None
        #self.gt_occupied_cells = None
        self.optimfunc_occupied_cells = None
        self.mt_inst_occupied_cells = None
        self.gen_image_base64 = None
        
        if self.use_oracle_code:
            gtcode_function = self.board_info["code"]["single_turn"]["function"]
        else:
            if self.avail_skills_code and self.board_info["combo_name"] in self.avail_skills_code:
                gtcode_function = self.avail_skills_code[self.board_info["combo_name"]]
            else:
                #Some error has happened - abort the game
                logger.error(f"Ground truth code for the skill {self.board_info['combo_name']} is not available. Defaulting to oracle code.")
                gtcode_function = self.board_info["code"]["single_turn"]["function"]
                self.used_oracle_code_as_skill_not_available = True
        self.gt_usage = self.board_info["code"]["single_turn"]["usage"]

        self.gtcode = {"function": gtcode_function, "usage": self.gt_usage}
        self.gamedata["used_gtcode_for_validation"] = self.gtcode

        #player_a_goal will be set to goal that was prepared during instance creation (send as input to this function). This is okay as long as we have only one game. If multiple games (reconstruct, reuse,repeat) clubbed together then it will be a problem.
        if self.player_a_goal is None:
            logger.info(f"Player A goal from reuse data is None, preparing ASCII representation from gtcode for variant {variant}.")
            skill_name = self.board_info["combo_name"]
            colors = self.board_info["colors"] if "colors" in self.board_info else None
            repeat_locations = [self.board_info["repeat_locations"]] if "repeat_locations" in self.board_info else None
            target_board_rep, goldboard, *_ = self.prepare_ascii_rep.get_ascii_representation_rb(self.board_info["size"], self.gtcode, skill_name, colors, repeat_locations)
            #ascii_rep_board, board_rep = self.prepare_ascii_rep.get_ascii_representation(self.gtcode, self.board_info["size"])
            self.player_a_goal = target_board_rep#ascii_rep_board #Layer-wise representation
        logger.info(f"Player A Goal:Prepared ASCII representation for variant {variant}:\n{self.player_a_goal}")

        #self.gt_occupied_cells = self.prepare_ascii_rep.get_occupied_cells(self.gtcode, self.board_info["size"])
        if self.skillandtargetcellsdiscrep:
            self.optimfunc_occupied_cells = self.prepare_ascii_rep.get_occupied_cells(self.gtcode, self.board_info["size"])
            if self.optimfunc_occupied_cells:
                self.optimfunc_occupied_cells = {k: [list(x) for x in v] for k, v in self.optimfunc_occupied_cells.items()}
            else:
                logger.info(f"Optim function occupied cells is empty for variant {variant} with skill {self.board_info['combo_name']}.GTCode:\n{self.gtcode}")
        
        self.gt_occupied_cells = {k: [list(x) for x in v] for k, v in self.gt_occupied_cells.items()}

        self.board_info["locations"] = {"row": self.board_info["x"][0]+1, "col": self.board_info["y"][0]+1}
        self.board_info["funcusage"] = self.gt_usage
        return True, variant
    
    def _set_current_task_context(self, optim_step: bool=False) -> Dict:
        status, variant = self._prepare_boardinfo(self.reuse_data["goal"])
        if not status:
            #raise GameError("No board info available to start the game.")
            logger.error("No board info available to start the game.")
            return None
        logger.info(f"Starting game with variant: {variant} for the combo_name: {self.board_info['combo_name']}")

        #self.max_task_turns = self.config_max_turns
        self.variant = variant
        self.current_task = "reuse"
        self.max_task_turns = self.reuse_data["max_task_turns"]
        self.gt_image_base64 = self.reuse_data["gt_image_base64"]
        self.empty_board_base64 = self.reuse_data["empty_board_base64"]                
        #p1_data = self.prompt_a["reuse"] + "\n" + self.reuse_data["details"]
        p1_data = self.reuse_data["details"]

        #self.current_task_turns = 0
        return p1_data

    def _on_before_game(self) -> None:
        """Actions to perform before starting the game (mandatory)."""

        p1_data = self._set_current_task_context(optim_step=False)
        if p1_data is None:
            logger.error(f"Game setup failed")
            self.aborted = True
            return        
        prompt_message = self.prompt_a[self.current_task]+"\n"+str(p1_data)

        if self.player_a_type == "human" and self.use_images_in_human_prompts:
            #Add GT image filename in html format
            safe_text = escape(prompt_message+"\n\nReference Images are given below:\n\n")
            data_uri_shapes = self.shapes_references_base64
            data_uri_1 = self.gt_image_base64#f"data:image/png;base64,{self.gt_image_base64}"
            data_uri_2 = self.empty_board_base64#f"data:image/png;base64,{self.empty_board_base64}"
            #p1_messages = f"""<div>{safe_text}</div><div style="display:flex; gap:8px; align-items:center;"><img src="{data_uri_1}" #width="200" height="200" /> <img src="{data_uri_2}" width="200" height="200" /></div> """
            p1_messages = f"""<div>{safe_text}</div><img src="{data_uri_shapes}" width="400" height="60"/><div style="display:flex; gap:8px; align-items:flex-start;"><figure style="text-align:center;"><figcaption style="font-size:14px; margin-bottom:4px;">Goal Grid</figcaption><img src="{data_uri_1}" width="350" height="320" /></figure><figure style="text-align:center;"><figcaption style="font-size:14px; margin-bottom:4px;">Current Player Grid</figcaption><img src="{data_uri_2}" width="350" height="320" /></figure></div>"""

            #p1_messages = self.prompt_a[self.current_task]+"\n"
        else:
            p1_messages = self.prompt_a[self.current_task]+"\n"+str(p1_data)

        self.set_context_for(self.player_a, p1_messages)
        #gt_cells, _ = self.prepare_ascii_rep.get_ascii_representation(self.gtcode, self.board_info["size"])
        #logger.info(f"Ground truth cells:\n{gt_cells}")
        logger.info(f"Ground truth Layerwise Representation:\n{self.goal_layerwise_rep}")
        #self.gt_occupied_cells = self.prepare_ascii_rep.get_occupied_cells(self.gtcode, self.board_info["size"])
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
        self.gamedata["play_turns"] = self.play_turns_total#self.current_round
        self.gamedata["reuse_success"] = self.reuse_success
        self.gamedata["play_turns_reuse"] = self.play_turns_reuse
        self.gamedata["play_turns_total"] = self.play_turns_total 
        self.gamedata["reuse_genresponse"] = self.reuse_details
        self.gamedata["used_oracle_code_as_skill_not_available"] = self.used_oracle_code_as_skill_not_available

        self._log_game_end()       

    def _set_pass_turn(self, player: Player, pass_turn) -> None:
        #Set the turn to be passed for a player (game specific).
        logger.info(f"Setting pass turn for player: {player}, pass_turn: {pass_turn}")
        self.set_pass_turn = pass_turn

    def _should_pass_turn(self):
        #Currently not checking for any condition to pass the turn
        logger.info(f"Checking if turn should be passed: {self.set_pass_turn}")
        return self.set_pass_turn       

    def _does_game_proceed(self) -> None:
        """Check if the game loop should continue (game specific)."""
        logger.info(f"Inside _does_game_proceed: Current Round: {self.current_round}, Current_task_turns: {self.current_task_turns}, Max Task Runs: {self.max_task_turns} aborted: {self.aborted}, lose: {self.lose}, success: {self.success}")
        # Determine if the game should proceed. This is also called once initially.
        #if self.current_round < self.n_turns and not self.aborted and not self.lose and not self.success:
        if self.current_task_turns < self.max_task_turns and not self.aborted and not self.lose and not self.success:       
            logger.info(f"Game continues: {self.current_task_turns} < {self.max_task_turns}")
            return True
        else:
            logger.info(f"Game does not proceed further: either max turns reached or game ended with success/lose/aborted.")

        
        if self.aborted:# and self.current_round == self.n_turns:
            #self.reuse_details["play_reuse_turns"] = self.current_task_turns
            self.reuse_aborted = True
            #self.aborted = False
        else:
            if not self.reuse_success:
                self.lose = True
            else:
                self.success = True

        logger.info(f"Game status: {self.success}, {self.lose}, {self.aborted}")

        if self.success:
            action_type = "info"
            action_content = "The game is successful; reuse task completed successfully."

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
        logger.info(f"Advancing game with parsed response: player = {player}, parsed_response = {parsed_response}, current_turn = {self.current_turn}, current_task_turns = {self.current_task_turns}")
        if player == self.player_a:
            # The validitiy of the generated code can be checked during scoring.
            # If there are no issues in format, the game is considered successful.
            if parsed_response.get("status") == "success":
                logger.info("Inside advance_game: Player A response validation is success.")
                if self.reuse_success:
                    self.success = True
                else:
                    logger.info(f"Something went wrong.. Reuse Success: {self.reuse_success}")
                    self.lose = True

            elif parsed_response.get("status") == "failure":
                logger.info("Inside advance_game: Player A response validation failed.")
                error = parsed_response.get("error", "Unknown error")
                self.gamedata["loss_reason"] = error                

                action_type = "info"
                action_content = "Player grid does not match the goal grid, but user responded with DONE; lost the game."

                self.log_to_self(action_type, action_content)                

                # Game should be aborted irrespective of the current task because there is some issue in handling user response
                self.aborted = True
                #self.lose = True
                self._set_pass_turn(self.player_a, False)


        elif player == self.player_b:
            if parsed_response["status"] == "failure":
                logger.info(f"Player B response validation failed with error: {parsed_response['error']}")
                # No need to check if retry is allowed here; it is handled in _set_violated_req_count() which triggers parseerror
                self._set_pass_turn(self.player_b, False)
            else:
                self._set_pass_turn(self.player_b, True)
                self.correct_response = True
                # increment current turn:
                self.current_turn += 1
                self.current_task_turns += 1
                # increment complete turns:
                self.complete_turns += 1
                #self.gencode = parsed_response
                # set the current turn's score to 1:
                self.turn_scores[self.current_turn-1] = 1


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
                  'content': error}
        self.log_event(from_='GM', to='GM', action=action)


    def _update_task_progress(self, status_type: str, status_data = None) -> None:
        #if status_data is None:
        #    return
        use_dict = self.reuse_details

        if status_type == "clarification":
            use_dict["used_clarification"] = True
            use_dict["num_clarifications"] += 1

        elif status_type == "clear":
            use_dict["used_clear"] = True
            use_dict["num_clears"] += 1

        elif status_type == "undo":
            use_dict["used_undo"] = True
            use_dict["num_undos"] += 1

        elif status_type == "remove":
            use_dict["used_remove"] = True
            use_dict["num_removes"] += 1

        elif status_type == "move":
            use_dict["used_move"] = True
            use_dict["num_moves"] += 1

        elif status_type == "code_execution" and status_data:
            use_dict["used_move"] = True if status_data.get("move", 0) else False
            use_dict["num_moves"] += status_data.get("move", 0)
            use_dict["used_remove"] = True if status_data.get("remove", 0) else False
            use_dict["num_removes"] += status_data.get("remove", 0)
            use_dict["used_undo"] = True if status_data.get("undo", 0) else False
            use_dict["num_undos"] += status_data.get("undo", 0)
            use_dict["used_clear"] = True if status_data.get("clear", 0) else False
            use_dict["num_clears"] += status_data.get("clear", 0)



    def _set_parsed_req_count(self) -> None:
        # increase the counter of requests that conform to form rules
        self.parsed_request_count += 1  

        # log the event that the string was valid (no strange characters)
        action = {'type': 'valid response', 'content': 'response conforms to rules'}
        self.log_event(from_='GM', to='GM', action=action)

    def _set_violated_req_count(self, error) -> None:
        # increase the counter of requests that violate the move format rule
        self.violated_request_count += 1
        retry = False
        if error:
            if self.num_retry and self.current_retry < self.num_retry:
                logger.info(f"Response did not conform to rules. Reprobing the player. Current retry: {self.current_retry+1}")
                self.log_event(from_='GM', to='GM', action={'type': 'info', 'content': 'Response did not conform to rules. Reprobing the player.'})
                self.current_retry += 1
                retry = True
            else:
                logger.error(f"Response did not conform to rules. Reprobing tries exceeded.")
                self.log_event(from_='GM', to='GM', action={'type': 'info', 'content': 'Response did not conform to rules. Reprobing tries exceeded.'})
                # Not checking for optimization flow here as it is handled separately

        if not retry:
            raise ParseError(error)


    def _validate_playerb_response(self, model_response: str) -> bool:
        parse_b, error = None, None
        logger.info(f"Validating response from player B: {model_response}")


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

    
    def _get_current_difference_grid(self, gen_occupied_cells=None) -> str:
        return self.difference_grid

    def _get_current_filled_grid(self, gen_occupied_cells=None) -> str:
        return self.player_grid

        if self.variant == "simple":
            diff_grid = self.prepare_ascii_rep.get_layer_representation_diff(self.gt_occupied_cells, gen_occupied_cells)
        else:
            repeat_locations = [[skill['x'], skill['y']] for skill in self.reuse_skills]
            diff_grid = self.prepare_ascii_rep.get_layer_representation_diff_rb(self.board_info['repeat_locations'], repeat_locations)


    def _get_shapes_colors_data(self, shapes_list):
        shapes_info = {"W": "w", "N": "n", "S": "s", "L": "bh", "T": "bv", "R": "bh", "B": "bv"}
        colors_info = {"r": "red", "g": "green", "b": "blue", "y": "yellow"}
        shape_data = ""
        colors_list = []
        for tpl in shapes_list:
            if tpl[0] in shapes_info:
                shape_data += shapes_info[tpl[0]]
            if tpl[1] in colors_info:
                colors_list.append(colors_info[tpl[1]])
        return shape_data, colors_list





    def _get_difference_grid(self):
        if self.gt_occupied_cells is None or self.player_occupied_cells is None:
            return None

        difference_grid = self.prepare_ascii_rep.get_layer_representation_diff(self.gt_occupied_cells, self.player_occupied_cells)

        return difference_grid


    def _prepare_playera_turn_response(self, user_instruction):
        p1_data = self.turn_prompt_b[self.current_task]
        use_current_grid = self._get_current_filled_grid()

        #if self.use_skills:
        #    p1_data += f"\n\nLEARNED SKILLS:\n{self.avail_skills}"

        p1_data += f"\n\nUser Instruction:\n{user_instruction}" + f"\n\nCurrent Grid:\n{use_current_grid}"

        return p1_data
    
    def _handle_task_completion(self):
        p2_prompt = None
        use_board = None

        use_board = self.genboard[self.current_task]

        #parse_a = self._validate_reconstruction(use_board)
        parse_a, _ = self._validate_game(use_board)

        if parse_a["status"] == "failure":
            logger.error(f"Task {self.current_task} task validation failed with error: {parse_a['error']}")
        else:
            logger.info(f"Task {self.current_task} validation succeeded.")
            parse_a["status"] = "success"
            p2_prompt = None
            self.reuse_success = True
            self.play_turns_reuse = self.current_task_turns
            self.play_turns_total += self.current_task_turns
            logger.info(f"Reuse task successful., turns taken: {self.current_task_turns}")

            self.current_task_turns = 0
            self.current_retry = 0

        return parse_a, p2_prompt
        
    def _prepare_function_header(self, func_name: str, shapes_list) -> str:
        return f"# Optimized function for placing an object {func_name}\n" + f"# This object consists the following shapes:\n# {shapes_list}\n" 
        
    def _validate_reconstruction(self, genboard) -> Dict:
        """Run the game validation process."""
        logger.info("Starting game validation process")

        result = {"status": "failure", "details": None, "error": None}

        gt_cells = self.gt_occupied_cells#self.prepare_ascii_rep.get_ascii_representation_forvalidation(self.gtcode, self.board_info["size"])
        gen_cells = self.player_occupied_cells#self.prepare_ascii_rep.get_ascii_representation_from_board_forvalidation(genboard, self.board_info["size"])

        #gt_cells = {k: [list(x) for x in v] for k, v in gt_cells.items()}
        #gen_cells = {k: [list(x) for x in v] for k, v in gen_cells.items()}

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

    def _prepare_playerb_turn_response(self, gen_image_base64, difference_grid, clarification, reconstruction_status, cfq):

        if cfq:
            if self.player_a_type == "human":
                if clarification:
                    clarification_text = f'<span style="color: red; font-size: 20px;">{clarification}</span>'
                else:
                    clarification_text = "None"
            else:
                clarification_text = clarification
            ack_text = "None"
        else:
            clarification_text = "None"
            if self.player_a_type == "human":
                if clarification:
                    ack_text = f'<span style="color: red; font-size: 20px;">{clarification}</span>'
                else:
                    ack_text = "None"
            else:
                ack_text = clarification

        #logger.info(f"_prepare_playerb_turn_response: Reconstruction status: {reconstruction_status}, {type(reconstruction_status)}")
        reconstruction_text = reconstruction_status
        if self.player_a_type == "human":
            if reconstruction_status == "True":
                reconst_color = "blue"
            else:
                reconst_color = "red"
            reconstruction_text = f'<span style="color: {reconst_color}; font-size: 15px;f">{reconstruction_status}</span>'

        clarification_header = "Clarification"
        ack_header = "Acknowledgement"
        p2_data = f"\nGoal Grid Layerwise Representation:\n{self.goal_layerwise_rep}\nGoal:{self.player_a_goal}\nDifference grid: {difference_grid}\n{clarification_header}: {clarification_text}\n{ack_header}: {ack_text}\nReconstruction Status: {reconstruction_text}"

        if self.player_a_type == "human":
            #Add GT image filename in html format
            safe_text = p2_data+"\n\nReference Images are given below:\n\n"                
            #if clarification:
            #    safe_text = p2_data+"\n\nReference Images are given below:\n\n"                
            #else:
            #    safe_text = escape(str(p2_data)+"\n\nReference Images are given below:\n\n")
            data_uri_shapes = self.shapes_references_base64
            data_uri_1 = self.gt_image_base64#f"data:image/png;base64,{self.gt_image_base64}"
            if gen_image_base64:
                data_uri_2 = f"data:image/png;base64,{gen_image_base64}"
            else:
                data_uri_2 = self.empty_board_base64#f"data:image/png;base64,{self.empty_player_board_base64}"
            #p1_messages = f"""<div>{safe_text}</div><div style="display:flex; gap:8px; align-items:center;"><img src="{data_uri_1}" #width="200" height="200" /> <img src="{data_uri_2}" width="200" height="200" /></div> """
            p2_message = f"""<div>{safe_text}</div><img src="{data_uri_shapes}" width="400" height="60"/><div style="display:flex; gap:8px; align-items:flex-start;"><figure style="text-align:center;"><figcaption style="font-size:14px; margin-bottom:4px;">Goal Grid</figcaption><img src="{data_uri_1}" width="350" height="320" /></figure><figure style="text-align:center;"><figcaption style="font-size:14px; margin-bottom:4px;">Current Player Grid</figcaption><img src="{data_uri_2}" width="350" height="320" /></figure></div>"""
        else:
            p2_message = p2_data

        return p2_message


    def _prepare_playerb_clarification_response(self, details: str, cfq: bool) -> str:
        #gen_image_base64 = None#self.genresponse[-1]["gen_image_base64"] if self.genresponse and len(self.genresponse) > 0 and "gen_image_base64" in self.genresponse[-1] else None
        gen_image_base64 = self.gen_image_base64
        diff_grid = self.difference_grid#self._get_difference_grid()#self._get_current_filled_grid()
        reconstruction_complete = "True" if self.player_grid_match_status else "False"

        return self._prepare_playerb_turn_response(gen_image_base64, diff_grid, details, reconstruction_complete, cfq)


    def _compare_reuse_function_data(self, details, func_name_list, func_colors_list, func_x_list, func_y_list):

        if "clear(" in details:
            logger.info("Clear function detected in reuse task; resetting reuse skills.")
            self.reuse_skills = []


        for func_name, func_colors, func_x, func_y in zip(func_name_list, func_colors_list, func_x_list, func_y_list):
            self.reuse_skills.append({"name": func_name, "colors": func_colors, "x": int(func_x), "y": int(func_y)})
        return True, None 


    def _execute_playerb_code_response(self, variant: str, details: str) -> Tuple[str, str]:
        if self.genboard[self.current_task] is None:
            board_gen_call = None
        else:
            board_gen_call = copy.deepcopy(self.genboard[self.current_task])

        if not self.use_skills:
            return self.prepare_ascii_rep.execute_generated_response(details, self.board_info["size"], board_gen_call)
        else:
            return self.prepare_ascii_rep.execute_generated_response_skill(details, self.board_info["size"], board_gen_call, self.gtcode["function"])    

    
    def _prepare_level_rep(self, board_gen_call: Dict) -> Tuple[str, List]:
        gen_ascii_rep, gen_occupied_cells = None, None

        logger.info("Calling get_ascii_representation to generate ASCII representation")
        gen_ascii_rep, gen_occupied_cells = self.prepare_ascii_rep.get_ascii_representation_from_board_layers(board_gen_call, self.board_info["size"])

        return gen_ascii_rep, gen_occupied_cells


    def _prepare_playerb_code_response(self, variant: str, details: str) -> str:
        logger.info(f"Current Task: {self.current_task}, Preparing player B code response for execution:\n{details}")
        try:
            board_gen_call, error, code_stats = self._execute_playerb_code_response(variant, details)
            if error:
                logger.error("Error executing generated response.")
                self.genresponse[self.current_task][-1]
                self.genresponse[self.current_task][-1]["code_execution_error"].append({"error": error, "response": {"status": "code", "details": details}})
                self.genresponse[self.current_task][-1]["reconstruction_complete"] = "False"
                self.genresponse[self.current_task][-1]["optim_func_compare_status"] = False

                return None, error

            self._update_task_progress("code_execution", code_stats)
              
            logger.info("Storing generated board state.")
            self.genboard[self.current_task] = copy.deepcopy(board_gen_call)

            logger.info("Preparing ASCII representation from generated board state.")
            gen_ascii_rep, gen_occupied_cells = self._prepare_level_rep(board_gen_call)
            if gen_ascii_rep is None:
                if gen_occupied_cells is None:
                    logger.error("Error generating ASCII representation from generated response.")
                    return None, "Error generating ASCII representation from generated response."

            logger.info(f"Generated ASCII representation:\n{gen_ascii_rep}")

            #logger.info(self.genresponse[self.current_task])

            logger.info("Updating player grid with generated ASCII representation.")
            self.player_grid = gen_ascii_rep
            self.player_occupied_cells = gen_occupied_cells
            if self.player_occupied_cells:
                self.player_occupied_cells = {k: [list(x) for x in v] for k, v in self.player_occupied_cells.items()}

            if self.use_images_in_human_prompts:
                self.gen_image_base64 = self.prepare_ascii_rep.get_image_gen_board(self.genboard[self.current_task], "turn_playerb_board.png")
            else:
                self.gen_image_base64 = None

            self.genresponse[self.current_task][-1]["ascii_rep"] = gen_ascii_rep
            self.genresponse[self.current_task][-1]["occupied_cells"] = gen_occupied_cells
            self.genresponse[self.current_task][-1]["code_stats"] = code_stats
            self.genresponse[self.current_task][-1]["gen_image_base64"] = self.gen_image_base64
            #self.genresponse[self.current_task][-1]["code_execution_error"] = None


            diff_grid = self._get_difference_grid()
            self.difference_grid = diff_grid
            
            result, optim_func_compare_status = self._validate_game(self.genboard[self.current_task])
            self.genresponse[self.current_task][-1]["optim_func_compare_status"] = optim_func_compare_status

            if result["status"] == "success":
                self.player_grid_match_status = True
            reconstruction_complete = "True" if self.player_grid_match_status else "False"
            self.genresponse[self.current_task][-1]["reconstruction_complete"] = reconstruction_complete

            p2_response = self._prepare_playerb_turn_response(self.genresponse[self.current_task][-1]["gen_image_base64"], diff_grid,
                                                              None, reconstruction_complete,False)

            return p2_response, None

        except Exception as e:
            logger.error(f"Error executing generated response: {e}")
            return None, "Error executing generated response from player B."

    def _validate_game(self, genboard) -> Dict:
        """Run the game validation process."""
        logger.info(f"Current Task: {self.current_task}, Starting game validation process")

        result = {"status": "failure", "details": None, "error": None}

        #Always GT cells as the actual target cells: No need to use the gen code exec cells
        gt_cells = self.gt_occupied_cells
        gen_cells = self.player_occupied_cells#self.prepare_ascii_rep.get_ascii_representation_from_board_forvalidation(genboard, self.board_info["size"])


        #gt_cells = {k: [list(x) for x in v] for k, v in gt_cells.items()}
        #gen_cells = {k: [list(x) for x in v] for k, v in gen_cells.items()}

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

        optim_func_compare_status = False
        if self.skillandtargetcellsdiscrep:
            if self.optimfunc_occupied_cells and gen_cells:
                if self.optimfunc_occupied_cells == gen_cells:
                    logger.info("The generated board matches the occupied cells of the optimization function, even though it does not match the ground truth.")
                    optim_func_compare_status = True


        return result, optim_func_compare_status


    def _parse_response(self, player: Player, response: str) -> str:
        # increase the number of API requests:
        self.request_count += 1
        logger.debug(f"Current  task turn: {self.current_task_turns} Overall turns: {self.current_turn}, Received response from player: {player}:{type(response)}\n{response}")
        parse_a = {"status": "failure", "details": None, "error": None}
        if player == self.player_a:
            user_instruction = self._handle_playera_response(response)
            logger.info(f"Parsed response from player A: {user_instruction}")

            action = {'type': 'parsed response',
                    'content': f"User instruction:\n{user_instruction}"}
            self.log_event(from_='GM', to='GM', action=action)

            if user_instruction is None:
                error = "Response from player A does not conform to required format."
                # TODO: The retry count will be unnecessarily incremented even if the violation is from Player A
                # So far, PLayer A never caused a violation, so we havent seen its repurcussions
                self._set_violated_req_count(error)
                parse_a["error"] = error
                return parse_a

            action = {'type': 'get message',
                      'content': user_instruction}
            #self.log_event(from_='Player 1', to='GM', action=action)
            self._set_parsed_req_count()  

            if user_instruction.upper() == "DONE":
                logger.info(f"Current turn: {self.current_task_turns}, Received 'DONE' from player A, validating the game.")
                parse_a, p2_prompt = self._handle_task_completion()
                logger.info(f"output of task completion: parse_a: {parse_a}")
                if parse_a["status"] == "on-going":
                    error = f"Unexpected success status received before completing all tasks. Current task: {self.current_task}"
                    logger.error(error)
                    raise GameError(error)

                elif parse_a["status"] == "success":
                    game_status = "Reuse task completed successfully."
                    logger.info(game_status)
                    action = {'type': 'info', 'content': game_status}
                    self.log_event(from_='GM', to='GM', action=action)

                else:
                    # This means the game is lost and the game master will handle it in advance_game
                    pass

            elif user_instruction.upper() == "SKILL_UNKNOWN":
                logger.info(f"Current turn: {self.current_task_turns}, Received 'SKILL_UNKNOWN' from player A.")
                parse_a["status"] = "failure"
                parse_a["details"] = None
                parse_a["error"] = "The requested skill is unknown or not available for reuse."

            else:
                # Task is still on-going
                self._set_pass_turn(self.player_a, True)
                p2_prompt = self._prepare_playera_turn_response(user_instruction)
                logger.info(f"Setting context for player B with parsed response from player A")
                self.set_context_for(self.player_b, p2_prompt)
                parse_a["status"] = "on-going"
                parse_a["details"] = user_instruction

                self.genresponse[self.current_task].append({"current_turn": self.current_task_turns, "instruction": user_instruction,
                                                            "code_execution_error": []})
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

                self.set_context_for(self.player_b, error_prompt)
                self._set_violated_req_count(error)
                return parse_b

            if parse_b["status"] == "clarification":
                try:
                    self._update_task_progress("clarification")
                    response_b = self._prepare_playerb_clarification_response(parse_b["details"], True)
                    parsed_response = f"Clarification question:\n{parse_b['details']}"
                    self.genresponse[self.current_task][-1]["clarification"] = parse_b["details"]
                    self.genresponse[self.current_task][-1]["response"] = parse_b                

                except GameError as e:
                    error = f"Game setup failed: {str(e)}"
                    logger.error(error)
                    #self._on_game_error(e)
                    #return                    

            elif parse_b["status"] == "acknowledgement":
                try:
                    self._update_task_progress("acknowledgement")
                    response_b = self._prepare_playerb_clarification_response(parse_b["details"], False)
                    parsed_response = f"Clarification question:\n{parse_b['details']}"    
                    self.genresponse[self.current_task][-1]["acknowledgement"] = parse_b["details"]                
                    self.genresponse[self.current_task][-1]["response"] = parse_b                      
                except GameError as e:
                    error = f"Game setup failed: {str(e)}"
                    logger.error(error)
                    #self._on_game_error(e)
            else:
                parsed_response = f"Code to execute:\n{parse_b['details']}"
                response_b, error = self._prepare_playerb_code_response(self.variant, parse_b["details"])
            
            action = {'type': 'parsed response',
                    'content': parsed_response}
            self.log_event(from_='GM', to='GM', action=action)            

            if error:
                action = {'type': 'error while executing parsed response',
                        'content': error}
                self.log_event(from_='GM', to='GM', action=action) 
                error_prompt = f"Error during execution of the code:\n{error}\nPlease correct the code to fix the error."                       

                parse_b = {"status": "failure", "details": None, "error": error}
                self.set_context_for(self.player_b, error_prompt)
                self._set_violated_req_count(error)
            else:
                self._set_parsed_req_count()
                self.current_retry = 0
                self.genresponse[self.current_task][-1]["response"] = parse_b

                logger.info(f"Setting context for player A with parsed response from player B") 
                p1_prompt = self.turn_prompt_a[self.current_task]+"\n"+response_b                   
                self.set_context_for(self.player_a, p1_prompt)

            return parse_b
    
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

    def _log_game_end(self) -> None:
        """Aux to log variables needed for scoring (firstlast specific)"""
        #self.log_key("Played turns", self.current_turn)
        self.log_key("Played turns", self.play_turns_total)
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



class SkillReuseScorer(GameScorer):
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

     
class SkillReuseBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    # copy this, replacing the name of the game master in the return statement
    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> DialogueGameMaster:
        return SkillReuseMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return SkillReuseScorer(self.game_name, experiment, game_instance)      
