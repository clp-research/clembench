import os
import random
from typing import List, Dict
from retry import retry

import json
import clemcore.backends as backends
from clemcore.clemgame import Player
from clemcore.backends import Model


import logging
logger = logging.getLogger(__name__)

    

class InstructionGiver(Player):
    def __init__(self, model: Model, player: str,):
        # always initialise the Player class with the model_name argument
        # if the player is a program and you don't want to make API calls to
        # LLMS, use model_name="programmatic"
        logger.info(f"InstructionGiver __init__ with model: {model}, player: {player}")
        super().__init__(model)

        self.player: str = player

        # a list to keep the dialogue history
        self.history: List = []
        self.current_turn = 0
        self.dspylm = None
        self.player_module = None


    def get_player_type(self) -> str:
        if self.model is None:
            return None

        if isinstance(self.model, backends.CustomResponseModel):
            return "programmatic"

        elif isinstance(self.model, backends.HumanModel) or self.model.model_spec["backend"].lower() == "slurk":
            return "human"
        else:
            return "others"



    # implement this method as you prefer, with these same arguments
    def _custom_response(self, context) -> str:
        """Return a mock message with the suitable output format."""
        if self.player == 'A':
            if self.current_turn == 0:
                self.current_turn += 1
                return "place a yellow washer in 1st row, 1st column"
            else:
                return "DONE"


class InstructionFollower(Player):
    def __init__(self, model: Model, player: str):
        # always initialise the Player class with the model_name argument
        # if the player is a program and you don't want to make API calls to
        # LLMS, use model_name="programmatic"
        super().__init__(model)

        self.player: str = player

        # a list to keep the dialogue history
        self.history: List = []
        self.current_turn = 0
        self.player_module = None


    def get_player_type(self) -> str:
        if self.model is None:
            return None

        return "others"

        if isinstance(self.model, backends.CustomResponseModel):
            return "programmatic"

        elif isinstance(self.model, backends.HumanModel) or self.model.model_spec["backend"].lower() == "slurk":
            return "human"
        else:
            return "others"


    # implement this method as you prefer, with these same arguments
    def _custom_response(self, context) -> str:
        """Return a mock message with the suitable output format."""
        if self.player == 'B':
            if self.current_turn == 0:
                self.current_turn += 1
                code_output = json.dumps({"status": "code", "details": "put(board, shape='washer', color='yellow', x=0, y=0)"})
                return code_output
            else:
                clarification_output = json.dumps({"status": "clarification", "details": "Isn't the mock test over?"})
                return clarification_output
