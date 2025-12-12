import os
import random
from typing import List, Dict
from retry import retry

import json
import clemcore.backends as backends
from clemcore.clemgame import Player
from clemcore.backends import Model

from instructiongiver import ProgrammerSignature
from instructionfollower import CobotSignature

os.environ["LITELLM_LOG"] = "DEBUG"
os.environ["LITELLM_VERBOSE"] = "True"

import dspy

import logging
logger = logging.getLogger(__name__)



PROVIDER_NAME = "generic_openai_compatible"
API_KEY = "api_key"
BASE_URL = "base_url"

#dspy.configure_cache(
#    enable_memory_cache=False,
#    enable_disk_cache=False,
#) 

class ProgrammerModule(dspy.Module):
    def __init__(self, dspylm, use_dspy_history: bool):
        super().__init__()
        self.lm = dspylm
        self.use_dspy_history = use_dspy_history
        logger.info(f"ProgrammerModule: Setting, self.lm:{self.lm}")
        self.predict = dspy.Predict(ProgrammerSignature)
        #Do not skip this step, if you do, need to change the input fields in ProgrammerSignature        
        self.conv_history = None#dspy.History(messages=[])

    def forward(self, prompt):
        logger.info(f"ProgrammerModule forward called with prompt length: {len(prompt)}, {type(prompt)}, {self.lm}, {getattr(dspy.settings, 'lm', None)}")
        if self.lm is None:
            raise RuntimeError("ProgrammerModule.forward called with no LM (mock mode).")        
        with dspy.context(lm=self.lm):
            logger.info(f"[GiverModule.forward] dspy.settings.lm INSIDE context: {getattr(dspy.settings, 'lm', None)}, {self.lm},")            
            #result = self.predict(prompt=prompt, history=self.conv_history)
            result = self.predict(prompt=prompt)
            if self.use_dspy_history:
                self.conv_history.messages.append({"prompt": prompt, **result})
            return result
    
class CobotModule(dspy.Module):
    def __init__(self, dspylm, use_dspy_history: bool):
        super().__init__()
        self.lm = dspylm
        self.use_dspy_history = use_dspy_history
        logger.info(f"CobotModule: Setting, self.lm:{self.lm}")
        self.predict = dspy.Predict(CobotSignature)
        #Do not skip this step, if you do, need to change the input fields in CobotSignature
        self.conv_history = None#dspy.History(messages=[])        

    def forward(self, prompt):
        logger.info(f"CobotModule forward called with prompt length: {len(prompt)}, {type(prompt)}, self.lm:{self.lm}")
        if self.lm is None:
            raise RuntimeError("CobotModule.forward called with no LM (mock mode).")       
        with dspy.context(lm=self.lm):        
            #logger.info(f"[CobotModule] calling predict() with prompt:\n{prompt}")        
            #result = self.predict(prompt=prompt, history=self.conv_history)
            result = self.predict(prompt=prompt)
            if self.use_dspy_history:
                self.conv_history.messages.append({"prompt": prompt, **result})
            return result

class LoggingLM(dspy.LM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn = 0
    
    @retry(tries=3, delay=90, logger=logger)     
    def __call__(self, *args, **kwargs):
        # This is where DSPy calls the LM.
        #logger.info("=== DSPy LM CALL ===")
        #logger.info(f"args: {args}")
        #logger.info(f"kwargs: {kwargs}")
        #logger.info("====================")
        """
        if 'messages' in kwargs:
            for message in kwargs['messages']:
                if 'role' in message and 'content' in message:
                    with open(f"dspy_lm_messages_new_{self.turn +1}.log", "a") as f:
                        f.write(f"Turn: {self.turn+1}, Role: {message['role']}\nContent: {message['content']}\n\n")
        self.turn += 1
        """        
        return super().__call__(*args, **kwargs)
    

class InstructionGiver(Player):
    def __init__(self, model: Model, player: str, test_variant: str, use_dspy: bool, use_dspy_history: bool):
        # always initialise the Player class with the model_name argument
        # if the player is a program and you don't want to make API calls to
        # LLMS, use model_name="programmatic"
        logger.info(f"InstructionGiver __init__ with model: {model}, player: {player}, test_variant: {test_variant}")
        super().__init__(model)

        self.player: str = player
        self.test_variant = test_variant
        self.use_dspy = use_dspy
        self.use_dspy_history = use_dspy_history

        # a list to keep the dialogue history
        self.history: List = []
        self.current_turn = 0
        self.dspylm = None
        self.player_module = None

        logger.info("Configuring dspy for InstructionGiver")
        if self.use_dspy:
            self._configure_dspy(model)

    def _configure_dspy(self, model: Model):
        creds = backends.load_credentials(PROVIDER_NAME)
        api_key = creds[PROVIDER_NAME][API_KEY]
        base_url=creds[PROVIDER_NAME][BASE_URL]

        logger.info(f"Configuring dspy for model: {model}, model_spec: {model.model_spec}")

        if 'model_id' in model.model_spec:
            model_name = f"openai/{model.model_spec['model_id']}"#"openrouter/qwen/qwen3-coder-flash"
            logger.info(f"Configuring dspy with base_url: {base_url}")

            #self.dspylm = dspy.LM(model_name, api_key=api_key,
            self.dspylm = LoggingLM(model_name, api_key=api_key,
                                        api_base=base_url,
                                        temperature=0.0, max_tokens=32000,
                                        cache=False)

            #dspy.configure(lm=self.dspylm)
            #dspy.settings.trace = []
            self.player_module = ProgrammerModule(self.dspylm, self.use_dspy_history)
            player_prompt = self.player_module.predict.signature.instructions
            self.set_player_prompt(player_prompt)            
        else:
            self.dspylm = None
            player_prompt = ProgrammerSignature.instructions
            self.set_player_prompt(player_prompt)

    def get_player_type(self) -> str:
        return "others"
        if self.model is None:
            return None

        if isinstance(self.model, backends.CustomResponseModel):
            return "programmatic"

        elif isinstance(self.model, backends.HumanModel) or self.model.model_spec["backend"].lower() == "slurk":
            return "human"
        else:
            return "others"


    def set_player_prompt(self, prompt: str) -> None:
        self.player_prompt = prompt

    def get_player_prompt(self) -> str:
        return self.player_prompt

    def __call__(self, context: Dict, memorize: bool = True) -> str:
        # Override to use dspy module
        if isinstance(self.model, backends.CustomResponseModel):
            response_text = self._custom_response(context)
            metadata = dict(prompt=context['content'], response_object=response_text)
            self.perceive_response(response_text, memorize=memorize, metadata=metadata)
            return response_text     

        if self.use_dspy and self.player_module:
            logger.info(f"InstructionGiver __call__ : probing the model, {type(context['content'])}")#\n{context['content']}")            
            result = self.player_module(prompt=context['content'])
            logger.debug(f"InstructionGiver __call__ model_response: {result}")
            last_trace = dspy.settings.trace[-1]
            self.set_player_prompt(last_trace[1]["prompt"])

            logger.debug(f"InstructionFollower __call__ model_response: {result}")
            response_text = result.instruction

            metadata = dict(prompt=context['content'], response_object=result)
            self.perceive_response(response_text, memorize=memorize, metadata=metadata)

            return response_text
        else:
            logger.info(f"Inside Giver Super Call: {type(context)}")#\n{context}")
            return super().__call__(context, memorize)

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
    def __init__(self, model: Model, player: str, test_variant: str, use_dspy: bool, use_dspy_history: bool):
        # always initialise the Player class with the model_name argument
        # if the player is a program and you don't want to make API calls to
        # LLMS, use model_name="programmatic"
        super().__init__(model)

        self.player: str = player
        self.test_variant = test_variant
        self.use_dspy = use_dspy
        self.use_dspy_history = use_dspy_history

        # a list to keep the dialogue history
        self.history: List = []
        self.current_turn = 0
        self.player_module = None
        if self.use_dspy:
            logger.info("Configuring dspy for InstructionFollower")            
            self._configure_dspy(model)

    def _configure_dspy(self, model: Model):
        #if 'model_id' not in model.model_spec:
        #    #model_name = model.model_spec["model_name"]
        #    #No need to configure dspy if model_id is not available
        #    return

        creds = backends.load_credentials(PROVIDER_NAME)
        api_key = creds[PROVIDER_NAME][API_KEY]
        base_url=creds[PROVIDER_NAME][BASE_URL]

        logger.info(f"Configuring dspy for model: {model}, model_spec: {model.model_spec}")

        if 'model_id' in model.model_spec:
            model_name = f"openai/{model.model_spec['model_id']}"#"openrouter/qwen/qwen3-coder-flash"
            logger.info(f"Configuring dspy with model_name: {model_name}, api_base: {base_url}")

            self.dspylm = LoggingLM(model_name, api_key=api_key,
                                        api_base=base_url,
                                        temperature=0.0, max_tokens=32000,
                                        cache=False)
            #dspy.configure(lm=self.dspylm)
            logger.info(f"Setting, self.dspylm:{self.dspylm}")
            self.player_module = CobotModule(self.dspylm, self.use_dspy_history)
            player_prompt = self.player_module.predict.signature.instructions

        else:
            self.dspylm = None
            player_prompt = CobotSignature.instructions
        self.set_player_prompt(player_prompt)

    def get_player_type(self) -> str:
        return "others"        
        if self.model is None:
            return None

        if isinstance(self.model, backends.CustomResponseModel):
            return "programmatic"

        elif isinstance(self.model, backends.HumanModel) or self.model.model_spec["backend"].lower() == "slurk":
            return "human"
        else:
            return "others"


    def set_player_prompt(self, prompt: str):
        self.player_prompt = prompt

    def get_player_prompt(self) -> str:
        return self.player_prompt

    def __call__(self, context: Dict, memorize: bool = True) -> str:
        # Override to use dspy module
        if isinstance(self.model, backends.CustomResponseModel):
            response_text = self._custom_response(context)

            metadata = dict(prompt=context['content'], response_object=response_text)
            self.perceive_response(response_text, memorize=memorize, metadata=metadata)

            return response_text


        if self.use_dspy and self.player_module:
            logger.info(f"InstructionFollower __call__ : probing the model")
            result = self.player_module(prompt=context['content'])
            last_trace = dspy.settings.trace[-1]
            self.set_player_prompt(last_trace[1]["prompt"])

            logger.debug(f"InstructionFollower __call__ model_response: {result}")
            response_text = result.player_response

            metadata = dict(prompt=context['content'], response_object=result)
            self.perceive_response(response_text, memorize=memorize, metadata=metadata)

            return response_text
        else:
            # TODO: If human is acting as Instruction Follower, need to format the response to JSON so that GM can parse it
            # Currently human response contains a text or a code snippet. Need regex to find out the response type and create JSON accordingly
            logger.info(f"Inside Follower Super Call: {type(context)}")#\n{context}")            
            return super().__call__(context, memorize)

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
