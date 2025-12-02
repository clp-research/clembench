import re
from typing import List, Dict
import json
import openai
from matplotlib.style import context
import dspy
import clemcore.backends as backends
from clemcore.backends import Model
from utils.codeoptimizer import OptimizerSignature
from retry import retry

import logging

logger = logging.getLogger(__name__)


PROVIDER_NAME = "generic_openai_compatible"
API_KEY = "api_key"
BASE_URL = "base_url"

class OptimizerModule(dspy.Module):
    def __init__(self, dspylm, use_history: bool):
        super().__init__()
        self.lm = dspylm
        self.use_history = use_history
        logger.info(f"OptimizerModule: Setting, self.lm:{self.lm}")
        self.predict = dspy.Predict(OptimizerSignature)
        #Do not skip this step, if you do, need to change the input fields in OptimizerSignature        
        self.conv_history = None#dspy.History(messages=[])

    def forward(self, prompt):
        if self.lm is None:
            raise RuntimeError("OptimizerModule.forward called with no LM (mock mode).")          
        with dspy.context(lm=self.lm):
            logger.info(f"[Optimizer] calling predict()")
            #result = self.predict(prompt=prompt, history=self.conv_history)
            result = self.predict(prompt=prompt)
            if self.use_history:
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

        if 'messages' in kwargs:
            for message in kwargs['messages']:
                if 'role' in message and 'content' in message:
                    with open(f"dspy_lm_messages_optim_new_{self.turn +1}.log", "a") as f:
                        f.write(f"Turn: {self.turn+1}, Role: {message['role']}\nContent: {message['content']}\n\n")
        self.turn += 1

        return super().__call__(*args, **kwargs)    
    

class LLMBackEnd:
    def __init__(self, model: Model):
        self.model = model
        logger.info(f"LLMBackEnd initialized with model: {model}, model_spec: {model.model_spec}")

        creds = backends.load_credentials(PROVIDER_NAME)
        api_key = creds[PROVIDER_NAME][API_KEY]
        base_url=creds[PROVIDER_NAME][BASE_URL]

        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

    @retry(tries=3, delay=90, logger=logger)
    def generate(self, prompt: List[Dict]) -> str:
        model_name = f"{self.model.model_spec['model_id']}"
        api_response = self.client.chat.completions.create(model=model_name, messages=prompt,temperature=0.0, max_tokens=32000)
        message = api_response.choices[0].message
        response_text = message.content.strip()
        return response_text



class CodeOptimizer:
    def __init__(self, model: Model, test_variant: str, use_dspy: bool, use_dspy_history: bool):
        # always initialise the Player class with the model_name argument
        # if the player is a program and you don't want to make API calls to
        # LLMS, use model_name="programmatic"

        self.model = model
        self.test_variant = test_variant
        self.use_dspy = use_dspy
        self.use_dspy_history = use_dspy_history
        # a list to keep the dialogue history
        self.history: List = []
        self.current_turn = 0
        self.optimizer_prompt = None
        self.optimizer_module = None
        if self.use_dspy:
            logger.info("Configuring dspy for CodeOptimizer")                
            self._configure_dspy(model)
        else:
            logger.info("CodeOptimizer initialized with LLMBackend")                
            self.llmclient = LLMBackEnd(model)

    def _configure_dspy(self, model: Model):
        creds = backends.load_credentials(PROVIDER_NAME)
        api_key = creds[PROVIDER_NAME][API_KEY]
        base_url=creds[PROVIDER_NAME][BASE_URL]

        logger.info(f"Configuring dspy for model: {model}, model_spec: {model.model_spec}")

        if 'model_id' in model.model_spec:
            model_name = f"openai/{model.model_spec['model_id']}"#"openrouter/qwen/qwen3-coder-flash"
            logger.info(f"Configuring dspy with model_name: {model_name}, base_url: {base_url}")

            self.dspylm = LoggingLM(model_name, api_key=api_key,
                                        api_base=base_url,
                                        temperature=0.0, max_tokens=32000,
                                        cache=False)
            self.optimizer_module = OptimizerModule(self.dspylm, self.use_dspy_history)
            optimizer_prompt = self.optimizer_module.predict.signature.instructions

        else:
            self.dspylm = None
            optimizer_prompt = OptimizerSignature.instructions            

        self.set_optimizer_prompt(optimizer_prompt)

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

    def set_optimizer_prompt(self, prompt: str):
        self.optimizer_prompt = prompt

    def get_optimizer_prompt(self) -> str:
        return self.optimizer_prompt
    
    def get_optimizer_history(self) -> List:
        return self.history
    
    def update_optimizer_history(self, data) -> List:
        self.history.append(data)
  
    
    def run(self, optimizer_input: str) -> str:
        #logger.info(f"CodeOptimizer optimize called with function_name: {function_name}, inst_code_pairs: {inst_code_pairs}")
        model_output = None
        if self.use_dspy and self.optimizer_module:
            logger.info(f"CodeOptimizer run() : probing the model")            
            result = self.optimizer_module(prompt=optimizer_input)
            last_trace = dspy.settings.trace[-1]
            logger.info(last_trace[1])
            #Check how to get the prompt used by dspy
            #self.set_optimizer_prompt(last_trace[1]["prompt"])            
            logger.info(f"Code Optimizer __call__ model_response: {result}")
            model_output = result.optimized_function
        else:
            logger.error("CodeOptimizer.optimize called with LLMBackend")
            optim_prompt = self.get_optimizer_history()
            optim_prompt.extend([{"role": "user", "content": optimizer_input}])
            model_output = self.llmclient.generate(prompt=optim_prompt)


        if self.current_turn == 0:
            self.history.append({"role": "user", "content": optimizer_input})
            self.history.append({"role": "assistant", "content": model_output})
            self.current_turn += 1
        else:
            self.history.append({"role": "user", "content": optimizer_input})
            self.history.append({"role": "assistant", "content": model_output})
            self.current_turn += 1

        metadata = dict(prompt=self.history[-2]["content"], response_object=model_output)                

        return model_output, metadata