from typing import List, Dict, Tuple, Any
from retry import retry

import json
import openai
import backends

logger = backends.get_logger(__name__)

MODEL_GPT_4 = "gpt-4"
MODEL_GPT_35 = "gpt-3.5-turbo"
MODEL_GPT_3 = "text-davinci-003"
SUPPORTED_MODELS = [MODEL_GPT_4, MODEL_GPT_35, MODEL_GPT_3]

NAME = "openai"


class OpenAI(backends.Backend):

    def __init__(self):
        creds = backends.load_credentials(NAME)
        if "organisation" in creds[NAME]:
            openai.organization = creds[NAME]["organisation"]
        openai.api_key = creds[NAME]["api_key"]
        self.chat_models: List = ["gpt-4", "gpt-3.5-turbo"]
        self.temperature: float = -1.

    def list_models(self):
        models = openai.Model.list()
        names = [item["id"] for item in models["data"]]
        names = sorted(names)
        [print(n) for n in names]

    @retry(tries=3, delay=0, logger=logger)
    def generate_response(self, messages: List[Dict], model: str) -> Tuple[str, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: chat-gpt for chat-completion, otherwise text completion
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"
        if model in self.chat_models:
            # chat completion
            prompt = messages
            api_response = openai.ChatCompletion.create(model=model, messages=prompt,
                                                        temperature=self.temperature, max_tokens=100)
            message = api_response["choices"][0]["message"]
            if message["role"] != "assistant":  # safety check
                raise AttributeError("Response message role is " + message["role"] + " but should be 'assistant'")
            response_text = message["content"].strip()
            response = json.loads(api_response.__str__())

        else:  # default (text completion)
            prompt = "\n".join([message["content"] for message in messages])
            api_response = openai.Completion.create(model=model, prompt=prompt,
                                                    temperature=self.temperature, max_tokens=100)
            response = json.loads(api_response.__str__())
            response_text = api_response["choices"][0]["text"].strip()
        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS
