from typing import List, Dict

from clemgame.clemgame import Player


class Guesser(Player):
    def __init__(self, model_name: str = None):
        super().__init__(model_name)

    def __call__(self, messages: List[Dict], turn_idx) -> str:
        # assert self.backend in ["human", "llm", "mock"], f"Invalid player role {self.backend}, please check the config file"
        if self.model_name == "human":
            guess_word = input("Enter your guess: ")
            # Repeating the same to maintain similar results w.r.t LLM mode
            return [guess_word], guess_word, guess_word
        return super().__call__(messages, turn_idx)

    def _custom_response(self, messages, turn_idx) -> str:
        # Repeating the same to maintain similar results w.r.t LLM mode
        dummy_response = "guess:<dummy> explanation:<dummy>"
        return dummy_response
