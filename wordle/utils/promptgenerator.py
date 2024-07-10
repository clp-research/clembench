from typing import Dict, List

from clemgame import get_logger
from games.wordle.utils.prompt_truncator import num_tokens_from_messages

logger = get_logger(__name__)


class PromptGenerator:
    def __init__(
        self,
        system_definition: str,
        guesser_prompt: List,
        guesser_critic_prompt: List,
        target_word_clue: str,
        use_system_message: bool,
        use_clue: bool,
        use_error_explanation: bool,
        use_critic: bool,
        max_token_limit_openai_models: int,
    ):
        self.system_definition = system_definition
        self.guesser_prompt = guesser_prompt
        self.guesser_critic_prompt = guesser_critic_prompt

        self.use_system_message = use_system_message
        self.use_error_explanation = use_error_explanation
        self.use_clue = use_clue
        self.target_word_clue = target_word_clue
        self.use_critic = use_critic
        self.max_token_limit_openai_models = max_token_limit_openai_models

    def create(
        self,
        guess: str = "",
        explanation: str = "",
        guess_feedback: str = "",
        prompt: List = [],
        agreement: str = "",
        agree_explanation: str = "",
    ):
        utterance = []
        if not prompt:
            # Initial Prompt: Add system message and game rules
            if self.use_system_message:
                prompt.extend([{"role": "system", "content": self.system_definition}])
                prompt.extend(self.guesser_prompt)
            else:
                prompt.extend([{"role": "user", "content": self.system_definition}])
                prompt[-1]["content"] = (
                    prompt[-1]["content"] + self.guesser_prompt[0]["content"]
                )
            if self.use_clue:
                # Add a clue
                prompt[-1]["content"] = (
                    prompt[-1]["content"] + "clue: " + self.target_word_clue + "\n"
                )
            utterance = prompt.copy()
        else:
            # Continuing with the previous conversation
            # Add the user input
            utterance.append(
                {
                    "role": "assistant",
                    "content": "guess: "
                    + guess
                    + "\nexplanation: "
                    + explanation
                    + "\n",
                }
            )
            if self.use_critic:
                if agreement == "do_not_use":
                    utterance.append({"role": "user", "content": ""})
                else:
                    # utterance.append({"role": "user", "content": "Here is my opinion for your guess. You can keep your initial guess or change it\
                    #                  after considering the clue and agreement.\nguess_agreement:" + agreement + "\nagreement_explanation:" + agree_explanation+"\n"})
                    utterance.append(
                        {
                            "role": "user",
                            "content": "clue: "
                            + self.target_word_clue
                            + "\n"
                            + "guess_agreement: "
                            + agreement
                            + "\nagreement_explanation: "
                            + agree_explanation
                            + "\n",
                        }
                    )
                # utterance[-1]["content"] = utterance[-1]["content"] + " guess_agreement:<" + agreement + "> agreement_explanation:<" + agree_explanation+">"
                if guess_feedback:
                    # utterance.append({"role": "user", "content": "guess_feedback: "+guess_feedback})
                    utterance[-1]["content"] = (
                        utterance[-1]["content"] + "guess_feedback: " + guess_feedback
                    )
            else:
                utterance.append(
                    {"role": "user", "content": "guess_feedback: " + guess_feedback}
                )
            prompt.extend(utterance)
        return utterance

    def recreate(
        self,
        error: str,
        guess: str,
        explanation: str,
        prompt: List,
        agreement: str = "",
        agree_explanation: str = "",
        for_critic: bool = False,
    ):
        utterance = []

        if for_critic:
            utterance.append(
                {
                    "role": "assistant",
                    "content": "agreement: "
                    + agreement
                    + "\nexplanation: "
                    + agree_explanation
                    + "\n",
                }
            )
            utterance.append(
                {
                    "role": "user",
                    "content": "Provide your response only in this format:\nagreement: yes or no\nexplanation: details\nPlease try again",
                }
            )
            prompt.extend(utterance)
            return utterance

        if self.use_critic:
            # Add a clue
            utterance.append(
                {
                    "role": "assistant",
                    "content": "clue: "
                    + self.target_word_clue
                    + "\n"
                    + "guess: "
                    + guess
                    + "\nexplanation: "
                    + explanation
                    + "\n",
                }
            )
        else:
            # Add the model guessed word
            utterance.append(
                {
                    "role": "assistant",
                    "content": "guess: "
                    + guess
                    + "\nexplanation: "
                    + explanation
                    + "\n",
                }
            )

        # If no feedback to be given, retry the same prompt
        if self.use_error_explanation:
            message_format_details = "Provide your response only in this format:\nguess: word\nexplanation: details\nDo not generate any other text. Please try again."
            # Can add feedback to the prompt
            if error == "INVALID_WORD_LENGTH":
                error_message = (
                    "The guess should have exactly 5 letters. Please try again."
                )
            elif error == "INVALID_WORD":
                error_message = (
                    "The guess should contain only letters. Please try again."
                )
            elif error == "NOT_VALID_ENGLISH_WORD":
                error_message = (
                    "Your guess is not a valid word for this game. Please try again."
                )
            elif error == "INVALID_FORMAT":
                error_message = message_format_details

        else:
            # No error explanation to be given
            error_message = "Guess an English five-letter word.\n" + message_format_details

        utterance.append(
            {
                "role": "user",
                "content": error_message,
            }
        )
        prompt.extend(utterance)
        return utterance

    def create_critic_prompt(
        self,
        guess: str,
        explanation: str,
        guess_feedback: str,
        prompt: List,
        agreement: str,
        agreement_explanation: str,
    ):
        utterance = []
        if not prompt:
            # Add the user input
            utterance.append(
                {
                    "role": "user",
                    "content": self.guesser_critic_prompt[0]["content"]
                    + f"clue: {self.target_word_clue}\n"
                    + f"guess: {guess}\nexplanation: {explanation}\n",
                }
            )
        else:
            #if agreement == "do_not_use":
            #    utterance.append({"role": "user", "content": ""})
            #else:
            utterance.append(
                {
                    "role": "assistant",
                    "content": "agreement: "
                    + agreement
                    + "\nexplanation: "
                    + agreement_explanation
                    + "\n",
                }
            )
            utterance.append(
                {
                    "role": "user",
                    "content": f"clue: {self.target_word_clue}\n"
                    + f"guess: {guess}\nexplanation: {explanation}\n",
                }
            )
            if guess_feedback:
                utterance[-1]["content"] = (
                    utterance[-1]["content"] + "guess_feedback: " + guess_feedback
                )
        prompt.extend(utterance)
        return utterance

    def tailor_prompt(self, prompt, model_name):
        # Guesser/Critic response may take upto 100 tokens for response
        # [4096-100 = 3996-100(because actual model computed tokens may be a bit different) = 3896 tokens is the hardlimit]
        try:
            num_tokens = num_tokens_from_messages(prompt, model_name)
            used_truncation = False
            base_prompt = prompt
            tailor_prompt = []

            while num_tokens > self.max_token_limit_openai_models:
                logger.info(
                    "Current num_tokens {%d} exceeded limit {%d}",
                    num_tokens,
                    self.max_token_limit_openai_models,
                )
                # used_truncation = True
                tailor_prompt = [prompt[0]]
                tailor_prompt.extend(prompt[3:])
                num_tokens = num_tokens_from_messages(tailor_prompt, model_name)
                logger.info("Num Tokens reduced to: %s", num_tokens)
                prompt = tailor_prompt
            base_prompt[:] = prompt

        except (ValueError, NotImplementedError):
            # Model is not supported for counting tokens - return as is
            pass
