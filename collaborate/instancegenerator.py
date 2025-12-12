import random
import string
from typing import Dict
import os

from clemcore.clemgame import GameInstanceGenerator
from utils.prepareasciirep import PrepareASCIIRep


# set the name of the game in the script, as you named the directory
# this name will be used everywhere, including in the table of results
#GAME_NAME = "imageccbts"
# we will create 10 instances for each experiment; vary this as you wish
#N_INSTANCES = 10
# if the generation involves randomness, remember to set a random seed
SEED = 123

LANGUAGE = "en"


class CollaborateInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        # always do this to initialise GameInstanceGenerator
        super().__init__(os.path.dirname(__file__))
        self.game_name = "cocoreuse"
        self.prepare_ascii_rep = PrepareASCIIRep()


    def _prepare_prompts(self, fill_labels) -> Dict[str, str]:

        prompt_files = {
            "prompt_a": f"resources/initial_prompts/{LANGUAGE}/initial_prompt_a.template",
            "turn_prompt_a": f"resources/initial_prompts/{LANGUAGE}/turn_prompt_a.template",
            "prompt_b": f"resources/initial_prompts/{LANGUAGE}/initial_prompt_b.template",
            "prompt_b_optim": f"resources/initial_prompts/{LANGUAGE}/initial_prompt_co.template",
            "turn_prompt_b": f"resources/initial_prompts/{LANGUAGE}/turn_prompt_b.template",
            "turn_prompt_b_optim": f"resources/initial_prompts/{LANGUAGE}/turn_prompt_co.template",
            "turn_prompt_a_newboard": f"resources/initial_prompts/{LANGUAGE}/turn_prompt_a_newboard.template",            
        }

        promptsdict = {}
        for key, file_path in prompt_files.items():
            prompt_template = self.load_template(file_path)
            #if "turn" in key:
            #    promptsdict[key] = prompt_template#self.create_prompt(prompt_template)
            #else:
            promptsdict[key] = self.create_prompt(prompt_template, **fill_labels)

        return promptsdict


    def _prepare_samples_labels(self, varconfig: dict) -> Dict[str, str]:
        samples = {}

        if not varconfig:
            raise ValueError("varconfig is empty or None")

        if "TRAIN_DATA_FILE_NAME" not in varconfig or "TEST_DATA_FILE_NAME" not in varconfig:
            raise ValueError("TRAIN_DATA_FILE_NAME or TEST_DATA_FILE_NAME not found in varconfig")


        #print(varconfig["TRAIN_DATA_FILE_NAME"], varconfig["TEST_DATA_FILE_NAME"])

        #No validation samples for human-written instructions
        if varconfig["TRAIN_DATA_FILE_NAME"] == "":
            train_samples = []
        else:
            train_samples = self.load_json(
                f'resources/data/{LANGUAGE}/{varconfig["TRAIN_DATA_FILE_NAME"]}'
            )

        if varconfig["TEST_DATA_FILE_NAME"] == "":
            test_samples = []
        else:
            test_samples = self.load_json(
                f'resources/data/{LANGUAGE}/{varconfig["TEST_DATA_FILE_NAME"]}'
            )

        samples = {
            "train": train_samples,
            "test": test_samples,
            "prompt_incontext_labels": {"NUM_INCONTEXT_SAMPLES": varconfig["NUM_INCONTEXT_SAMPLES"]}
        }

        return samples


    # define on_generate, a mandatory method
    def on_generate(self, seed: int, **kwargs):
        num_instances = 0

        config = self.load_json(
            f"resources/config/{LANGUAGE}/taskconfig.json")
        
        num_ic_samples = config["collaborate"]["NUM_INCONTEXT_SAMPLES"]
        ic_type = f"fs_{num_ic_samples}" if num_ic_samples > 0 else "zs"
        experiment = self.add_experiment(f"collaborate_{ic_type}")
        samples = self._prepare_samples_labels(config["collaborate"])


        tot_instances = 0
        for sample_id, sample_data in samples["test"].items():
            simple_boards = sample_data["simple"]
            regular_boards = sample_data["regular"]
            metadata = sample_data["metadata"]

            if metadata["num_simple_boards"] != config["max_boards_sb"]:
                print(f"Skipping sample {sample_id} due to less simple boards: {metadata['num_simple_boards']}")
                continue
            if metadata["num_regular_boards"] != config["max_boards_rb"]:
                print(f"Skipping sample {sample_id} due to less regular boards: {metadata['num_regular_boards']}")
                continue

            """
            num_simple_boards = {k: len(simple_boards[k]) for k in simple_boards}
            print(num_simple_boards)
            if any(v != config["max_boards_sb"] for v in num_simple_boards.values()):
                print(f"Skipping sample {sample_id} due to less simple boards: {num_simple_boards}")
                continue

            num_regular_boards = {k: len(regular_boards[k]) for k in regular_boards}
            print(num_regular_boards)
            input()
            if any(v != config["max_boards_rb"] for v in num_regular_boards.values()):
                print(f"Skipping sample {sample_id} due to less regular boards: {num_regular_boards}")
                continue

            input()
            """            
            promptsdict = self._prepare_prompts(samples["prompt_incontext_labels"])            

            instance = self.add_game_instance(experiment, tot_instances)
            instance["data"] = {}
            instance["data"]["simple_boards"] = simple_boards
            instance["data"]["regular_boards"] = regular_boards
            instance["data"]["metadata"] = metadata            
            instance["data"]["max_boards_sb"] = config["max_boards_sb"]
            instance["data"]["max_boards_rb"] = config["max_boards_rb"]
            instance["data"]["use_error_feedback"] = config["use_error_feedback"]
            instance["data"]["use_simple_reuse"] = config["use_simple_reuse"]
            instance["data"]["use_regular_challenging"] = config["use_regular_challenging"]
            instance["data"]["use_regular"] = config["use_regular"]
            instance["data"]["use_dspy_collab"] = config["use_dspy_collab"]
            instance["data"]["use_dspy_collab_history"] = config["use_dspy_collab_history"]
            instance["data"]["use_dspy_collab_retry"] = config["use_dspy_collab_retry"]
            instance["data"]["num_dspy_collab_retry"] = config["num_dspy_collab_retry"]
            instance["data"]["use_sandbox_llm"] = config["use_sandbox_llm"]
            instance["data"]["num_collab_retry"] = config["num_collab_retry"]
            instance["data"]["num_collab_optim_turns"] = config["num_collab_optim_turns"]
            instance["data"]["n_turns"] = config["max_turns"]
            instance["data"]["prompts_dict"] = promptsdict
            tot_instances += 1

            if tot_instances == 1:
                break


        print(f"Generated {tot_instances} instances for experiment collaborate_{ic_type}")



    # an additional method, specific for our example
    def create_prompt(self, prompt: str, **kwargs) -> str:
        """Replace a prompt template with slot values."""
        #print(prompt,"\n")
        #print(kwargs)
        #input()
        text = string.Template(prompt).substitute(**kwargs)
        return text


if __name__ == "__main__":
    random.seed(SEED)
    # always call this, which will actually generate and save the JSON file
    CollaborateInstanceGenerator().generate(seed=SEED)
