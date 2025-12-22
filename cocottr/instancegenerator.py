import random
import string
from typing import Dict
import os
import json

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


class CocoTTRInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        # always do this to initialise GameInstanceGenerator
        super().__init__(os.path.dirname(__file__))
        self.game_name = "cocottr"
        self.prepare_ascii_rep = PrepareASCIIRep()


    def _prepare_prompts(self, fill_labels) -> Dict[str, str]:

        prompt_files = {
            "prompt_a_reconst": f"resources/initial_prompts/{LANGUAGE}/initial_prompt_a_reconst.template",
            "turn_prompt_a_reconst": f"resources/initial_prompts/{LANGUAGE}/turn_prompt_a_reconst.template",
            "prompt_b_reconst": f"resources/initial_prompts/{LANGUAGE}/initial_prompt_b_reconst.template",
            "turn_prompt_b_reconst": f"resources/initial_prompts/{LANGUAGE}/turn_prompt_b_reconst.template",

            "prompt_b_optim": f"resources/initial_prompts/{LANGUAGE}/initial_prompt_b_optim.template",
            "turn_prompt_b_optim": f"resources/initial_prompts/{LANGUAGE}/turn_prompt_b_optim.template",

            "prompt_a_reuse": f"resources/initial_prompts/{LANGUAGE}/initial_prompt_a_reuse.template",
            "turn_prompt_a_reuse": f"resources/initial_prompts/{LANGUAGE}/turn_prompt_a_reuse.template",
            "prompt_b_reuse": f"resources/initial_prompts/{LANGUAGE}/initial_prompt_b_reuse.template",            
            "turn_prompt_b_reuse": f"resources/initial_prompts/{LANGUAGE}/turn_prompt_b_reuse.template",

            "prompt_a_repeat": f"resources/initial_prompts/{LANGUAGE}/initial_prompt_a_repeat.template",
            "turn_prompt_a_repeat": f"resources/initial_prompts/{LANGUAGE}/turn_prompt_a_repeat.template",
            "prompt_b_repeat": f"resources/initial_prompts/{LANGUAGE}/initial_prompt_b_repeat.template",
            "turn_prompt_b_repeat": f"resources/initial_prompts/{LANGUAGE}/turn_prompt_b_repeat.template",
        }

        promptsdict = {}
        for key, file_path in prompt_files.items():
            prompt_template = self.load_template(file_path)
            #if "turn" in key:
            #    promptsdict[key] = prompt_template#self.create_prompt(prompt_template)
            #else:
            promptsdict[key] = self.create_prompt(prompt_template, **fill_labels)

        return promptsdict


    def _get_grid_data(self, data):

        if 'rows' in data and 'cols' in data:
            rows = data['rows']
            cols = data['cols']
            gridsize = f"{rows}x{cols}"
        else:
            rows = 8
            cols = 8
            gridsize = f"{rows}x{cols}"

        board_size = {"rows": rows,"cols": cols}
        colors = data["colors"]
        if "x" in data and "y" in data:
            location = [data["x"][0], data["y"][0]]
        else:
            location = data["repeat_locations"]
        skill_name = data["combo_name"]
        return board_size, gridsize, colors, location, skill_name


    def _prepare_reconst_data(self, data):

        board_size, gridsize, colors, location, skill_name = self._get_grid_data(data)
        gt_code = data["dialogues"]["single_turn"]["instructions"][0]["<Editor>"]
        ascii_rep_board, _ = self.prepare_ascii_rep.get_ascii_representation(gt_code, board_size)
        
        details = f"Grid size: {gridsize}\nObject name: {skill_name}\nColors: {colors}\nLocation: {location}\nGoal:\n{ascii_rep_board}\n"
        return {"details": details, "goal": ascii_rep_board, "gt_code": gt_code}

    
    def _prepare_optim_inst_code_pairs(self, data):
        inst_code_pairs = []
        for turn in data["dialogues"]["multi_turn"]["instructions"]:
            instruction = turn["<Programmer>"]
            code_snippet = turn["<Editor>"]
            inst_code_pairs.append({"instruction": instruction, "code_snippet": code_snippet})
        return inst_code_pairs

    def _prepare_optim_data(self, data):
        gt_function = data["code"]["single_turn"]["function"]
        gt_usage = data["code"]["single_turn"]["usage"]
        gt_code_st = {"func": gt_function, "usage": gt_usage}

        function_signature = f"Function signature: def {data['combo_name']}(board, colors, x, y)\nFunction Usage: {gt_usage}"
        inst_code_pairs = self._prepare_optim_inst_code_pairs(data)
        details = function_signature + "\n"

        return {"details": details, "optim_gt": gt_code_st, "inst_code_pairs": inst_code_pairs}
    

    def _prepare_reuse_data(self, data):
        board_size, gridsize, colors, location, skill_name = self._get_grid_data(data)

        gt_code = {"function": data["code"]["single_turn"]["function"], "usage": data["code"]["single_turn"]["function"]}

        repeat_locations = [location]

        target_board_rep, *_ = self.prepare_ascii_rep.get_ascii_representation_rb(board_size, gt_code, skill_name, colors, repeat_locations)

        details = f"Grid size: {gridsize}\nObject name: {skill_name}\nColors: {colors}\nLocation: {location}\nGoal:\n{target_board_rep}\n"
        return {"details": details, "goal": target_board_rep, "gt_code": gt_code}



    def _prepare_repeat_data(self, data):
        repeat_locations = data["repeat_locations"]
        board_size, gridsize, colors, location, skill_name = self._get_grid_data(data)

        gt_code = {"function": data["code"]["function"], "usage": data["code"]["output"]}
        
        target_board_rep, *_ = self.prepare_ascii_rep.get_ascii_representation_rb(board_size, gt_code, skill_name, colors, repeat_locations)        

        details = f"Grid size: {gridsize}\nObject name: {skill_name}\nColors: {colors}\nLocation: {location}\nGoal:\n{target_board_rep}\nDifference grid: None\nClarification: None"

        return {"details": details, "goal": target_board_rep, "gt_code": gt_code}





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

        config = self.load_json(
            f"resources/config/{LANGUAGE}/taskconfig.json")
        
        num_ic_samples = config["cocottr"]["NUM_INCONTEXT_SAMPLES"]
        ic_type = f"fs_{num_ic_samples}" if num_ic_samples > 0 else "zs"
        experiment = self.add_experiment(f"cocottr_{ic_type}")
        samples = self._prepare_samples_labels(config["cocottr"])

        tot_instances = 0
        for sample in samples["test"]:
            promptsdict = self._prepare_prompts(samples["prompt_incontext_labels"])

            print(sample.keys())
            instance = self.add_game_instance(experiment, tot_instances)
            instance["data"] = {}
            instance["data"]["boards"] = sample
            instance["data"]["reconst_data"] = self._prepare_reconst_data(sample["simple"])
            instance["data"]["optim_data"] = self._prepare_optim_data(sample["simple"])
            instance["data"]["reuse_data"] = self._prepare_reconst_data(sample["simple_reuse"])#self._prepare_reuse_data(sample["simple_reuse"])
            instance["data"]["repeat_data"] = self._prepare_repeat_data(sample["regular"])
            instance["data"]["use_error_feedback"] = config["use_error_feedback"]
            instance["data"]["use_images_in_human_prompts"] = config["use_images_in_human_prompts"]
            instance["data"]["use_optimizer"] = config["use_optimizer"]
            instance["data"]["num_retry"] = config["num_retry"]
            instance["data"]["num_optim_turns"] = config["num_optim_turns"]
            instance["data"]["use_sandbox_llm"] = config["use_sandbox_llm"]
            instance["data"]["n_turns"] = config["max_turns"]
            instance["data"]["prompts_dict"] = promptsdict
            tot_instances += 1

            if tot_instances == 12:
                break

        print(f"Generated {tot_instances} instances for experiment cocottr_{ic_type}")

        tot_instances = 0



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
    CocoTTRInstanceGenerator().generate(seed=SEED)        
