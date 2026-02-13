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
        self.game_name = "cocottroptimizer"
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


    def _prepare_reconst_data(self, data, max_task_turns, empty_board_encoding):

        board_size, gridsize, colors, location, skill_name = self._get_grid_data(data)
        gt_code = data["dialogues"]["single_turn"]["instructions"][0]["<Editor>"]
        ascii_rep_board, goldboard = self.prepare_ascii_rep.get_ascii_representation(gt_code, board_size)
        gt_image_base64, empty_board_base64 = None, None

        if goldboard is not None and empty_board_encoding is not None:
            self.prepare_ascii_rep.saveboard(goldboard, "tempboard.png")
            gt_image_base64 = self.prepare_ascii_rep.encode_image_to_base64("tempboard.png")
            empty_board_base64 = empty_board_encoding

        if gt_image_base64 is None:
            print("GT image generation failed for reconst data preparation.")               
        
        details = f"Grid size: {gridsize}\nObject name: {skill_name}\nColors: {colors}\nLocation: {location}\nGoal:\n{ascii_rep_board}\n"
        return {"details": details, "goal": ascii_rep_board, "gt_code": gt_code, "max_task_turns": max_task_turns,
                "gt_image_base64": gt_image_base64, "empty_board_base64": empty_board_base64}

    
    def _prepare_optim_inst_code_pairs(self, data):
        inst_code_pairs = []
        for turn in data["dialogues"]["multi_turn"]["instructions"]:
            instruction = turn["<Programmer>"]
            code_snippet = turn["<Editor>"]
            inst_code_pairs.append({"instruction": instruction, "code_snippet": code_snippet})
        return inst_code_pairs

    def _prepare_optim_data(self, data, max_task_turns):
        gt_function = data["code"]["single_turn"]["function"]
        gt_usage = data["code"]["single_turn"]["usage"]
        gt_code_st = {"func": gt_function, "usage": gt_usage}

        function_signature = f"Function signature: def {data['combo_name']}(board, colors, x, y)\nFunction Usage: {gt_usage}"
        inst_code_pairs = self._prepare_optim_inst_code_pairs(data)
        details = function_signature + "\n"

        return {"details": details, "optim_gt": gt_code_st, "inst_code_pairs": inst_code_pairs, "max_task_turns": max_task_turns}
    

    def _prepare_reuse_data(self, data, max_task_turns, empty_board_encoding):
        board_size, gridsize, colors, location, skill_name = self._get_grid_data(data)

        gt_code = {"function": data["code"]["single_turn"]["function"], "usage": data["code"]["single_turn"]["usage"]}

        repeat_locations = [location]

        target_board_rep, goldboard, *_ = self.prepare_ascii_rep.get_ascii_representation_rb(board_size, gt_code, skill_name, colors, repeat_locations)

        gt_image_base64, empty_board_base64 = None, None

        if goldboard is not None and empty_board_encoding is not None:
            self.prepare_ascii_rep.saveboard(goldboard, "tempboard11.png")
            gt_image_base64 = self.prepare_ascii_rep.encode_image_to_base64("tempboard11.png")
            empty_board_base64 = empty_board_encoding        

        if gt_image_base64 is None:
            print("GT image generation failed for reuse data preparation.")


        details = f"Grid size: {gridsize}\nObject name: {skill_name}\nColors: {colors}\nLocation: {location}\nGoal:\n{target_board_rep}\n"
        return {"details": details, "goal": target_board_rep, "gt_code": gt_code, "max_task_turns": max_task_turns,
                "gt_image_base64": gt_image_base64, "empty_board_base64": empty_board_base64}



    def _prepare_repeat_data(self, data, max_task_turns, empty_board_encoding):
        repeat_locations = data["repeat_locations"]
        board_size, gridsize, colors, location, skill_name = self._get_grid_data(data)

        gt_code = {"function": data["code"]["function"], "usage": data["code"]["output"]}
        
        target_board_rep, goldboard, *_ = self.prepare_ascii_rep.get_ascii_representation_rb(board_size, gt_code, skill_name, colors, repeat_locations)        
        gt_image_base64, empty_board_base64 = None, None

        if goldboard is not None and empty_board_encoding is not None:
            self.prepare_ascii_rep.saveboard(goldboard, "tempboard.png")
            gt_image_base64 = self.prepare_ascii_rep.encode_image_to_base64("tempboard.png")
            empty_board_base64 = empty_board_encoding

        if gt_image_base64 is None:
            print("GT image generation failed for repeat data preparation.")


        details = f"Grid size: {gridsize}\nObject name: {skill_name}\nColors: {colors}\nLocation: {location}\nGoal:\n{target_board_rep}\nDifference grid: None\nClarification: None"

        return {"details": details, "goal": target_board_rep, "gt_code": gt_code, "max_task_turns": max_task_turns,
                "gt_image_base64": gt_image_base64, "empty_board_base64": empty_board_base64}





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
    
    def _prepare_empty_board_encoding(self, board_size):
        empty_board_image_filepath = "empty_player_board.png"
        empty_player_board = self.prepare_ascii_rep.get_empty_board(board_size)
        self.prepare_ascii_rep.saveboard(empty_player_board, empty_board_image_filepath)
        return self.prepare_ascii_rep.encode_image_to_base64(empty_board_image_filepath)

    def _get_shapes_references_base64(self) -> str:
        return "data:image/png;base64,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"    


    # define on_generate, a mandatory method
    def on_generate(self, seed: int, **kwargs):

        config = self.load_json(
            f"resources/config/{LANGUAGE}/taskconfig.json")
        
        num_ic_samples = config["cocottr"]["NUM_INCONTEXT_SAMPLES"]
        ic_type = f"fs_{num_ic_samples}" if num_ic_samples > 0 else "zs"
        experiment = self.add_experiment(f"cocottr_{ic_type}")
        samples = self._prepare_samples_labels(config["cocottr"])
        reconst_turns = config["num_reconst_turns"]
        reuse_turns = config["num_reuse_turns"]
        repeat_turns = config["num_repeat_turns"]
        optim_turns = config["num_optim_turns"]

        tot_instances = 0
        for sample in samples["test"]:
            promptsdict = self._prepare_prompts(samples["prompt_incontext_labels"])

            #print(sample.keys())
            instance = self.add_game_instance(experiment, tot_instances)

            board_size = {"rows": sample["simple"]['rows'], "cols": sample["simple"]['cols']}
            empty_player_board = self._prepare_empty_board_encoding(board_size)
            
            instance["data"] = {}
            instance["data"]["boards"] = sample

            use_images_in_human_prompts = config["use_images_in_human_prompts"]
            if not use_images_in_human_prompts:
                empty_player_board = None

            instance["data"]["reconst_data"] = self._prepare_reconst_data(sample["simple"], reconst_turns, empty_player_board)
            instance["data"]["optim_data"] = self._prepare_optim_data(sample["simple"], optim_turns)
            instance["data"]["reuse_data"] = self._prepare_reuse_data(sample["simple_reuse"], reuse_turns, empty_player_board)
            instance["data"]["repeat_data"] = self._prepare_repeat_data(sample["regular"], repeat_turns, empty_player_board)
            instance["data"]["use_error_feedback"] = config["use_error_feedback"]
            instance["data"]["use_images_in_human_prompts"] = config["use_images_in_human_prompts"]
            instance["data"]["use_optimizer"] = config["use_optimizer"]
            instance["data"]["num_retry"] = config["num_retry"]
            instance["data"]["num_optim_turns"] = config["num_optim_turns"]
            instance["data"]["use_sandbox_llm"] = config["use_sandbox_llm"]
            instance["data"]["n_turns"] = config["max_turns"]
            instance["data"]["prompts_dict"] = promptsdict
            instance["data"]["shapes_references_base64"] = self._get_shapes_references_base64()
            tot_instances += 1

            if tot_instances == 5:
                break

        print(f"Generated {tot_instances} instances for experiment cocottr_optimizer_{ic_type}")

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
