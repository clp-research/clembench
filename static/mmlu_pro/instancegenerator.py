import ast
import glob
import importlib
import json
import os
import types
from pathlib import Path
from typing import Dict, List

import yaml
from clemcore.clemgame import GameInstanceGenerator
from datasets import Dataset, load_dataset
from jinja2 import Template
from tqdm import tqdm
import traceback

def to_function(function_path: str):
    # Split into module and function name
    module_name, func_name = function_path.rsplit(".", 1)

    # Import the module
    module = importlib.import_module(module_name)

    # Extract the function object from the module
    return getattr(module, func_name)


class MmluProGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(os.path.dirname(__file__))
        self.dataset = None
        self.fewshow_dataset = None

    def load_yaml_with_include(self, filepath: Path):
        filepath = filepath.resolve()
        base_dir = filepath.parent
        data = yaml.safe_load(filepath.read_text())
        include_path = data.pop('include', None)
        if include_path:
            include_file = base_dir / include_path
            included_data = yaml.safe_load(include_file.read_text())
            data = {**included_data, **data}  # Main file overrides included
        return data

    def _load_dataset_for(self, experiment_config: Dict, reuse=True) -> Dataset:
        """Load the dataset from HuggingFace."""
        dataset_path = experiment_config["dataset_path"]
        dataset_name = experiment_config.get("dataset_name", None)
        split_name = experiment_config.get("test_split", lambda: experiment_config["validation_split"])
        if self.dataset is None or not reuse:
            self.dataset = load_dataset(dataset_path, name=dataset_name, split=split_name)
        return self.dataset

    def to_experiment_name(self, experiment_config: Dict):
        if "dataset_name" in experiment_config:
            return experiment_config["dataset_name"]
        # Use the task name e.g. playschool_eq_bench
        task_name = experiment_config["task"].removeprefix("playschool_")
        return task_name

    def to_prompt(self, experiment_config: Dict):
        if self.fewshow_dataset is None:
            self.fewshow_dataset = load_dataset(experiment_config["dataset_path"],
                                                split=experiment_config["fewshot_split"])
        samples = self.fewshow_dataset
        samples = samples.filter(lambda x: x["category"] == experiment_config["task_alias"])
        input_fn = to_function(experiment_config["fewshot_config"]["doc_to_text"])
        num_samples = experiment_config["num_fewshot"]
        selection = samples.select(range(num_samples))
        initial_prompt = experiment_config["description"] + "\n"
        initial_prompt += "".join([input_fn(sample) for sample in selection])
        initial_prompt += "{{input}}"
        return initial_prompt

    def to_choices(self, experiment_config: Dict) -> str:
        return ast.literal_eval(experiment_config["doc_to_choices"])

    def to_target(self, experiment_config: Dict, sample: Dict) -> Dict:
        """ A list of verifiable instructions along with their kwargs"""
        target_column = experiment_config["doc_to_target"]
        target = sample[target_column]
        return target

    def to_input(self, experiment_config: Dict, sample: Dict) -> str:
        input_fn = to_function(experiment_config["doc_to_text"])
        text = input_fn(sample)
        return text

    def to_regex_pattern(self, experiment_config: Dict) -> str:
        filters = experiment_config["filter_list"][0]
        regex_pattern = filters["filter"][0]["regex_pattern"]
        return regex_pattern

    def on_generate(self, seed: int, **kwargs):
        experiment_config_file_paths = [Path(fp) for fp in glob.glob("resources/configs/*.yaml")]
        experiment_config_file_paths = [fp for fp in experiment_config_file_paths
                                        if not fp.name.startswith("_")]  # ignore auxiliary files
        for experiment_config_file_path in tqdm(experiment_config_file_paths):
            try:
                experiment_config = self.load_yaml_with_include(experiment_config_file_path)
                experiment_name = self.to_experiment_name(experiment_config)
                experiment = self.add_experiment(experiment_name)
                experiment["meta"] = experiment_config
                experiment["initial_prompt"] = self.to_prompt(experiment_config)
                experiment["choices"] = self.to_choices(experiment_config)
                experiment["regex_pattern"] = self.to_regex_pattern(experiment_config)
                samples = self._load_dataset_for(experiment["meta"])
                samples = samples.filter(lambda x: x["category"] == experiment_config["task_alias"])
                for task_id, sample in enumerate(samples):
                    task = self.add_game_instance(experiment, task_id)
                    task["input"] = self.to_input(experiment_config, sample)
                    task["target"] = self.to_target(experiment_config, sample)
            except Exception as e:
                print("Error:", experiment_config_file_path)
                print(e)
                traceback.print_exc()


def main():
    MmluProGameInstanceGenerator().generate("instances.json", seed=42)


if __name__ == "__main__":
    main()
