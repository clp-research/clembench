import ast
import glob
import os
from pathlib import Path
from typing import Optional, Dict, List

import yaml
from clemcore.clemgame import GameInstanceGenerator
from datasets import Dataset, load_dataset
from jinja2 import Template
from tqdm import tqdm


class CLadderGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(os.path.dirname(__file__))

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

    def _load_dataset_for(self, experiment_config) -> Dataset:
        """Load the dataset from HuggingFace."""
        dataset_path = experiment_config["dataset_path"]
        dataset_name = experiment_config["dataset_name"]
        split_name = experiment_config.get("test_split", lambda: experiment_config["validation_split"])
        return load_dataset(dataset_path, name=dataset_name, split=split_name)

    def to_experiment_name(self, experiment_config: Dict):
        if "dataset_name" in experiment_config:
            return experiment_config["dataset_name"]
        # Use the task name e.g. playschool_eq_bench
        task_name = experiment_config["task"].removeprefix("playschool_")
        return task_name

    def to_prompt(self, experiment_config: Dict):
        return experiment_config["doc_to_text"]  # Note: Template with is a {{prompt}} field

    def to_choices(self, experiment_config: Dict) -> List:
        # Render the template to evaluate the expression inside {{ ... }}
        choices_text = Template(experiment_config["doc_to_choice"]).render()
        # Convert the rendered string to a Python list
        return ast.literal_eval(choices_text)

    def to_target(self, experiment_config: Dict, sample: Dict) -> str:
        target_column = experiment_config["doc_to_target"]
        target = sample[target_column]
        return target

    def to_input(self, experiment_config: Dict, sample: Dict) -> Dict:
        return sample["prompt"]

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
                samples = self._load_dataset_for(experiment["meta"])
                for task_id, sample in enumerate(samples):
                    task = self.add_game_instance(experiment, task_id)
                    task["input"] = self.to_input(experiment_config, sample)
                    task["target"] = self.to_target(experiment_config, sample)
            except Exception as e:
                print("Error:", experiment_config_file_path)
                print(e)


def main():
    CLadderGameInstanceGenerator().generate("instances.json", seed=42)


if __name__ == "__main__":
    main()
