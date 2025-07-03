import glob
import os
from pathlib import Path
from typing import Optional, Dict

import yaml
from clemcore.clemgame import GameInstanceGenerator
from datasets import Dataset, load_dataset
from jinja2 import Template
from tqdm import tqdm


class BbhFewshowGameInstanceGenerator(GameInstanceGenerator):

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
        split_name = experiment_config["test_split"] \
            if "test_split" in experiment_config else experiment_config["validation_split"]
        return load_dataset(dataset_path, name=dataset_name, split=split_name)

    def to_fewshot_prompt(self, experiment_config: Dict):
        # Add the initial description
        prompt = experiment_config["description"]
        # Add the few shot example with answers
        template = Template(f"{experiment_config['doc_to_text']} {experiment_config['doc_to_target']}")
        prompt += "\n\n".join([template.render(**example)
                               for example in experiment_config["fewshot_config"]["samples"]])
        # Add the final template for the instance to be rendered during the benchmark run
        prompt += f"\n\n{experiment_config['doc_to_text']}"
        return prompt

    def on_generate(self, seed: int, **kwargs):
        experiment_config_file_paths = [Path(fp) for fp in glob.glob("resources/configs/*.yaml")]
        experiment_config_file_paths = [fp for fp in experiment_config_file_paths
                                        if not fp.name.startswith("_")]  # ignore auxiliary files
        for experiment_config_file_path in tqdm(experiment_config_file_paths):
            try:
                experiment_config = self.load_yaml_with_include(experiment_config_file_path)
                experiment = self.add_experiment(experiment_config["dataset_name"])
                experiment["meta"] = experiment_config
                experiment["initial_prompt"] = self.to_fewshot_prompt(experiment_config)
                samples = self._load_dataset_for(experiment["meta"])
                for task_id, sample in enumerate(samples):
                    task = self.add_game_instance(experiment, task_id)
                    task["input"] = sample["input"]
                    task["target"] = sample["target"]
            except Exception as e:
                print("Error:", experiment_config_file_path)
                print(e)


def main():
    BbhFewshowGameInstanceGenerator().generate("instances.json", seed=42)


if __name__ == "__main__":
    main()
