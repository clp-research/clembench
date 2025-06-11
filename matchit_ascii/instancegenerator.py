import os
import pandas as pd
import json
from typing import Dict

from clemcore.clemgame import GameInstanceGenerator

# n instances to be generated
N: int = 10  # max = len(similar_grid_1) = 27, if not using other grid pairs
# paths to image pair tables
# PATH_PAIRS: str = "resources/grid_pairs/grid-pairs.csv"
PATH_PAIRS: str = os.path.join("resources", "grid_pairs", "grid-pairs.csv")
# PATH_GRIDS: str = "resources/grid_pairs/grids_matchit.json"
PATH_GRIDS: str = os.path.join("resources", "grid_pairs", "grids_matchit.json")

# how many questions can each player ask?
DEC_TURN: int = 3
# should the players be informed about the number of questions they can ask?
INFO_NUM_QUESTIONS: bool = False

# SEED: int = 42  # seed for old/v1.6 instances
# SEED: int = 123 # v2.0

# Flags that have to be at the beginning of each response; are also specified in the prompts
FLAGS: Dict = {"description": "DESCRIPTION:", "question": "QUESTION:", "answer": "ANSWER:", "decision": "DECISION:"}
SOL_SAME: str = "same grid"
SOL_DIFF: str = "different grids"


class MatchItAsciiInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed: int, **kwargs):
        print("current path:", self.game_path)
        # df = pd.read_csv(PATH_PAIRS, index_col = 0)
        df = pd.read_csv(os.path.join(self.game_path, PATH_PAIRS), index_col=0)
        diffs = df[df.category == "different_grid"].sample(n=N, random_state=seed)
        sims1 = df[df.category == "similar_grid_1"].sample(n=N, random_state=seed)
        sims2 = df[df.category == "similar_grid_2"].sample(n=N, random_state=seed)
        sams = df[df.category == "same_grid"].sample(n=N, random_state=seed)

        # with open("resources/grid_pairs/grids_matchit.json") as file:
        with open(os.path.join(self.game_path, PATH_GRIDS)) as file:
            grid_dict = json.load(file)

        initial_prompt = (self.load_template('resources/prompts/initial_prompt.template')
                          .replace("$FLAG$", FLAGS["description"]))
        q_reprompt = (self.load_template('resources/prompts/q_reprompt.template')
                      .replace("$FLAG$", FLAGS["question"]))
        d_reprompt = (self.load_template('resources/prompts/d_reprompt.template')
                      .replace("$SOL_SAME$", SOL_SAME)
                      .replace("$SOL_DIFF$", SOL_DIFF)
                      .replace("$FLAG$", FLAGS["decision"]))
        a_request = (self.load_template('resources/prompts/a_request.template')
                     .replace("$FLAG$", FLAGS["answer"]))
        desc_intro = self.load_template('resources/prompts/description_introduction.template')

        if INFO_NUM_QUESTIONS:
            sentence_num_questions = (self.load_template('resources/prompts/info_num_questions.template')
                                      .replace("$DEC_TURN$", str(DEC_TURN)))
            initial_prompt = initial_prompt.replace("$NUM_QUESTIONS$", sentence_num_questions)
        else:
            initial_prompt = initial_prompt.replace("$NUM_QUESTIONS$", "")

        experiments = {"same_grid": (sams, SOL_SAME),
                       "similar_grid_1": (sims1, SOL_DIFF),
                       "similar_grid_2": (sims2, SOL_DIFF),
                       "different_grid": (diffs, SOL_DIFF)}

        for exp_name in experiments.keys():
            experiment = self.add_experiment(exp_name)
            game_id = 0
            experiment["initial_prompt"] = initial_prompt
            experiment["q_reprompt"] = q_reprompt
            experiment["d_reprompt"] = d_reprompt
            experiment["a_request"] = a_request
            experiment["desc_intro"] = desc_intro
            experiment["flags"] = FLAGS
            experiment["solution"] = experiments[exp_name][1]
            if experiment["solution"] == SOL_SAME:
                experiment["wrong_solution"] = SOL_DIFF
            else:
                experiment["wrong_solution"] = SOL_SAME

            for index, row in experiments[exp_name][0].iterrows():
                instance = self.add_game_instance(experiment, game_id)
                id_a, id_b = row["grid1"], row["grid2"]
                instance["grid_a"] = grid_dict[str(id_a)]
                instance["grid_b"] = grid_dict[str(id_b)]

                instance["decision_turn"] = DEC_TURN

                game_id += 1


if __name__ == "__main__":
    MatchItAsciiInstanceGenerator().generate(seed=123)
