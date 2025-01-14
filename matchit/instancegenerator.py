import pandas as pd
from clemcore.clemgame import GameInstanceGenerator
from clemcore.utils import file_utils

GAME_NAME: str = "matchit"
# n instances to be generated
N: int = 10 # max: len(similar_images.csv) = 161, if not using other image pairs
# paths to image pair tables
# PATH_DIFF: str = "resources/image_pairs/different_images.csv"
PATH_DIFF: str = file_utils.file_path("resources/image_pairs/different_images.csv", "../clemgames/matchit")
# PATH_SIM: str = "resources/image_pairs/similar_images.csv"
PATH_SIM: str = file_utils.file_path("resources/image_pairs/similar_images.csv", "../clemgames/matchit")

#how many questions can each player ask?
DEC_TURN: int = 3
# should the players be informed about the number of questions they can ask?
INFO_NUM_QUESTIONS: bool = False

# SEED: int = 42  # seed for old/v1.6 instances
SEED: int = 123

# Flags that have to be at the beginning of each response; are also specified in the prompts
FLAGS: dict[str, str] = {"description": "DESCRIPTION:", "question": "QUESTION:", "answer": "ANSWER:", "decision": "DECISION:"}
SOL_SAME: str = "same image"
SOL_DIFF: str = "different images"

class MatchItInstanceGenerator(GameInstanceGenerator):
    def __init__(self, game_name):
        super().__init__(game_name)
        self.game_name = game_name

    def on_generate(self): 

        differents = pd.read_csv(PATH_DIFF)
        diffs = differents.sample(n = N, random_state = SEED)

        similars = pd.read_csv(PATH_SIM)
        sims = similars.sample(n = N, random_state= SEED)[["url1", "url2"]]
        
        # same images get sampled from the same df as different image, just doubling url1
        sames = differents[~differents.url1.isin(diffs.url1)]
        sams = sames.sample(n = N, random_state= SEED)[["url1"]]
        sams["url2"] = sams[["url1"]]

        # initial_prompt = self.load_template('resources/initial_prompts/initial_prompt.template').replace("$FLAG$", FLAGS["description"])
        initial_prompt = self.load_template(file_utils.file_path('resources/initial_prompts/initial_prompt.template', "../clemgames/matchit")).replace("$FLAG$", FLAGS["description"])

        # desc_intro = self.load_template('resources/initial_prompts/description_introduction.template')
        desc_intro = self.load_template(file_utils.file_path('resources/initial_prompts/description_introduction.template', "../clemgames/matchit"))

        # sentence_num_questions = self.load_template('resources/initial_prompts/info_num_questions.template').replace("$DEC_TURN$", str(DEC_TURN))
        sentence_num_questions = self.load_template(file_utils.file_path('resources/initial_prompts/info_num_questions.template', "../clemgames/matchit")).replace("$DEC_TURN$", str(DEC_TURN))

        if INFO_NUM_QUESTIONS:
            initial_prompt = initial_prompt.replace("$NUM_QUESTIONS$", sentence_num_questions)
        else:
            initial_prompt = initial_prompt.replace("$NUM_QUESTIONS$", "")


        # q_reprompt = self.load_template('resources/initial_prompts/q_reprompt.template').replace("$FLAG$", FLAGS["question"])
        q_reprompt = self.load_template(file_utils.file_path('resources/initial_prompts/q_reprompt.template', "../clemgames/matchit")).replace("$FLAG$", FLAGS["question"])
        # d_reprompt = self.load_template('resources/initial_prompts/d_reprompt.template').replace("$SOL_SAME$", SOL_SAME).replace("$SOL_DIFF$", SOL_DIFF).replace("$FLAG$", FLAGS["decision"])
        d_reprompt = self.load_template(file_utils.file_path('resources/initial_prompts/d_reprompt.template', "../clemgames/matchit")).replace("$SOL_SAME$", SOL_SAME).replace("$SOL_DIFF$", SOL_DIFF).replace("$FLAG$", FLAGS["decision"])
        # a_request = self.load_template('resources/initial_prompts/a_request.template').replace("$FLAG$", FLAGS["answer"])
        a_request = self.load_template(file_utils.file_path('resources/initial_prompts/a_request.template', "../clemgames/matchit")).replace("$FLAG$", FLAGS["answer"])


        experiments = {"same_image": (sams, SOL_SAME), 
                       "similar_image": (sims, SOL_DIFF), 
                       "different_image": (diffs, SOL_DIFF)}


        for exp_name in experiments.keys(): 
            experiment =  self.add_experiment(exp_name)
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
                image_a, image_b = row["url1"], row["url2"]
                if image_a.startswith("http"):
                    instance["image_a"] = image_a
                else:
                    # instance["image_a"] = "games/matchit/resources/images/" + image_a
                    instance["image_a"] = file_utils.file_path('resources/images/', "../clemgames/matchit") + image_a
                if image_b.startswith("http"):
                    instance["image_b"] = image_b
                else:
                    # instance["image_b"] = "games/matchit/resources/images/" + image_b
                    instance["image_b"] = file_utils.file_path('resources/images/', "../clemgames/matchit") + image_b

                instance["decision_turn"] = DEC_TURN

                game_id += 1


if __name__ == "__main__":
    MatchItInstanceGenerator(GAME_NAME).generate()
