import os
from clemcore.clemgame import GameInstanceGenerator


class CloudGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        # always do this to initialise GameInstanceGenerator
        super().__init__(os.path.dirname(os.path.abspath(__file__)))

    def on_generate(self):
      
        prompt = self.load_template('resources/initial_prompts/prompt.template')
        experiments = {"no": self.load_file('resources/no_clouds.txt').strip('\n').split('\n'),
                       "yes": self.load_file('resources/yes_clouds.txt').strip('\n').split('\n') }

        for exp in experiments.keys():
            experiment = self.add_experiment(exp)
            game_id = 0
            for inst in experiments[exp]:
                game_id = game_id
                instance = self.add_game_instance(experiment, game_id)
                instance["image"] = "resources/images/" + inst
                instance["prompt"] = prompt
                game_id += 1


if __name__ == '__main__':
    # always call this, which will actually generate and save the JSON file
    CloudGameInstanceGenerator().generate()

