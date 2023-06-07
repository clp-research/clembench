"""
Randomly generate templates for the private/shared game.

Creates files in ./instances and ./requests
"""
from random import randint

import random

import clemgame
from clemgame.clemgame import GameInstanceGenerator

random.seed(123)
N_INSTANCES = 10

logger = clemgame.get_logger(__name__)
GAME_NAME = "referencegame"

def update_grid(grid:str, position):

    update_row_index, update_cell_index = position

    rows = grid.split('\n')
    updated_grid = ''

    for r_index in range(0, len(rows)):

        if r_index != update_row_index:
            updated_grid += rows[r_index] + '\n'
        else:
            updated_row = ''
            cells = rows[r_index].split(' ')
            for c_index in range(0, len(cells)):
                if c_index != update_cell_index:
                    updated_row += cells[c_index] + ' '
                else:
                    # update the cell with empty symbol
                    updated_row += '▢' + ' '
            updated_grid += updated_row.strip() + '\n'

    return updated_grid.strip()

def manipulate_grid(grid:str, edit_distance:int, existing_grids:list):
    rows = grid.split('\n')

    while True:

        valid_positions = []
        for r_index in range(0, len(rows)):
            cells = rows[r_index].split(' ')
            for c_index in range(0, len(cells)):

                if cells[c_index] != '▢':
                    valid_positions.append((r_index, c_index))

        updated_grid = grid
        for i in range(0, edit_distance):
            random_position = randint(0, len(valid_positions) - 1)
            updated_grid = update_grid(updated_grid, valid_positions[random_position])
            valid_positions.pop(random_position)

        if updated_grid not in existing_grids:
            break

    return updated_grid

def generate_samples(grids, edit_distance, number_of_samples):
    samples = []
    enough_samples_generated = False
    while len(samples) < number_of_samples:

        for target_grid in grids:

            if target_grid.count('X') < 10:
                continue

            second_grid = manipulate_grid(target_grid, edit_distance, [target_grid])
            third_grid = manipulate_grid(target_grid, edit_distance, [target_grid, second_grid])

            samples.append((target_grid, second_grid, third_grid))

            if len(samples) == number_of_samples:
                enough_samples_generated = True
                break

        if enough_samples_generated:
            break
    return samples


def select_grids(grids: dict, difficulty_levels):

    first_grid_level, second_grid_level, third_grid_level = difficulty_levels

    first_grid_index = randint(1, len(grids[first_grid_level]))

    first_grid = grids[first_grid_level][str(first_grid_index)]

    while True:
        second_grid_index = randint(1, len(grids[second_grid_level]))
        second_grid = grids[second_grid_level][str(second_grid_index)]

        if second_grid != first_grid:
            break

    while True:
        third_grid_index = randint(1, len(grids[third_grid_level]))
        third_grid = grids[third_grid_level][str(third_grid_index)]

        if third_grid != first_grid and third_grid != second_grid:
            break

    return first_grid, second_grid, third_grid

def assign_grids(selected_grids: list, target_grid):
    # assing the random one as target one and the rest as 2nd and 3rd grid
    random.shuffle(selected_grids)

    target_grid_index_name = ''

    if selected_grids[0] == target_grid:
        target_grid_index_name = 'first'
    elif selected_grids[1] == target_grid:
        target_grid_index_name = 'second'
    else:
        target_grid_index_name = 'third'

    return selected_grids[0], selected_grids[1], selected_grids[2], target_grid_index_name

class ImageGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(GAME_NAME)

    def on_generate(self):

        player_a_prompt_header = self.load_template(f"resources/initial_prompts/player_a_prompt_header.template")
        player_b_prompt_header = self.load_template(f"resources/initial_prompts/player_b_prompt_header.template")
        grids = self.load_json("resources/grids_v02.json")

        configs = [('hard_grids', 2, 20), ('hard_grids', 4, 20)]

        for config in configs:
            name, edit_distance, number_of_samples = config

            samples = generate_samples(grids[name], edit_distance, number_of_samples)

            experiment = self.add_experiment(name+'_edit_distance_'+str(edit_distance))

            game_counter = 0
            for sample in samples:
                target_grid, second_grid, third_grid = sample

                game_instance = self.add_game_instance(experiment, game_counter)

                game_instance["edit_distance_interval"] = [edit_distance, edit_distance]

                game_instance["player_1_prompt_header"] = player_a_prompt_header.replace('TARGET_GRID',
                                                                                         target_grid).replace(
                    'SECOND_GRID', second_grid).replace('THIRD_GRID', third_grid)
                game_instance['player_1_target_grid'] = target_grid
                game_instance['player_1_second_grid'] = second_grid
                game_instance['player_1_third_grid'] = third_grid

                # randomly shuffle the selected grids
                first_grid, second_grid, third_grid, target_grid_name = assign_grids(
                    [target_grid, second_grid, third_grid],
                    target_grid)
                game_instance["player_2_prompt_header"] = player_b_prompt_header.replace('FIRST_GRID',
                                                                                         first_grid).replace(
                    'SECOND_GRID', second_grid).replace('THIRD_GRID', third_grid)
                game_instance['player_2_first_grid'] = first_grid
                game_instance['player_2_second_grid'] = second_grid
                game_instance['player_2_third_grid'] = third_grid
                game_instance['target_grid_name'] = target_grid_name


if __name__ == '__main__':
    ImageGameInstanceGenerator().generate()
