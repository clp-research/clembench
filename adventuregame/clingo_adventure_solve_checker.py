"""
Script to check correct ASP encoding for adventure solving.
Was extensively used to create the ASP encodings for action definitions.
"""

from clingo.control import Control

# init clingo controller in 'all model output' mode:
ctl = Control(["0"])

# load ASP encoding:
with open("adventure_solve_asp_example.lp", 'r', encoding='utf-8') as lp_file:
    example_lp = lp_file.read()

# add encoding to clingo controller:
ctl.add(example_lp)

# ground the encoding:
ctl.ground()

# report successful grounding:
print("Grounded!")
# for the complexity of these text adventures, even extensive ones, grounding should be finished in under a minutes
# if it does, the encoding likely needs improvements - some early versions took more than a minute, while the current
# version, as used for adventure solving and shown in adventure_solve_asp_example.lp, takes milliseconds

# solve encoding, collect produced models:
models = list()
with ctl.solve(yield_=True) as solve:
    for model in solve:
        # print("model:", model)
        model_split = model.__str__().split()
        models.append(model_split)
    satisfiable = solve.get()
    if satisfiable == "SAT":
        print("Adventure can be solved.")
    elif satisfiable == "UNSAT":
        print("Adventure can NOT be solved.")
# the last model in the models list is the optimal solution

def convert_action_to_tuple(action: str):
    action_splice = action[9:-1]
    action_split = action_splice.split(",")
    action_split[0] = int(action_split[0])
    action_tuple = tuple(action_split)
    return action_tuple


def convert_mutable_to_tuple(mutable: str):
    mutable_split = mutable.split("_t(")
    mutable_type = mutable_split[0]
    mutable_pred = mutable_split[1][:-1].split(",")
    if len(mutable_pred) == 3:
        mutable_list = [int(mutable_pred[0]), mutable_type, mutable_pred[1], mutable_pred[2]]
    elif len(mutable_pred) == 2:
        mutable_list = [int(mutable_pred[0]), mutable_type, mutable_pred[1]]
    mutable_tuple = tuple(mutable_list)
    return mutable_tuple


# show actions and mutable facts sorted by turn for all models:
for model_idx, model in enumerate(models):
    actions_list: list = list()
    mutable_list: list = list()
    turns_list: list = list()
    print(f"\nModel {model_idx}")
    for fact in model:
        if "action_t" in fact:
            actions_list.append(convert_action_to_tuple(fact))
        elif "_t(" in fact:
            mutable_list.append(convert_mutable_to_tuple(fact))
        if "turn" in fact:
            turns_list.append(int(fact.split("(")[1][:-1]))

    turns_list.sort()
    # as turns limit actions, the effect of the last action is seen at the 'turn after the last turn':
    turns_list.append(turns_list[-1]+1)

    actions_list.sort(key=lambda turn: turn[0])
    # filter out specific action types for inspection:
    actions_filter = ["go", "open", "close", "take", "put"]

    mutable_list.sort(key=lambda turn: turn[0])
    # filter out specific mutable types for inspection:
    mutable_filter = ["in", "on", "open", "closed"]
    # filter out specific entities for inspection:
    mutable_entities = ["sandwich1"]

    for turn in turns_list:
        for mutable in mutable_list:
            if mutable[1] in mutable_filter and mutable[0] == turn:
                if mutable_entities:
                    if mutable[2] in mutable_entities:
                        print("mutabl", mutable)
                else:
                    print("mutabl", mutable)
        for action in actions_list:
            if action[1] in actions_filter and action[0] == turn:
                print("action", action)
        print()

