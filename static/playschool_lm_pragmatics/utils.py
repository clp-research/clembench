import ast

def doc_to_choice(doc) -> list:
    choices = doc["randomized_option_order"]
    choices = [str(i) for i in ast.literal_eval(choices)]
    return choices

def doc_to_target(doc) -> int:
    correct_answer = doc['randomized_true_answer']
    options = ast.literal_eval(doc['randomized_option_order'])
    return options.index(correct_answer)