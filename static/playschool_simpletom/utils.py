def doc_to_choice(doc: dict) -> list[str]:
    """Return all of the accepted answers as choices."""
    choices = doc["choices"]

    # Remove prefixes from all choices
    choices_str = []

    for i, choice_str in enumerate(choices["text"]):
        choices_str.append(f"({choices['label'][i]}): {choice_str}")

    return choices_str

def doc_to_target(doc: dict) -> int:
    choice_labels = doc["choices"]["label"]
    answer = doc["answerKey"]

    return choice_labels.index(answer)

def doc_to_text(doc: dict) -> str:

    choices = "\n".join(doc_to_choice(doc))

    return f"Given the following story, answer the question by giving the correct answer choice, (A) or (B).\nStory: {doc['story']}\nQuestion: {doc['question']}\n{choices}\nAnswer:"
