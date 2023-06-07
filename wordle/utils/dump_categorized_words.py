import clemgame
from clemgame import file_utils

logger = clemgame.get_logger(__name__)


def read_file_contents(filename, file_ext="txt"):
    if file_ext == "csv":
        words_dict = {}
        words_list = file_utils.load_csv(f"resources/{filename}", "wordle")
        if filename == "nytcrosswords.csv":
            for word in words_list:
                words_dict[word[1].lower().strip()] = word[2].lower().strip()
        elif filename == "unigram_freq.csv":
            words_list = words_list[1:]
            for word, freq in words_list:
                words_dict[word.lower().strip()] = freq
        else:
            return words_list
        return words_dict

    elif file_ext == "txt":
        words = file_utils.load_file(f"resources/{filename}", "wordle")
        words_list = words.split("\n")
        words_list = [word.lower().strip() for word in words_list]
        return words_list


def get_freq(word_list, freq_dict, clue_dict):
    """
    Returns a dictionary of words and their frequencies
    Checks if the target word is present in the frequency dictionary and the clue dictionary
    """
    data = {}
    for word in word_list:
        if word in freq_dict and word in clue_dict:
            data[word] = freq_dict[word]
    return data


def write_to_file(data, filename):
    # Write the data to a file
    with open(filename, "w") as fp:
        fp.write("\n".join(data))


def start_word_categorization():
    unigram_freq_file = "unigram_freq.csv"
    all_possible_wordle_words_files = "wordle_recognized_english_words.txt"
    target_words_file = "wordle_target_words.txt"
    clue_file = "nytcrosswords.csv"

    unigram_freq = read_file_contents(unigram_freq_file, file_ext="csv")
    # all_wordle_words = read_file_contents(all_possible_wordle_words_files)
    target_words = read_file_contents(target_words_file)
    clue_words = read_file_contents(clue_file, file_ext="csv")

    target_words_freq_dict = get_freq(target_words, unigram_freq, clue_words)
    # Sort the dictionary by value
    sorted_dict = sorted(target_words_freq_dict.items(), key=lambda x: x[1])

    # Divide the dictionary into 3 parts
    easy_words = sorted_dict[: int(len(sorted_dict) / 3)]
    medium_words = sorted_dict[
        int(len(sorted_dict) / 3) : int(2 * len(sorted_dict) / 3)
    ]
    hard_words = sorted_dict[int(2 * len(sorted_dict) / 3) :]

    print(f"Easy Words:: Length = {len(easy_words)}, {easy_words[:5]}")
    print(f"Medium Words:: Length = {len(medium_words)}, {medium_words[:5]}")
    print(f"Hard Words:: Length = {len(hard_words)}, {hard_words[:5]}")

    easy_words = [word[0] for word in easy_words]
    medium_words = [word[0] for word in medium_words]
    hard_words = [word[0] for word in hard_words]

    write_to_file(
        easy_words,
        "/home/admin/Desktop/codebase/cllm-eval-games/wordle/git_code/untouched_code/101_clembench/games/wordle/resources/easy_words.txt",
    )
    write_to_file(
        medium_words,
        "/home/admin/Desktop/codebase/cllm-eval-games/wordle/git_code/untouched_code/101_clembench/games/wordle/resources/medium_words.txt",
    )
    write_to_file(
        hard_words,
        "/home/admin/Desktop/codebase/cllm-eval-games/wordle/git_code/untouched_code/101_clembench/games/wordle/resources/hard_words.txt",
    )


if __name__ == "__main__":
    start_word_categorization()
