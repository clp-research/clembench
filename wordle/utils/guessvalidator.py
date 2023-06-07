import random

# from random import choice


class GuessValidator:
    def __init__(self, target_word):
        self.target_word = target_word

    def get_target_word(self):
        return self.target_word

    def validate(self, guessed_word, target_word=""):
        """
        Compare the guessed word with the target word
        Return a list of tuples with the letter and the color

        green - correct letter in correct position
        yellow - correct letter in wrong position
        red - wrong letter
        """
        if not target_word:
            target_word = self.target_word

        response = ""
        # Check if the input word is the target word
        if guessed_word == target_word:
            response = [l + "<green>" for l in guessed_word]
            response = " ".join(response)
            return response

        marked_target_positions = []
        for index in range(len(guessed_word)):
            letter = guessed_word[index]
            if letter in target_word:
                # Check if the letter is in the correct position
                target_index = target_word.find(letter)
                if target_index == -1:
                    # Letter is not in the target word
                    response += letter + "<red> "
                else:
                    while target_index in marked_target_positions:
                        target_index = target_word.find(letter, target_index + 1)
                        if target_index == -1:
                            break

                    if target_index == -1:
                        # No more occurences of the letter in the target word
                        response += letter + "<red> "
                    else:
                        if target_index == index:
                            # Letter is in the target word and in the correct position
                            response += letter + "<green> "
                            marked_target_positions.append(target_index)
                        else:
                            # Letter is in the target word but not in the correct position
                            if guessed_word[target_index] == letter:
                                # There is another occurence of the letter in the guessed word
                                response += letter + "<red> "
                            else:
                                response += letter + "<yellow> "
                                marked_target_positions.append(target_index)
            else:
                # Letter is not in the target word
                response += letter + "<red> "
        return response.strip()
