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

        result = ["⬜"] * len(target_word)
        target_list = list(target_word)
        # Step 1: Mark greens (🟩) and track available letters in the target
        for i in range(len(target_word)):
            if guessed_word[i] == target_list[i]:
                result[i] = "🟩"
                target_list[i] = None  # Mark as used to prevent duplicate processing

        # Step 2: Mark yellows (🟨), ensuring we don't overcount letters
        for i in range(len(target_word)):
            if result[i] == "🟩":
                continue  # Skip already marked greens
            if guessed_word[i] in target_list:  # Check if letter is available in the target
                result[i] = "🟨"
                target_list[target_list.index(guessed_word[i])] = None  # Mark as used to avoid overmarking

        # Prepare the response
        for i in range(len(target_word)):
            if result[i] == "🟩":
                response += guessed_word[i] + "<green> "
            elif result[i] == "🟨":
                response += guessed_word[i] + "<yellow> "
            else:
                response += guessed_word[i] + "<red> "
        return response.strip()


if __name__ == "__main__":
    '''
    tests = [('stoop', 'boost'), ('round', 'broad'), ('hello', 'world'), ('spare', 'spree'), ('sweep', 'clean')]
    tests = [('spree', 'spare'), ('spree', 'smaer'), ('spree', 'seats'), ('spree', 'elbow'),
             ('spree', 'erase'), ('spree', 'cheer'),  ('spree', 'sweep'), ('spree', 'spear'),
             ('spree', 'speer'), ('spree', 'spree')]
    tests = [('store', 'proud'), ('large', 'stare'), ('clean', 'damps'), ('shore', 'heals'), ('first', 'bunks')]
    '''
    tests = [('strap', 'spree'), ('strap', 'start'), ('strap', 'fluff'), ('strap', 'hello'),
             ('strap', 'error'), ('strap', 'zappy'), ('strap', 'smash'), ('strap', 'banal'),
             ('strap', 'strap')]

    for target_word, guessed_word in tests:
        guess = GuessValidator(target_word)
        print(f"Target: {target_word}, Guessed: {guessed_word}, Result: {guess.validate(guessed_word)}")
