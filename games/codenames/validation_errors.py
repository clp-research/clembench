from enum import Enum
from .constants import CLUEGIVER, GUESSER

class ValidationErrorTypes(str, Enum):
    RAMBLING_ERROR = "rambling error"
    PREFIX_ERROR = "prefix error"
    WRONG_NUMBER_OF_GUESSES = "wrong number of guesses"
    NO_CORRECT_GUESS = "no correct guess"
    INVALID_GUESS = "invalid guess"
    RELATED_CLUE_ERROR = "clue is morphologically related to word on the board"
    TOO_FEW_TEXT = "answer only contained one line"
    CLUE_CONTAINS_SPACES = "clue contains spaces"
    CLUE_CONTAINS_INVALID_CHARACTERS = "clue contains non-alphabetical characters"
    GUESS_CONTAINS_INVALID_CHARACTERS = "guess contains non-alphabetical characters"
    CLUE_CONTAINS_NUMBER_OF_TARGETS = "clue line contains the number of targets"
    CLUE_ON_BOARD = "clue is word on board"
    INVALID_TARGET = "target is invalid"

# general class
class ValidationError(Exception):
    def __init__(self, player, error_type, utterance, message="Response does not follow the rules and is hence invalid."):
        super().__init__(message)
        self.player = player
        self.type = error_type
        self.utterance = utterance
        self.message = message
        
    def get_dict(self):
        return {"player": self.player,
                "type": self.type,
                "utterance": self.utterance,
                "message": self.message}

class PrefixError(ValidationError):
    def __init__(self, player, utterance, message, prefix):
        super().__init__(player, ValidationErrorTypes.PREFIX_ERROR, utterance, message)
        self.prefix = prefix

    def get_dict(self):
        result = super().get_dict()
        result["prefix"] = self.prefix
        return result

# Guesser errors

class MissingGuessPrefix(PrefixError):
    def __init__(self, utterance, prefix):
        message = f"Your guesses did not start with the correct prefix ({prefix})."
        super().__init__(GUESSER, utterance, message, prefix)

class GuesserRamblingError(ValidationError):
    def __init__(self, utterance):
        message = f"Your answer contained more than one line, please only give one round of guesses on one line."
        super().__init__(GUESSER, ValidationErrorTypes.RAMBLING_ERROR, utterance, message)

class WrongNumberOfGuessesError(ValidationError):
    def __init__(self, utterance, guesses, number_of_allowed_guesses):
        message = f"You made too many guesses ({len(guesses)}). You are only allowed to make {number_of_allowed_guesses} guesses!"
        super().__init__(GUESSER, ValidationErrorTypes.WRONG_NUMBER_OF_GUESSES, utterance, message)

        self.guesses = guesses
        self.number_of_allowed_guesses = number_of_allowed_guesses

    def get_dict(self):
        result = super().get_dict()
        result["guesses"] = self.guesses
        result["number of allowed guesses"] = self.number_of_allowed_guesses
        return result

class NoCorrectGuessError(ValidationError):
    def __init__(self, utterance, guesses, remaining_words):
        message = f"All of your guesses are incorrect and are not present anymore. You can only choose words as guesses from the remaining words and at least one guess has to be correct."
        super().__init__(GUESSER, ValidationErrorTypes.NO_CORRECT_GUESS, utterance, message)
        self.guesses = guesses
        self.remaining_words = remaining_words

    def get_dict(self):
        result = super().get_dict()
        result["guesses"] = self.guesses
        result["remaining words"] = self.remaining_words
        return result
    
class GuessContainsInvalidCharacters(ValidationError):
    def __init__(self, utterance, guess):
        message = f"Guessed word '{guess}' contains invalid characters, only put your target word, no other characters around it (apart from commas)."
        super().__init__(GUESSER, ValidationErrorTypes.GUESS_CONTAINS_INVALID_CHARACTERS, utterance, message)
        self.guess = guess
    
    def get_dict(self):
        result = super().get_dict()
        result["guess"] = self.guess
        return result
    
class InvalidGuessError(ValidationError):
    def __init__(self, utterance, guess, board):
        message = f"Guessed word '{guess}' was not listed, you can only guess words provided in the lists."
        super().__init__(GUESSER, ValidationErrorTypes.INVALID_GUESS, utterance, message)
        self.guess = guess
        self.board = board

    def get_dict(self):
        result = super().get_dict()
        result["guess"] = self.guess
        result["board"] = self.board
        return result

# Cluegiver errors

class RelatedClueError(ValidationError):
    def __init__(self, utterance, clue, similar_board_word):
        message = f"Your clue '{clue}' is morphologically similar to the word {similar_board_word}, please choose another clue word."
        super().__init__(CLUEGIVER, ValidationErrorTypes.RELATED_CLUE_ERROR, utterance, message)
        self.clue = clue
        self.similar_board_word = similar_board_word

    def get_dict(self):
        result = super().get_dict()
        result["clue"] = self.clue
        result["similar board word"] = self.similar_board_word
        return result

class TooFewTextError(ValidationError):
    def __init__(self, utterance):
        message = f"Your answer did not contain clue and targets on two separate lines."
        super().__init__(CLUEGIVER, ValidationErrorTypes.TOO_FEW_TEXT, utterance, message)

class CluegiverRamblingError(ValidationError):
    def __init__(self, utterance):
        message = f"Your answer contained more than two lines, please only give one clue and your targets on two separate lines."
        super().__init__(CLUEGIVER, ValidationErrorTypes.RAMBLING_ERROR, utterance, message)

class InvalidTargetError(ValidationError):
    def __init__(self, utterance, target, board):
        message = f"Targeted word '{target}' was not listed, you can only target words provided in the lists."
        super().__init__(CLUEGIVER, ValidationErrorTypes.INVALID_TARGET, utterance, message)
        self.target = target
        self.board = board

    def get_dict(self):
        result = super().get_dict()
        result["target"] = self.target
        result["board"] = self.board
        return result

class ClueOnBoardError(ValidationError):
    def __init__(self, utterance, clue, board):
        message = f"Clue '{clue}' is one of the words on the board, please come up with a new word."
        super().__init__(CLUEGIVER, ValidationErrorTypes.CLUE_ON_BOARD, utterance, message)
        self.clue = clue
        self.board = board

    def get_dict(self):
        result = super().get_dict()
        result["clue"] = self.clue
        result["board"] = self.board
        return result
    
class ClueContainsNumberOfTargets(ValidationError):
    def __init__(self, utterance, clue):
        message = f"Only provide the clue word and the target words in the requested format, do not add the number of targets on your own."
        super().__init__(GUESSER, ValidationErrorTypes.CLUE_CONTAINS_NUMBER_OF_TARGETS, utterance, message)
        self.clue = clue

    def get_dict(self):
        result = super().get_dict()
        result["clue"] = self.clue
        return result

class ClueContainsSpaces(ValidationError):
    def __init__(self, utterance, clue):
        message = f"Clue '{clue}' contains spaces and thus is not a single word."
        super().__init__(CLUEGIVER, ValidationErrorTypes.CLUE_CONTAINS_SPACES, utterance, message)
        self.clue = clue

    def get_dict(self):
        result = super().get_dict()
        result["clue"] = self.clue
        return result

class ClueContainsNonAlphabeticalCharacters(ValidationError):
    def __init__(self, utterance, clue):
        message = f"Clue '{clue}' contains non-alphabetical characters, please give a correct single English word."
        super().__init__(CLUEGIVER, ValidationErrorTypes.CLUE_CONTAINS_INVALID_CHARACTERS, utterance, message)
        self.clue = clue

    def get_dict(self):
        result = super().get_dict()
        result["clue"] = self.clue
        return result
    
class MissingCluePrefix(PrefixError):
    def __init__(self, utterance, prefix):
        message = f"Your clue did not start with the correct prefix ({prefix})."
        super().__init__(CLUEGIVER, utterance, message, prefix)

class MissingTargetPrefix(PrefixError):
    def __init__(self, utterance, prefix):
        message = f"Your targets did not start with the correct prefix ({prefix})."
        super().__init__(CLUEGIVER, utterance, message, prefix)