LANG_CONFIG = {
    "en": {
        "guesswhat": {
            "QUESTION": "QUESTION:",
            "ANSWER": "ANSWER:",
            "GUESS": "GUESS:",

            "ANSWER_VARIATIONS": [
                "ANSWER: yes",
                "ANSWER: no",
                "ANSWER: Yes.",
                "ANSWER: Yes",
                "ANSWER: No.",
                "ANSWER: No"
            ],

            "PATTERNS": {
                "LETTER": "(does the target word (start with the letter|contain the letter)|does the target word have the letter\\s*[a-z]|is the (first|second|third|fourth|fifth) letter of the target word [a-z])",
                "DIRECT": "^is the target word\\s*(['\"])[^'\"]+?\\1\\s*\\?",
                "LENGTH": "does the target word (have|contain) (more|less|exactly) \\d+ (letters|letter)",
                "SYLLABLE": "does the target word (have|contain) (more than|less than|exactly) (one|two|three|four|five|six|seven|eight|nine|ten|\\d+) (syllable|syllables)",
                "POS": "^is the target word (a|an)\\s+(noun|verb|adjective|adverb|pronoun|preposition|conjunction|interjection)\\s*\\?"
            }
        }
    },
    "hu": {
        "guesswhat": {
            "QUESTION": "KÉRDÉS:",
            "ANSWER": "VÁLASZ:",
            "GUESS": "TIPP:",

            "ANSWER_VARIATIONS": [
                "VÁLASZ: igen",
                "VÁLASZ: nem",
                "VÁLASZ: Igen",
                "VÁLASZ: Nem"
            ],

            "PATTERNS": {
                "LETTER": "(a célszó (betűvel kezdődik|tartalmazza a betűt))",
                "DIRECT": "^a célszó\\s*(['\"])[^'\"]+?\\1\\s*\\?",
                "LENGTH": "a célszó (több|kevesebb|pontosan) \\d+ betűből áll",
                "SYLLABLE": "a célszó (több mint|kevesebb mint|pontosan) \\d+ szótagos",
                "POS": "^a célszó (főnév|ige|melléknév)\\s*\\?"
            }
        }
    }
}
