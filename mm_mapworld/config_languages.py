LANG_CONFIG = {
    "en": {
        "DONE": "DONE",
        "MOVE": "GO",
        "DIRECTIONS": ["north", "east", "west", "south"],
        "prompt_dir": "en",
        "DIR_TO_DELTA": {
            "north": (-1, 0),
            "south": (1, 0),
            "east": (0, 1),
            "west": (0, -1)
        },
    },
    "hu": {
        "DONE": "KÉSZ",
        "MOVE": "MENJ",
        "DIRECTIONS": ["észak", "kelet", "nyugat", "dél"],
        "prompt_dir": "hu",
        "DIR_TO_DELTA": {
            "észak": (-1, 0),
            "dél": (1, 0),
            "kelet": (0, 1),
            "nyugat": (0, -1)
        },
    },
}