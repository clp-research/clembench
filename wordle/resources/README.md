# Wordle Game Word Lists

This directory contains 5-letter noun word lists for Wordle-style games in 28 languages (EU-24 + Russian + Turkish + Urdu + Arabic).

## Overview

Each language has frequency-ordered noun lists suitable for Wordle gameplay, split into two difficulty levels:
- **Easy words** (`easy_words.txt`): Top 100 most frequent 5-letter nouns
- **Medium words** (`medium_words.txt`): Additional 5-letter nouns for extended gameplay

## Directory Structure

```
wordle/
├── README.md                                    # This file
├── extract_wordle_words_from_universal_dependencies.py  # Extraction script
└── resources/                                   # Word lists by language
    ├── ar/
    │   ├── easy_words.txt
    │   └── medium_words.txt
    ├── bg/
    │   ├── easy_words.txt
    │   └── medium_words.txt
    └── ... (28 languages total)
```

## Supported Languages

- **EU-24**: bg, cs, da, de, el, en, es, et, fi, fr, ga, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, sk, sl, sv
- **Additional**: ar (Arabic), ru (Russian), tr (Turkish), ur (Urdu)

All languages use **5-letter words** for consistency with the classic Wordle format.

## Data Format

Each text file contains one word per line, ordered by frequency (most common first):

```
house
water
party
night
...
```

### File Types

- `easy_words.txt`: First 100 most frequent words (easier to guess)
- `medium_words.txt`: Remaining words (more challenging)

## Extraction Process

The word lists are automatically generated from **Universal Dependencies v2.17** treebanks using the `extract_wordle_words_from_universal_dependencies.py` script:

### 1. Data Source
- Downloads Universal Dependencies v2.17 treebanks (if not present)
- Uses annotated CoNLL-U format files with linguistic information
- Source: [Universal Dependencies Project](https://universaldependencies.org/)

### 2. Noun Extraction
- Parses CoNLL-U files for each language
- Filters for tokens tagged as `NOUN` (part-of-speech)
- Extracts lemmas (base forms) rather than inflected forms
- Example: "houses" → "house", "children" → "child"

### 3. Length Filtering
- Selects only words with exactly 5 letters
- Counts character length after cleaning (removing punctuation)

### 4. Frequency Counting
- Counts occurrences of each unique noun across all treebanks
- More frequent nouns appear multiple times in training data
- Reflects real-world usage patterns

### 5. Word List Generation
- Sorts nouns by frequency (highest to lowest)
- Top 100 → `easy_words.txt`
- Remaining → `medium_words.txt`

### 6. Extract allowed_words.txt 
We used Wikipedia dumps for each language.


## Data Source

**Universal Dependencies v2.17**
- Multilingual annotated corpora
- 146+ treebanks across 122 languages
- CoNLL-U format with POS tags, lemmas, dependencies
- License: CC BY-SA 4.0
- Website: https://universaldependencies.org/

## Usage

### In a Wordle Game

1. Load the appropriate word list for the target language
2. For easier gameplay, use `easy_words.txt`
3. For challenging gameplay, use `medium_words.txt`
4. Select a random word as the target
5. Validate player guesses against the word list

### Example (Python)

```python
import random

# Load easy words for English
with open('../../../../Downloads/conceptnet-words-categories-extractor-main-2/wordle/resources/en/easy_words.txt', 'r',
          encoding='utf-8') as f:
    words = [line.strip() for line in f]

# Pick a random word
target_word = random.choice(words)
print(f"Target word: {target_word}")
```

## Regeneration

To regenerate the word lists:

```bash
cd wordle
python3 extract_wordle_words_from_universal_dependencies.py
```

### Prerequisites

- Python 3.6+
- Internet connection (for first-time UD download)
- ~500MB disk space for UD data

### First Run

The script will automatically:
1. Download Universal Dependencies v2.17 data (~200MB)
2. Extract the treebanks
3. Process all 28 languages
4. Generate word lists in `resources/`
