# Taboo Game Word Lists

This directory contains word lists for the Taboo game in 28 languages (EU-24 + Russian + Urdu + Arabic + Maltese).

## Overview

Each language has a set of target words with 3 related "taboo" words that players cannot use when describing the target word. The word lists are split into two difficulty levels:
- **High frequency**: Top 50 most common nouns with relations
- **Low frequency**: Additional nouns for extended gameplay

## Directory Structure

```
taboo/
├── README.md                                    # This file
└── resources/                                   # Final taboo word lists
    ├── ar/
    │   ├── high_frequency_taboo_words.json
    │   └── low_frequency_taboo_words.json
    ├── bg/
    │   ├── high_frequency_taboo_words.json
    │   └── low_frequency_taboo_words.json
    └── ... (28 languages total)
```

## Supported Languages

- **EU-24**: bg, cs, da, de, el, en, es, et, fi, fr, ga, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, sk, sl, sv
- **Additional**: ar (Arabic), ru (Russian), tr (Turkish), ur (Urdu)

## Data Format

Each JSON file contains an array of word entries with this structure:

```json
{
  "target_word": "winter",
  "related_word": ["cold", "snow", "season"],
  "target_word_stem": "winter",
  "related_word_stem": ["cold", "snow", "season"]
}
```

- `target_word`: The word players must describe
- `related_word`: Array of 3 taboo words that cannot be used
- `target_word_stem`: Stem/lemma form of the target word
- `related_word_stem`: Stem/lemma forms of the related words

## Curation Process

### Automated Curation (25 languages)

For most languages (bg, cs, da, de, el, en, es, et, fi, fr, ga, hu, it, lt, lv, nl, pl, pt, ro, ru, sk, sl, sv, tr, ur), the curation process is fully automated:

1. **Noun Extraction** (`translate_target_words.py`)
   - translates "safe_nouns.json" to the target language"

2. **Relation Extraction** (`taboo_extract_relations_from_conceptnet.py`)
   - Loads frequent nouns from CSV files
   - Merges with nouns from Universal Dependencies treebanks
   - Queries ConceptNet 5.7 for semantic relations
   - Selects 3 related words per target using multiple relation types:
     - `/r/RelatedTo` - General semantic relatedness
     - `/r/Synonym` - Synonyms
     - `/r/IsA` - Category membership
     - `/r/HasA` - Part-whole relations
     - `/r/PartOf` - Whole-part relations
     - `/r/UsedFor` - Purpose/function
     - `/r/CapableOf` - Capabilities
     - `/r/Antonym` - Opposites
     - `/r/DerivedFrom` - Word origins
     - `/r/SimilarTo` - Similarity
     - `/r/MadeOf` - Material composition
   - Samples highest-weight words from different relations
   - Generates final JSON files in `resources/{lang}/`

3. Computes both embedding and edit distance similarity (current and lemma) between target and taboo words (`taboo_compute_similarity.py`)
4. Sample candidates from existing word relations and their similarities (`taboo_sample_game_words.py`)
5. Generate sample where there aren't enough samples (`generate_taboo_list_with_gemini.py`), these languages have less than 20 samples:

LANGUAGES = {
"sl": "Slovenian",
"sk": "Slovak",
'ga': 'Irish',
'mt': 'Maltese',
'da': 'Danish',
'lv': 'Latvian',
'ar': 'Arabic',
'hr': 'Croatian',
'bg': 'Bulgarian',
'et': 'Estonian',
'lt': 'Lithuanian'
}
