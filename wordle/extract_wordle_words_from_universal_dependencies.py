#!/usr/bin/env python3
"""
Script to extract Wordle-suitable words from Universal Dependencies v2.17 treebanks.

This script:
1. Downloads the UD v2.17 data (or uses existing data)
2. Reads .conllu files from each language treebank
3. Extracts NOUN lemmas (base forms) that match Wordle word length
4. Counts their frequencies
5. Saves first 100 most frequent as easy_words.txt
6. Saves remaining words as medium_words.txt
"""

import os
import sys
import tarfile
import zipfile
import subprocess
from collections import Counter
from pathlib import Path
import re

# Configuration
# Dictionary with language code, whether to lowercase nouns, and Wordle word length
LANGUAGES = {
    'ar': {'name': 'Arabic', 'lowercase': True, 'word_length': 5},
    'bg': {'name': 'Bulgarian', 'lowercase': True, 'word_length': 5},
    'cs': {'name': 'Czech', 'lowercase': True, 'word_length': 5},
    'da': {'name': 'Danish', 'lowercase': True, 'word_length': 5},
    'de': {'name': 'German', 'lowercase': False, 'word_length': 5},  # German nouns stay capitalized
    'el': {'name': 'Greek', 'lowercase': True, 'word_length': 5},
    'en': {'name': 'English', 'lowercase': True, 'word_length': 5},
    'es': {'name': 'Spanish', 'lowercase': True, 'word_length': 5},
    'et': {'name': 'Estonian', 'lowercase': True, 'word_length': 5},
    'fi': {'name': 'Finnish', 'lowercase': True, 'word_length': 5},
    'fr': {'name': 'French', 'lowercase': True, 'word_length': 5},
    'ga': {'name': 'Irish', 'lowercase': True, 'word_length': 5},
    'hr': {'name': 'Croatian', 'lowercase': True, 'word_length': 5},
    'hu': {'name': 'Hungarian', 'lowercase': True, 'word_length': 5},
    'it': {'name': 'Italian', 'lowercase': True, 'word_length': 5},
    'lt': {'name': 'Lithuanian', 'lowercase': True, 'word_length': 5},
    'lv': {'name': 'Latvian', 'lowercase': True, 'word_length': 5},
    'mt': {'name': 'Maltese', 'lowercase': True, 'word_length': 5},
    'nl': {'name': 'Dutch', 'lowercase': True, 'word_length': 5},
    'pl': {'name': 'Polish', 'lowercase': True, 'word_length': 5},
    'pt': {'name': 'Portuguese', 'lowercase': True, 'word_length': 5},
    'ro': {'name': 'Romanian', 'lowercase': True, 'word_length': 5},
    'ru': {'name': 'Russian', 'lowercase': True, 'word_length': 5},
    'sk': {'name': 'Slovak', 'lowercase': True, 'word_length': 5},
    'sl': {'name': 'Slovenian', 'lowercase': True, 'word_length': 5},
    'sv': {'name': 'Swedish', 'lowercase': True, 'word_length': 5},
    'tr': {'name': 'Turkish', 'lowercase': True, 'word_length': 5},
    'ur': {'name': 'Urdu', 'lowercase': True, 'word_length': 5}
}

# Mapping from ISO 639-1 codes to UD treebank folder prefixes
LANGUAGE_MAPPING = {
    'ar': 'UD_Arabic',
    'bg': 'UD_Bulgarian',
    'cs': 'UD_Czech',
    'da': 'UD_Danish',
    'de': 'UD_German',
    'el': 'UD_Greek',
    'en': 'UD_English',
    'es': 'UD_Spanish',
    'et': 'UD_Estonian',
    'fi': 'UD_Finnish',
    'fr': 'UD_French',
    'ga': 'UD_Irish',
    'hr': 'UD_Croatian',
    'hu': 'UD_Hungarian',
    'it': 'UD_Italian',
    'lt': 'UD_Lithuanian',
    'lv': 'UD_Latvian',
    'mt': 'UD_Maltese',
    'nl': 'UD_Dutch',
    'pl': 'UD_Polish',
    'pt': 'UD_Portuguese',
    'ro': 'UD_Romanian',
    'ru': 'UD_Russian',
    'sk': 'UD_Slovak',
    'sl': 'UD_Slovenian',
    'sv': 'UD_Swedish',
    'tr': 'UD_Turkish',
    'ur': 'UD_Urdu'
}

# UD download URL
UD_DOWNLOAD_URL = "https://lindat.mff.cuni.cz/repository/server/api/core/items/b4fcb1e0-f4b2-4939-80f5-baeafda9e5c0/allzip?handleId=11234/1-6036"
UD_ZIP_FILE = "ud-v2.17-allzip.zip"
UD_TGZ_FILE = "ud-treebanks-v2.17.tgz"


def download_ud_data():
    """Download UD v2.17 data if not already present."""
    if not os.path.exists(UD_TGZ_FILE):
        if not os.path.exists(UD_ZIP_FILE):
            print(f"Downloading UD v2.17 data...")
            try:
                subprocess.run([
                    'curl', '-L', '-o', UD_ZIP_FILE, UD_DOWNLOAD_URL
                ], check=True)
                print(f"Downloaded: {UD_ZIP_FILE}")
            except subprocess.CalledProcessError as e:
                print(f"Error downloading data: {e}")
                sys.exit(1)
        
        # Extract .tgz from .zip
        print(f"Extracting .tgz file from {UD_ZIP_FILE}...")
        with zipfile.ZipFile(UD_ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"Extracted: {UD_TGZ_FILE}")
        
    return UD_TGZ_FILE


def clean_noun(noun):
    """
    Clean a noun by removing trailing/leading non-alphabetic characters.
    
    Args:
        noun: The noun to clean
        
    Returns:
        Cleaned noun or None if it should be filtered out
    """
    if not noun:
        return None
    
    # Remove leading and trailing non-alphabetic characters (periods, hyphens, etc.)
    # Keep internal punctuation for hyphenated words, apostrophes, etc.
    cleaned = noun.strip()
    
    # Strip leading/trailing non-alphabetic characters
    while cleaned and not cleaned[0].isalnum():
        cleaned = cleaned[1:]
    while cleaned and not cleaned[-1].isalnum():
        cleaned = cleaned[:-1]
    
    if not cleaned:
        return None
    
    # Filter out if it's only numbers
    if cleaned.isdigit():
        return None
    
    # Filter out if it's only punctuation (shouldn't happen after above, but be safe)
    if all(not c.isalnum() for c in cleaned):
        return None
    
    # Filter out very short entries that are likely artifacts (single character)
    if len(cleaned) == 1 and not cleaned.isalpha():
        return None
    
    return cleaned


def check_word_length(word, target_length):
    """
    Check if a word matches the target length for Wordle.
    
    Args:
        word: The word to check
        target_length: The target length for this language
        
    Returns:
        True if word matches the target length, False otherwise
    """
    if not word:
        return False
    
    return len(word) == target_length


def parse_conllu_line(line, lang_code='en'):
    """
    Parse a CoNLL-U format line and extract lemma if it's a NOUN with correct length.
    
    CoNLL-U format:
    ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
    
    Example:
    9   Narren   Narr   NOUN   NN   _   5   conj   _   _
    
    Returns lemma if UPOS is NOUN and matches word length, else None.
    """
    line = line.strip()
    
    # Skip comments and empty lines
    if not line or line.startswith('#'):
        return None
    
    fields = line.split('\t')
    
    # CoNLL-U has 10 fields
    if len(fields) < 10:
        return None
    
    # Skip multi-word tokens (ID contains '-' or '.')
    token_id = fields[0]
    if '-' in token_id or '.' in token_id:
        return None
    
    form = fields[1]  # Actual word
    lemma = fields[2]  # Base form
    POS_tag = fields[3]   # Part of speech
    
    # Return lemma only if it's a NOUN
    if POS_tag != 'NOUN':
        return None
    
    # If lemma is missing (underscore), use the actual word form
    if lemma == '_':
        lemma = form
    
    # Clean the noun
    lemma = clean_noun(lemma)
    if not lemma:
        return None
    
    # Filter out "unknown" for non-English languages
    if lang_code != 'en' and lemma.lower() == 'unknown':
        return None
    
    # Apply lowercase based on language settings
    lang_config = LANGUAGES.get(lang_code, {})
    if lang_config.get('lowercase', True):
        lemma = lemma.lower()
    
    # Check if word matches the target length for Wordle
    target_length = lang_config.get('word_length', 5)
    if not check_word_length(lemma, target_length):
        return None
    
    return lemma


def extract_nouns_from_tgz(tgz_path, lang_code):
    """
    Extract nouns from CoNLL-U files in the .tgz archive for a specific language.
    
    Args:
        tgz_path: Path to the .tgz file
        lang_code: ISO 639-1 language code
        
    Returns:
        Counter object with noun frequencies
    """
    noun_counter = Counter()
    
    # Get the UD folder prefix for this language
    folder_prefix = LANGUAGE_MAPPING.get(lang_code)
    if not folder_prefix:
        print(f"Warning: No mapping found for language code '{lang_code}'")
        return noun_counter
    
    lang_name = LANGUAGES.get(lang_code, {}).get('name', lang_code)
    word_length = LANGUAGES.get(lang_code, {}).get('word_length', 5)
    print(f"Processing {lang_code} ({lang_name} - {folder_prefix}, {word_length}-letter words)...")
    
    try:
        with tarfile.open(tgz_path, 'r:gz') as tar:
            # Get all members that belong to this language's treebanks
            members = [m for m in tar.getmembers() 
                      if folder_prefix in m.name and m.name.endswith('.conllu')]
            
            if not members:
                print(f"  No .conllu files found for {lang_code}")
                return noun_counter
            
            print(f"  Found {len(members)} .conllu file(s)")
            
            for member in members:
                print(f"  Processing: {member.name}")
                
                # Extract and read the file
                f = tar.extractfile(member)
                if f is None:
                    continue
                
                # Read and parse the file
                try:
                    content = f.read().decode('utf-8')
                    lines = content.split('\n')
                    
                    for line in lines:
                        lemma = parse_conllu_line(line, lang_code)
                        if lemma:
                            noun_counter[lemma] += 1
                    
                except UnicodeDecodeError:
                    print(f"  Warning: Could not decode {member.name}")
                    continue
    
    except Exception as e:
        print(f"Error processing {lang_code}: {e}")
    
    return noun_counter


def save_wordle_words(noun_counter, lang_code, output_dir='resources'):
    """
    Save Wordle words to easy_words.txt and medium_words.txt files.
    
    Args:
        noun_counter: Counter object with noun frequencies
        lang_code: ISO 639-1 language code
        output_dir: Base output directory
    """
    # Create output directory
    lang_output_dir = Path(output_dir) / lang_code
    lang_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by frequency (highest to lowest)
    sorted_nouns = noun_counter.most_common()
    
    if not sorted_nouns:
        print(f"  No words to save for {lang_code}")
        return
    
    # Split into easy (first 100) and medium (remaining)
    easy_words = [word for word, count in sorted_nouns[:100]]
    medium_words = [word for word, count in sorted_nouns[100:]]
    
    # Save easy words
    easy_file = lang_output_dir / 'easy_words.txt'
    with open(easy_file, 'w', encoding='utf-8') as f:
        for word in easy_words:
            f.write(f"{word}\n")
    
    print(f"  Saved {len(easy_words)} easy words to: {easy_file}")
    
    # Save medium words (if any)
    if medium_words:
        medium_file = lang_output_dir / 'medium_words.txt'
        with open(medium_file, 'w', encoding='utf-8') as f:
            for word in medium_words:
                f.write(f"{word}\n")
        
        print(f"  Saved {len(medium_words)} medium words to: {medium_file}")
    else:
        print(f"  No medium words to save (total words < 100)")
    
    print(f"  Total unique words: {len(sorted_nouns)}")
    print(f"  Total word occurrences: {sum(noun_counter.values())}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Universal Dependencies v2.17 Wordle Word Extractor")
    print("=" * 80)
    print()
    
    # Download if needed
    data_source = download_ud_data()
    
    print()
    print(f"Processing {len(LANGUAGES)} language(s)...")
    print()
    
    # Process each language
    for lang_code in LANGUAGES.keys():
        print("-" * 80)
        
        noun_counter = extract_nouns_from_tgz(data_source, lang_code)
        
        if noun_counter:
            save_wordle_words(noun_counter, lang_code)
        else:
            print(f"  No words extracted for {lang_code}")
        
        print()
    
    print("=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
