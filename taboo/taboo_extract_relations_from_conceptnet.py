#!/usr/bin/env python3
"""
Script to extract taboo game word lists from ConceptNet 5.7 using noun frequency lists.

This script:
1. Loads pre-existing noun frequency lists from top_nouns_frequency folder
2. Extracts nouns from Universal Dependencies treebanks
3. Merges both noun lists
4. Finds related words from ConceptNet using multiple relations
5. Samples words with highest weights from different relations
6. Creates two JSON files: high_frequency and low_frequency
"""

import gzip
import csv
import json
import sys
import os
import tarfile
from typing import Dict, List, Set, Tuple
from collections import Counter, defaultdict
from pathlib import Path

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
CONCEPTNET_FILE = 'conceptnet-assertions-5.7.0.csv.gz'
UD_TGZ_FILE = 'ud-treebanks-v2.17.tgz'

# EU-24 + Urdu + Russian + Arabic language codes
TARGET_LANGUAGES = [
    'ar', 'bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr',
    'ga', 'hr', 'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt',
    'ro', 'ru', 'sk', 'sl', 'sv', 'tr', 'ur'
]

# Language mapping from ISO codes to UD folder prefixes
LANGUAGE_MAPPING = {
    'ar': 'UD_Arabic', 'bg': 'UD_Bulgarian', 'cs': 'UD_Czech', 'da': 'UD_Danish',
    'de': 'UD_German', 'el': 'UD_Greek', 'en': 'UD_English',
    'es': 'UD_Spanish', 'et': 'UD_Estonian', 'fi': 'UD_Finnish',
    'fr': 'UD_French', 'ga': 'UD_Irish', 'hr': 'UD_Croatian',
    'hu': 'UD_Hungarian', 'it': 'UD_Italian', 'lt': 'UD_Lithuanian',
    'lv': 'UD_Latvian', 'mt': 'UD_Maltese', 'nl': 'UD_Dutch',
    'pl': 'UD_Polish', 'pt': 'UD_Portuguese', 'ro': 'UD_Romanian',
    'ru': 'UD_Russian', 'sk': 'UD_Slovak', 'sl': 'UD_Slovenian',
    'sv': 'UD_Swedish', 'tr': 'UD_Turkish', 'ur': 'UD_Urdu'
}

# Language-specific settings with character length limits and single_word_check flag
# single_word_check: True for languages that tokenize by space, False for languages like Chinese, Japanese, Arabic
LANGUAGES = {
    'ar': {'lowercase': True, 'min_len': 2, 'max_len': 10, 'single_word_check': False},
    'bg': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'cs': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'da': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'de': {'lowercase': False, 'min_len': 4, 'max_len': 12, 'single_word_check': True},
    'el': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'en': {'lowercase': True, 'min_len': 3, 'max_len': 8, 'single_word_check': True},
    'es': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'et': {'lowercase': True, 'min_len': 3, 'max_len': 12, 'single_word_check': True},
    'fi': {'lowercase': True, 'min_len': 3, 'max_len': 12, 'single_word_check': True},
    'fr': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'ga': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'hr': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'hu': {'lowercase': True, 'min_len': 3, 'max_len': 12, 'single_word_check': True},
    'it': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'lt': {'lowercase': True, 'min_len': 3, 'max_len': 12, 'single_word_check': True},
    'lv': {'lowercase': True, 'min_len': 3, 'max_len': 12, 'single_word_check': True},
    'mt': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'nl': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'pl': {'lowercase': True, 'min_len': 3, 'max_len': 12, 'single_word_check': True},
    'pt': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'ro': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'ru': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'sk': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'sl': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'sv': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'tr': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': True},
    'ur': {'lowercase': True, 'min_len': 3, 'max_len': 10, 'single_word_check': False}
}

# List of ConceptNet relations to use (no weight thresholds - accept all)
VALID_RELATIONS = [
    '/r/Synonym', '/r/RelatedTo', '/r/IsA', '/r/HasA', '/r/PartOf',
    '/r/UsedFor', '/r/CapableOf', '/r/Antonym', '/r/DerivedFrom', 
    '/r/SimilarTo', '/r/MadeOf'
]

# Number of taboo words per target
NUM_TABOO_WORDS = 3

# Maximum words to take from a single relation
MAX_WORDS_PER_RELATION = 2

# High frequency threshold (first N words WITH relations)
HIGH_FREQUENCY_THRESHOLD = 50

OUTPUT_DIR = 'resources'

# Top nouns frequency folder
TOP_NOUNS_FOLDER = 'top_nouns_frequency'

# -------------------------------------------------------------------
# SYSTEM SETUP
# -------------------------------------------------------------------
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)



def clean_noun(noun):
    """Clean a noun by removing non-alphabetic characters."""
    if not noun:
        return None
    
    cleaned = noun.strip()
    while cleaned and not cleaned[0].isalnum():
        cleaned = cleaned[1:]
    while cleaned and not cleaned[-1].isalnum():
        cleaned = cleaned[:-1]
    
    if not cleaned or cleaned.isdigit():
        return None
    if all(not c.isalnum() for c in cleaned):
        return None
    if len(cleaned) == 1 and not cleaned.isalpha():
        return None
    
    return cleaned


def parse_conllu_line(line, lang_code='en'):
    """Parse a CoNLL-U format line and extract lemma if it's a NOUN."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    
    fields = line.split('\t')
    if len(fields) < 10:
        return None
    
    token_id = fields[0]
    if '-' in token_id or '.' in token_id:
        return None
    
    form = fields[1]
    lemma = fields[2]
    pos_tag = fields[3]
    
    if pos_tag != 'NOUN':
        return None
    
    if lemma == '_':
        lemma = form
    
    lemma = clean_noun(lemma)
    if not lemma:
        return None
    
    if lang_code != 'en' and lemma.lower() == 'unknown':
        return None
    
    lang_config = LANGUAGES.get(lang_code, {})
    if lang_config.get('lowercase', True):
        lemma = lemma.lower()
    
    # Check length constraints
    min_len = lang_config.get('min_len', 3)
    max_len = lang_config.get('max_len', 15)
    if len(lemma) < min_len or len(lemma) > max_len:
        return None
    
    return lemma


def load_noun_frequencies(lang_code):
    """Load noun frequencies from the top_nouns_frequency folder."""
    noun_file = Path(TOP_NOUNS_FOLDER) / f"{lang_code}_nouns.csv"
    
    if not noun_file.exists():
        print(f"  Warning: No noun frequency file found for {lang_code}")
        return Counter()
    
    print(f"  Loading noun frequencies from {noun_file}")
    noun_counter = Counter()
    
    try:
        with open(noun_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row['word'].strip()
                try:
                    count = int(row['count'])
                except:
                    count = 1
                
                # Apply language-specific constraints
                lang_config = LANGUAGES.get(lang_code, {})
                if lang_config.get('lowercase', True):
                    word = word.lower()
                
                # Check length constraints
                min_len = lang_config.get('min_len', 3)
                max_len = lang_config.get('max_len', 15)
                if len(word) < min_len or len(word) > max_len:
                    continue
                
                noun_counter[word] += count
        
        print(f"    Loaded {len(noun_counter)} unique nouns, {sum(noun_counter.values())} total frequency")
    except Exception as e:
        print(f"    Error loading noun frequencies: {e}")
    
    return noun_counter


def extract_nouns_from_ud(lang_code):
    """Extract ALL nouns from UD treebanks with frequencies."""
    noun_counter = Counter()
    folder_prefix = LANGUAGE_MAPPING.get(lang_code)
    
    if not folder_prefix:
        return noun_counter
    
    print(f"  Extracting nouns from UD treebanks ({folder_prefix})...")
    
    try:
        with tarfile.open(UD_TGZ_FILE, 'r:gz') as tar:
            members = [m for m in tar.getmembers() 
                      if folder_prefix in m.name and m.name.endswith('.conllu')]
            
            for member in members:
                f = tar.extractfile(member)
                if f is None:
                    continue
                
                try:
                    content = f.read().decode('utf-8')
                    for line in content.split('\n'):
                        lemma = parse_conllu_line(line, lang_code)
                        if lemma:
                            noun_counter[lemma] += 1
                except UnicodeDecodeError:
                    continue
    except Exception as e:
        print(f"    Error: {e}")
    
    print(f"    Found {len(noun_counter)} unique nouns from UD, {sum(noun_counter.values())} occurrences")
    return noun_counter


def merge_noun_sources(lang_code):
    """Merge nouns from frequency file and UD treebanks."""
    print(f"  Merging noun sources for {lang_code}...")
    
    # Load from frequency file
    freq_nouns = load_noun_frequencies(lang_code)
    
    # Extract from UD
    ud_nouns = extract_nouns_from_ud(lang_code)
    
    # Merge: add UD nouns to frequency nouns
    for noun, count in ud_nouns.items():
        freq_nouns[noun] += count
    
    print(f"    Total: {len(freq_nouns)} unique nouns, {sum(freq_nouns.values())} total frequency")
    return freq_nouns


def parse_uri(uri: str) -> dict:
    """Parse ConceptNet URI into components."""
    if not uri.startswith('/c/'):
        return None
    
    parts = uri.split('/')
    if len(parts) < 4:
        return None
    
    return {
        'lang': parts[2],
        'text': parts[3],
        'pos': parts[4] if len(parts) > 4 else None
    }


def is_single_word(text: str) -> bool:
    """Check if text is a single word (no spaces)."""
    return ' ' not in text.strip()


def extract_related_words_from_conceptnet(target_nouns: Set[str], lang_code: str):
    """
    Extract related words from ConceptNet for target nouns.
    Returns dict: {target_word: {relation_type: [(word, weight)]}}
    """
    print(f"  Extracting related words from ConceptNet for {lang_code}...")
    
    # Structure: {target: {relation_type: [(word, weight)]}}
    relations_data = defaultdict(lambda: defaultdict(list))
    
    line_count = 0
    
    try:
        with gzip.open(CONCEPTNET_FILE, 'rt', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            
            for row in reader:
                line_count += 1
                if line_count % 1_000_000 == 0:
                    print(f"    Processed {line_count:,} assertions...", end='\r')
                
                if len(row) < 5:
                    continue
                
                uri, relation, start_node, end_node, json_metadata = row
                
                # Filter by relevant relations
                if relation not in VALID_RELATIONS:
                    continue
                
                start = parse_uri(start_node)
                end = parse_uri(end_node)
                
                if not start or not end:
                    continue
                
                # Must be same language
                if start['lang'] != lang_code or end['lang'] != lang_code:
                    continue
                
                # Get weight
                try:
                    meta = json.loads(json_metadata)
                    weight = meta.get('weight', 0)
                except json.JSONDecodeError:
                    continue
                
                start_text = start['text'].replace('_', ' ')
                end_text = end['text'].replace('_', ' ')
                
                # Get language config to check if we should filter by single words
                lang_config = LANGUAGES.get(lang_code, {})
                check_single_word = lang_config.get('single_word_check', True)
                
                # Filter to single words only if language supports space-based tokenization
                if check_single_word and not is_single_word(end_text):
                    continue
                
                # If start is one of our target nouns, add end as related
                if start_text in target_nouns and start_text != end_text:
                    relations_data[start_text][relation].append((end_text, weight))
                
                # For bidirectional relations (Synonym, RelatedTo)
                if relation in ['/r/Synonym', '/r/RelatedTo']:
                    if check_single_word and not is_single_word(start_text):
                        continue
                    if end_text in target_nouns and start_text != end_text:
                        relations_data[end_text][relation].append((start_text, weight))
    
    except Exception as e:
        print(f"\n    Error: {e}")
    
    print(f"\n    Found relations for {len(relations_data)} target words")
    return relations_data


def build_taboo_lists(target_nouns_freq: Counter, relations_data: dict, lang_code: str):
    """Build taboo word lists by sampling from different relations."""
    print(f"  Building taboo lists for {lang_code}...")
    
    all_entries = []
    
    # Process nouns in frequency order
    sorted_nouns = target_nouns_freq.most_common()
    
    for target_word, freq in sorted_nouns:
        if target_word not in relations_data:
            continue
        
        # Get relations for this word
        word_relations = relations_data[target_word]
        
        # Sample taboo words from different relations
        # Strategy: Try to get one from each relation first, then fill up to MAX_WORDS_PER_RELATION
        taboo_words = []
        used_relations = []
        
        # Priority order for sampling (will use highest weight words from each relation)
        relation_priority = VALID_RELATIONS
        
        # First pass: get one word from each available relation
        for relation in relation_priority:
            if len(taboo_words) >= NUM_TABOO_WORDS:
                break
            
            if relation in word_relations and word_relations[relation]:
                sorted_words = sorted(word_relations[relation], key=lambda x: x[1], reverse=True)
                word, weight = sorted_words[0]
                if word not in taboo_words:
                    taboo_words.append(word)
                    used_relations.append((relation, 1))  # Track we used 1 word from this relation
        
        # Second pass: if we still need more words, take up to MAX_WORDS_PER_RELATION from relations
        if len(taboo_words) < NUM_TABOO_WORDS:
            for relation in relation_priority:
                if len(taboo_words) >= NUM_TABOO_WORDS:
                    break
                
                if relation not in word_relations or not word_relations[relation]:
                    continue
                
                # Find how many we already used from this relation
                used_count = sum(count for rel, count in used_relations if rel == relation)
                
                if used_count < MAX_WORDS_PER_RELATION:
                    sorted_words = sorted(word_relations[relation], key=lambda x: x[1], reverse=True)
                    # Skip words we already added, take next ones
                    for word, weight in sorted_words:
                        if word not in taboo_words and len(taboo_words) < NUM_TABOO_WORDS:
                            if used_count < MAX_WORDS_PER_RELATION:
                                taboo_words.append(word)
                                used_count += 1
        
        # If we don't have enough, skip this target word
        if len(taboo_words) < NUM_TABOO_WORDS:
            continue
        
        # Limit to NUM_TABOO_WORDS
        taboo_words = taboo_words[:NUM_TABOO_WORDS]
        
        entry = {
            "target_word": target_word,
            "related_word": taboo_words,
            "target_word_stem": target_word,  # Using same for now
            "related_word_stem": taboo_words   # Using same for now
        }
        
        all_entries.append(entry)
    
    # Split into high and low frequency based on first N entries WITH relations
    high_freq_data = all_entries[:HIGH_FREQUENCY_THRESHOLD]
    low_freq_data = all_entries[HIGH_FREQUENCY_THRESHOLD:]
    
    print(f"    High frequency: {len(high_freq_data)} words")
    print(f"    Low frequency: {len(low_freq_data)} words")
    
    return high_freq_data, low_freq_data


def save_results(high_freq_data: list, low_freq_data: list, lang_code: str):
    """Save results to two separate JSON files."""
    lang_dir = Path(OUTPUT_DIR) / lang_code
    lang_dir.mkdir(parents=True, exist_ok=True)
    
    if high_freq_data:
        high_freq_file = lang_dir / 'high_frequency_taboo_words.json'
        with open(high_freq_file, 'w', encoding='utf-8') as f:
            json.dump(high_freq_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(high_freq_data)} entries to: {high_freq_file}")
    
    if low_freq_data:
        low_freq_file = lang_dir / 'low_frequency_taboo_words.json'
        with open(low_freq_file, 'w', encoding='utf-8') as f:
            json.dump(low_freq_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(low_freq_data)} entries to: {low_freq_file}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("ConceptNet Taboo Game Word Extractor (Noun Frequencies + UD)")
    print("=" * 80)
    print()
    
    for lang_code in TARGET_LANGUAGES:
        print(f"\nProcessing {lang_code}...")
        print("-" * 80)
        
        try:
            # Step 1: Merge noun sources (frequency file + UD)
            target_nouns_freq = merge_noun_sources(lang_code)
            if not target_nouns_freq:
                print(f"  No nouns found for {lang_code}")
                continue
            
            target_nouns_set = set(target_nouns_freq.keys())
            
            # Step 2: Extract related words from ConceptNet
            relations_data = extract_related_words_from_conceptnet(target_nouns_set, lang_code)
            if not relations_data:
                print(f"  No relations found for {lang_code}")
                continue
            
            # Step 3: Build taboo lists
            high_freq_data, low_freq_data = build_taboo_lists(
                target_nouns_freq, relations_data, lang_code
            )
            
            # Step 4: Save results
            save_results(high_freq_data, low_freq_data, lang_code)
            
        except Exception as e:
            print(f"  Error processing {lang_code}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print()
    print("=" * 80)
    print("Extraction complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
