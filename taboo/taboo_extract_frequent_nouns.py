import pandas as pd
import stanza
import os
import requests
import xml.etree.ElementTree as ET
from io import StringIO

# 1. Define Target Languages
target_langs = {
    'ar': 'Arabic', 'bg': 'Bulgarian', 'hr': 'Croatian', 'cs': 'Czech', 'da': 'Danish',
    'nl': 'Dutch', 'en': 'English', 'et': 'Estonian', 'fi': 'Finnish',
    'fr': 'French', 'de': 'German', 'el': 'Greek', 'hu': 'Hungarian',
    'ga': 'Irish', 'it': 'Italian', 'lv': 'Latvian', 'lt': 'Lithuanian',
    'mt': 'Maltese', 'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian',
    'ru': 'Russian',
    'sk': 'Slovak', 'sl': 'Slovenian', 'es': 'Spanish', 'sv': 'Swedish',
    'tr': 'Turkish', 'ur': 'Urdu'
}
# Ensure output directory exists
output_dir = "taboo_frequent_nouns"
os.makedirs(output_dir, exist_ok=True)


# --- Helper: Custom Handler for Maltese (Apertium) ---
def process_maltese(lang_code, lang_name):
    print(f"\n[{lang_code.upper()}] Processing {lang_name} using Apertium Dictionary...")
    url = "https://raw.githubusercontent.com/apertium/apertium-mlt/refs/heads/master/apertium-mlt.mlt.dix"

    try:
        print(f"   Downloading: {url}")
        response = requests.get(url)
        response.raise_for_status()

        # Parse XML
        root = ET.fromstring(response.content)
        noun_data = []
        unique_words = []

        # Apertium structure: <e><p><l>word</l><r>...<s n="n"/>...</r></p></e>
        for e in root.findall('.//e'):
            r_tag = e.find('.//r')
            l_tag = e.find('.//l')

            if r_tag is not None and l_tag is not None:
                symbols = [s.get('n') for s in r_tag.findall('s')]

                # Check for Common Noun ('n') or Proper Noun ('np')
                if 'n' in symbols:
                    word = l_tag.text
                    if word:
                        if word not in unique_words:
                            unique_words.append(word)
                            noun_data.append({
                                'word': word,
                                'count': 0  # No frequency in dictionary
                            })

        if noun_data:
            save_results(lang_code, noun_data)
        else:
            print(f"   ⚠️ No nouns extracted for {lang_name}.")

    except Exception as e:
        print(f"   ❌ Error processing {lang_name}: {e}")


# --- Helper: Custom Handler for Irish (MichMech) ---
def process_irish(lang_code, lang_name):
    print(f"\n[{lang_code.upper()}] Processing {lang_name} using MichMech Frequency List...")
    url = "https://raw.githubusercontent.com/michmech/irish-word-frequency/refs/heads/master/frequency.txt"

    try:
        print(f"   Downloading: {url}")
        response = requests.get(url)
        response.raise_for_status()

        data = response.text
        lines = data.split('\n')

        parsed_data = []
        for line in lines:
            stripped_line = line.strip()

            # Skip empty lines or comments (lines starting with //)
            if not stripped_line or stripped_line.startswith('//'):
                continue

            # Split by TAB as per the file format
            # Format: rank [0] \t lemma [1] \t frequency [2] \t window [3]
            parts = line.split('\t')

            if len(parts) >= 3:
                try:
                    word = parts[1].strip()
                    # The frequency is in the 3rd column (index 2)
                    count = int(parts[2].strip())

                    parsed_data.append({'word': word, 'count': count})
                except ValueError:
                    continue  # Skip lines where count isn't an integer or format is unexpected

        df = pd.DataFrame(parsed_data)
        print(f"   Extracted {len(df)} rows of raw data.")

        # Apply Stanza to filter these words for nouns
        if not df.empty:
            apply_stanza_and_save(lang_code, lang_name, df)
        else:
            print(f"   ⚠️ No valid data parsed for {lang_name}. Check delimiter.")

    except Exception as e:
        print(f"   ❌ Error processing {lang_name}: {e}")


# --- Helper: Standard Handler for other languages ---
def process_standard(lang_code, lang_name):
    print(f"\n[{lang_code.upper()}] Processing {lang_name} using Orgtre OpenSubtitles...")
    csv_url = f"https://raw.githubusercontent.com/orgtre/top-open-subtitles-sentences/main/bld/top_words/{lang_code}_top_words.csv"

    try:
        print(f"   Downloading: {csv_url}")
        response = requests.get(csv_url)
        response.raise_for_status()
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        df = df.dropna(subset=['word'])

        apply_stanza_and_save(lang_code, lang_name, df)

    except Exception as e:
        print(f"   ❌ Error processing {lang_name}: {e}")


# --- Core Logic: Apply Stanza to a DataFrame ---
def apply_stanza_and_save(lang_code, lang_name, df):
    try:
        # Initialize Stanza
        print('Downloading the stanza model')
        stanza.download(lang_code, processors='tokenize,pos,lemma', verbose=False, model_dir='stanza_resources')
        print('Loading the stanza model')
        nlp = stanza.Pipeline(lang=lang_code, processors='tokenize,pos,lemma', verbose=False, use_gpu=True,
                              model_dir='stanza_resources')
    except Exception as e:
        print(f"   ❌ Error loading Stanza model for {lang_name}: {e}")
        return

    noun_data = []
    unique_words = []
    print(f"   Tagging {len(df)} words...")

    for _, row in df.iterrows():
        word = str(row['word'])

        if len(word) < 2: continue

        doc = nlp(word)

        if doc.sentences and doc.sentences[0].words:
            token = doc.sentences[0].words[0]
            # UPOS tags: NOUN (common).
            if token.upos in ['NOUN']:
                lemma = token.lemma
                if lemma not in unique_words:
                    unique_words.append(lemma)
                    noun_data.append({
                        'word': token.lemma,
                        'count': row['count']
                    })

    save_results(lang_code, noun_data)


# --- Save Utility ---
def save_results(lang_code, data_list):
    if data_list:
        output_file = os.path.join(output_dir, f"{lang_code}_nouns.csv")
        df = pd.DataFrame(data_list)
        df.to_csv(output_file, index=False)
        print(f"   ✅ Saved {len(df)} nouns to: {output_file}")
    else:
        print(f"   ⚠️ No nouns found for {lang_code}.")


# --- Main Dispatcher ---
def process_language(lang_code, lang_name):
    if lang_code == 'ga':
        process_irish(lang_code, lang_name)
    elif lang_code == 'mt':
        process_maltese(lang_code, lang_name)
    else:
        process_standard(lang_code, lang_name)


# --- Execution ---
if __name__ == "__main__":
    print(f"Starting extraction for {len(target_langs)} languages...\n")

    for code, name in target_langs.items():
        process_language(code, name)

    print("\nProcessing complete.")