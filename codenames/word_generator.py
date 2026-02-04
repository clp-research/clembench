import os
import json
from openai import OpenAI

# ---------------- CONFIGURATION ---------------- #

# PASTE YOUR GOOGLE API KEY HERE
API_KEY = "YOUR_GEMINI_API_KEY"

# Model Name
MODEL_NAME = "gemini-2.0-flash"

# Base URL for Google's OpenAI-compatible endpoint
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Paths
SOURCE_DIR = os.path.join("../../../Downloads/conceptnet-words-categories-extractor-main/codenames/resources", "wordlists")
FILE_ORIGINAL = "original.json"
FILE_CATEGORIES = "categories.json"

# ---------------- LANGUAGE DATA ---------------- #

LANGUAGES = {
    'ar': {'name': 'Arabic', 'lowercase': True},
    'bg': {'name': 'Bulgarian', 'lowercase': True},
    'cs': {'name': 'Czech', 'lowercase': True},
    'da': {'name': 'Danish', 'lowercase': True},
    'de': {'name': 'German', 'lowercase': False},  # German nouns stay capitalized
    'el': {'name': 'Greek', 'lowercase': True},
    'en': {'name': 'English', 'lowercase': True},
    'es': {'name': 'Spanish', 'lowercase': True},
    'et': {'name': 'Estonian', 'lowercase': True},
    'fi': {'name': 'Finnish', 'lowercase': True},
    'fr': {'name': 'French', 'lowercase': True},
    'ga': {'name': 'Irish', 'lowercase': True},
    'hr': {'name': 'Croatian', 'lowercase': True},
    'hu': {'name': 'Hungarian', 'lowercase': True},
    'it': {'name': 'Italian', 'lowercase': True},
    'lt': {'name': 'Lithuanian', 'lowercase': True},
    'lv': {'name': 'Latvian', 'lowercase': True},
    'mt': {'name': 'Maltese', 'lowercase': True},
    'nl': {'name': 'Dutch', 'lowercase': True},
    'pl': {'name': 'Polish', 'lowercase': True},
    'pt': {'name': 'Portuguese', 'lowercase': True},
    'ro': {'name': 'Romanian', 'lowercase': True},
    'ru': {'name': 'Russian', 'lowercase': True},
    'sk': {'name': 'Slovak', 'lowercase': True},
    'sl': {'name': 'Slovenian', 'lowercase': True},
    'sv': {'name': 'Swedish', 'lowercase': True},
    'tr': {'name': 'Turkish', 'lowercase': True},
    'ur': {'name': 'Urdu', 'lowercase': True}
}

# ---------------- INITIALIZATION ---------------- #

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


def load_source_json(filename):
    """Loads the source JSON file from resources/wordlists/."""
    path = os.path.join(SOURCE_DIR, filename)
    if not os.path.exists(path):
        print(f"Error: Source file not found at {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def translate_content(content, filename, lang_code, lang_info):
    """
    Sends JSON content to Gemini for translation with specific preservation rules.
    """
    lang_name = lang_info['name']
    is_lowercase = lang_info['lowercase']

    # Casing instruction
    casing_instruction = "Output the translated words in LOWERCASE." if is_lowercase else "Keep standard noun capitalization (e.g., Capitalize nouns for German)."

    # Specific instructions based on file type
    if filename == FILE_ORIGINAL:
        specific_instruction = """
        This is a flat list under the key "words".
        1. Keep the key "words" exactly as it is (do NOT translate it).
        2. Translate the list of strings inside "words" into the target language.
        """
    elif filename == FILE_CATEGORIES:
        specific_instruction = """
        This file contains categories.
        1. Keep the root key "words" exactly as it is.
        2. Keep the sub-category keys (e.g., "a bird", "places") exactly as they are in English (do NOT translate the keys).
        3. ONLY translate the list of strings associated with each sub-category.
        """
    else:
        specific_instruction = "Translate the values."

    prompt = f"""
    You are a professional translator for {lang_name}.

    Task:
    Translate the values within the provided JSON object into {lang_name}, following strict structural rules.

    Rules:
    {specific_instruction}
    - {casing_instruction}
    - Return ONLY valid JSON.
    - Do not add markdown formatting like ```json.

    Input JSON:
    {json.dumps(content, ensure_ascii=False)}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs strict JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        result_content = response.choices[0].message.content
        return json.loads(result_content)

    except Exception as e:
        print(f"Error translating {filename} for {lang_name}: {e}")
        return None


def save_translated_file(lang_code, filename, content):
    """Saves the translated JSON to resources/{lang_code}/{filename}."""
    directory = os.path.join("../../../Downloads/conceptnet-words-categories-extractor-main/codenames/resources", lang_code)
    os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)


# ---------------- MAIN LOOP ---------------- #

def main():
    if API_KEY == "YOUR_GEMINI_API_KEY":
        print("Please set your API_KEY at the top of the script.")
        return

    # Load source files once
    original_data = load_source_json(FILE_ORIGINAL)
    categories_data = load_source_json(FILE_CATEGORIES)

    if not original_data or not categories_data:
        print("Aborting: Could not load source files.")
        return

    print(f"Loaded source files. Starting translation for {len(LANGUAGES)} languages...")

    for lang_code, lang_info in LANGUAGES.items():
        print(f"--- Processing {lang_info['name']} ({lang_code}) ---")

        # 1. Translate original.json
        trans_original = translate_content(original_data, FILE_ORIGINAL, lang_code, lang_info)
        if trans_original:
            save_translated_file(lang_code, FILE_ORIGINAL, trans_original)
            print(f"Saved {FILE_ORIGINAL}")

        # 2. Translate categories.json
        trans_categories = translate_content(categories_data, FILE_CATEGORIES, lang_code, lang_info)
        if trans_categories:
            save_translated_file(lang_code, FILE_CATEGORIES, trans_categories)
            print(f"Saved {FILE_CATEGORIES}")


if __name__ == "__main__":
    main()