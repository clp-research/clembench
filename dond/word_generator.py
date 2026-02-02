import os
import json
from openai import OpenAI

# ---------------- CONFIGURATION ---------------- #

# PASTE YOUR GOOGLE API KEY HERE
API_KEY = "YOUR_GEMINI_API_KEY"

# Model Name (Update this string if the specific version changes)
# Currently using gemini-2.0-flash as the standard reference for the next flash model.
# If you specifically need 1.5, change to "gemini-1.5-flash".
MODEL_NAME = "gemini-2.0-flash"

# Base URL for Google's OpenAI-compatible endpoint
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

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


def generate_word_list(lang_code, lang_info):
    """
    Queries Gemini via OpenAI API to generate singular and plural noun lists.
    """
    lang_name = lang_info['name']
    is_lowercase = lang_info['lowercase']

    # Constructing the casing instruction
    casing_instruction = "Output all words in LOWERCASE." if is_lowercase else "Keep standard noun capitalization (e.g., capitalize nouns if required by the language)."

    prompt = f"""
    You are a linguistic data generator for {lang_name}.

    Task:
    1. Generate a list of 20 everyday, concrete objects (nouns) that are suitable for trading (e.g., fruits, office supplies, small tools, common household items).
    2. Provide two versions for every noun: the Singular form and the Plural form.
    3. If the language does not have a plural form for a specific word, repeat the singular form in the plural list.
    4. {casing_instruction}

    Output Format:
    Return ONLY a raw JSON object with exactly this structure:
    {{
      "singular": ["item1", "item2", ...],
      "plural": ["item1s", "item2s", ...]
    }}

    Ensure both lists have exactly 20 items and the indexes match (index 0 in plural corresponds to index 0 in singular).
    """

    try:
        print(f"Processing {lang_name} ({lang_code})...")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs strict JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        # Validation
        if "singular" not in data or "plural" not in data:
            raise ValueError("JSON missing required keys")

        if len(data["singular"]) != len(data["plural"]):
            print(f"Warning: Length mismatch for {lang_name}. Truncating to shortest list.")
            min_len = min(len(data["singular"]), len(data["plural"]))
            data["singular"] = data["singular"][:min_len]
            data["plural"] = data["plural"][:min_len]

        return data["singular"], data["plural"]

    except Exception as e:
        print(f"Error processing {lang_name}: {e}")
        return None, None


def save_files(lang_code, singular_list, plural_list):
    """
    Saves the lists to the specified directory structure.
    """
    # Create directory: resources/{lang_code}
    directory = os.path.join("resources", lang_code)
    os.makedirs(directory, exist_ok=True)

    # Path definitions
    path_singular = os.path.join(directory, "possible_items.json")
    path_plural = os.path.join(directory, "possible_items_plural.json")

    # Write Singular
    with open(path_singular, 'w', encoding='utf-8') as f:
        json.dump(singular_list, f, ensure_ascii=False, indent=4)

    # Write Plural
    with open(path_plural, 'w', encoding='utf-8') as f:
        json.dump(plural_list, f, ensure_ascii=False, indent=4)

    print(f"Saved files for {lang_code}.")


# ---------------- MAIN LOOP ---------------- #

def main():
    if API_KEY == "YOUR_GEMINI_API_KEY":
        print("Please set your API_KEY at the top of the script.")
        return

    # Ensure resources folder exists
    os.makedirs("resources", exist_ok=True)

    for lang_code, lang_info in LANGUAGES.items():
        singular, plural = generate_word_list(lang_code, lang_info)

        if singular and plural:
            save_files(lang_code, singular, plural)
        else:
            print(f"Skipping {lang_info['name']} due to errors.")


if __name__ == "__main__":
    main()