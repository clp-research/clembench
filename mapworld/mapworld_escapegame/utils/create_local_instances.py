import os
import json
import shutil
import requests
from urllib.parse import urlparse

# Paths
INPUT_JSON = os.path.join("mapworld", "escapegame", "in", "instances.json")
OUTPUT_JSON = os.path.join("mapworld", "escapegame", "in", "instances_local.json")
IMAGES_DIR = os.path.join("mapworld", "escapegame", "resources", "images")

# Create base images directory if it doesn't exist
os.makedirs(IMAGES_DIR, exist_ok=True)

# Load the JSON data
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Save the updated JSON data
def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Given a URL, download the image preserving the URL's folder structure under IMAGES_DIR
def download_image(url):
    parsed = urlparse(url)
    path_parts = parsed.path.split('/')
    try:
        # find index of 'images' in path
        idx = path_parts.index('images')
        sub_dirs = path_parts[idx+1:-1]  # folders after 'images', excluding filename
        filename = path_parts[-1]
    except ValueError:
        # fallback: just use filename
        sub_dirs = []
        filename = os.path.basename(parsed.path)

    # Construct local directory and file path
    local_dir = os.path.join(IMAGES_DIR, *sub_dirs)
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)

    if not os.path.exists(local_path):
        print(f"Downloading {url} -> {local_path}")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
    else:
        print(f"Already exists: {local_path}")

    # Return the local path to be inserted into JSON
    return local_path

# Recursively update any URL fields in the JSON structure
def update_urls(obj):
    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, str) and val.startswith("http"):
                obj[key] = download_image(val)
            else:
                update_urls(val)
    elif isinstance(obj, list):
        for item in obj:
            update_urls(item)


def main():
    data = load_json(INPUT_JSON)
    update_urls(data)
    save_json(data, OUTPUT_JSON)
    print(f"Updated JSON written to {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
