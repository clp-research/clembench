import os
import json

# Path to your folder
folder = os.path.join("mapworld", "escapegame")


def reformat_jsons(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Re-save with indent=4
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)

                    print(f"Updated: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    reformat_jsons(folder)
