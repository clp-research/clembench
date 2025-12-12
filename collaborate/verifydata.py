import os
import json


def verify(filepath):
    with open(filepath, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON format: {e}")
            return False, f"Invalid JSON format: {e}"
        
    print(f"Verifying data from {filepath}...")
    failed_instances = {}
    for game_id, game_data in data.items():
        sb_data = game_data.get('simple', {})
        rb_data = game_data.get('regular', {})

        sb_combos = list(sb_data.keys())
        rb_combos = list(rb_data.keys())
        print(f"Game ID: {game_id}, Simple Combos: {sb_combos}, Regular Combos: {rb_combos}")

        if sb_combos != rb_combos:
            print(f"Combo mismatch for game ID {game_id}:")
            failed_instances[game_id] = {'simple_combos': sb_combos, 'regular_combos': rb_combos, "error": "Combo mismatch"}
            input()
            continue

        for combo in sb_combos:
            sb_combo_data = sb_data.get(combo, {})
            rb_combo_data = rb_data.get(combo, {})

            #print(sb_data.keys(), rb_data.keys())
            if not sb_combo_data or not rb_combo_data:
                print(f"Missing data for combo {combo} in game ID {game_id}")
                print(sb_combo_data.keys())
                print(rb_combo_data.keys())
                input()

            if sb_combo_data == rb_combo_data:
                print(f"Exact data for the combo in both simple and regular for the combo name:{combo}")
                print(sb_combo_data["code"]["single_turn_sc"]["function"])
                print(rb_combo_data["code"]["single_turn_sc"]["function"])
                input()
            else:
                if sb_combo_data["shapes"] == rb_combo_data["shapes"]:
                    continue
                else:
                    print(f"Data mismatch for game ID {game_id}, combo {combo}: {sb_combo_data['shapes']}, {rb_combo_data['shapes']}")
                    input()
                    if game_id not in failed_instances:
                        failed_instances[game_id] = {'simple_combos': sb_combos, 'regular_combos': rb_combos, "error": "Item mismatch",
                                                    "missed_combos": []}
                    failed_instances[game_id][combo] = {"simple_data": sb_combo_data, "regular_data": rb_combo_data}
                    failed_instances[game_id]["missed_combos"].append(combo)
        

    if failed_instances:
        print("Verification failed for the following instances:")
        for game_id, details in failed_instances.items():
            print(f"Game ID: {game_id}")

        with open("verification_failures.json", 'w') as outfile:
            json.dump(failed_instances, outfile, indent=4)

    else:
        print("All data verified successfully!")


if __name__ == "__main__":
    #verify("collab_data_single_reuse.json")
    #verify("collab_data_mixed.json")
    #verify("collab_data_regular.json")
    #verify("collab_data_regular_co.json")
    verify("collab_data_regular_both.json")    
