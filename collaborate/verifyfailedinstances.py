import os
import json



with open("debug_failed_combos.json", 'r', encoding='utf-8') as f:
    failed_combos = json.load(f)

with open("debug_rb_input.json", 'r', encoding='utf-8') as f:
    rb_inputs = json.load(f)


for combo_name in failed_combos:
    if combo_name[0] not in rb_inputs:
        print(f"Combo {combo_name} not found in rb_inputs")
        continue
    rb_combo_data = rb_inputs[combo_name[0]]
    if not rb_combo_data:
        print(f"No RB data available for combo {combo_name[0]}")
        continue

    if len(rb_combo_data) > 2:
        print(f"Enough RB samples for combo {combo_name[0]}, available: {len(rb_combo_data)}")

print("Verification complete.")