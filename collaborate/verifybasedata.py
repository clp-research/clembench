import os
import json


with open("resources/data/en/rb_co_test.json") as f:
    data = json.load(f)



vdata = {}
for board_type, objs_type in data.items():
    for boardobj, num_shapes in objs_type.items():
        for total_shapes, combo_names in num_shapes.items():
            for combo_name in num_shapes[total_shapes]:
                if combo_name not in vdata:
                    vdata[combo_name] = {}
                else:
                    print(f"Duplicate combo name found: {combo_name}")
                    input()
                for index, tsample in enumerate(num_shapes[total_shapes][combo_name]):
                    vdata[combo_name].update({f"{index+1}": tsample["shapes"]})

#with open("base_verify_sb_so_val.json", "w") as f:
#    json.dump(vdata, f, indent=4)

for combo_name, combo_dict in vdata.items():
    if not combo_dict:
        print(f"Empty combo found: {combo_name}")
        continue
    combo_shapes = combo_dict["1"]

    for key_index, data in combo_dict.items():
        if combo_shapes != data:
            print(f"Mismatch found in combo {combo_name} between samples 1 and {key_index}")
            input() 
