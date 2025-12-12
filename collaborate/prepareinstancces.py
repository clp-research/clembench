import os
import json
import random
from typing import Dict, List


SEED = 42


class PrepareCollabData:
    def __init__(self, configfile):
        self.config = self.readfile(configfile)

    def readfile(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    
    def writefile(self, filename, data):
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)

    def combineall(self, data: Dict) -> List:
        combined_samples = {}
        for board_type, objs_type in data.items():
            for boardobj, num_shapes in objs_type.items():
                for total_shapes, combo_names in num_shapes.items():
                    for combo_name in num_shapes[total_shapes]:
                        combined_samples.update({combo_name: num_shapes[total_shapes][combo_name]})
        return combined_samples
    
    def _getsamplessb_old(self, max_num_boards, data, collab_data, common_combos):
        # Gather all current samples to avoid duplicates
        current_samples = {}
        for instance in collab_data:
            current_samples.update(collab_data[instance]["simple"])

        combo_names_data = list(data.keys())

        available_combos = {name: data[name] for name in common_combos if name not in current_samples}
        print(f"Simple Boards: Available Combos: {len(available_combos)}")

        # Filter data to only those not in current_samples
        #available_samples = [sample for sample in data if sample not in current_samples]

        available_combo_names = list(available_combos.keys())
        if len(available_combo_names) < max_num_boards:
            print("Not enough unique simple samples available to select.")
            return None
        
        # Randomly sample max_numb_boards from available combo_names
        random.seed(SEED)        
        selected_combos = random.sample(available_combo_names, max_num_boards)

        # Now randomly select 1 sample for each selected combo_name
        selected_simple = {}

        for combo in selected_combos:
            selected_simple[combo] = random.choice(available_combos[combo])

        return selected_simple
    
    def _getsamplessb(self, data, selected_combos, max_num_boards):

        # Now randomly select n sample for each selected combo_name
        selected_simple = {}

        for combo in selected_combos:
            selected_simple[combo] = random.sample(data[combo], max_num_boards)

        return selected_simple    

    def _getsamplesrb(self, max_num_boards, data, collab_data):

        # Find the RB samples that have same combo names as the selected SB samples
        selected_combos = {"regular": {}, "regular_challenging": {}, "simple_reuse": {}}
        for combo_name, combo_data_list in collab_data.items():
            for combo_data in combo_data_list:
                sb_combo_data = combo_data
                if combo_name in data["regular"]:
                    rb_combo_data = data["regular"][combo_name]
                    if rb_combo_data:
                        # Check if shapes match
                        for rb_item in rb_combo_data:
                            if rb_item["shapes"] == sb_combo_data["shapes"]:
                                if combo_name not in selected_combos["regular"]:
                                    selected_combos["regular"][combo_name] = []
                                #selected_combos["regular"][combo_name] = selected_combos["regular"].get(combo_name, [])
                                selected_combos["regular"][combo_name].append(rb_item)
                                print(f"Matched RB item for combo {combo_name} added to regular samples.")

                if combo_name in data["regular_challenging"]:
                    rb_combo_data = data["regular_challenging"][combo_name]
                    if rb_combo_data:
                        # Check if shapes match
                        for rb_item in rb_combo_data:
                            if rb_item["shapes"] == sb_combo_data["shapes"]:
                                if combo_name not in selected_combos["regular_challenging"]:
                                    selected_combos["regular_challenging"][combo_name] = []
                                #selected_combos["regular_challenging"][combo_name] = selected_combos["regular_challenging"].get(combo_name, [])
                                selected_combos["regular_challenging"][combo_name].append(rb_item)


                #selected_combos["regular"][combo_name] = data["regular"].get(combo_name, {})
                #selected_combos["regular_challenging"][combo_name] = data["regular_challenging"].get(combo_name, {})

                #if self.config["use_simple_reuse"]:
                # Also include simple samples if reuse is allowed
                selected_combos["simple_reuse"][combo_name] = data["simple"].get(combo_name, {})

        num_rb_samples = {}
        num_rb_challenging_samples = {}
        num_sb_samples = {}
        for combo_name in collab_data:
            num_rb_samples[combo_name] = len(selected_combos["regular"].get(combo_name, []))
            num_rb_challenging_samples[combo_name] = len(selected_combos["regular_challenging"].get(combo_name, []))
            num_sb_samples[combo_name] = len(selected_combos["simple_reuse"].get(combo_name, []))


        #For each combo_name, check if there are enough samples available
        if self.config["use_regular"]:
            #check num_rb_samples per combo_name and check if less than max_num_boards
            insufficient_regular = [name for name, count in num_rb_samples.items() if count < max_num_boards]
            if insufficient_regular:
                print(f"Not enough unique regular samples available to select for combos: {insufficient_regular}")
                return None, None
        if self.config["use_regular_challenging"]:
            insufficient_regular_challenging = [name for name, count in num_rb_challenging_samples.items() if count < max_num_boards]
            if insufficient_regular_challenging:
                print(f"Not enough unique regular challenging samples available to select for combos: {insufficient_regular_challenging}")
                return None, None
        if self.config["use_simple_reuse"]:
            insufficient_simple_reuse = [name for name, count in num_sb_samples.items() if count < max_num_boards]
            if insufficient_simple_reuse:
                print(f"Not enough unique simple reuse samples available to select for combos: {insufficient_simple_reuse}")
                return None, None

        """
        if self.config["use_regular"] and len(selected_combos["regular"]) < max_num_boards or \
           self.config["use_regular_challenging"] and len(selected_combos["regular_challenging"]) < max_num_boards or \
           self.config["use_simple_reuse"] and len(selected_combos["simple_reuse"]) < max_num_boards:
            print(f"Not enough unique samples available to select based on the enabled board types. {collab_data.keys()}")
            print(f"Regular: {len(selected_combos['regular'])}, Regular Challenging: {len(selected_combos['regular_challenging'])}, Simple Reuse: {len(selected_combos['simple_reuse'])}, Required: {max_num_boards}")
            return None, None
        """
        
        # Select regular boards or challenging boards randomly
        if self.config["use_regular"] and self.config["use_regular_challenging"] and self.config["use_simple_reuse"]:
            board_type_choice = random.choice(["regular", "regular_challenging", "simple_reuse"])
        elif self.config["use_regular"] and self.config["use_regular_challenging"]:
            board_type_choice = random.choice(["regular", "regular_challenging"])
        elif self.config["use_regular"] and self.config["use_simple_reuse"]:
            board_type_choice = random.choice(["regular", "simple_reuse"])
        elif self.config["use_regular_challenging"] and self.config["use_simple_reuse"]:
            board_type_choice = random.choice(["regular_challenging", "simple_reuse"])
        elif self.config["use_regular"]:
            board_type_choice = "regular"
        elif self.config["use_regular_challenging"]:
            board_type_choice = "regular_challenging"
        elif self.config["use_simple_reuse"]:
            board_type_choice = "simple_reuse"
        else:
            print("No board types are enabled in the configuration.")
            return None, None

        selected_combos = selected_combos[board_type_choice]


        # Now randomly select n samples for each selected combo_name
        selected_regular = {}
        sb_combo_data = list(collab_data.values())  
      
        for combo in selected_combos:
            if selected_combos[combo]:  # Check if there are available samples
                if board_type_choice != "simple_reuse":
                    selected_regular[combo] = random.sample(selected_combos[combo], max_num_boards)
                else:
                    unused_samples = []
                    for item in selected_combos[combo]:
                        for sb_item in sb_combo_data:
                            if item["combo_name"] == sb_item["combo_name"] and item["shapes"] == sb_item["shapes"]:
                                #self.writefile("debug_item.json", selected_combos[combo])
                                #self.writefile("debug_sb_item.json", sb_item)
                                if item["colors"] != sb_item["colors"] and (item["x"] != sb_item["x"] or item["y"] != sb_item["y"]):
                                    #self.writefile("debug_item.json", item)
                                    #self.writefile("debug_sb_item.json", sb_item)
                                    #input("Check the files")
                                    unused_samples.append(item)
                    
                    if not unused_samples:
                        #print(f"No unused samples available for combo {combo} in simple reuse.")
                        #input()
                        return None, None
                    else:
                        #print(f"Found {len(unused_samples)} unused samples for combo {combo} in simple reuse.")
                        #self.writefile("debug_unused_samples.json", unused_samples)
                        #input("Check the debug_unused_samples.json file")
                        pass
                    combo_data = unused_samples
                    selected_regular[combo] = random.sample(combo_data, max_num_boards)
            else:
                print(f"No available samples for combo {combo} in {board_type_choice}.")
                return None, None

        return selected_regular, board_type_choice
    
    def _get_common_combos(self, data_simple, data_regular, data_regular_challenging):
        combos_simple = set(data_simple.keys())
        combos_regular = set(data_regular.keys())
        combos_regular_challenging = set(data_regular_challenging.keys())

        common_combos = combos_simple.intersection(combos_regular).intersection(combos_regular_challenging)
        return list(common_combos)


    def run(self, input_files: Dict[str, str], output_file: str):

        if not isinstance(input_files, dict):
            raise ValueError("Input files should be provided as a dictionary.")
        
        if "simple" not in input_files or "regular" not in input_files:
            raise ValueError("Input files must include 'simple' and 'complex' keys.")
        data_simple = self.readfile(input_files["simple"])
        data_regular = self.readfile(input_files["regular"])
        data_regular_challenging = self.readfile(input_files["regular_challenging"])

        data_simple_all = self.combineall(data_simple)
        data_regular_all = self.combineall(data_regular)
        data_regular_challenging_all = self.combineall(data_regular_challenging)

        with open("debug_sb_input.json", 'w', encoding='utf-8') as f:
            json.dump(data_simple_all, f, indent=4)
        with open("debug_rb_input.json", 'w', encoding='utf-8') as f:
            json.dump(data_regular_all, f, indent=4)
        with open("debug_rb_challenging_input.json", 'w', encoding='utf-8') as f:
            json.dump(data_regular_challenging_all, f, indent=4)

        #self.writefile("debug_sb.json", data_simple_all)
        #self.writefile("debug_rb.json", data_regular_all)
        #self.writefile("debug_rb_challenging.json", data_regular_challenging_all)        


        # Select 'N' random samples from the simple boards and 'M' from the regular boards
        max_boards_sb = self.config["max_boards_sb"]
        max_boards_rb = self.config["max_boards_rb"]

        min_instances = self.config["min_instances"]

        num_instances = 0
        collab_data = {}

        common_combos = self._get_common_combos(data_simple_all, data_regular_all, data_regular_challenging_all)
        print(f"Total common combos across SB and RB datasets: {len(common_combos)}")
        if len(common_combos) < max_boards_sb * min_instances: 
            print("Not enough common combo names across SB and RB datasets to prepare the requested number of instances.")
            return

        combo_instances = common_combos.copy()
        combo_instances = sorted(combo_instances)
        random.seed(SEED)
        random.shuffle(combo_instances)
        max_groups = len(combo_instances) // max_boards_sb
        sets_of_n = [combo_instances[i*max_boards_sb:(i+1)*max_boards_sb] for i in range(max_groups)]
        print(f"Total sets of {max_boards_sb} combos prepared: {len(sets_of_n)}, combos: {sets_of_n}")
        self.writefile("debug_sets_of_n.json", sets_of_n)



        instance_num = 1
        collab_data = {}
        failed_combos = []
        for index in range(len(sets_of_n)):
            #Select samples that are not already selected

            simple_boards = self._getsamplessb(data_simple_all, sets_of_n[index], max_boards_sb)
            if simple_boards is None:
                break            
            
            #print(f"Selected Simple Boards for {index+1}: {list(simple_boards.keys())}")
            if max_boards_rb > 0:
                regular_data = {"regular": data_regular_all, "regular_challenging": data_regular_challenging_all, "simple": data_simple_all}
                regular_boards, board_type_choice = self._getsamplesrb(max_boards_rb, regular_data, simple_boards)
                if regular_boards is None:
                    #print("Could not select regular boards, skipping this instance.")
                    #collab_data[instance_num] = {"simple": {}, "regular": {}, "metadata": {}}
                    failed_combos.append(sets_of_n[index])
                    continue
            else:
                regular_boards = {}
                board_type_choice = "simple"

            collab_data[instance_num] = {"simple": {}, "regular": {}, "metadata": {}}
            collab_data[instance_num]["simple"].update(simple_boards)
            collab_data[instance_num]["regular"].update(regular_boards)
            collab_data[instance_num]["metadata"].update({"board_type": board_type_choice, "num_simple_boards": max_boards_sb, "num_regular_boards": max_boards_rb})

            instance_num += 1

        if failed_combos:
            print(f"Failed to prepare instances for {len(failed_combos)} sets of combos")
            with open("debug_failed_combos.json", 'w', encoding='utf-8') as f:
                json.dump(failed_combos, f, indent=4)

        print(f"Total instances prepared: {instance_num - 1}, writing to {output_file}")
        self.writefile(output_file, collab_data)

        #for instance in collab_data:
        #     print(f"Instance: {instance}: SB Keys: {list(collab_data[instance]['simple'].keys())}, RB Keys: {list(collab_data[instance]['regular'].keys())}, Metadata: {collab_data[instance]['metadata']}")




if __name__ == "__main__":
    input_files = {
        "simple": "resources/data/en/sb_so_test.json",
        "regular": "resources/data/en/rb_so_test.json",
        "regular_challenging": "resources/data/en/rb_co_test.json"
    }
    output_file = "collab_data_regular.json"

    codata = PrepareCollabData("resources/config/en/taskconfig.json")
    codata.run(input_files, output_file)


