import os
import json
from utils.prepareasciirep import PrepareASCIIRep

class PrepareDataForClemTesting:
    def __init__(self):
        self.prepareasciirep = PrepareASCIIRep()

    def readfile(self, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data

    def writefile(self, filename, data):
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)    


        

    def _executeandcomparesameobject(self, traincode, testcode):
        trainfunction = traincode["single_turn"]["function"]
        trainusage = traincode["single_turn"]["usage"]
        if "single_turn" in testcode:
            testfunction = testcode["single_turn"]["function"]
            testusage = testcode["single_turn"]["usage"]
        else:
            testfunction = testcode["function"]
            testusage = testcode["output"]

        train_verify = {"function": trainfunction, "usage": trainusage}
        test_verify = {"function": testfunction, "usage": trainusage} #pass trainusage so that we can compare the cells directly for colors, locations, and shapes
        board_size = {"rows":8, "cols":8}
        train_cells = self.prepareasciirep.get_occupied_cells(train_verify, board_size)
        test_cells = self.prepareasciirep.get_occupied_cells(test_verify, board_size)
        if train_cells != test_cells:
            print(f"Object mismatch: Train cells {train_cells} vs Test cells {test_cells}")
            input()
            return False
        return True

        
    
    def run(self, testfile):
        if not testfile:
            raise ValueError("Testfile must be provided.")
        
        test_data = self.readfile(testfile)
        if test_data is None:
            raise ValueError("Failed to read data file.")
        
        instancedata = []

        for board_type, objs_type in test_data.items():
            for boardobj, num_shapes in objs_type.items():
                for total_shapes, combo_names in num_shapes.items():
                    for combo_name in num_shapes[total_shapes]:
                        #print(len(num_shapes[total_shapes][combo_name]))
                        for index, tsample in enumerate(num_shapes[total_shapes][combo_name]):
                            instancedata.append({'simple_reuse': tsample})
                                
        print(f"Total instances prepared: {len(instancedata)}")
        self.writefile("resources/data/en/simplereusedata.json", instancedata)


if __name__ == "__main__":
    pdc = PrepareDataForClemTesting()
    pdc.run("resources/data/en/sb_so_test.json")