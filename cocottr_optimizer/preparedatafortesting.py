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

        
    
    def run(self, trainfile, testfiles):
        if not trainfile or not testfiles:
            raise ValueError("Both trainfile and testfiles must be provided.")
        
        train_data = self.readfile(trainfile)
        test_data_list = {tftype: self.readfile(testfile) for tftype, testfile in testfiles.items()}

        if train_data is None or any(td is None for td in test_data_list.values()):
            raise ValueError("Failed to read one or more data files.")
        

        instancedata = []

        for board_type, objs_type in train_data.items():
            for boardobj, num_shapes in objs_type.items():
                for total_shapes, combo_names in num_shapes.items():
                    for combo_name in num_shapes[total_shapes]:
                        #print(len(num_shapes[total_shapes][combo_name]))
                        for index, tsample in enumerate(num_shapes[total_shapes][combo_name]):
                            instancedata.append({'simple': tsample})                            
                            for tftype, test_data in test_data_list.items():
                                try:
                                    if tftype == "simple":
                                        test_sample = test_data[board_type][boardobj][total_shapes][combo_name][index]
                                    else:
                                        test_sample = test_data["regular"]["simple"][total_shapes][combo_name][index]
                                    if not test_sample:
                                        raise ValueError(f"Empty test sample for {board_type}, {boardobj}, {total_shapes}, {combo_name}, index {index} in {tftype} data.")
                                    
                                    if test_sample["combo_name"] != tsample["combo_name"]:
                                        raise ValueError(f"Combo name mismatch for {board_type}, {boardobj}, {total_shapes}, {combo_name}, index {index} in {tftype} data.")
                                    
                                    if test_sample["shapes"] != tsample["shapes"]:
                                        raise ValueError(f"Shapes mismatch for {board_type}, {boardobj}, {total_shapes}, {combo_name}, index {index} in {tftype} data.")

                                    #compare the same object are not
                                    code_status = self._executeandcomparesameobject(tsample["code"], test_sample["code"])
                                    if not code_status:
                                        print(f"Object mismatch for {board_type}, {boardobj}, {total_shapes}, {combo_name}, index {index} in {tftype} data.")
                                        # delete the last instance added since the object is not same in test and train
                                        instancedata.pop()
                                        break


                                    if tftype == "simple":
                                        if test_sample["x"] == tsample["x"] and test_sample["y"] == tsample["y"]:
                                            raise ValueError(f"Position not varied for {board_type}, {boardobj}, {total_shapes}, {combo_name}, index {index} in {tftype} data.")
                                    else:
                                        if len(test_sample["repeat_locations"]) == 1:
                                            raise ValueError(f"Only one repeat location for {board_type}, {boardobj}, {total_shapes}, {combo_name}, index {index} in {tftype} data.")

                                    if tftype == "simple":
                                        instancedata[-1]['simple_reuse'] = test_sample
                                    else:
                                        instancedata[-1]['regular'] = test_sample

                                except (KeyError, IndexError):
                                    raise ValueError(f"Missing test sample for {board_type}, {boardobj}, {total_shapes}, {combo_name}, index {index} in {tftype} data.")
                                
                                
        print(f"Total instances prepared: {len(instancedata)}")
        self.writefile("cocottrdata.json", instancedata)



if __name__ == "__main__":
    pdc = PrepareDataForClemTesting()
    pdc.run(
        trainfile="resources/data/en/sb_so_train.json",
        testfiles={
            "simple": "resources/data/en/sb_so_test.json",
            "regular": "resources/data/en/rb_so_test.json"
        })