import json
with open("instances.json", "r") as file:
    instances = json.load(file)

with open("generous_instances.json", "r") as second_file:
    generous_instances = json.load(second_file)

for i in range(len(instances["experiments"])):
    if not str(instances["experiments"][i]["game_instances"]) == str(generous_instances["experiments"][i]["game_instances"]):
        print(i, instances["experiments"][i]["variable"], generous_instances["experiments"][i]["name"])