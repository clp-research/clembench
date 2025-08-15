import json

with open("curated_home_deliver_three_adventures_v2_2_a.json") as infile:
    curated_adventures = json.load(infile)

# print(curated_adventures)

for difficulty in curated_adventures:
    print(difficulty)
    for adventure in curated_adventures[difficulty]:
        adventure['prompt_template_set'] = 'home_delivery'
        print(adventure)

with open("curated_home_deliver_three_adventures_v2_2.json", 'w') as outfile:
    json.dump(curated_adventures, outfile)