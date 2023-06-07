"""
Script that checks whether no value overlap in the slot values.
"""

from collections import Counter
import json

from games.privateshared.constants import EXPERIMENTS

for experiment in EXPERIMENTS:
    print(f'Checking {experiment}...')
    with open(f'games/privateshared/resources/texts/{experiment}/slot_values.json', 'r') as file:
        slotvalues = json.load(file)

    all_values = [w for vlist in slotvalues.values() for w in vlist]

    counts = Counter(all_values)
    for value, count in counts.items():
        if count > 1:
            print('\t', value)

    for w1 in all_values:
        for w2 in all_values:
            if w1 in w2 and w1 != w2:
                print('\t', w1, w2)
