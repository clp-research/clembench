import json


class CompareDiffInstances:
    def __init__(self):
        pass

    def readfile(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    

    def run(self, file1, file2):
        file1data = self.readfile(file1)
        file2data = self.readfile(file2)

        file1instances = {}
        for data in file1data:
            for boardtype, boarddata in data.items():
                if boardtype not in file1instances:
                    file1instances[boardtype] = {}
                comboname = boarddata['combo_name']
                shapes = boarddata['shapes']
                colors = boarddata['colors']
                x = boarddata['x'][0]
                y = boarddata['y'][0]
                if comboname not in file1instances[boardtype]:
                    file1instances[boardtype][comboname] = []
                file1instances[boardtype][comboname].append({'shapes': shapes, 'colors': colors, 'x': x, 'y': y})

        file2instances = {}
        for data in file2data:
            for boardtype, boarddata in data.items():
                if boardtype not in file2instances:
                    file2instances[boardtype] = {}
                comboname = boarddata['combo_name']
                shapes = boarddata['shapes']
                colors = boarddata['colors']
                x = boarddata['x'][0]
                y = boarddata['y'][0]
                if comboname not in file2instances[boardtype]:
                    file2instances[boardtype][comboname] = []
                file2instances[boardtype][comboname].append({'shapes': shapes, 'colors': colors, 'x': x, 'y': y})

        for boardtype, combos in file1instances.items():
            for comboname, data in combos.items():
                if boardtype not in file2instances or comboname not in file2instances[boardtype]:
                    print(f"Combo name {comboname} of board type {boardtype} found in file1 but not in file2")
                    input()
                    continue

                file2combo = file2instances[boardtype][comboname]
                if data['shapes'] != file2combo['shapes']:
                    print(f"Shapes mismatch for combo {comboname} of board type {boardtype}: file1 shapes {data['shapes']} vs file2 shapes {file2combo['shapes']}")
                    input()
                if data['colors'] != file2combo['colors']:
                    print(f"Colors mismatch for combo {comboname} of board type {boardtype}: file1 colors {data['colors']} vs file2 colors {file2combo['colors']}")
                    input()
                if data['x'] != file2combo['x'] or data['y'] != file2combo['y']:
                    print(f"Location mismatch for combo {comboname} of board type {boardtype}: file1 location ({data['x']}, {data['y']}) vs file2 location ({file2combo['x']}, {file2combo['y']})")
                    input()



if __name__ == "__main__":
    cdi = CompareDiffInstances()
    cdi.run("resources/data/en/10instancesdata/simplereusedata.json", "resources/data/en/simplereusedata.json")