import numpy as np


def convert_pairs(pairs_path, db):
    def name_comp(name_, id_):
        return f'./{db}/{name_}_{int(id_):04d}.jpg'

    def lfw(line_):
        entries = line_.split()
        if len(entries) == 3:
            name, ida, idb = entries
            fna = name_comp(name, ida)
            fnb = name_comp(name, idb)
        else:
            namea, ida, nameb, idb = entries
            fna = name_comp(namea, ida)
            fnb = name_comp(nameb, idb)
        return [fna, fnb], len(entries) == 3

    func_dict = {'lfw': lfw}
    func = func_dict[db]
    result = []
    matches = []
    for line in open(pairs_path):
        pair, match = func(line)
        result += pair
        matches.append(match)
    return np.unique(result), list(zip(result[::2], result[1::2])), np.array(matches)
