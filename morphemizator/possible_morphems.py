import os
from collections import Counter


FREQ_THR = 5

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
data_path = os.path.join(project_path + '/morphemizator/data/train_24_01.txt')


def read_file(path2data_file):
    morphems = []
    file_object = open(path2data_file, "r")
    for line in file_object.readlines():
        line = line.strip()
        p = line.split('\t')
        try:
            for m in p[1].split():
                morphems.append(m.split('_')[0])
        except IndexError:
            pass
    return morphems


data = read_file(data_path)
data = Counter(data)

with open(os.path.join(project_path + '/morphemizator/data/', 'possible_morphems.txt'), 'w') as outfile:
    for m in data:
        if data[m] > FREQ_THR:
            outfile.write(m + '\n')
    outfile.close()
