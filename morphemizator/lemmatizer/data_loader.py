import os
import itertools
from collections import namedtuple


DataObject = namedtuple('DataObject', ['token_id', 'forma', 'lemma', 'pos', 'morpho'])


class DataLoader:
    def __init__(self, language):
        self.path2data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../')) + \
                              '/lemmatizer/data/train_%s.txt' % (language,)

    def read_file(self):
        strings = []
        file_object = open(self.path2data_file, "r")
        for line in file_object.readlines():
            line = line.strip()
            if not line.startswith('#'):
                strings.append(line)
        return strings

    def form_sentences(self, data):
        sentences = [[t.split('\t') for t in list(y)] for x, y in itertools.groupby(data, lambda z: z == '') if not x]
        return [[DataObject(t[0], t[1], t[2], t[3], t[5]) for t in s] for s in sentences]


if __name__ == '__main__':
    data_loader = DataLoader()
    strings = data_loader.read_file()
    sentences = data_loader.form_sentences(strings)
