import os
import numpy as np

from MorphemRuEval2019.lemmatizer.tools import Tools


class LemmaDataSetStatistics:
    def __init__(self):
        project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        self.tools = Tools()
        self.path2lemma_data_set_file = project_path + '/data/lemma_data_set.txt'

        self.data = self.tools.read_file(self.path2lemma_data_set_file)

    def get_stata(self):
        data_tokens = sorted(set([t for p in self.data for t in p.split()]))
        data_chars = sorted(set([char for p in self.data for t in p.split() for char in t]))
        len_tokens = [len(t) for t in data_tokens]

        print('Voc tokens size: ', len(data_tokens))
        print('Voc chars size: ', len(data_chars))
        print('Max token len: ', max(len_tokens))
        print('Min token len: ', min(len_tokens))
        print('Average token len: ', np.average(len_tokens))


if __name__ == '__main__':
    lemma_data_set_statistics = LemmaDataSetStatistics()
    lemma_data_set_statistics.get_stata()
