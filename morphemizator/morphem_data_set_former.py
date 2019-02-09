import os
import re
import distutils.dir_util
import argparse
from collections import namedtuple

from sklearn.model_selection import KFold, train_test_split


DataObject = namedtuple('DataObject', ['forma', 'morphems'])


class MorphemsDataSetCreator:
    def __init__(self, folds_n):
        self.project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        self.path2data_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../')) + '/morphemizator/data/train_24_01.txt'

        self.folds_n = folds_n
        self.sentences = self.read_file()
        self.corrupted_lemmas = ['_']

    def read_file(self):
        strings = []
        file_object = open(self.path2data_file, "r")
        for line in file_object.readlines():
            line = line.strip()
            strings.append(line)
        return strings

    def form_sent_pairs(self):
        data_set_forma2morphs_sentences = list()
        for sent in self.sentences:
            sent = sent.split('\t')
            if sent != ['']:
                data_set_forma2morphs_sentences.append(sent)
        return data_set_forma2morphs_sentences

    def save_data_set(self, data, project_path, prefix, data_type):
        for el in (('src', 0), ('trg', 1)):
            with open(project_path + '%s_%s.%s' % (el[0], data_type, prefix), 'w') as outfile:
                for pair in data:
                    outfile.write(pair[el[1]] + '\n')
                outfile.close()

    def save_spec_test_format(self, data, project_path, prefix, data_type):
        with open(project_path + '%s.%s' % (data_type, prefix), 'w') as outfile:
            for s_pair in data:

                if data_type.endswith('INPUT'):
                    file_string = "%s	_" % (s_pair[0],)
                    outfile.write(file_string + '\n')

                else:
                    file_string = "%s" % (s_pair[1],)
                    outfile.write(file_string + '\n')

            outfile.close()

    def get_morph_borders(self, data):
        borders = []
        for pair in data:
            forma = pair[0]
            morphems = [m.split('_')[0] for m in pair[1].split()]
            bs = self.find_char_positions(morphems, forma)

            if len(bs) > 1:
                borders_indexes = [1 if m[-1] == s else 0 for m in morphems for s in m]
                borders_indexes = borders_indexes[:-1] + [0]
                chars = list(forma)

                try:
                    borders.append((' '.join(list(forma)),
                                    ' '.join([chars[index_char] + '_' if borders_indexes[index_char] == 1
                                    else chars[index_char] for index_char in range(len(borders_indexes))])))
                except IndexError:
                    print(chars, borders_indexes, morphems)
            else:
                borders.append((' '.join(list(forma)), ' '.join(list(forma))))

        return borders

    def find_char_positions(self, patterns, initial_text):
        re_str = '(' + '|'.join(patterns) + ')'
        reg_pat = re.compile(re_str)
        strings_indexes = [[el.group(), el.start(), el.end()] for el in reg_pat.finditer(initial_text)]
        return strings_indexes

    def split_data(self, data):
        kf = KFold(n_splits=self.folds_n, shuffle=True, random_state=1024)
        f_n = 0

        for train_index, test_index in kf.split(data):

            X_train, X_test = [data[i] for i in train_index], [data[i] for i in test_index]
            X_test, X_valid = train_test_split(X_test, random_state=1024)
            project_path = os.path.join(self.project_path + '/morphemizator/folds/%s/' % (f_n,))
            distutils.dir_util.mkpath(project_path)

            data_set = {

                "tokens_chars": {
                    "train": self.get_morph_borders(X_train),
                    "test": self.get_morph_borders(X_test),
                    "valid": self.get_morph_borders(X_valid)
                }

            }

            for ds in data_set:
                self.save_data_set(data_set[ds]['train'], project_path, 'train', ds)
                self.save_data_set(data_set[ds]['test'], project_path, 'test', ds)
                self.save_data_set(data_set[ds]['valid'], project_path, 'valid', ds)

                print('Fold #%s; train: %s; test: %s: valid: %s; data_type: %s;' % (
                    f_n, len(data_set[ds]['train']), len(data_set[ds]['test']), len(data_set[ds]['valid']), ds))

            # self.save_spec_test_format(data_set['tokens_chars']['test'], project_path, 'txt', 'EXERCISE_INPUT')
            # self.save_spec_test_format(data_set['tokens_chars']['test'], project_path, 'txt', 'EXERCISE_TEST')

            print('#' * 100)

            f_n += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data for Evaluator')
    parser.add_argument('--folds_n', type=int, required=True)
    args = parser.parse_args()

    pos_data_set_creator = MorphemsDataSetCreator(folds_n=args.folds_n)
    data_set_forma2lpos_sentences = pos_data_set_creator.form_sent_pairs()
    pos_data_set_creator.split_data(data_set_forma2lpos_sentences)
