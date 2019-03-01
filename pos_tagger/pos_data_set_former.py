import os
import json
import distutils.dir_util
import argparse
from collections import Counter

from sklearn.model_selection import KFold, train_test_split

from data_loader import DataLoader


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class LemmaDataSetCreator:
    def __init__(self, folds_n, language):
        data_loader = DataLoader(language=language)
        self.folds_n = folds_n
        self.language = language
        strings = data_loader.read_file()
        self.sentences = data_loader.form_sentences(strings)
        self.stop_lemma = ['()', '_']

        self.project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        self.corrupted_lemmas = ['_', 'UNKN']

        self.pos_language_labels = {
            "эвенкийский": ["INTJ", "ADV", "PART", "VERB", "PRON", "NOUN", "ADJ", "DET", "CCONJ", "NUM", "SCONJ"],
            "селькупский": ["NOUN", "VERB", "ADV", "CCONJ", "DET", "PART", "PRON", "NUM", "ADJ", "INTJ", "ADP"],
            "вепсский": ["NOUN", "ADV", "VERB", "NUM", "PRON", "PART", "CCONJ", "AUX", "SCONJ", "INTJ", "ADP", "ADJ"],
            "карельский_кар": ["CCONJ", "NOUN", "VERB", "PRON", "AUX", "ADJ", "ADV", "NUM", "ADP", "INTJ"],
            "карельский_ливвик": ["ADV", "NOUN", "ADJ", "SCONJ", "ADP", "NUM", "CCONJ", "PART", "VERB", "PRON", "INTJ"],
            "карельский_людик": ["ADV", "PRON", "VERB", "NOUN", "ADJ", "NUM", "CCONJ", "INTJ", "AUX", "PART", "ADP"]
        }

        for pos in self.pos_language_labels[language]:
            data_set_forma2lemma_sentences, labels_stata = self.form_sent_pairs(pos=pos)

            print(language, pos, labels_stata)

            if sum([labels_stata[el] for el in labels_stata if el != 'X']) >= 100:
                self.split_data(data_set_forma2lemma_sentences, pos=pos)

        print('%' * 100)

    def form_sent_pairs(self, pos):

        labels_stata = []

        data_set_forma2lemma_sentences = list()
        for sent in self.sentences:
            sentence_forma = ' '.join([t.forma + '%' + t.morpho.replace('_', 'X')
                                       if t.pos == pos else t.forma + '%' + 'X' for t in sent])
            sentence_lemma = ' '.join([t.lemma for t in sent if t.lemma not in self.corrupted_lemmas])

            for el in sentence_forma.split():
                try:
                    labels_stata.append(el.split('%')[1])
                except IndexError:
                    pass

            if [sentence_forma, sentence_lemma] != ['', '']:
                data_set_forma2lemma_sentences.append(sentence_forma + '@' + sentence_lemma)
        return data_set_forma2lemma_sentences, Counter(labels_stata)

    def sent_pair2sentence_tokens(self, data):
        sent = []
        for p in data:
            try:
                r = [t.split('%') for t in p.split('@')[0].split()]
                sent.append((' '.join([el[0] for el in r]), ' '.join([el[1] for el in r])))
            except IndexError:
                pass
        return sent

    def split_data(self, data, pos):
        kf = KFold(n_splits=self.folds_n, shuffle=True, random_state=1024)
        f_n = 0

        for train_index, test_index in kf.split(data):

            X_train, X_test = [data[i] for i in train_index], [data[i] for i in test_index]
            X_test, X_valid = train_test_split(X_test, random_state=1024)
            project_path = os.path.join(self.project_path + '/pos_tagger/folds_%s_morpho/%s/%s/' % (
                self.language, f_n, pos))
            distutils.dir_util.mkpath(project_path)

            data_set = {

                "sentence_tokens_pos": {
                    "train": self.sent_pair2sentence_tokens(X_train),
                    "test": self.sent_pair2sentence_tokens(X_test),
                    "valid": self.sent_pair2sentence_tokens(X_valid)
                }
            }

            for ds in data_set:
                self.save_data_set(data_set[ds]['train'], project_path, 'train', ds)
                self.save_data_set(data_set[ds]['test'], project_path, 'test', ds)
                self.save_data_set(data_set[ds]['valid'], project_path, 'valid', ds)

                # print('Fold #%s; train: %s; test: %s: valid: %s; data_type: %s;' % (
                #     f_n, len(data_set[ds]['train']), len(data_set[ds]['test']), len(data_set[ds]['valid']), ds))

            self.save_spec_test_format(data_set['sentence_tokens_pos']['test'], project_path, 'txt', 'EXERCISE_INPUT')

            # print('#' * 100)

            f_n += 1

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
                    for t in s_pair[0].split(' '):

                        try:
                            file_string = "_	%s	_	%s	_	_	_	_	_	_" % (t.split('%')[0], t.split('%')[1])
                        except IndexError:
                            file_string = "_	%s	_	_	_	_	_	_	_	_" % (t.split('%')[0],)

                        outfile.write(file_string + '\n')
                else:

                    forms = s_pair[0].split(' ')
                    lemmas = s_pair[1].split(' ')

                    for index_t in range(len(forms)):
                        try:
                            file_string = "_	%s	%s	%s	_	_	_	_	_	_" % (
                                forms[index_t].split('%')[0],
                                lemmas[index_t],
                                forms[index_t].split('%')[1]
                            )
                        except IndexError:
                            file_string = "_	%s	_	%s	_	_	_	_	_	_" % (
                                forms[index_t].split('%')[0],
                                lemmas[index_t])

                        outfile.write(file_string + '\n')

                outfile.write('\n')
            outfile.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data for Evaluator')
    parser.add_argument('--folds_n', type=int, required=True)
    parser.add_argument('--language', type=str, required=True)
    args = parser.parse_args()

    LemmaDataSetCreator(folds_n=args.folds_n, language=args.language)
