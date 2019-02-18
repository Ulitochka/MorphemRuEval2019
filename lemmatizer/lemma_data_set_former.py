import os
import re
import json
import distutils.dir_util
import argparse

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
        self.path2paradigms = self.project_path + '/lemmatizer/data/paradigms_%s.json' % (language,)
        self.form2lemmas = self.project_path + '/lemmatizer/data/form2lemmas_%s.json' % (language,)
        self.corrupted_lemmas = ['_', 'UNKN']

    def form_sent_pairs(self):
        data_set_forma2lemma_sentences = list()
        for sent in self.sentences:
            sentence_forma = ' '.join([t.forma + '%' + t.pos for t in sent if t.pos not in self.corrupted_lemmas])
            sentence_lemma = ' '.join([t.lemma for t in sent if t.lemma not in self.corrupted_lemmas])
            if [sentence_forma, sentence_lemma] != ['', '']:
                data_set_forma2lemma_sentences.append(sentence_forma + '@' + sentence_lemma)
        return data_set_forma2lemma_sentences

    def sent_pair2tokens_chars(self, data, use_pos=True):
        result = []
        for s_pair in data:
            # print(s_pair)
            for t_p in list(zip(s_pair.split('@')[0].split(' '), s_pair.split('@')[1].split(' '))):
                if use_pos:
                    try:

                        src_str = ' '.join(list(t_p[0].split('%')[0]))
                        srs_pos = t_p[0].split('%')[1]
                        trg_str = ' '.join(list(t_p[1]))

                        if [src_str] != ['']:
                            result.append((src_str + ' ' + srs_pos, trg_str))

                    except IndexError:
                        pass
                else:
                    result.append((' '.join(list(t_p[0].split('%')[0])), ' '.join(list(t_p[1]))))
        return result

    def sent_pair2sentence_tokens(self, data):
        return [s_pair.split('@') for s_pair in data]

    def split_data(self, data):
        kf = KFold(n_splits=self.folds_n, shuffle=True, random_state=1024)
        f_n = 0

        for train_index, test_index in kf.split(data):

            X_train, X_test = [data[i] for i in train_index], [data[i] for i in test_index]
            X_test, X_valid = train_test_split(X_test, random_state=1024)
            project_path = os.path.join(self.project_path + '/lemmatizer/folds_%s/%s/' % (self.language, f_n,))
            distutils.dir_util.mkpath(project_path)

            data_set = {

                "token_chars_pos": {
                    "train": self.sent_pair2tokens_chars(X_train, use_pos=True),
                    "test": self.sent_pair2tokens_chars(X_test, use_pos=True),
                    "valid": self.sent_pair2tokens_chars(X_valid, use_pos=True)
                },
                "sentence_tokens": {
                    "train": self.sent_pair2sentence_tokens(X_train),
                    "test": self.sent_pair2sentence_tokens(X_test),
                    "valid": self.sent_pair2sentence_tokens(X_valid)
                }
            }

            for ds in data_set:
                self.save_data_set(data_set[ds]['train'], project_path, 'train', ds)
                self.save_data_set(data_set[ds]['test'], project_path, 'test', ds)
                self.save_data_set(data_set[ds]['valid'], project_path, 'valid', ds)

                print('Fold #%s; train: %s; test: %s: valid: %s; data_type: %s;' % (
                    f_n, len(data_set[ds]['train']), len(data_set[ds]['test']), len(data_set[ds]['valid']), ds))

            self.save_spec_test_format(data_set['sentence_tokens']['test'], project_path, 'txt', 'EXERCISE_INPUT')
            self.save_spec_test_format(data_set['sentence_tokens']['test'], project_path, 'txt', 'EXERCISE_TEST')

            print('#' * 100)

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
                            file_string = "_	%s	%s	_	_	_	_	_	_	_" % (
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

    lemma_data_set_creator = LemmaDataSetCreator(folds_n=args.folds_n, language=args.language)
    data_set_forma2lemma_sentences = lemma_data_set_creator.form_sent_pairs()
    lemma_data_set_creator.split_data(data_set_forma2lemma_sentences)
