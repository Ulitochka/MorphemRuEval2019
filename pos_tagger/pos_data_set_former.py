import os
import json
from tqdm import tqdm
import distutils.dir_util
import argparse
from collections import Counter

from sklearn.model_selection import KFold, train_test_split
import pandas as pd

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

        for pos in tqdm(self.pos_language_labels[language], desc=language):
            data_set_forma2lemma_sentences, labels_stata = self.form_sent_pairs(pos=pos)
            if sum([labels_stata[el] for el in labels_stata if el != 'X']) >= 100:
                self.split_data(data_set_forma2lemma_sentences, pos=pos)

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
                sent.append(
                    (' '.join(['I_' + el[1].replace('PROPN', 'NOUN') for el in r]),
                     ' '.join([el[0] for el in r]))
                )
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

            x_train = data_set["sentence_tokens_pos"]["train"] + data_set["sentence_tokens_pos"]["valid"]
            x_test = data_set["sentence_tokens_pos"]["test"]

            train_df = pd.DataFrame(x_train, columns=["0", "1"])
            train_df.to_csv(project_path + "/train.csv", index=False)

            valid_df = pd.DataFrame(x_test, columns=["0", "1"])
            valid_df.to_csv(project_path + "/valid.csv", index=False)

            f_n += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data for Evaluator')
    parser.add_argument('--folds_n', type=int, required=True)
    parser.add_argument('--language', type=str, required=True)
    args = parser.parse_args()

    LemmaDataSetCreator(folds_n=args.folds_n, language=args.language)
