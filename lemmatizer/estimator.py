import os
import argparse
from itertools import groupby

from tools import Tools


class Estimator:
    def __init__(self, language):
        self.language = language
        self.project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../lemmatizer/folds_%s/' %
                                                         (self.language,)))
        self.folds_n = 10
        self.tools = Tools()

    def compare(self, predict, true):
        predict = [[t.split('\t') for t in list(y)] for x, y in groupby(predict, lambda z: z == '') if not x]
        true = [[t.split('\t') for t in list(y)] for x, y in groupby(true, lambda z: z == '') if not x]

        predict = [[w for w in s if w[3] != 'X'] for s in predict]
        true = [[w for w in s if w[3] != 'X'] for s in true]

        acc_per_forms = 0
        total_forms = 0

        acc_per_sent = 0
        total_sent = 0

        for index_sent in range(len(true)):
            true_sent_forms = 0
            all_sent_forms = len(true[index_sent])

            if len(true[index_sent]) != len(predict[index_sent]):
                print(true[index_sent], index_sent)
                assert 0

            for index_t in range(len(true[index_sent])):
                if true[index_sent][index_t] == predict[index_sent][index_t]:
                    true_sent_forms += 1

            acc_per_forms += true_sent_forms
            total_forms += all_sent_forms

            if true_sent_forms == all_sent_forms:
                acc_per_sent += 1
            total_sent += 1

        acc_per_forms = acc_per_forms / total_forms
        acc_per_sent = acc_per_sent / total_sent

        return {'acc_per_forms': acc_per_forms, 'acc_per_sent': acc_per_sent}

    def estimate(self):
        results = {}
        for folds in range(self.folds_n):
            fold_path = os.path.join(self.project_path, str(folds))

            try:
                results[folds] = {
                    'predict': self.tools.read_file(os.path.join(fold_path, 'EXERCISE_PRED.txt')),
                    'test': self.tools.read_file(os.path.join(fold_path, 'EXERCISE_TEST.txt'))
                }

            except FileNotFoundError:
                pass

        for f in results:
            metrics = self.compare(results[f]['predict'], results[f]['test'])
            print('Language# %s; Fold#: %s; Acc per forms: %s; Acc per sentences: %s' % (
                self.language, f, metrics['acc_per_forms'], metrics['acc_per_sent']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data for Evaluator')
    parser.add_argument('--language', type=str, required=True)
    args = parser.parse_args()

    estimator = Estimator(language=args.language)
    estimator.estimate()
