import os
import logging
import argparse

from models.modules import BertNerData as NerData
from models.modules.models.bert_models import BertBiLSTMAttnCRF
from models.modules import NerLearner
from models.modules.data.bert_data import get_bert_data_loader_for_predict
from models.modules.train.train import validate_step
from pos_tagger.tools import Tools

import torch
from sklearn_crfsuite.metrics import flat_classification_report


torch.cuda.set_device(0)


class PosTaggerTrainer:
    def __init__(self, target_language, bert_path):
        """
        Possible models:
            BertBiLSTMCRF: Encoder + Decoder (BiLSTM + CRF)
            BertBiLSTMAttnCRF: Encoder + Decoder (BiLSTM + MultiHead Attention + CRF)
            BertBiLSTMAttnNMT: Encoder + Decoder (LSTM + Bahdanau Attention - NMT Decode)
            BertBiLSTMAttnCRFJoint: Encoder + Decoder (BiLSTM + MultiHead Attention + CRF) + PoolingLinearClassifier
            BertBiLSTMAttnNMTJoint: Encoder + Decoder (LSTM + Bahdanau Attention - NMT Decode) + LinearClassifier
        """

        self.pos_language_labels = {
            "эвенкийский": ["ADV", "VERB", "PRON", "NOUN", "ADJ", "DET"],
            "селькупский": ["NOUN", "VERB", "ADV", "PRON", "ADP"],
            "вепсский": ["NOUN", "ADV", "VERB", "NUM", "PRON", "AUX", "ADJ"],
            "карельский_кар": ["NOUN", "VERB", "PRON", "AUX", "ADJ"],
            "карельский_ливвик": ["NOUN", "ADJ", "NUM", "VERB", "PRON"],
            "карельский_людик": ["VERB", "NOUN", "PRON"]
        }

        self.target_language = target_language
        self.max_seq_len = 128
        self.enc_hidden_dim = 256
        self.n_epochs = 2

        self.tools = Tools()

        for pos in self.pos_language_labels[target_language]:

            self.tools.init_logging('morpho_log_%s_%s' % (target_language, pos))

            for fold in [0, 1, 2, 3]:

                self.fold = fold
                self.pos = pos

                self.project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
                self.data_path = os.path.join(self.project_path + '/pos_tagger/folds_%s_morpho/%s/%s/' % (
                    target_language, self.fold, self.pos))
                self.models = os.path.join(self.project_path + '/pos_tagger/models/')
                self.bert_model_dir = bert_path
                self.stop_labels = ['<pad>', '[CLS]', 'X', 'I_X']

                self.init_checkpoint_pt = os.path.join(self.bert_model_dir, "pytorch_model.bin")
                self.bert_config_file = os.path.join(self.bert_model_dir, "bert_config.json")
                self.vocab_file = os.path.join(self.bert_model_dir, "vocab.txt")

                self.data, self.support_labels = self.get_dataset()
                self.model = self.get_model()
                self.learner = self.train()
                self.predict()

    def get_dataset(self):
        data = NerData.create(os.path.join(self.data_path + "train.csv"),
                              os.path.join(self.data_path + "valid.csv"),
                              self.vocab_file,
                              max_seq_len=self.max_seq_len)
        support_labels = [l for l in data.id2label if l not in self.stop_labels]
        return data, support_labels

    def get_model(self):
        return BertBiLSTMAttnCRF.create(len(self.data.label2idx),
                                        self.bert_config_file,
                                        self.init_checkpoint_pt,
                                        enc_hidden_dim=256)

    def train(self):
        learner = NerLearner(self.model,
                             self.data,
                             best_model_path=os.path.join(self.models, "bilstm_attn_cased_%s_morpho_%s.cpt" % (
                                 self.target_language, self.pos)),
                             lr=0.001,
                             clip=1.0,
                             sup_labels=self.support_labels,
                             t_total=self.n_epochs * len(self.data.train_dl))
        learner.fit(self.n_epochs, target_metric='f1')
        return learner

    def custom_voting_choicer(self, tok_map, labels):
        label = []
        # print(tok_map)
        # print(labels)
        for origin_idx in tok_map:
            if origin_idx >= 0:
                label.append(labels[origin_idx])
                # print(origin_idx, labels[origin_idx])
        assert "[SEP]" not in label
        # print(label, '\n')
        return label

    def custom_bert_labels2tokens(self, dl, labels, fn=custom_voting_choicer):
        res_tokens = []
        res_labels = []
        for f, l in zip(dl.dataset, labels):
            label = fn(f.tok_map, l)
            res_tokens.append(f.tokens[1:])
            res_labels.append(label)
        return res_tokens, res_labels

    def predict(self):

        dl = get_bert_data_loader_for_predict(os.path.join(self.data_path + "/valid.csv"), self.learner)
        self.learner.load_model()
        preds = self.learner.predict(dl)

        pred_tokens, pred_labels = self.custom_bert_labels2tokens(dl,
                                                                  preds,
                                                                  fn=self.custom_voting_choicer)

        true_tokens, true_labels = self.custom_bert_labels2tokens(dl,
                                                                  [x.labels for x in dl.dataset],
                                                                  fn=self.custom_voting_choicer)

        assert pred_tokens == true_tokens
        tokens_report = flat_classification_report(true_labels, pred_labels, labels=self.learner.sup_labels, digits=3)

        logging.info('#' * 100)
        logging.info('Language: ' + self.target_language)
        logging.info('POS: ' + self.pos)
        logging.info('Fold: ' + str(self.fold))
        logging.info(tokens_report)
        logging.info('#' * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data for Evaluator')
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--bert_path', type=str, required=True)
    args = parser.parse_args()
    target_language = args.language
    bert_path = args.bert_path

    pos_tagger_trainer = PosTaggerTrainer(target_language=target_language, bert_path=bert_path)
