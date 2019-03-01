import os

import pandas as pd


pos_language_labels = {
            "эвенкийский": ["INTJ", "ADV", "PART", "VERB", "PRON", "NOUN", "ADJ", "DET", "CCONJ", "NUM", "SCONJ"],
            "селькупский": ["NOUN", "VERB", "ADV", "CCONJ", "DET", "PART", "PRON", "NUM", "ADJ", "INTJ", "ADP"],
            "вепсский": ["NOUN", "ADV", "VERB", "NUM", "PRON", "PART", "CCONJ", "AUX", "SCONJ", "INTJ", "ADP", "ADJ"],
            "карельский_кар": ["CCONJ", "NOUN", "VERB", "PRON", "AUX", "ADJ", "ADV", "NUM", "ADP", "INTJ"],
            "карельский_ливвик": ["ADV", "NOUN", "ADJ", "SCONJ", "ADP", "NUM", "CCONJ", "PART", "VERB", "PRON", "INTJ"],
            "карельский_людик": ["ADV", "PRON", "VERB", "NOUN", "ADJ", "NUM", "CCONJ", "INTJ", "AUX", "PART", "ADP"]
        }


for l in ["карельский_людик", "карельский_ливвик", "эвенкийский", "селькупский", "карельский_кар", "вепсский"]:
    for f_n in [0, 1, 2, 3]:
        project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        for pos in pos_language_labels[l]:

            try:

                data_path = os.path.join(project_path + '/pos_tagger/folds_%s_morpho/%s/%s' % (l, f_n, pos))

                data_set = {
                    "train": {
                        "x": "src_sentence_tokens_pos.train",
                        "y": "trg_sentence_tokens_pos.train",
                        "data": []
                    },

                    "valid": {
                        "x": "src_sentence_tokens_pos.valid",
                        "y": "trg_sentence_tokens_pos.valid",
                        "data": []
                    },

                    "test": {
                        "x": "src_sentence_tokens_pos.test",
                        "y": "trg_sentence_tokens_pos.test",
                        "data": []
                    }
                }


                def read_file(path2data_file):
                    strings = []
                    file_object = open(path2data_file, "r")
                    for line in file_object.readlines():
                        line = line.strip()
                        strings.append(line)
                    return strings


                for s in data_set:
                    for f in data_set[s]:
                        x = read_file(os.path.join(data_path, data_set[s]["x"]))
                        y = read_file(os.path.join(data_path, data_set[s]["y"]))
                        data_set[s]["data"] = [
                            [el for el in list(zip(w[0].split(), w[1].split())) if el != (' ', ' ')] for w in list(zip(x, y))
                        ]

                train = data_set["train"]["data"] + data_set["valid"]["data"]
                test = data_set["test"]["data"]

                x_train = [[' '.join(['I_' + el[1].replace('PROPN', 'NOUN') for el in s]), ' '.join([el[0] for el in s])] for s in train]
                x_test = [[' '.join(['I_' + el[1].replace('PROPN', 'NOUN') for el in s]), ' '.join([el[0] for el in s])] for s in test]
                max_len = max([len(s[1].split()) for s in x_test + x_train])

                print(x_train[10])
                print(x_test[10])
                print(max_len)

                train_df = pd.DataFrame(x_train, columns=["0", "1"])
                train_df.to_csv(data_path + "train.csv", index=False)

                valid_df = pd.DataFrame(x_test, columns=["0", "1"])
                valid_df.to_csv(data_path + "valid.csv", index=False)

            except FileNotFoundError:
                pass
