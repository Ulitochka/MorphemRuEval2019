import os

import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from bayes_opt import BayesianOptimization

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


f_n = 0
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
project_path = os.path.join(project_path + '/morphemizator/folds_эвенкийский/%s/' % (f_n,))


data_set = {
    "train": {
        "x": "src_tokens_chars.train",
        "y": "trg_tokens_chars.train",
        "data": None
    },

    "valid": {
        "x": "src_tokens_chars.valid",
        "y": "trg_tokens_chars.valid",
        "data": None
    },

    "test": {
        "x": "src_tokens_chars.test",
        "y": "trg_tokens_chars.test",
        "data": None
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
        x = read_file(os.path.join(project_path, data_set[s]["x"]))
        y = read_file(os.path.join(project_path, data_set[s]["y"]))
        data_set[s]["data"] = [
            [el for el in list(zip(list(w[0]), list(w[1]))) if el != (' ', ' ')] for w in list(zip(x, y))
        ]

train = data_set["train"]["data"] + data_set["valid"]["data"]
test = data_set["test"]["data"]

########################################################################################################################

vowels_0 = ["a", "ɑ", "æ", "ɛ", "e", "i", "ɨ", "o", "ɵ", "u", "ʉ"]
vowels_1 = ["ɐ", "ə", "ɪ", "ɨ", "ʉ", "ʊ"]
r_vowels = ["е", 'и', 'о',  'а', 'у', 'ы', 'э', 'я']
spec_chars = ["ː"]


def char_type(char):
    if char in vowels_0 + vowels_1:
        return "v"
    if char in spec_chars:
        return 's'
    else:
        return "c"


def char_position(i):
    return i


def char2features(word, i):
    char = word[i][0]

    features = {
        'bias': 1.0,
        'char.lower()': char.lower(),
        'char.type()': char_type(char),
        'char.position()': char_position(i)
    }

    if i > 0:
        char1 = word[i-1][0]
        features.update({'-1:char.lower()': char1.lower(),
                         '-1:char.type()': char_type(char1),
                         '-1:char.position()': char_position(i-1)
                         })
    if i > 1:
        char1 = word[i-2][0]
        features.update({'-2:char.lower()': char1.lower(),
                         '-2:char.type()': char_type(char1),
                         '-2:char.position()': char_position(i-2)
                         })
    if i > 2:
        char1 = word[i-3][0]
        features.update({'-3:char.lower()': char1.lower(),
                         '-3:char.type()': char_type(char1),
                         '-3:char.position()': char_position(i-3)
                         })
    if i > 3:
        char1 = word[i-4][0]
        features.update({'-4:char.lower()': char1.lower(),
                         '-4:char.type()': char_type(char1),
                         '-4:char.position()': char_position(i-4)
                         })
    if i > 4:
        char1 = word[i-5][0]
        features.update({'-5:char.lower()': char1.lower(),
                         '-5:char.type()': char_type(char1),
                         '-5:char.position()': char_position(i-5)
                         })
    if i > 5:
        char1 = word[i-6][0]
        features.update({'-6:char.lower()': char1.lower(),
                         '-6:char.type()': char_type(char1),
                         '-6:char.position()': char_position(i-6)
                         })
    if i > 6:
        char1 = word[i-7][0]
        features.update({'-7:char.lower()': char1.lower(),
                         '-7:char.type()': char_type(char1),
                         '-7:char.position()': char_position(i-7)
                         })
    if i > 7:
        char1 = word[i-8][0]
        features.update({'-8:char.lower()': char1.lower(),
                         '-8:char.type()': char_type(char1),
                         '-8:char.position()': char_position(i-8)
                         })
    else:
        features['BOS'] = True

    if i < len(word)-1:
        char1 = word[i+1][0]
        features.update({'+1:char.lower()': char1.lower(),
                         '+1:char.type()': char_type(char1),
                         '+1:char.position()': char_position(i+1)
                         })
    if i < len(word)-2:
        char1 = word[i+2][0]
        features.update({'+2:char.lower()': char1.lower(),
                         '+2:char.type()': char_type(char1),
                         '+2:char.position()': char_position(i+2)
                         })
    if i < len(word)-3:
        char1 = word[i+3][0]
        features.update({'+3:char.lower()': char1.lower(),
                         '+3:char.type()': char_type(char1),
                         '+3:char.position()': char_position(i+3)
                         })
    if i < len(word)-4:
        char1 = word[i+4][0]
        features.update({'+4:char.lower()': char1.lower(),
                         '+4:char.type()': char_type(char1),
                         '+4:char.position()': char_position(i+4)
                         })
    if i < len(word)-5:
        char1 = word[i+5][0]
        features.update({'+5:char.lower()': char1.lower(),
                         '+5:char.type()': char_type(char1),
                         '+5:char.position()': char_position(i+5)
                         })
    if i < len(word)-6:
        char1 = word[i+6][0]
        features.update({'+6:char.lower()': char1.lower(),
                         '+6:char.type()': char_type(char1),
                         '+6:char.position()': char_position(i+6)})
    if i < len(word)-7:
        char1 = word[i+7][0]
        features.update({'+7:char.lower()': char1.lower(),
                         '+7:char.type()': char_type(char1),
                         '+7:char.position()': char_position(i+7)})
    else:
        features['EOS'] = True

    return features


def word2features(word):
    return [char2features(word, i) for i in range(len(word))]


def word2labels(word):
    return [label for char, label in word]


X_train = [word2features(w) for w in train]
y_train = [word2labels(w) for w in train]

X_test = [word2features(w) for w in test]
y_test = [word2labels(w) for w in test]


print('X_train: ', len(X_train))
print('y_test: ', len(y_test))

########################################################################################################################

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
    all_possible_states=True
)

crf.fit(X_train, y_train)

########################################################################################################################

labels = list(crf.classes_)
labels.remove('0')
print('aim labels:', labels)

y_pred = crf.predict(X_test)
metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)

sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))

########################################################################################################################

algo = {0: "lbfgs", 1: "l2sgd", 2: "ap", 3: "pa", 4: "arow"}


def optimise_crf(x_train, y_train, x_test, y_test, C1, C2):

    def target(C1, C2):
        clf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=C1,
            c2=C2,
            max_iterations=100,
            all_possible_transitions=True,
            all_possible_states=True)

        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        result = metrics.flat_f1_score(y_test, predictions, average='weighted', labels=labels)
        return result

    bo = BayesianOptimization(f=target, pbounds={'C1': C1, 'C2': C2}, random_state=1024)
    bo.maximize(init_points=10, n_iter=100)
    max_target = max([t['target'] for t in bo.res])
    aim_params = [el for el in bo.res if el['target'] == max_target][0]
    print(aim_params)
    return aim_params


optimise_crf(x_train=X_train,
             y_train=y_train,
             x_test=X_test,
             y_test=y_test,
             C1=(0.5, 1.0),
             C2=(0.05, 1.0))
