import os
import math
import argparse

from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import *
from keras.models import *
from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import *
from keras.utils import to_categorical
import keras.initializers
import gc; gc.collect()
from keras import backend as K
from sklearn.metrics import classification_report, f1_score
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss

parser = argparse.ArgumentParser()

parser.add_argument("--hidden_states", type=int, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--char_random_emb_size", type=int, required=True)
args = parser.parse_args()

HIDDEN_STATES = args.hidden_states
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
CHAR_RANDOM_EMB_SIZE = args.char_random_emb_size

f_n = 0
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
data_path = os.path.join(project_path + '/morphemizator/folds/%s/' % (f_n,))
model_path = os.path.join(project_path + '/morphemizator/models/')

USE_CRF = True


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
        x = read_file(os.path.join(data_path, data_set[s]["x"]))
        y = read_file(os.path.join(data_path, data_set[s]["y"]))
        data_set[s]["data"] = [
            [el for el in list(zip(w[0].split(), w[1].split())) if el != (' ', ' ')] for w in list(zip(x, y))
        ]

train = data_set["train"]["data"] + data_set["valid"]["data"]
test = data_set["test"]["data"]

########################################################################################################################

unique_chars = sorted(set([el for w in train + test for ch in w for el in ch[0]]))
unique_tokens = sorted(set([el[0] for w in train + test for el in w]))
unique_labels = sorted(set([el[1] for w in train + test for el in w]))
max_w_len = max([len(t) for t in unique_tokens])
max_s_len = max(len(s) for s in train + test)
print('max_w_len: ', max_w_len)
print('max_s_len: ', max_s_len)

chars2id = {ch: i + 1 for i, ch in enumerate(unique_chars)}
tokens2id = {t: i + 1 for i, t in enumerate(unique_tokens)}
labels2ind = {l: i + 1 for i, l in enumerate(unique_labels)}
ind2labels = {i + 1: l for i, l in enumerate(unique_labels)}
print("unique_chars: ", len(unique_chars), unique_chars[:10])
print("unique_tokens: ", len(unique_tokens), unique_tokens[:10])
print("unique_labels: ", len(unique_labels))

x_train_chars = [[[chars2id.get(ch) for ch in el[0]] for el in w] for w in train]
x_train_tokens = [[tokens2id.get(el[0]) for el in w] for w in train]
y_train = [[labels2ind.get(el[1]) for el in w] for w in train]
x_test_chars = [[[chars2id.get(ch) for ch in el[0]] for el in w] for w in test]
x_test_tokens = [[tokens2id.get(el[0]) for el in w] for w in test]
y_test = [[labels2ind.get(el[1]) for el in w] for w in test]
print('#' * 100)
print('x_train:', train[:4])
print('x_train_chars: ', x_train_chars[:4])
print('x_train_tokens: ', x_train_tokens[:4])

x_train_tokens_pad = pad_sequences(x_train_tokens, maxlen=max_s_len, value=0, padding='pre', truncating='pre')
x_train_chars_pad = pad_sequences([pad_sequences(el, maxlen=max_w_len, value=0, padding='pre', truncating='pre') for el in x_train_chars])
y_train_pad = pad_sequences(y_train, maxlen=max_s_len, value=0, padding='pre', truncating='pre')
x_test_tokens_pad = pad_sequences(x_test_tokens, maxlen=max_s_len, value=0, padding='pre', truncating='pre')
x_test_chars_pad = pad_sequences([pad_sequences(el, maxlen=max_w_len, value=0, padding='pre', truncating='pre') for el in x_test_chars])
y_test_pad = pad_sequences(y_test, maxlen=max_s_len, value=0, padding='pre', truncating='pre')
print('x_train_tokens_padding: ', x_train_tokens_pad[1])
print('x_train_chars_padding: ', x_train_chars_pad[1], x_train_chars_pad[1].shape)
print('x_train_tokens_pad: ', x_train_tokens_pad.shape)
print('x_train_chars_pad: ', x_train_chars_pad.shape)

y_train = to_categorical(y_train_pad, len(labels2ind)+1)
y_test = to_categorical(y_test_pad, len(labels2ind)+1)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)

########################################################################################################################


def char_matrix(unique_ch):
    embed_vocab = list()
    base_vector = np.zeros(len(unique_chars))
    embed_vocab.append(base_vector)
    for ch in unique_ch:
        features_per_char = np.zeros(len(unique_chars))
        features_per_char[unique_ch.index(ch)] = 1
        embed_vocab.append(features_per_char)
    return np.array(embed_vocab).astype('int8')


def char_emb_random(unique_ch):
    char_embedding_matrix = []
    char_embedding_matrix.append(np.zeros(CHAR_RANDOM_EMB_SIZE))
    for ch in unique_ch:
        limit = math.sqrt(3.0 / CHAR_RANDOM_EMB_SIZE)
        char_vector = np.random.uniform(-limit, limit, CHAR_RANDOM_EMB_SIZE)
        char_embedding_matrix.append(char_vector)
    return np.array(char_embedding_matrix)


def char_matrix_per_token(unique_symbols, unique_tokens):
    embed_vocab = list()
    base_vector = np.zeros(len(unique_symbols) * max_w_len)
    embed_vocab.append(base_vector)
    for tokens in unique_tokens:
        features_per_token = np.array([], dtype='int8')
        for index_chars in range(0, max_w_len):
            array_char = np.zeros((len(unique_symbols),))
            try:
                array_char[unique_symbols.index(tokens[index_chars])] = 1
            except IndexError:
                pass
            features_per_token = np.append(features_per_token, array_char)
        embed_vocab.append(features_per_token)
    return np.array(embed_vocab).astype('int8')


emb_oh_per_char = char_matrix(unique_chars)
print("emb_oh_per_char: ", emb_oh_per_char.shape)

emb_random_per_char = char_emb_random(unique_chars)
print("emb_random_per_char: ", emb_random_per_char.shape)

emb_char_per_token = char_matrix_per_token(unique_chars, unique_tokens)
print("emb_char_per_token: ", emb_char_per_token.shape)

########################################################################################################################


class ReversedLSTM(LSTM):
    def __init__(self, units, **kwargs):
        kwargs['go_backwards'] = True
        super().__init__(units, **kwargs)

    def call(self, inputs, **kwargs):
        y_rev = super().call(inputs, **kwargs)
        return K.reverse(y_rev, 1)


input_char_per_token = Input((max_s_len,), name='input_char_per_token')
char_per_token_emb = Embedding(
    input_dim=len(unique_tokens) + 1,
    output_dim=emb_char_per_token.shape[1],
    input_length=max_s_len,
    weights=[emb_char_per_token],
    mask_zero=False,
    trainable=False
    )(input_char_per_token)

cnn_outputs_char_per_token = []
for el in ((20, 1), (40, 2), (60, 3), (80, 4)):
    cnns_char_per_token = Conv1D(filters=el[0], kernel_size=el[1], padding="same", strides=1)(char_per_token_emb)
    cnns_char_per_token = BatchNormalization()(cnns_char_per_token)
    cnns_char_per_token = Activation('tanh')(cnns_char_per_token)
    cnn_outputs_char_per_token.append(cnns_char_per_token)
cnns_char_per_token = concatenate(cnn_outputs_char_per_token, axis=-1)

hway_input_char_per_token = Input(shape=(K.int_shape(cnns_char_per_token)[-1],))
gate_bias_init_char_per_token = keras.initializers.Constant(-2)
transform_gate_char_per_token = Dense(
    units=K.int_shape(cnns_char_per_token)[-1], bias_initializer=gate_bias_init_char_per_token, activation='sigmoid') \
    (hway_input_char_per_token)
carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(K.int_shape(cnns_char_per_token)[-1],))(transform_gate_char_per_token)
h_transformed_char_per_token = Dense(units=K.int_shape(cnns_char_per_token)[-1])(hway_input_char_per_token)
h_transformed_char_per_token = Activation('relu')(h_transformed_char_per_token)
transformed_gated_char_per_token = Multiply()([transform_gate_char_per_token, h_transformed_char_per_token])
carried_gated_char_per_token = Multiply()([carry_gate, hway_input_char_per_token])
outputs_char_per_token = Add()([transformed_gated_char_per_token, carried_gated_char_per_token])
highway_char_per_token = Model(inputs=hway_input_char_per_token, outputs=outputs_char_per_token)
chars_pre_token_vectors = highway_char_per_token(cnns_char_per_token)


input_char = Input((max_s_len, max_w_len,), name='input_char_emb')
char_per_emb = TimeDistributed(Embedding(
    input_dim=len(unique_chars) + 1,
    output_dim=emb_random_per_char.shape[1],
    input_length=max_w_len,
    weights=[emb_random_per_char],
    mask_zero=False,
    trainable=True
    ))(input_char)

cnn_outputs_char = []
for el in ((20, 1), (40, 2), (60, 3), (80, 4)):
    cnns_char = TimeDistributed(Conv1D(filters=el[0], kernel_size=el[1], padding="same", strides=1))(char_per_emb)
    cnns_char = TimeDistributed(BatchNormalization())(cnns_char)
    cnns_char = TimeDistributed(Activation('tanh'))(cnns_char)
    cnns_char = TimeDistributed(GlobalMaxPooling1D())(cnns_char)
    cnn_outputs_char.append(cnns_char)
cnns_char = concatenate(cnn_outputs_char, axis=-1)

hway_input_char = Input(shape=(K.int_shape(cnns_char)[-1],))
gate_bias_init_char = keras.initializers.Constant(-2)
transform_gate_char = Dense(
    units=K.int_shape(cnns_char)[-1], bias_initializer=gate_bias_init_char, activation='sigmoid')(hway_input_char)
carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(K.int_shape(cnns_char)[-1],))(transform_gate_char)
h_transformed_char = Dense(units=K.int_shape(cnns_char)[-1])(hway_input_char)
h_transformed_char = Activation('relu')(h_transformed_char)
transformed_gated_char = Multiply()([transform_gate_char, h_transformed_char])
carried_gated_char = Multiply()([carry_gate, hway_input_char])
outputs_char = Add()([transformed_gated_char, carried_gated_char])
highway_char = Model(inputs=hway_input_char, outputs=outputs_char)
chars_vectors = TimeDistributed(highway_char)(cnns_char)


word_vects = concatenate([chars_vectors, chars_pre_token_vectors], axis=-1)

if USE_CRF:
    lstm_enc, fh, fc, bh, bc = Bidirectional(LSTM(HIDDEN_STATES, return_sequences=True, return_state=True))(word_vects)
    lstm_dec = Bidirectional(LSTM(HIDDEN_STATES, return_sequences=True))(lstm_enc, initial_state=[bh, bc, fh, fc])
    lyr_crf = CRF(len(labels2ind) + 1)
    outputs = lyr_crf(lstm_dec)
    model_mk2 = Model(inputs=[input_char_per_token, input_char], outputs=outputs)
    model_mk2.compile(optimizer='adam', loss=lyr_crf.loss_function)
else:
    lstm_forward_1 = LSTM(256, return_sequences=True, name='LSTM_1_forward')(word_vects)
    lstm_backward_1 = ReversedLSTM(256, return_sequences=True, name='LSTM_1_backward')(word_vects)
    layer = concatenate([lstm_forward_1, lstm_backward_1], name="BiLSTM_input")
    layer = Bidirectional(LSTM(256, return_sequences=True, name='LSTM_1'))(layer)

    layer = TimeDistributed(Dense(128))(layer)
    layer = TimeDistributed(Dropout(0.5))(layer)
    layer = TimeDistributed(BatchNormalization())(layer)
    layer = TimeDistributed(Activation('relu'))(layer)

    outputs = []
    loss = {}
    prev_layer_name = 'shifted_pred_prev'
    next_layer_name = 'shifted_pred_next'
    prev_layer = Dense(len(labels2ind) + 1, activation='softmax', name=prev_layer_name)
    next_layer = Dense(len(labels2ind) + 1, activation='softmax', name=next_layer_name)
    outputs.append(prev_layer(Dense(128, activation='relu')(lstm_backward_1)))
    outputs.append(next_layer(Dense(128, activation='relu')(lstm_forward_1)))
    loss[prev_layer_name] = loss[next_layer_name] = 'categorical_crossentropy'
    model_mk2 = Model(inputs=[input_char_per_token, input_char], outputs=outputs)
    model_mk2.compile(Adam(clipnorm=5.), loss=loss)


#######################################################################################################################


early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,
                               verbose=1,
                               mode='min')

model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_path + '/model.pkl'),
                                   monitor='val_loss',
                                   verbose=0,
                                   save_weights_only=False,
                                   save_best_only=True,
                                   mode='min')

cb_redlr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001, verbose=1)

model_mk2.fit(
    x=[x_train_tokens_pad, x_train_chars_pad],
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    verbose=1,
    shuffle=True,
    callbacks=[model_checkpoint, early_stopping])

#######################################################################################################################


def preparation_data_to_score(yh, pr):
    yh = yh.argmax(2)
    pr = [list(np.argmax(el, axis=1)) for el in pr]
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr


custom_objects = {'CRF': CRF, 'crf_loss': crf_loss}

model = load_model(os.path.join(model_path + '/model.pkl'), custom_objects=custom_objects)
pr = model.predict([x_test_tokens_pad, x_test_chars_pad], verbose=1)
y_test, pr = preparation_data_to_score(y_test, pr)
y_test = [ind2labels[l] for l in y_test]
pr = [ind2labels.get(l, "ROOT") for l in pr]
print(classification_report(y_test, pr, digits=4))
print(f1_score(y_test, pr, average='macro'))
