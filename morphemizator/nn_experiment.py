import os

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

unique_chars = {ch: i + 1 for i, ch in enumerate(sorted(set([ch[0] for w in train + test for ch in w])))}
unique_labels = sorted(set([ch[1] for w in train + test for ch in w]))

labels2ind = {l: i + 1 for i, l in enumerate(unique_labels)}
ind2labels = {i+ 1: l  for i, l in enumerate(unique_labels)}
print('unique_labels: ', len(labels2ind))

x_train = [[unique_chars.get(ch[0]) for ch in w] for w in train]
y_train = [[labels2ind.get(ch[1]) for ch in w] for w in train]

x_test = [[unique_chars.get(ch[0]) for ch in w] for w in test]
y_test = [[labels2ind.get(ch[1]) for ch in w] for w in test]

max_w_len = max([len(w) for w in train])
print('max_token_len: ', max_w_len)

x_train = pad_sequences(x_train, maxlen=max_w_len, value=0, padding='pre', truncating='pre')
y_train = pad_sequences(y_train, maxlen=max_w_len, value=0, padding='pre', truncating='pre')
y_train = to_categorical(y_train, len(labels2ind)+1)

x_test = pad_sequences(x_test, maxlen=max_w_len, value=0, padding='pre', truncating='pre')
y_test = pad_sequences(y_test, maxlen=max_w_len, value=0, padding='pre', truncating='pre')
y_test = to_categorical(y_test, len(labels2ind)+1)

print('ch2ind:', x_train[0])
print('l2ind:', y_train[2])

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


emb_oh = char_matrix(list(unique_chars.keys()))
print("emb_oh: ", emb_oh.shape)

########################################################################################################################

input_char_emb = Input((max_w_len,), name='input_char_emb')
char_emb = Embedding(
    input_dim=len(unique_chars) + 1,
    output_dim=emb_oh.shape[1],
    input_length=max_w_len,
    weights=[emb_oh],
    mask_zero=True,
    trainable=False
    )(input_char_emb)
rnn_layer = Bidirectional(GRU(
    128,
    return_sequences=True,
    activation='tanh',
    recurrent_activation="hard_sigmoid", ))(char_emb)
emb_out = Dropout(0.5)(rnn_layer)
dense_network = TimeDistributed(Dense(64, activation='relu'))(emb_out)
output = Dense(3, activation='softmax')(dense_network)
model_mk1 = Model(inputs=input_char_emb, outputs=output)
model_mk1.compile(optimizer='adam', loss="categorical_crossentropy")

########################################################################################################################


class ReversedLSTM(LSTM):
    def __init__(self, units, **kwargs):
        kwargs['go_backwards'] = True
        super().__init__(units, **kwargs)

    def call(self, inputs, **kwargs):
        y_rev = super().call(inputs, **kwargs)
        return K.reverse(y_rev, 1)


input_char_emb = Input((max_w_len,), name='input_char_emb')

char_emb = Embedding(
    input_dim=len(unique_chars) + 1,
    output_dim=emb_oh.shape[1],
    input_length=max_w_len,
    weights=[emb_oh],
    mask_zero=False,
    trainable=False
    )(input_char_emb)

cnn_outputs = []
for el in ((20, 1), (40, 2), (60, 3), (80, 4)):
    cnns = Conv1D(filters=el[0], kernel_size=el[1], padding="same", strides=1)(char_emb)
    cnns = BatchNormalization()(cnns)
    cnns = Activation('tanh')(cnns)
    cnn_outputs.append(cnns)
cnns = concatenate(cnn_outputs, axis=-1, name='cnn_concat')

hway_input = Input(shape=(K.int_shape(cnns)[-1],))
gate_bias_init = keras.initializers.Constant(-2)
transform_gate = Dense(units=K.int_shape(cnns)[-1], bias_initializer=gate_bias_init, activation='sigmoid')(hway_input)
carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(K.int_shape(cnns)[-1],))(transform_gate)
h_transformed = Dense(units=K.int_shape(cnns)[-1])(hway_input)
h_transformed = Activation('relu')(h_transformed)
transformed_gated = Multiply()([transform_gate, h_transformed])
carried_gated = Multiply()([carry_gate, hway_input])
outputs = Add()([transformed_gated, carried_gated])
highway = Model(inputs=hway_input, outputs=outputs)
chars_vectors = highway(cnns)


if USE_CRF:
    lstm_enc, fh, fc, bh, bc = Bidirectional(LSTM(256, return_sequences=True, return_state=True))(chars_vectors)
    lstm_dec = Bidirectional(LSTM(256, return_sequences=True))(lstm_enc, initial_state=[bh, bc, fh, fc])
    lyr_crf = CRF(len(labels2ind) + 1)
    outputs = lyr_crf(lstm_dec)
    model_mk2 = Model(inputs=input_char_emb, outputs=outputs)
    model_mk2.compile(optimizer='adam', loss=lyr_crf.loss_function)
else:
    lstm_forward_1 = LSTM(256, return_sequences=True, name='LSTM_1_forward')(chars_vectors)
    lstm_backward_1 = ReversedLSTM(256, return_sequences=True, name='LSTM_1_backward')(chars_vectors)
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
    model_mk2 = Model(inputs=input_char_emb, outputs=outputs)
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
    x=x_train,
    y=y_train,
    batch_size=32,
    epochs=50,
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
pr = model.predict(x_test, verbose=1)
y_test, pr = preparation_data_to_score(y_test, pr)
y_test = [ind2labels[l] for l in y_test]
pr = [ind2labels.get(l, "ROOT") for l in pr]
print(classification_report(y_test, pr, digits=4))