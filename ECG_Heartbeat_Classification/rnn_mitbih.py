import pandas as pd
import numpy as np

import tensorflow as tf
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import f1_score, accuracy_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional

df_train = pd.read_csv("../data/heartbeat/mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("../data/heartbeat/mitbih_test.csv", header=None)

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]


def get_model():
    nclass = 5
    inp = Input(shape=(187, 1))

    x = LSTM(32, name='LSTM1', return_sequences=True)(inp)
    x = Dropout(0.5)(x)
    x = LSTM(16, name='LSTM2')(x)
    x = Dropout(0.5)(x)

    x = Dense(18, name='Dense1', activation='relu')(x)
    x = Dense(8, name='Dense2', activation='relu')(x)
    x = Dense(nclass, name='Output', activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

def get_another_model():
    nclass = 5
    inp = Input(shape=(187, 1))

    layer = LSTM(64, name='LSTM1', return_sequences=True, dropout=0.2)  # (inp)
    x = Bidirectional(layer, name='BiRNN1')(inp)
    layer = LSTM(128, name='LSTM2', dropout=0.2)  # (x)
    x = Bidirectional(layer, name='BiRNN2')(x)
    x = Dense(64, name='Dense1', activation='relu')(x)
    x = Dense(16, name='Dense2', activation='relu')(x)
    x = Dense(nclass, name='Output', activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

model = get_another_model()
file_path = "rnn_lstm_bidirectional_mitbih.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
model.load_weights(file_path)

pred_test = model.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)

f1 = f1_score(Y_test, pred_test, average="macro")

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)