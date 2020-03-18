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


def rnn_lstm(nclass, input_shape=(187, 1), hidden_layers=[64, 128, 64, 16], dropout=0.2, loss=losses.sparse_categorical_crossentropy):
    """
    RNN with Long Short Term Memory (LSTM) Units
    :param nclass:
    :param input_shape:
    :param layers:
    :param dropout:
    :param loss:
    :return:
    """
    inp = Input(shape=input_shape)

    x = LSTM(hidden_layers[0], name='LSTM1', return_sequences=True, dropout=dropout)(inp)
    x = LSTM(hidden_layers[1], name='LSTM2', dropout=dropout)(x)
    x = Dense(hidden_layers[2], name='Dense1', activation='relu')(x)
    x = Dense(hidden_layers[3], name='Dense2', activation='relu')(x)
    x = Dense(nclass, name='Output', activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=loss, metrics=['acc'])
    model.summary()
    return model


def rnn_lstm_bidir(nclass, input_shape=(187, 1), hidden_layers=[64, 128, 64, 16], dropout=0.2, loss=losses.sparse_categorical_crossentropy):
    """
    Bidirectional RNN with Long Short Term Memory (LSTM) Units
    :param nclass:
    :param input_shape:
    :param layers:
    :param dropout:
    :return:
    """
    inp = Input(shape=input_shape)

    layer = LSTM(hidden_layers[0], name='LSTM1', return_sequences=True, dropout=dropout)  # (inp)
    x = Bidirectional(layer, name='BiRNN1')(inp)
    layer = LSTM(hidden_layers[1], name='LSTM2', dropout=dropout)  # (x)
    x = Bidirectional(layer, name='BiRNN2')(x)
    x = Dense(hidden_layers[2], name='Dense1', activation='relu')(x)
    x = Dense(hidden_layers[3], name='Dense2', activation='relu')(x)
    x = Dense(nclass, name='Output', activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=loss, metrics=['acc'])
    model.summary()
    return model


def rnn_gru(nclass, input_shape=(187, 1), hidden_layers=[64, 128, 64, 16], dropout=0.2, loss=losses.sparse_categorical_crossentropy):
    """
    RNN with Gated Rectified Units
    :param nclass:
    :param input_shape:
    :param layers:
    :param dropout:
    :return:
    """
    inp = Input(shape=input_shape)

    x = GRU(hidden_layers[0], name='GRU1', return_sequences=True, dropout=dropout)(inp)
    x = GRU(hidden_layers[1], name='GRU2', dropout=dropout)(x)
    x = Dense(hidden_layers[2], name='Dense1', activation='relu')(x)
    x = Dense(hidden_layers[3], name='Dense2', activation='relu')(x)
    x = Dense(nclass, name='Output', activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=loss, metrics=['acc'])
    model.summary()
    return model


def rnn_gru_bidir(nclass, input_shape=(187, 1), hidden_layers=[64, 128, 64, 16], dropout=0.2, loss=losses.sparse_categorical_crossentropy):
    """
    Bidirectional RNN with Gated Rectified Units
    :param nclass:
    :param input_shape:
    :param layers:
    :param dropout:
    :return:
    """
    inp = Input(shape=input_shape)

    layer = GRU(hidden_layers[0], name='GRU1', return_sequences=True, dropout=dropout)  # (inp)
    x = Bidirectional(layer, name='BiRNN1')(inp)
    layer = GRU(hidden_layers[1], name='GRU2', dropout=dropout)  # (x)
    x = Bidirectional(layer, name='BiRNN2')(x)
    x = Dense(hidden_layers[2], name='Dense1', activation='relu')(x)
    x = Dense(hidden_layers[3], name='Dense2', activation='relu')(x)
    x = Dense(nclass, name='Output', activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=loss, metrics=['acc'])
    model.summary()
    return model


def get_model(nclass, input_shape=(187, 1)):
    """
    1d ConvNet
    :param nclass:
    :param input_shape:
    :return:
    """
    inp = Input(shape=input_shape)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model