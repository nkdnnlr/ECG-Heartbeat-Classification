import os

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
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score



def run(model, file_path, X, Y):
    """
    Define checkpoints and run model
    :param model:
    :param file_path:
    :param X:
    :param Y:
    :return:
    """
    # Set global random seed for reproducibility
    tf.set_random_seed(42)

    # Define callbacks
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]  # early

    # Train model
    model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)

    return model


def test_binary(Y_test, pred_test, verbose=True):
    """
    Test binary classification problem
    :param Y_test:
    :param pred_test:
    :param verbose:
    :return:
    """
    pred_test = (pred_test > 0.5).astype(np.int8)

    # Calculate scores
    f1 = f1_score(Y_test, pred_test)
    acc = accuracy_score(Y_test, pred_test)
    auroc = roc_auc_score(Y_test, pred_test)
    auprc = average_precision_score(Y_test, pred_test)

    if verbose:
        print("Test f1 score : %s " % f1)
        print("Test accuracy score : %s " % acc)
        print("AUROC score : %s " % auroc)
        print("AUPRC accuracy score : %s " % auprc)


def test_nonbinary(Y_test, pred_test, verbose=True):
    """
    Test binary classification problem
    :param Y_test:
    :param pred_test:
    :param verbose:
    :return:
    """
    pred_test = np.argmax(pred_test, axis=-1)

    # Calculate scores
    f1 = f1_score(Y_test, pred_test, average="macro")
    acc = accuracy_score(Y_test, pred_test)

    if verbose:
        print("Test f1 score : %s " % f1)
        print("Test accuracy score : %s " % acc)