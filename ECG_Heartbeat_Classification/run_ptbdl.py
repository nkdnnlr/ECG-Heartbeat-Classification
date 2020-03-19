import os

import pandas as pd
import numpy as np
import tensorflow as tf

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional
from sklearn.model_selection import train_test_split

from utils import get_model

models = [#'rnn_lstm', \
          # 'rnn_lstm_bidir', \
          # 'rnn_gru', \
          'rnn_gru_bidir',\
          'rnn_gru_bidir_transfer',\
          ]

def run(model, X, Y, file_path):
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]  # early
    model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)

# Set global random seed for reproducibility
tf.random.set_seed(42)  ## ATTENTION: Unfortunately this has been added after the model has run.

# Make directory
model_directory = "../models"
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

df_1 = pd.read_csv("../data/heartbeat/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("../data/heartbeat/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])
df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]
Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

#
# LSTM
if 'rnn_lstm' in models:
    model = get_model.rnn_lstm(nclass=1, dense_layers=[64, 16, 8], binary=True)
    file_name = "ptbdb_rnn_lstm"
    file_path = os.path.join(model_directory, file_name + ".h5")
    run(model, X, Y, file_path)
    model.load_weights(file_path)
    # Save the entire model as a SavedModel.
    model.save(os.path.join(model_directory, file_name))

    # Test and print out scores
    pred_test = model.predict(X_test)
    pred_test = (pred_test > 0.5).astype(np.int8)

    # Calculate scores
    f1 = f1_score(Y_test, pred_test)
    acc = accuracy_score(Y_test, pred_test)
    auroc = roc_auc_score(Y_test, pred_test)
    auprc = average_precision_score(Y_test, pred_test)

    print("Test f1 score : %s " % f1)
    print("Test accuracy score : %s " % acc)
    print("AUROC score : %s " % auroc)
    print("AUPRC accuracy score : %s " % auprc)

#
# GRU
if 'rnn_gru' in models:
    model = get_model.rnn_gru(nclass=1, dense_layers=[64, 16, 8], binary=True)
    file_name = "ptbdb_rnn_gru"
    file_path = os.path.join(model_directory, file_name + ".h5")
    run(model, X, Y, file_path)
    model.load_weights(file_path)
    # Save the entire model as a SavedModel.
    model.save(os.path.join(model_directory, file_name))

    # Test and print out scores
    pred_test = model.predict(X_test)
    pred_test = (pred_test > 0.5).astype(np.int8)

    # Calculate scores
    f1 = f1_score(Y_test, pred_test)
    acc = accuracy_score(Y_test, pred_test)
    auroc = roc_auc_score(Y_test, pred_test)
    auprc = average_precision_score(Y_test, pred_test)

    print("Test f1 score : %s " % f1)
    print("Test accuracy score : %s " % acc)
    print("AUROC score : %s " % auroc)
    print("AUPRC accuracy score : %s " % auprc)

#
# Bidirectional GRU
if 'rnn_gru_bidir' in models:
    model = get_model.rnn_gru_bidir(nclass=1, dense_layers=[64, 16, 8], binary=True)
    file_name = "ptbdb_rnn_gru_bidir"
    file_path = os.path.join(model_directory, file_name + ".h5")
    run(model, X, Y, file_path)
    model.load_weights(file_path)
    # Save the entire model as a SavedModel.
    model.save(os.path.join(model_directory, file_name))

    # Test and print out scores
    pred_test = model.predict(X_test)
    pred_test = (pred_test > 0.5).astype(np.int8)

    # Calculate scores
    f1 = f1_score(Y_test, pred_test)
    acc = accuracy_score(Y_test, pred_test)
    auroc = roc_auc_score(Y_test, pred_test)
    auprc = average_precision_score(Y_test, pred_test)

    print("Test f1 score : %s " % f1)
    print("Test accuracy score : %s " % acc)
    print("AUROC score : %s " % auroc)
    print("AUPRC accuracy score : %s " % auprc)

#
# Transfer Learning
if 'rnn_gru_bidir_transfer' in models:
    base_model = get_model.rnn_gru_bidir(nclass=5, dense_layers=[64, 16], binary=False)
    file_name = "rnn_bidirectional_mitbih"
    file_path = os.path.join(model_directory, file_name + ".h5")
    base_model.load_weights(file_path)

    model = get_model.transfer_learning(nclass=1, base_model=base_model, loss=losses.binary_crossentropy)
    file_name = "ptbdb_rnn_gru_bidir_transfer"
    # file_name = "baseline_rnn_bidir_ptbdb"
    file_path = file_name + ".h5"
    run(model, X, Y, file_path)
    model.load_weights(file_path)
    # Save the entire model as a SavedModel.
    model.save(os.path.join(model_directory, file_name))

    # Test and print out scores
    pred_test = model.predict(X_test)
    pred_test = (pred_test > 0.5).astype(np.int8)

    # Calculate scores
    f1 = f1_score(Y_test, pred_test)
    acc = accuracy_score(Y_test, pred_test)
    auroc = roc_auc_score(Y_test, pred_test)
    auprc = average_precision_score(Y_test, pred_test)

    print("Test f1 score : %s " % f1)
    print("Test accuracy score : %s " % acc)
    print("AUROC score : %s " % auroc)
    print("AUPRC accuracy score : %s " % auprc)


print("Done.")

