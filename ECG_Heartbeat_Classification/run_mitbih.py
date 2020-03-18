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

from utils import get_model

def run(model, file_path):
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]  # early

    model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
    model.load_weights(file_path)


# Make directory
model_directory = "../models"
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

df_train = pd.read_csv("../data/heartbeat/mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("../data/heartbeat/mitbih_test.csv", header=None)

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]
Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

#
# LSTM
model = get_model.rnn_lstm(nclass=5)
file_name = "mitbih_rnn_lstm"
file_path = file_name + ".h5"
run(model, file_path)
# Save the entire model as a SavedModel.
model.save(os.path.join(model_directory, file_name))

# Test and print out scores
pred_test = model.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)
f1 = f1_score(Y_test, pred_test, average="macro")
print("Test f1 score : %s "% f1)
acc = accuracy_score(Y_test, pred_test)
print("Test accuracy score : %s "% acc)

#
# GRU
model = get_model.rnn_gru(nclass=5)
file_name = "mitbih_rnn_gru"
file_path = file_name + ".h5"
run(model, file_path)
# Save the entire model as a SavedModel.
model.save(os.path.join(model_directory, file_name))

# Test and print out scores
pred_test = model.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)
f1 = f1_score(Y_test, pred_test, average="macro")
print("Test f1 score : %s "% f1)
acc = accuracy_score(Y_test, pred_test)
print("Test accuracy score : %s "% acc)

#
# Bidirectional GRU
model = get_model.rnn_gru_bidir(nclass=5)
file_name = "mitbih_rnn_gru_bidir"
file_path = file_name + ".h5"
run(model, file_path)
# Save the entire model as a SavedModel.
model.save(os.path.join(model_directory, file_name))

# Test and print out scores
pred_test = model.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)
f1 = f1_score(Y_test, pred_test, average="macro")
print("Test f1 score : %s "% f1)
acc = accuracy_score(Y_test, pred_test)
print("Test accuracy score : %s "% acc)



