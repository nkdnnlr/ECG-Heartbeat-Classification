import os

import pandas as pd
import numpy as np

from keras import losses
from sklearn.model_selection import train_test_split

from utils import get_model, helpers



#
# LSTM
# if "rnn_lstm" in models:
def run_lstm(model_directory, X, Y, X_test, Y_test):
    model = get_model.rnn_lstm(nclass=1, dense_layers=[64, 16, 8], binary=True)
    file_name = "ptbdb_rnn_lstm"
    file_path = os.path.join(model_directory, file_name + ".h5")
    model = helpers.run(model, file_path, X, Y)
    model.load_weights(file_path)

    # Save the entire model as a SavedModel.
    model.save(os.path.join(model_directory, file_name))

    # Make predictions on test set
    pred_test = model.predict(X_test)

    # Evaluate predictions
    helpers.test_binary(Y_test, pred_test)

#
# GRU
# if "rnn_gru" in models:
def run_gru(model_directory, X, Y, X_test, Y_test):
    model = get_model.rnn_gru(nclass=1, dense_layers=[64, 16, 8], binary=True)
    file_name = "ptbdb_rnn_gru"
    file_path = os.path.join(model_directory, file_name + ".h5")
    model = helpers.run(model, file_path, X, Y)
    model.load_weights(file_path)

    # Save the entire model as a SavedModel.
    model.save(os.path.join(model_directory, file_name))

    # Make predictions on test set
    pred_test = model.predict(X_test)

    # Evaluate predictions
    helpers.test_binary(Y_test, pred_test)

#
# Bidirectional LSTM
# if "rnn_lstm_bidir" in models:
def run_lstm_bidir(model_directory, X, Y, X_test, Y_test):
    model = get_model.rnn_lstm_bidir(nclass=1, dense_layers=[64, 16, 8], binary=True)
    file_name = "ptbdb_rnn_lstm_bidir"
    file_path = os.path.join(model_directory, file_name + ".h5")
    model = helpers.run(model, file_path, X, Y)
    model.load_weights(file_path)

    # Save the entire model as a SavedModel.
    model.save(os.path.join(model_directory, file_name))

    # Make predictions on test set
    pred_test = model.predict(X_test)

    # Evaluate predictions
    helpers.test_binary(Y_test, pred_test)


#
# Bidirectional GRU
# if "rnn_gru_bidir" in models:
def run_gru_bidir(model_directory, X, Y, X_test, Y_test):
    model = get_model.rnn_gru_bidir(nclass=1, dense_layers=[64, 16, 8], binary=True)
    file_name = "ptbdb_rnn_gru_bidir"
    file_path = os.path.join(model_directory, file_name + ".h5")
    model = helpers.run(model, file_path, X, Y)
    model.load_weights(file_path)

    # Save the entire model as a SavedModel.
    model.save(os.path.join(model_directory, file_name))

    # Make predictions on test set
    pred_test = model.predict(X_test)

    # Evaluate predictions
    helpers.test_binary(Y_test, pred_test)

#
# Transfer Learning
# if "rnn_gru_bidir_transfer" in models:
def run_transfer_learning(base_model, model_directory, X, Y, X_test, Y_test):
    model = get_model.transfer_learning(
        nclass=1, base_model=base_model, loss=losses.binary_crossentropy
    )
    file_name = "ptbdb_rnn_gru_bidir_transfer"
    file_path = file_name + ".h5"
    model = helpers.run(model, file_path, X, Y)
    model.load_weights(file_path)

    # Save the entire model as a SavedModel.
    model.save(os.path.join(model_directory, file_name))

    # Make predictions on test set
    pred_test = model.predict(X_test)

    # Evaluate predictions
    helpers.test_binary(Y_test, pred_test)

if __name__ == '__main__':
    models = [
        "rnn_lstm",
        "rnn_gru",
        "rnn_gru_bidir",
        "rnn_lstm_bidir",
        "rnn_transferlearning",
    ]

    # Make directory
    model_directory = "./models"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    df_1 = pd.read_csv("../data/ECG_Heartbeat_Classification/ptbdb_normal.csv", header=None)
    df_2 = pd.read_csv("../data/ECG_Heartbeat_Classification/ptbdb_abnormal.csv", header=None)
    df = pd.concat([df_1, df_2])
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1337, stratify=df[187]
    )

    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]
    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    if "rnn_lstm" in models:
        run_lstm(model_directory, X, Y, X_test, Y_test)
    if "rnn_gru" in models:
        run_gru(model_directory, X, Y, X_test, Y_test)
    if "rnn_lstm_bidir" in models:
        run_lstm_bidir(model_directory, X, Y, X_test, Y_test)
    if "rnn_gru_bidir" in models:
        run_gru_bidir(model_directory, X, Y, X_test, Y_test)
    if "rnn_transferlearning" in models:
        base_model = get_model.rnn_gru_bidir(
            nclass=5, dense_layers=[64, 16, 8], binary=False
        )
        file_name = "mitbih_rnn_gru_bidir"
        file_path = os.path.join(model_directory, file_name + ".h5")
        base_model.load_weights(file_path)

        run_transfer_learning(base_model, model_directory, X, Y, X_test, Y_test)