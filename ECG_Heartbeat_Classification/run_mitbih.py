import os

import pandas as pd
import numpy as np

from utils import get_model, helpers

models = [
    "rnn_lstm",
    "rnn_gru",
    "rnn_gru_bidir",
    "rnn_lstm_bidir",
]

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
if "rnn_lstm" in models:
    model = get_model.rnn_lstm(nclass=5, dense_layers=[64, 16, 8])
    file_name = "mitbih_rnn_lstm"
    file_path = os.path.join(model_directory, file_name + ".h5")
    model = helpers.run(model, file_path, X, Y)
    model.load_weights(file_path)

    # Save the entire model as a SavedModel.
    model.save(os.path.join(model_directory, file_name))

    # Make predictions on test set
    pred_test = model.predict(X_test)

    # Evaluate predictions
    helpers.test_nonbinary(Y_test, pred_test, verbose=True)

#
# GRU
if "rnn_gru" in models:
    model = get_model.rnn_gru(nclass=5, dense_layers=[64, 16, 8])
    file_name = "mitbih_rnn_gru"
    file_path = os.path.join(model_directory, file_name + ".h5")
    model = helpers.run(model, file_path, X, Y)
    model.load_weights(file_path)

    # Save the entire model as a SavedModel.
    model.save(os.path.join(model_directory, file_name))

    # Make predictions on test set
    pred_test = model.predict(X_test)

    # Evaluate predictions
    helpers.test_nonbinary(Y_test, pred_test, verbose=True)


#
# Bidirectional LSTM
if "rnn_lstm_bidir" in models:
    model = get_model.rnn_lstm_bidir(nclass=5, dense_layers=[64, 16, 8])
    file_name = "mitbih_rnn_lstm_bidir"
    file_path = os.path.join(model_directory, file_name + ".h5")
    model = helpers.run(model, file_path, X, Y)
    model.load_weights(file_path)

    # Save the entire model as a SavedModel.
    model.save(os.path.join(model_directory, file_name))

    # Make predictions on test set
    pred_test = model.predict(X_test)

    # Evaluate predictions
    helpers.test_nonbinary(Y_test, pred_test, verbose=True)


#
# Bidirectional GRU
if "rnn_gru_bidir" in models:
    model = get_model.rnn_gru_bidir(nclass=5, dense_layers=[64, 16, 8])
    file_name = "mitbih_rnn_gru_bidir"
    file_path = os.path.join(model_directory, file_name + ".h5")
    model = helpers.run(model, file_path, X, Y)
    model.load_weights(file_path)

    # Save the entire model as a SavedModel.
    model.save(os.path.join(model_directory, file_name))

    # Make predictions on test set
    pred_test = model.predict(X_test)

    # Evaluate predictions
    helpers.test_nonbinary(Y_test, pred_test, verbose=True)

print("Done.")
