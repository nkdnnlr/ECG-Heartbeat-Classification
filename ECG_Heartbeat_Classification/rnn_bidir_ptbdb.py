import pandas as pd
import numpy as np

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate, Bidirectional, GRU, LSTM
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

df_1 = pd.read_csv("../data/heartbeat/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("../data/heartbeat/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])


Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]


def get_model():
    nclass = 2
    inp = Input(shape=(187, 1))

    layer = GRU(64, name='GRU1', return_sequences=True, dropout=0.2)#(inp)
    x = Bidirectional(layer, name='BiRNN1')(inp)
    layer = GRU(128, name='GRU2', dropout=0.2)#(x)
    x = Bidirectional(layer, name='BiRNN2')(x)
    x = Dense(64, name='Dense1', activation='relu')(x)
    x = Dense(16, name='Dense2', activation='relu')(x)
    x = Dense(8, name='Dense3', activation='relu')(x)
    x = Dense(nclass, name='Output', activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

model = get_model()
file_path = "baseline_rnn_bidir_ptbdb.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
model.load_weights(file_path)

pred_test = model.predict(X_test)
pred_test = (pred_test>0.5).astype(np.int8)

f1 = f1_score(Y_test, pred_test)

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)