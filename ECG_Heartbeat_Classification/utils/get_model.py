import tensorflow as tf
from keras import optimizers, losses, activations, models
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, Dropout, LSTM, GRU, Bidirectional, CuDNNLSTM, CuDNNGRU

gpu = tf.test.is_gpu_available()
print(f"GPU available:{gpu}")

if gpu is not False:
    LSTM = CuDNNLSTM
    GRU = CuDNNGRU


def rnn_lstm(nclass, input_shape=(187, 1), recurrent_layers=[64, 128], dense_layers=[64, 16], dropout=0.2, binary=False):
    """
    RNN with Long Short Term Memory (LSTM) Units
    :param nclass:
    :param input_shape:
    :param layers:
    :param dropout:
    :param loss:
    :return:
    """
    if not binary:
        loss = losses.sparse_categorical_crossentropy
        last_activation = activations.softmax
    else:
        loss = losses.binary_crossentropy
        last_activation = activations.sigmoid
    return_sequences = True

    inp = Input(shape=input_shape)
    x = inp
    for neurons in recurrent_layers:
        x = LSTM(neurons, return_sequences=return_sequences)(x)
        x = Dropout(rate=dropout)(x)
        return_sequences = False
    for neurons in dense_layers:
        x = Dense(neurons, activation='relu')(x)
    x = Dense(nclass, name='Output', activation=last_activation)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss=loss, metrics=['acc'])
    model.summary()
    return model


def rnn_lstm_bidir(nclass, input_shape=(187, 1), recurrent_layers=[64, 128], dense_layers=[64, 16], dropout=0.2, binary=False):
    """
    Bidirectional RNN with Long Short Term Memory (LSTM) Units
    :param nclass:
    :param input_shape:
    :param layers:
    :param dropout:
    :return:
    """
    if not binary:
        loss = losses.sparse_categorical_crossentropy
        last_activation = activations.softmax
    else:
        loss = losses.binary_crossentropy
        last_activation = activations.sigmoid
    return_sequences = True

    inp = Input(shape=input_shape)
    x = inp
    for neurons in recurrent_layers:
        layer = LSTM(neurons, return_sequences=return_sequences)
        x = Bidirectional(layer)(x)
        x = Dropout(rate=dropout)(x)
        return_sequences = False
    for neurons in dense_layers:
        x = Dense(neurons, activation='relu')(x)
    x = Dense(nclass, name='Output', activation=last_activation)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss=loss, metrics=['acc'])
    model.summary()
    return model

def rnn_gru(nclass, input_shape=(187, 1), recurrent_layers=[64, 128], dense_layers=[64, 16], dropout=0.2, binary=False):
    """
    RNN with Gated Rectified Units
    :param nclass:
    :param input_shape:
    :param layers:
    :param dropout:
    :return:
    """
    if not binary:
        loss = losses.sparse_categorical_crossentropy
        last_activation = activations.softmax
    else:
        loss = losses.binary_crossentropy
        last_activation = activations.sigmoid
    return_sequences = True

    inp = Input(shape=input_shape)
    x = inp
    for neurons in recurrent_layers:
        x = GRU(neurons, return_sequences=return_sequences)(x)
        x = Dropout(rate=dropout)(x)
        return_sequences = False
    for neurons in dense_layers:
        x = Dense(neurons, activation='relu')(x)
    x = Dense(nclass, name='Output', activation=last_activation)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss=loss, metrics=['acc'])
    model.summary()
    return model


def rnn_gru_bidir(nclass, input_shape=(187, 1), recurrent_layers=[64, 128], dense_layers=[64, 16], dropout=0.2, binary=False):
    """
    Bidirectional RNN with Gated Rectified Units
    :param nclass:
    :param input_shape:
    :param layers:
    :param dropout:
    :return:
    """
    if not binary:
        loss = losses.sparse_categorical_crossentropy
        last_activation = activations.softmax
    else:
        loss = losses.binary_crossentropy
        last_activation = 'sigmoid'
    return_sequences = True

    inp = Input(shape=input_shape)
    x = inp
    for neurons in recurrent_layers:
        print(neurons)
        layer = GRU(neurons, return_sequences=return_sequences)
        x = Bidirectional(layer)(x)
        x = Dropout(rate=dropout)(x)
        return_sequences = False
    for neurons in dense_layers:
        print(neurons)
        x = Dense(neurons, activation='relu')(x)
    x = Dense(nclass, name='Output', activation=last_activation)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss=loss, metrics=['acc'])
    model.summary()
    return model


def transfer_learning(nclass, base_model, loss=losses.binary_crossentropy):
    """
    Transfer learning approach. Take base model, clip last few layers, append new layers and freeze the old ones.
    :param nclass:
    :param base_model:
    :param loss:
    :return:
    """
    base_model.layers.pop()
    base_model.layers[-1].outbound_nodes = []
    base_model.outputs = [base_model.layers[-1].output]
    x = base_model.get_layer('dense_1').output

    inp = base_model.input

    # x = base_model.output
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(nclass, name='Output', activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=x)

    # Freeze layers
    for layer in base_model.layers:
        layer.trainable = False

    opt = optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss=loss, metrics=['acc'])
    model.summary()
    return model


def cnn_1d(nclass, input_shape=(187, 1)):
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
