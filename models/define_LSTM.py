from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Conv1D, ReLU

from tensorflow.keras import Model

def define_LSTM(data_in_shape):
    X_input = Input(shape=data_in_shape)
    X = Conv1D(filters=64, kernel_size=5, strides=1, padding='causal', activation='relu')(X_input)
    X = Bidirectional(LSTM(128, return_sequences=True))(X)
    X = Bidirectional(LSTM(128, return_sequences=True))(X)
    X = Bidirectional(LSTM(64, return_sequences=False))(X)
    X = Dense(512, activation='relu')(X)
    X = Dense(256, activation='relu')(X)
    X = Dense(128, activation='relu')(X)

    X_SBP = Dense(1, name='SBP')(X)
    X_DBP = Dense(1, name='DBP')(X)

    model = Model(inputs=X_input, outputs=[X_SBP, X_DBP], name='LSTM')

    return model




