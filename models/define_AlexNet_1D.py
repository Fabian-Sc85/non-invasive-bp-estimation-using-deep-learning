# Implements an AlexNet Conv-Net which performs HR, SBP and DBP regression based
# on spectrograms. The net's input are ppg time series.
#
# Arguments:
# data_in : Tensor containing the rppg time series. Size (batch_size, ppg_length, 1)
# fs : sampling frequency (default fs = 125 Hz)
# frame_len : length of the STFFT in seconds (default frame_len = 10)
# n_hop : length between frames during spectrogram calculation in samples (default n_hop = 1)
#
# A. Krizhevsky, I. Sutskever, und G. E. Hinton, „ImageNet classification with deep convolutional neural networks“,
# Commun. ACM, Bd. 60, Nr. 6, S. 84–90, Mai 2017, doi: 10.1145/3065386.
#
# Author: Fabian Schrumpf, MSc.
# Laboratory for Biosignal Processing; HTWK Leipzig (Leipzig University of
# Applied Sciences)
# email address: fabian.schrumpf@htwk-leipzig.de
# Website: https://labp.github.io/
# August 2020; Last revision: --

from tensorflow.keras.layers import Softmax, Permute, Input, Add, Conv1D, MaxPooling1D, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling1D, MaxPooling2D, GlobalMaxPooling2D, LeakyReLU, GlobalAveragePooling2D, ReLU, Dropout
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model
import tensorflow as tf

def AlexNet_1D(data_in_shape, num_output=2, dil=1, kernel_size=3, fs = 125, useMaxPooling=True, UseDerivative=False):

    # Define the input as a tensor with shape input_shape
    X_input = Input(shape=data_in_shape)

    if UseDerivative:
        dt1 = (X_input[:,1:] - X_input[:,:-1])*fs
        dt2 = (dt1[:,1:] - dt1[:,:-1])*fs

        dt1 = tf.pad(dt1, tf.constant([[0,0],[0,1],[0,0]]))
        dt2 = tf.pad(dt2, tf.constant([[0,0],[0,2],[0,0]]))

        X = tf.concat([X_input, dt1, dt2], axis=2)
    else:
        X=X_input


    # convolutional stage
    X = Conv1D(96, 7, strides=3, name='conv1', kernel_initializer=glorot_uniform(seed=0), padding="same")(X)
    if useMaxPooling:
        X = MaxPooling1D(3, strides=2, name="MaxPool1")(X)
    X = Activation(ReLU())(X)
    X = BatchNormalization(axis=-1, name='BatchNorm1')(X)

    X = Conv1D(256, kernel_size=kernel_size, strides=1, dilation_rate=dil, name='conv2', kernel_initializer=glorot_uniform(seed=0), padding="same")(X)
    if useMaxPooling:
        X = MaxPooling1D(3, strides=2, name="MaxPool2")(X)
    X = Activation(ReLU())(X)
    X = BatchNormalization(axis=-1, name='BatchNorm2')(X)

    X = Conv1D(384, kernel_size=kernel_size, strides=1, dilation_rate=dil, name='conv3', kernel_initializer=glorot_uniform(seed=0), padding="same")(X)
    X = Activation(ReLU())(X)
    X = BatchNormalization(axis=-1, name='BatchNorm3')(X)

    X = Conv1D(384, kernel_size=kernel_size, strides=1, dilation_rate=dil, name='conv4', kernel_initializer=glorot_uniform(seed=0), padding="same")(X)
    X = Activation(ReLU())(X)
    X = BatchNormalization(axis=-1, name='BatchNorm4')(X)

    X = Conv1D(256, kernel_size=kernel_size, strides=1, dilation_rate=dil, name='conv5', kernel_initializer=glorot_uniform(seed=0), padding="same")(X)
    if useMaxPooling:
        X = MaxPooling1D(3, strides=2, name="MaxPool5")(X)
    X = Activation(ReLU())(X)
    X = BatchNormalization(axis=-1, name='BatchNorm5')(X)

    # Fully connected stage
    X = Flatten()(X)
    X = Dense(4096, activation='relu', name='dense1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dropout(rate=0.5)(X)
    X = Dense(4096, activation='relu', name='dense2', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dropout(rate=0.5)(X)

    # Create model
    if num_output == 1:
        X_out = Dense(3, activation='softmax', name='out', kernel_initializer=glorot_uniform(seed=0))(X)
        model = Model(inputs=X_input, outputs=X_out, name='AlexNet_1D')
    else:
        # output stage
        X_SBP = Dense(1, activation='relu', name='SBP', kernel_initializer=glorot_uniform(seed=0))(X)
        X_DBP = Dense(1, activation='relu', name='DBP', kernel_initializer=glorot_uniform(seed=0))(X)
        model = Model(inputs=X_input, outputs=[X_SBP, X_DBP], name='AlexNet_1D')

    return model
