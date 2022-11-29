from __future__ import division, print_function
import numpy as np
np.random.seed(3)
import os

from scipy.signal import butter, lfilter, lfilter_zi, filtfilt, savgol_filter
from sklearn.preprocessing import MinMaxScaler
import sklearn as sk
import random
random.seed(3)

import scipy.io as sio
import matplotlib.pyplot as plt
import natsort as natsort
from scipy import signal
import math

import tensorflow as tf
# from tensorflow.keras.utils import multi_gpu_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import squeeze
from kapre import STFT, Magnitude, MagnitudeToDecibel
# from kapre.utils import Normalization2D
from tensorflow.keras.layers import Input, BatchNormalization, Lambda, AveragePooling2D, Flatten, Dense, Conv1D, Activation, add, AveragePooling1D, Dropout, Permute, concatenate, MaxPooling1D, LSTM, Reshape, GRU
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras import optimizers
#from tensorflow.keras.utils.vis_utils import plot_model

from tensorflow.keras.layers import Conv2D, MaxPooling2D

def diff(input, fs):
    dt = (input[:, 1:] - input[:, :-1]) * fs
    dt = tf.pad(dt, tf.constant([[0, 0], [0, 1], [0, 0]]))

    return dt

def mid_spectrogram_layer(input_x):
    l2_lambda = .001
    n_dft = 128
    n_hop = 64
    fmin = 0.0
    fmax = 50 / 2

    x = Permute((2, 1))(input_x)
    # x = input_x
    x = STFT(n_fft=n_dft, hop_length=n_hop, output_data_format='channels_last')(x)
    x = Magnitude()(x)
    #x = MagnitudeToDecibel()(x)
    #x = BatchNormalization()(x)
    # x = Normalization2D(str_axis='batch')(x)
    x = Flatten()(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)

    return x


def mid_spectrogram_LSTM_layer(input_x):
    l2_lambda = .001
    n_dft = 64

    n_hop = 64
    fmin = 0.0
    fmax = 50 / 2

    #x = Permute((2, 1))(input_x)
    x = input_x
    x = STFT(n_fft=n_dft, hop_length=n_hop, output_data_format='channels_last')(x)
    x = Magnitude()(x)
    x = MagnitudeToDecibel()(x)
   #x = BatchNormalization()(x)
    # x = Normalization2D(str_axis='batch')(x)
    # print(np.array(x).shape)
    # x = Reshape((2, 64))(x)
    # x = GRU(64)(x)
    x = Flatten()(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)

    return x


def single_channel_resnet(my_input, num_filters=64, num_res_blocks=4, cnn_per_res=3,
                          kernel_sizes=[8, 5, 3], max_filters=128, pool_size=3, pool_stride_size=2):

    #my_input = Input(shape=input_shape)
    # my_input = input_shape
    # my_input = ks.expand_dims(my_input, axis=2)

    for i in np.arange(num_res_blocks):
        if (i == 0):
            block_input = my_input
            x = BatchNormalization()(block_input)
        else:
            block_input = x

        for j in np.arange(cnn_per_res):
            x = Conv1D(num_filters, kernel_sizes[j], padding='same')(x)
            x = BatchNormalization()(x)
            if (j < cnn_per_res - 1):
                x = Activation('relu')(x)

        is_expand_channels = not (my_input.shape[0] == num_filters)

        if is_expand_channels:
            res_conn = Conv1D(num_filters, 1, padding='same')(block_input)
            res_conn = BatchNormalization()(res_conn)
        else:
            res_conn = BatchNormalization()(block_input)

        x = add([res_conn, x])
        x = Activation('relu')(x)

        if (i < 5):
            x = AveragePooling1D(pool_size=pool_size, strides=pool_stride_size)(x)

        num_filters = 2 * num_filters
        if max_filters < num_filters:
            num_filters = max_filters

    return my_input, x


def raw_signals_deep_ResNet(input, UseDerivative=False):
    fs=125

    inputs = []
    l2_lambda = .001
    channel_outputs = []
    num_filters = 32

    X_input = Input(shape=input)

    if UseDerivative:
        # fs = tf.constant(fs, dtype=float)
        X_dt1 = Lambda(diff, arguments={'fs': fs})(X_input)
        X_dt2 = Lambda(diff, arguments={'fs': fs})(X_dt1)
        X = [X_input, X_dt1, X_dt2]
    else:
        X = [X_input]

    num_channels = len(X)

    for i in np.arange(num_channels):
        channel_resnet_input, channel_resnet_out = single_channel_resnet(X[i], num_filters=num_filters,
                                                                     num_res_blocks=4, cnn_per_res=3,
                                                                     kernel_sizes=[8, 5, 5, 3],
                                                                     max_filters=64, pool_size=2, pool_stride_size=1)
        channel_outputs.append(channel_resnet_out)
        inputs.append(channel_resnet_input)

    spectral_outputs = []
    num_filters = 32
    for x in inputs:
        spectro_x = mid_spectrogram_LSTM_layer(x)
        spectral_outputs.append(spectro_x)

    # concateante the channel specific residual layers
    if num_channels > 1:
        x = concatenate(channel_outputs, axis=-1)
    else:
        x = channel_outputs[0]

    x = BatchNormalization()(x)
    x = GRU(65)(x)
    # x = Flatten()(x)
    x = BatchNormalization()(x)

    # join time-domain and frequnecy domain fully-conencted layers
    if num_channels > 1:
        s = concatenate(spectral_outputs, axis=-1)
    else:
        s = spectral_outputs[0]

    # s = Flatten()(s)
    #     x = Dense(128,activation="relu",kernel_regularizer=l2(l2_lambda)) (x)
    s = BatchNormalization()(s)
    # LETS DO OVERFIT
    x = concatenate([s, x])
    x = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(0.25)(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(0.25)(x)
    #output = Dense(2, activation="relu")(x)
    x = Flatten()(x)
    X_SBP = Dense(1, activation='linear', name='SBP')(x)
    X_DBP = Dense(1, activation='linear', name='DBP')(x)

    model = Model(inputs=X_input, outputs=[X_SBP, X_DBP], name="Slapnicar_Model")
    # model = multi_gpu_model(model, gpus=2)
    # optimizer = optimizers.Adadelta()
    # loss = ks.keras.losses.mean_absolute_error
    # model.compile(optimizer=optimizer, loss=loss, metrics=['mae', 'mae'])
    # print(model.summary())
    # plot_model(model=model, to_file='lstm_model.png', show_shapes=True, show_layer_names=True)
    return model


def one_chennel_resnet(input_shape,num_filters=16,num_res_blocks = 5,cnn_per_res = 3,
                        kernel_sizes = [3,3,3], max_filters = 64, pool_size = 3,
                        pool_stride_size = 2,num_classes=8):
    my_input  = Input(shape=(input_shape))
    for i in np.arange(num_res_blocks):
        if(i==0):
            block_input = my_input
            x = BatchNormalization()(block_input)
        else:
            block_input = x
        for j in np.arange(cnn_per_res):
            x = Conv1D(num_filters, kernel_sizes[j], padding='same')(x)
            x = BatchNormalization()(x)
            if(j<cnn_per_res-1):
                x = Activation('relu')(x)
        is_expand_channels = not (input_shape[0] == num_filters)
        if is_expand_channels:
            res_conn = Conv1D(num_filters, 1,padding='same')(block_input)
            res_conn = BatchNormalization()(res_conn)
        else:
            res_conn = BatchNormalization()(block_input)
        x = add([res_conn, x])
        x = Activation('relu')(x)
        if(i<5):
            x = MaxPooling1D(pool_size=pool_size,strides =pool_stride_size)(x)
        num_filters = 2*num_filters
        if max_filters<num_filters:
            num_filters = max_filters
    return my_input,x


def one_chennel_resnet_2D(input_shape, input_layer, num_filters=16,num_res_blocks = 5,cnn_per_res = 3,
                        kernel_sizes = [8, 5, 3], max_filters = 64, pool_size = (3,3),
                        pool_stride_size = 2, num_classes=8):
    kernel_sizes = [(8, 1), (5, 1), (3, 1)]
    my_input = input_layer
    for i in np.arange(num_res_blocks):
        if(i==0):
            block_input = my_input
            x = BatchNormalization()(block_input)
        else:
            block_input = x
        for j in np.arange(cnn_per_res):
            x = Conv2D(num_filters, kernel_sizes[j], padding='same')(x)
            x = BatchNormalization()(x)
            if(j<cnn_per_res-1):
                x = Activation('relu')(x)
        is_expand_channels = not (input_shape[0] == num_filters)
        if is_expand_channels:
            res_conn = Conv2D(num_filters, (1,1), padding='same')(block_input)
            res_conn = BatchNormalization()(res_conn)
        else:
            res_conn = BatchNormalization()(block_input)
        x = add([res_conn, x])
        x = Activation('relu')(x)
        if(i<5):
            x = MaxPooling2D(pool_size=pool_size,strides =pool_stride_size)(x)
        num_filters = 2*num_filters
        if max_filters<num_filters:
            num_filters = max_filters
    return my_input,x


def spectro_layer_mid(input_x,sampling_rate, ndft=0, num_classes=8):
    l2_lambda = .001
    if(ndft == 0):
        n_dft= 128
    else:
        n_dft = ndft
    # n_dft = 64
    n_hop = 64
    fmin=0.0
    fmax=sampling_rate//2

    x = Permute((2,1))(input_x)
    x = STFT(n_fft=n_dft, hop_length=n_hop, output_data_format='channels_last')(x)
    x = Magnitude()(x)
    #x = MagnitudeToDecibel()(x)
    #x = BatchNormalization()(x)    # x = Normalization2D(str_axis='batch')(x)
    channel_resnet_input,channel_resnet_out= one_chennel_resnet_2D((625, 1), x, num_filters=64,
                    num_res_blocks = 6,cnn_per_res = 3,kernel_sizes = [3,3,3,3],
                    max_filters = 32, pool_size = 1,
                    pool_stride_size =1,num_classes=8)
    channel_resnet_out = BatchNormalization()(channel_resnet_out)

    # x = Reshape((10, 65))(x)
    # x = GRU(65)(x)

    return channel_resnet_out


#  Class custom_callback is used logging data and other operations of the model while its in learning process.
class custom_callback(tf.keras.callbacks.Callback):
    model_name = ""
    path = ""
    best = 100

    def __init__(self, dir, model_name, treshold=25):
        self.model_name = model_name
        self.path = dir + model_name + "/"
        self.best = treshold

    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        # print(self.model.summary())
        return

    def on_train_end(self, logs={}):
        # n = np.arange(0, len(self.losses))
        # plt.figure()
        # plt.plot(n, self.losses, label="train_loss")
        # plt.plot(n, self.acc, label="train_acc")
        # plt.plot(n, self.val_losses, label="val_loss")
        # plt.plot(n, self.val_acc, label="val_acc")
        # plt.title("Training Loss and Acc")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss/Accuracy")
        # plt.legend()
        # plt.savefig(self.path + 'training_stats.png')
        # plt.close()
        # self.model.save(self.path + "model_arch.h5")
        return

    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('mean_absolute_error'))
        self.val_acc.append(logs.get('val_mean_absolute_error'))
        if(logs.get('val_mean_absolute_error') < self.best):
            print("val_mean_absolute_error improved from " + str(self.best) + " to " + str(logs.get('val_mean_absolute_error')) + "...")
            self.best = logs.get('val_mean_absolute_error')
            self.model.save_weights("./Models/" + self.model_name + "_weights.h5")
        else:
            print("val_mean_absolute_error has not improved from " + str(self.best) + "...")


    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

