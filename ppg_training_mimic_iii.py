""" train neural architectures using PPG data

This script trains a neural network using PPG data. The data is loaded from the the .tfrecord files created by the script
'hdf_to_tfrecord.py'. Four different neural architectures can be selected:

- AlexNet [1]
- ResNet [2]
- Architecture published by Slapnicar et al. (modified to work with Tensorflow 2.4.1) [3] The original code can be downloaded
  from https://github.com/gslapnicar/bp-estimation-mimic3
- LSTM network

A checkpoint callback is used to store the best network weights in terms of validation loss. These weights are subsequently
used to perform predictions on the test set. Test results are stored in a csv file for later evaluation.

References
[1] A. Krizhevsky, I. Sutskever, und G. E. Hinton, „ImageNet classification with deep convolutional neural networks“,
    Commun. ACM, Bd. 60, Nr. 6, S. 84–90, Mai 2017, doi: 10.1145/3065386.
[2] K. He, X. Zhang, S. Ren, und J. Sun, „Deep Residual Learning for Image Recognition“, in 2016 IEEE Conference on
    Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, Juni 2016, S. 770–778. doi: 10.1109/CVPR.2016.90.
[3] G. Slapničar, N. Mlakar, und M. Luštrek, „Blood Pressure Estimation from Photoplethysmogram Using a Spectro-Temporal
    Deep Neural Network“, Sensors, Bd. 19, Nr. 15, S. 3420, Aug. 2019, doi: 10.3390/s19153420.

File: prepare_MIMIC_dataset.py
Author: Dr.-Ing. Fabian Schrumpf
E-Mail: Fabian.Schrumpf@htwk-leipzig.de
Date created: 8/9/2021
Date last modified: 8/9/2021
"""

from os.path import expanduser, join
from os import environ
from sys import argv
from functools import partial
from datetime import datetime
import argparse

import tensorflow as tf
import pandas as pd
import numpy as np

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

from models.define_ResNet_1D import ResNet50_1D
from models.define_AlexNet_1D import AlexNet_1D
from models.define_LSTM import define_LSTM

from models.slapnicar_model import raw_signals_deep_ResNet

def read_tfrecord(example, win_len=875):
    tfrecord_format = (
        {
            'ppg': tf.io.FixedLenFeature([win_len], tf.float32),
            'label': tf.io.FixedLenFeature([2], tf.float32)
        }
    )
    parsed_features = tf.io.parse_single_example(example, tfrecord_format)

    return parsed_features['ppg'], (parsed_features['label'][0], parsed_features['label'][1])

def create_dataset(tfrecords_dir, tfrecord_basename, win_len=875, batch_size=32, modus='train'):

    pattern = join(tfrecords_dir, modus, tfrecord_basename + "_" + modus + "_?????_of_?????.tfrecord")
    dataset = tf.data.TFRecordDataset.list_files(pattern)

    if modus == 'train':
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=800,
            block_length=400)
    else:
        dataset = dataset.interleave(
            tf.data.TFRecordDataset)

    dataset = dataset.map(partial(read_tfrecord, win_len=win_len), num_parallel_calls=2)
    dataset = dataset.shuffle(4096, reshuffle_each_iteration=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.repeat()

    return dataset

def get_model(architecture, input_shape, UseDerivative=False):
    return {
        'resnet': ResNet50_1D(input_shape, UseDerivative=UseDerivative),
        'alexnet': AlexNet_1D(input_shape, UseDerivative=UseDerivative),
        'slapnicar' : raw_signals_deep_ResNet(input_shape, UseDerivative=UseDerivative),
        'lstm' : define_LSTM(input_shape)
    }[architecture]

def ppg_train_mimic_iii(architecture,
                        DataDir,
                        ResultsDir,
                        CheckpointDir,
                        tensorboard_tag,
                        tfrecord_basename,
                        experiment_name,
                        win_len=875,
                        batch_size=32,
                        lr = None,
                        N_epochs = 20,
                        Ntrain=1e6,
                        Nval=2.5e5,
                        Ntest=2.5e5,
                        UseDerivative=False,
                        earlystopping=True):

    # create datasets for training, validation and testing using .tfrecord files
    test_dataset = create_dataset(DataDir, tfrecord_basename, win_len=win_len, batch_size=batch_size,
                                  modus='test')
    train_dataset = create_dataset(DataDir, tfrecord_basename, win_len=win_len, batch_size=batch_size, modus='train')
    val_dataset = create_dataset(DataDir, tfrecord_basename, win_len=win_len, batch_size=batch_size,
                                 modus='val')


    data_in_shape = (win_len,1)

    # load the neurarchitecture
    model = get_model(architecture, data_in_shape, UseDerivative=UseDerivative)
    print(model.summary())
    # callback for logging training and validation results
    csvLogger_cb = tf.keras.callbacks.CSVLogger(
        filename=join(ResultsDir,experiment_name + '_learningcurve.csv')
    )

    # checkpoint callback
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=join(CheckpointDir , experiment_name + '_cb.h5'),
        save_best_only=True
    )

    # tensorboard callback
    tensorbard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=join(ResultsDir, 'tb', tensorboard_tag),
        histogram_freq=0,
        write_graph=False
    )

    # callback for early stopping if validation loss stops improving
    EarlyStopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # define Adam optimizer
    if lr is None:
        opt = tf.keras.optimizers.Adam()
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # compile model using mean squared error as loss function
    model.compile(
        optimizer=opt,
        loss = tf.keras.losses.mean_squared_error,
        metrics = [['mae'], ['mae']]
    )

    cb_list = [checkpoint_cb,
               tensorbard_cb,
               csvLogger_cb,
               EarlyStopping_cb if earlystopping == True else []]

    # Perform Training and Validation
    history = model.fit(
        train_dataset,
        steps_per_epoch=Ntrain//batch_size,
        epochs=N_epochs,
        validation_data=val_dataset,
        validation_steps=Nval//batch_size,
        callbacks=cb_list
    )

    # Predictions on the testset
    model.load_weights(checkpoint_cb.filepath)
    test_results = pd.DataFrame({'SBP_true' : [],
                                 'DBP_true' : [],
                                 'SBP_est' : [],
                                 'DBP_est' : []})

    # store predictions on the test set as well as the corresponding ground truth in a csv file
    test_dataset = iter(test_dataset)
    for i in range(int(Ntest//batch_size)):
        ppg_test, BP_true = test_dataset.next()
        BP_est = model.predict(ppg_test)
        TestBatchResult = pd.DataFrame({'SBP_true' : BP_true[0].numpy(),
                                        'DBP_true' : BP_true[1].numpy(),
                                        'SBP_est' : np.squeeze(BP_est[0]),
                                        'DBP_est' : np.squeeze(BP_est[1])})
        test_results = test_results.append(TestBatchResult)

    ResultsFile = join(ResultsDir,experiment_name + '_test_results.csv')
    test_results.to_csv(ResultsFile)

    idx_min = np.argmin(history.history['val_loss'])

    print(' Training finished')

    return history.history['SBP_mae'][idx_min], history.history['DBP_mae'][idx_min], history.history['val_SBP_mae'][idx_min], history.history['val_DBP_mae'][idx_min]

if __name__ == "__main__":

    if len(argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('ExpName', type=str, help="unique name for the training")
        parser.add_argument('datadir', type=str,
                            help="folder containing the train, val and test subfolders containing tfrecord files")
        parser.add_argument('resultsdir', type=str, help="Directory in which results are stored")
        parser.add_argument('chkptdir', type=str, help="directory used for storing model checkpoints")
        parser.add_argument('--arch', type=str, default="alexnet",
                            help="neural architecture used for training (alexnet (default), resnet,  slapnicar, lstm)")
        parser.add_argument('--lr', type=float, default=0.003, help="initial learning rate (default: 0.003)")
        parser.add_argument('--batch_size', type=int, default=32, help="batch size used for training (default: 32)")
        parser.add_argument('--winlen', type=int, default=875,
                            help="length of the ppg windows in samples (default: 875)")
        parser.add_argument('--epochs', type=int, default=60,
                            help="maximum number of epochs for training (default: 60)")
        parser.add_argument('--gpuid', type=str, default=None,
                            help="GPU-ID used for training in a multi-GPU environment (default: None)")
        args = parser.parse_args()

        architecture = args.arch
        experiment_name = args.ExpName
        experiment_name = datetime.now().strftime("%Y-%d-%m") + '_' + architecture + '_' + experiment_name
        DataDir = args.datadir
        ResultsDir = args.resultsdir
        CheckpointDir = args.chkptdir
        tb_tag = experiment_name
        lr = args.lr if args.lr > 0 else None
        batch_size = args.batch_size
        win_len = args.winlen
        N_epochs = args.epochs
        if args.gpuid is not None:
            environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    else:
        architecture = 'lstm'
        experiment_name = datetime.now().strftime("%Y-%d-%m") + '_' + architecture + '_' + 'mimic_iii_ppg_nonmixed_pretrain'
        HomePath = expanduser("~")
        DataDir = join(HomePath,'data','MIMIC-III_BP', 'tfrecords_nonmixed')
        ResultsDir = join(HomePath,'data','MIMIC-III_BP', 'results')
        CheckpointDir = join(HomePath,'data','MIMIC-III_BP', 'checkpoints')
        tb_tag = architecture + '_' + 'mimic_iii_ppg_pretrain'
        batch_size = 64
        win_len = 875
        lr = None
        N_epochs = 60

    tfrecord_basename = 'MIMIC_III_ppg'

    ppg_train_mimic_iii(architecture,
                        DataDir,
                        ResultsDir,
                        CheckpointDir,
                        tb_tag,
                        tfrecord_basename,
                        experiment_name,
                        win_len=win_len,
                        batch_size=batch_size,
                        lr=lr,
                        N_epochs=N_epochs,
                        UseDerivative=True,
                        earlystopping=False)