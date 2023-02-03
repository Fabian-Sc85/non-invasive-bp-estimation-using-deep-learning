""" Convert PPG/BP samples pairs to a binary format that is used for training neural networks using Tensorflow

This script reads a dataset consisting of PPG and BP samples from a .h5 file and converts them into a binary format that
can be used for as input data for a neural network during training. The dataset can be divided into training, validation
and test set by (i) dividing the dataset on a subject basis ensuring that data from one subject are not scattered across
training, validation and test set or (ii) dividing the dataset randomly.

File: prepare_MIMIC_dataset.py
Author: Dr.-Ing. Fabian Schrumpf
E-Mail: Fabian.Schrumpf@htwk-leipzig.de
Date created: 8/6/2021
Date last modified: 8/6/2021
"""

import matplotlib.pyplot as plt
import argparse
import pandas as pd
import h5py
import tensorflow as tf
# ks.enable_eager_execution()
import numpy as np
from sklearn.model_selection import train_test_split

from datetime import datetime
from os.path import expanduser, isdir, join
from os import mkdir
from sys import argv

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    #    if isinstance(value, type(ks.constant(0))):
    #        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def ppg_hdf2tfrecord(h5_file, tfrecord_path, samp_idx, weights_SBP=None, weights_DBP=None):
    # Function that converts PPG/BP sample pairs into the binary .tfrecord file format. This function creates a .tfrecord
    # file containing a defined number os samples
    #
    # Parameters:
    # h5_file: file containing ppg and BP data
    # tfrecordpath: full path for storing the .tfrecord files
    # samp_idx: sample indizes of the data in the .h5 file to be stored in the .tfrecord file
    # weights_SBP: sample weights for the systolic BP (optional)
    # weights_DBP: sample weights for the diastolic BP (optional)

    N_samples = len(samp_idx)
    # open the .h5 file and get the samples with the indizes specified by samp_idx
    with h5py.File(h5_file, 'r') as f:
        # load ppg and BP data as well as the subject numbers the samples belong to
        ppg_h5 = f.get('/ppg')
        BP = f.get('/label')
        subject_idx = f.get('/subject_idx')

        writer = tf.io.TFRecordWriter(tfrecord_path)

        # iterate over each sample index and convert the corresponding data to a binary format
        for i in np.nditer(samp_idx):

            ppg = np.array(ppg_h5[i,:])

            if weights_SBP is not None and weights_DBP is not None:
                weight_SBP = weights_SBP[i]
                weight_DBP = weights_DBP[i]
            else:
                weight_SBP = 1
                weight_DBP = 1

            target = np.array(BP[i,:], dtype=np.float32)
            sub_idx = np.array(subject_idx[i])

            # create a dictionary containing the serialized data
            data = \
                {'ppg': _float_feature(ppg.tolist()),
                 'label': _float_feature(target.tolist()),
                 'subject_idx': _float_feature(sub_idx.tolist()),
                 'weight_SBP': _float_feature([weight_SBP]),
                 'weight_DBP': _float_feature([weight_DBP]),
                 'Nsamples': _float_feature([N_samples])}

            # write data to the .tfrecord target file
            feature = tf.train.Features(feature=data)
            example = tf.train.Example(features=feature)
            serialized = example.SerializeToString()

            writer.write(serialized)

        writer.close()


def ppg_hdf2tfrecord_sharded(h5_file, samp_idx, tfrecordpath, Nsamp_per_shard, modus='train', weights_SBP=None,
                         weights_DBP=None):
    # Save PPG/BP pairs as .tfrecord files. Save defined number os samples per file (Sharding)
    # Weights can be defined for each sample
    #
    # Parameters:
    # h5_file: File that contains the whole dataset (in .h5 format), created by
    # samp_idx: sample indizes from the dataset in the h5. file that are used to create this tfrecords dataset
    # tfrecordpath: full path for storing the .tfrecord files
    # N_samp_per_shard: number of samples per shard/.tfrecord file
    # modus: define if the data is stored in the "train", "val" or "test" subfolder of "tfrecordpath"
    # weights_SBP: sample weights for the systolic BP (optional)
    # weights_DBP: sample weights for the diastolic BP (optional)

    base_filename = join(tfrecordpath, 'MIMIC_III_ppg')

    N_samples = len(samp_idx)

    # calculate the number of Files/shards that are needed to stroe the whole dataset
    N_shards = np.ceil(N_samples / Nsamp_per_shard).astype(int)

    # iterate over every shard
    for i in range(N_shards):
        idx_start = i * Nsamp_per_shard
        idx_stop = (i + 1) * Nsamp_per_shard
        if idx_stop > N_samples:
            idx_stop = N_samples

        idx_curr = samp_idx[idx_start:idx_stop]
        output_filename = '{0}_{1}_{2:05d}_of_{3:05d}.tfrecord'.format(base_filename,
                                                                       modus,
                                                                       i + 1,
                                                                       N_shards)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string, ': processing ',
              modus,
              ' shard ', str(i + 1), ' of ', str(N_shards))
        ppg_hdf2tfrecord(h5_file, output_filename, idx_curr, weights_SBP=weights_SBP, weights_DBP=weights_DBP)


def h5_to_tfrecords(SourceFile, tfrecordsPath, N_train=1e6, N_val=2.5e5, N_test=2.5e5,
                    divide_by_subject=True, save_tfrecords=True):
    N_train = int(N_train)
    N_val = int(N_val)
    N_test = int(N_test)

    tfrecord_path_train = join(tfrecordsPath, 'train')
    if not isdir(tfrecord_path_train):
        mkdir(tfrecord_path_train)
    tfrecord_path_val = join(tfrecordsPath, 'val')
    if not isdir(tfrecord_path_val):
        mkdir(tfrecord_path_val)
    tfrecord_path_test = join(tfrecordsPath, 'test')
    if not isdir(tfrecord_path_test):
        mkdir(tfrecord_path_test)

    csv_path = tfrecordsPath

    Nsamp_per_shard = 1000

    with h5py.File(SourceFile, 'r') as f:
        BP = np.array(f.get('/label'))
        BP = np.round(BP)
        BP = np.transpose(BP)
        subject_idx = np.squeeze(np.array(f.get('/subject_idx')))

    N_samp_total = BP.shape[1]
    subject_idx = subject_idx[:N_samp_total]

    # Divide the dataset into training, validation and test set
    # -------------------------------------------------------------------------------
    if divide_by_subject is True:
        valid_idx = np.arange(subject_idx.shape[-1])

        # divide the subjects into training, validation and test subjects
        subject_labels = np.unique(subject_idx)
        subjects_train_labels, subjects_val_labels = train_test_split(subject_labels, test_size=0.5)
        subjects_val_labels, subjects_test_labels = train_test_split(subjects_val_labels, test_size=0.5)

        # Calculate samples belong to training, validation and test subjects
        train_part = valid_idx[np.isin(subject_idx,subjects_train_labels)]
        val_part = valid_idx[np.isin(subject_idx,subjects_val_labels)]
        test_part = valid_idx[np.isin(subject_idx, subjects_test_labels)]

        # draw a number samples defined by N_train, N_val and N_test from the training, validation and test subjects
        idx_train = np.random.choice(train_part, N_train, replace=False)
        idx_val = np.random.choice(val_part, N_val, replace=False)
        idx_test = np.random.choice(test_part, N_test, replace=False)
    else:
        # Create a subset of the whole dataset by drawing a number of subjects from the dataset. The total number of
        # samples contributed by those subjects must equal N_train + N_val + _N_test
        subject_labels, SampSubject_hist = np.unique(subject_idx, return_counts=True)
        cumsum_samp = np.cumsum(SampSubject_hist)
        subject_labels_train = subject_labels[:np.nonzero(cumsum_samp>(N_train+N_val+N_test))[0][0]]
        idx_valid = np.nonzero(np.isin(subject_idx,subject_labels_train))[0]

        # divide subset randomly into training, validation and test set
        idx_train, idx_val = train_test_split(idx_valid, train_size= N_train, test_size=N_val+N_test)
        idx_val, idx_test = train_test_split(idx_val, test_size=0.5)

    # save ground truth BP values of training, validation and test set in csv-files for future reference
    BP_train = BP[:,idx_train]
    d = {"SBP": np.transpose(BP_train[0, :]), "DBP": np.transpose(BP_train[1, :])}
    train_set = pd.DataFrame(d)
    train_set.to_csv(csv_path + 'MIMIC-III_BP_trainset.csv')
    BP_val = BP[:,idx_val]
    d = {"SBP": np.transpose(BP_val[0, :]), "DBP": np.transpose(BP_val[1, :])}
    train_set = pd.DataFrame(d)
    train_set.to_csv(csv_path + 'MIMIC-III_BP_valset.csv')
    BP_test = BP[:,idx_test]
    d = {"SBP": np.transpose(BP_test[0, :]), "DBP": np.transpose(BP_test[1, :])}
    train_set = pd.DataFrame(d)
    train_set.to_csv(csv_path + 'MIMIC-III_BP_testset.csv')

    # create tfrecord dataset
    # ----------------------------
    if save_tfrecords:
        np.random.shuffle(idx_train)
        ppg_hdf2tfrecord_sharded(SourceFile, idx_test, tfrecord_path_test, Nsamp_per_shard, modus='test')
        ppg_hdf2tfrecord_sharded(SourceFile, idx_train, tfrecord_path_train, Nsamp_per_shard, modus='train')
        ppg_hdf2tfrecord_sharded(SourceFile, idx_val, tfrecord_path_val, Nsamp_per_shard, modus='val')
    print("Script finished")

if __name__ == "__main__":
    np.random.seed(seed=42)

    if len(argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('input', type=str, help="Path to the .h5 file containing the dataset")
        parser.add_argument('output', type=str, help="Target folder for the .tfrecord files")
        parser.add_argument('--ntrain', type=int, default=1e6,
                            help="Number of samples in the training set (default: 1e6)")
        parser.add_argument('--nval', type=int, default=2.5e5,
                            help="Number of samples in the validation set (default: 2.5e5)")
        parser.add_argument('--ntest', type=int, default=2.5e5,
                            help="Number of samples in the test set (default: 2.5e5)")
        parser.add_argument('--divbysubj', type=int, default=1,
                            help="Perform subject based (1) or sample based (0) division of the dataset")
        args = parser.parse_args()
        SourceFile = args.input
        tfrecordsPath = args.output
        divbysubj = True if args.divbysubj == 1 else False

        N_train = int(args.ntrain)
        N_val = int(args.nval)
        N_test = int(args.ntest)
    else:
        HomePath = expanduser("~")
        SourceFile = join(HomePath, 'data', 'MIMIC-III_BP', 'MIMIC-III_ppg_dataset.h5')
        tfrecordsPath = join(HomePath, 'test')
        divbysubj = True
        N_train = 1e6
        N_val = 2.5e5
        N_test = 2.5e5

    h5_to_tfrecords(SourceFile=SourceFile, tfrecordsPath=tfrecordsPath, divide_by_subject=divbysubj,
                      N_train=N_train, N_val=N_val, N_test=N_test)
