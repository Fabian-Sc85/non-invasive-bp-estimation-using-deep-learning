""" personalize neural architectures using data from test subjects

This script retrains a pretrained neural network using additional data from test subjects. The pretrained network resulted
from a PPG based training by the script 'ppg_training_mimic_iii.py'. Additional data can be the first 20 % of the test
subject's data or be comprised of randomly drawn 20 %. Validation is performed using the remaining 80 % of the data. The
script performs this personalization for a defined number of subjects separately and stores the results for further
analysis.

File: prepare_MIMIC_dataset.py
Author: Dr.-Ing. Fabian Schrumpf
E-Mail: Fabian.Schrumpf@htwk-leipzig.de
Date created: 8/10/2021
Date last modified: 8/10/2021
"""

from os.path import join, expanduser, isfile
from functools import partial
import argparse

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

from tensorflow.keras.layers import ReLU
from kapre import STFT, Magnitude, MagnitudeToDecibel
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def read_tfrecord(example, win_len=1875):
    tfrecord_format = (
        {
            'ppg': tf.io.FixedLenFeature([win_len], tf.float32),
            'label': tf.io.FixedLenFeature([2], tf.float32),
            'subject_idx': tf.io.FixedLenFeature([1], tf.float32)
        }
    )
    parsed_features = tf.io.parse_single_example(example, tfrecord_format)

    return parsed_features['ppg'], (parsed_features['label'][0], parsed_features['label'][1]), parsed_features['subject_idx']

def create_dataset(tfrecords_dir, tfrecord_basename, win_len=1875, batch_size=32, modus='train'):
    pattern = join(tfrecords_dir, modus, tfrecord_basename + "_" + modus + "_?????_of_?????.tfrecord")
    dataset = tf.data.TFRecordDataset.list_files(pattern)

    if modus == 'train':
        dataset = dataset.shuffle(100, reshuffle_each_iteration=True)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=800,
            block_length=100)
    else:
        dataset = dataset.interleave(
            tf.data.TFRecordDataset)

    dataset = dataset.map(partial(read_tfrecord, win_len=win_len), num_parallel_calls=4)
    dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.repeat()

    return dataset

def ppg_personalization_mimic_iii(DataDir,
                                  ResultsDir,
                                  ModelFile,
                                  CheckpointDir,
                                  tfrecord_basename,
                                  experiment_name,
                                  win_len=875,
                                  batch_size=32,
                                  lr = None,
                                  N_epochs = 40,
                                  Nsamp=2.5e5,
                                  Ntrials = 30,
                                  RandomPick = True):

    pd_col_names = ['subject', 'SBP_true', 'DBP_true', 'SBP_est_prepers', 'DBP_est_prepers', 'SBP_est_postpers', 'DBP_est_postpers']
    results = pd.DataFrame([], columns=pd_col_names)
    experiment_name = experiment_name + '_pers'

    # Load the test set from the .tfrecord files and save it as a .npz file for easier access
    if isfile(join(DataDir, experiment_name + "_dataset.npz")):
        npz_file = np.load(join(DataDir, experiment_name + "_dataset.npz"))
        ppg = npz_file['arr_0']
        BP = npz_file['arr_1']
        subject_idx = npz_file['arr_2']
    else:
        # Load test dataset for personalization
        dataset = create_dataset(DataDir, tfrecord_basename, win_len=win_len, batch_size=batch_size, modus='test')
        dataset = iter(dataset)
        ppg = np.empty(shape=(int(Nsamp), int(win_len)))
        BP = np.empty(shape=(int(Nsamp), 2))
        subject_idx = np.empty(shape=(int(Nsamp)))

        for i in range(int(Nsamp) // int(batch_size)):
            ppg_batch, BP_batch, subject_idx_batch = dataset.get_next()
            ppg[i * batch_size:(i + 1) * batch_size, :] = ppg_batch.numpy()
            BP[i * batch_size:(i + 1) * batch_size, :] = np.transpose(np.asarray(BP_batch))
            subject_idx[i * batch_size:(i + 1) * batch_size] = np.squeeze(subject_idx_batch.numpy())

        np.savez(join(DataDir, experiment_name + "_dataset.npz"), ppg, BP, subject_idx,['ppg', 'BP', 'subject_idx'])

    # draw test subjects randomly and save their ID for reproducibility
    subjects = np.unique(subject_idx)
    if isfile(join(ResultsDir,'ppg_personalization_subject_list.txt')):
        file = open(join(ResultsDir,'ppg_personalization_subject_list.txt'),'r')
        trial_subjects = file.read()
        trial_subjects = [int(float(i)) for i in trial_subjects.split('\n')[:-1]]
    else:
        trial_subjects = np.random.choice(subjects, size=Ntrials, replace=False)
        with open(join(ResultsDir,'ppg_personalization_subject_list.txt'),'w') as f:
            for item in trial_subjects:
                f.write(("%s\n" % item))

    # perform personalization for each test subject
    for subject in trial_subjects:
        print(f'Processing subject {subject} of {len(trial_subjects)}')

        ppg_trial = ppg[subject_idx==subject,:]
        BP_trial = BP[subject_idx==subject,:]
        Nsamp_trial = BP_trial.shape[0]
        N_train = int(np.round(0.2*Nsamp_trial))

        idx_test = np.arange(N_train+1,Nsamp_trial,2)
        ppg_test = ppg_trial[idx_test,:]
        BP_test = BP_trial[idx_test,:]

        ppg_trial = np.delete(ppg_trial, idx_test, axis=0)
        BP_trial = np.delete(BP_trial, idx_test, axis=0)

        # draw training data from the test subjct's data
        if RandomPick==True:
            idx_train, idx_val = train_test_split(range(ppg_trial.shape[0]), test_size=int(N_train), shuffle=True)
            ppg_train = ppg_trial[idx_train,:]
            BP_train = BP_trial[idx_train,:]
            ppg_val = ppg_trial[idx_val,:]
            BP_val = BP_trial[idx_val,:]
        else:
            ppg_train = ppg_trial[:N_train, :]
            BP_train = BP_trial[:N_train, :]
            ppg_val = ppg_trial[:N_train, :]
            BP_val = BP_trial[:N_train, :]

        # load model dependencies
        dependencies = {
            'ReLU': ReLU,
            'STFT': STFT,
            'Magnitude': Magnitude,
            'MagnitudeToDecibel': MagnitudeToDecibel
        }

        model = tf.keras.models.load_model(ModelFile, custom_objects=dependencies)

        # retrain only the last 7 layers
        for layer in model.layers[:-7]:
            layer.trainable = False

        if lr is None:
            opt = tf.keras.optimizers.Adam()
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.mean_squared_error,
            metrics=[['mae'], ['mae']]
        )

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=CheckpointDir + experiment_name + '.h5',
            save_best_only=True,
            save_weights_only=True
        )

        EarlyStopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # prediction on the test data prior to personalization
        SBP_val_prepers, DBP_val_prepers = model.predict(ppg_test)

        SBP_train = BP_train[:, 0]
        DBP_train = BP_train[:, 1]
        SBP_val = BP_val[:, 0]
        DBP_val = BP_val[:, 1]

        # perform personalization using 20% of the test subject's data
        history = model.fit(x=ppg_train, y=(SBP_train, DBP_train),
                            epochs=N_epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(ppg_val, (SBP_val, DBP_val)),
                            callbacks=[checkpoint_cb, EarlyStopping_cb])

        # prediction on the test data after personalization
        model.load_weights(checkpoint_cb.filepath)
        SBP_val_postpers, DBP_val_postpers = model.predict(ppg_test)

        # save predictions for later analysis
        results = results.append(pd.DataFrame(np.concatenate((
            subject*np.ones(shape=(BP_test.shape[0],1)),
            np.expand_dims(BP_test[:,0], axis=1),
            np.expand_dims(BP_test[:,1], axis=1),
            SBP_val_prepers,
            DBP_val_prepers,
            SBP_val_postpers,
            DBP_val_postpers
        ),axis=1), columns=pd_col_names))

        if RandomPick == True:
            results.to_csv(join(ResultsDir, experiment_name + '_random.csv'))
        else:
            results.to_csv(join(ResultsDir, experiment_name + '_first.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ExpName', type=str, help="Name of the training preceeded by the repsective date in the format MM-DD-YYYY")
    parser.add_argument('DataDir', type=str, help="folder containing the train, val and test subfolders containing tfrecord files")
    parser.add_argument('ResultsDir', type=str, help="Directory in which results are stored")
    parser.add_argument('ModelPath', type=str, help="Path where the model file used for personalization is located")
    parser.add_argument('chkptdir', type=str, help="directory used for storing model checkpoints")
    parser.add_argument('--lr', type=float, default=0.003, help="initial learning rate (default: 0.003)")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size used for training (default: 32)")
    parser.add_argument('--winlen', type=int, default=875, help="length of the ppg windows in samples (default: 875)")
    parser.add_argument('--epochs', type=int, default=1000, help="maximum number of epochs for training (default: 60)")
    parser.add_argument('--nsubj', type=int, default=20, help="Number subjects used for personalization (default :20)")
    parser.add_argument('--randompick', type=int, default=0, help="define wether data for personalization is drawn randomly (1) or comprises the first 20 %% of the test subject's data (0) (default: 0)")
    args = parser.parse_args()

    tfrecord_basename = 'MIMIC_III_ppg'

    ExpName = args.ExpName
    DataDir = args.DataDir
    ResultsDir = args.ResultsDir
    ModelPath = args.ModelPath
    CheckpointDir = args.chkptdir
    win_len = args.winlen
    lr = args.lr
    N_epochs = args.epochs
    N_trials = args.nsubj
    RandomPick = True if args.randompick == 1 else False

    ModelFile = join(ModelPath, ExpName + '_cb.h5')
    ppg_personalization_mimic_iii(DataDir,
                                  ResultsDir,
                                  ModelFile,
                                  CheckpointDir,
                                  tfrecord_basename,
                                  ExpNamewin_len=win_len,
                                  lr=lr,
                                  Ntrials=N_trials,
                                  N_epochs=N_epochs,
                                  RandomPick=False)

    #architecture = 'slapnicar'
    #date = "12-07-2021"
    #HomePath = expanduser("~")
    #experiment_name = "mimic_iii_ppg_nonmixed_pretrain"
    #ModelFile = join(HomePath, 'data', 'Sensors-Paper', 'ppg_pretrain',
    #                 date + "_" + architecture + "_" + experiment_name + '_cb.h5')
    #DataDir = join(HomePath,'data','MIMIC-III_BP', 'tfrecords_nonmixed')
    #ResultsDir = join(HomePath,'Arbeit','7_Paper', '2021_Sensors_BP_ML', 'results', 'ppg_personalization')
    #CheckpointDir = join(HomePath,'data','MIMIC-III_BP', 'checkpoints')
    #tfrecord_basename = 'MIMIC_III_ppg'

    #learning_rate = None

    #ppg_personalization_mimic_iii(DataDir,
    #                              ResultsDir,
    #                              ModelFile,
    #                              CheckpointDir,
    #                              tfrecord_basename,
    #                              date+'_' + architecture+ '_' +experiment_name,
    #                              win_len=875,
    #                              lr=learning_rate,
    #                              Ntrials=20,
    #                              N_epochs=100,
    #                              RandomPick=False)

