""" fine tune pretrained neural networks using rPPG data

This script trains neural network for the rPPG based prediction of blood pressure values using transfer learning. Neural
networks were previously trained using PPG data. rPPG data is provided via a hdf5 file. Validation is performed using a
leave-one-subject out cross validation scheme. Optionally, personalization of the neural network can be performed by using
a fraction of the test subject's data for training. The selection of this additional training data can be done randomly
or systematically

File: retrain_rppg_personalization.py
Author: Dr.-Ing. Fabian Schrumpf
E-Mail: Fabian.Schrumpf@htwk-leipzig.de
Date created: 9/9/2021
Date last modified: --
"""

from os.path import expanduser, join
import argparse
import h5py
import numpy as np
import pandas as pd
import tensorflow.keras as ks
import tensorflow as tf

np.random.seed(seed=42)

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

from kapre import STFT, Magnitude, MagnitudeToDecibel

from sklearn.model_selection import train_test_split

def retrain_rppg_personalization(data_file,
                                 ModelFile,
                                 experiment_name,
                                 checkpoint_path,
                                 ResultsPath,
                                 batch_size=32,
                                 lr = None,
                                 N_epochs = 40,
                                 PerformPersonalization=True,
                                 RandomPick=True):

    # modify the experiment name according to the selectes setting (with/without personalization)
    if PerformPersonalization:
        experiment_name = experiment_name + "_retrain_pers_" + ("random" if RandomPick else "first")
    else:
        experiment_name = experiment_name + "_retrain_pers_"

    # load rPPG data from the provided hdf5 files
    with h5py.File(data_file,'r') as f:
        rppg = f.get('rppg')
        BP = f.get('label')
        subjects = f.get('subject_idx')

        rppg = np.transpose(np.array(rppg), axes=(1,0))
        rppg = np.expand_dims(rppg, axis=2)
        BP = np.transpose(np.array(BP), axes=(1,0))
        subjects = np.array(subjects)

    subjects_list = np.unique(subjects)
    N_subjects = subjects_list.shape[-1]

    # column names for the csv files that store the results
    pd_col_names = ['subject', 'SBP_true', 'DBP_true', 'SBP_est_prepers', 'DBP_est_prepers', 'SBP_est_postpers',
                    'DBP_est_postpers']

    results = pd.DataFrame([], columns=pd_col_names)

    # import ML model
    dependencies = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel
    }
    model = ks.models.load_model(ModelFile, custom_objects=dependencies)

    # iterate over every subject and use it as a test subject
    for i in subjects_list:
        subjects_iter = subjects
        subjects_list_iter = subjects_list

        # determine index of the test subject and delete it from the subjects list
        idx_test = np.where(np.isin(subjects_iter, i))[-1]
        # subjects_iter = np.delete(subjects_iter, np.where(idx_test))
        subjects_list_iter = np.delete(subjects_list_iter, np.where(np.isin(subjects_list_iter,i)))

        # split remaining subejcts into training and validation set
        subjects_train, subjects_val = train_test_split(subjects_list_iter, test_size=0.2)

        idx_train = np.where(np.isin(subjects_iter, subjects_train))[-1]
        idx_val = np.where(np.isin(subjects_iter, subjects_val))[-1]
        #idx_train, idx_val = train_test_split(subjects_iter.astype(int), test_size=0.4)

        # if personalization is enabled: assign some data from the test subjects to the training set
        if PerformPersonalization:
            # choose data randomly or use first 20 % of the test subject's data
            if RandomPick:
                idx_test, idx_add_train = train_test_split(idx_test, test_size=0.2)
                idx_train = np.concatenate((idx_train, idx_add_train), axis=0)
            else:
                N_add_train = np.round(idx_test.shape[0]*0.2).astype(int)
                idx_add_train, idx_test = np.split(idx_test, [N_add_train])
                idx_train = np.concatenate((idx_train, idx_add_train), axis=0)


        SBP_train = BP[idx_train,0]
        DBP_train = BP[idx_train, 1]
        rppg_train = rppg[idx_train, :, :]

        idx_shuffle = np.random.permutation(SBP_train.shape[0] - 1)
        SBP_train = SBP_train[idx_shuffle]
        DBP_train = DBP_train[idx_shuffle]
        rppg_train = rppg_train[idx_shuffle,:,:]

        SBP_val = BP[idx_val, 0]
        DBP_val = BP[idx_val, 1]
        rppg_val = rppg[idx_val, :, :]
        SBP_test = BP[idx_test, 0]
        DBP_test = BP[idx_test, 1]
        rppg_test = rppg[idx_test, :, :]

        # model = ks.models.load_model(model_path + model_file, custom_objects=dependencies)
        model.load_weights(ModelFile)

        # Prediction on the test set BEFORE fine tuning
        SBP_val_prepers, DBP_val_prepers = model.predict(rppg_test)

        # set trainable flag of all layers to False except for the output layer
        for layer in model.layers[:-2]:
            layer.trainable = False

        if lr is None:
            opt = ks.optimizers.Adam()
        else:
            opt = ks.optimizers.Adam(learning_rate=lr)

        model.compile(optimizer=opt,
                      loss=ks.losses.mean_squared_error,
                      metrics=[['mae'], ['mae']])

        model.summary()

        # Checkpoint callback; Checkpoints are used to test the model using the test subject
        checkpoint_cb = ks.callbacks.ModelCheckpoint(
            filepath=join(checkpoint_path, experiment_name + '_cb.h5'),
            save_best_only=True,
            save_weights_only=True
        )

        # use early stopping to terminate the training if the validation error stops improving
        EarlyStopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # run the training
        history = model.fit(x=rppg_train, y=(SBP_train, DBP_train),
                            epochs=N_epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(rppg_val, (SBP_val, DBP_val)),
                            callbacks=[checkpoint_cb, EarlyStopping_cb])

        # load weights of the best epoch
        model.load_weights(checkpoint_cb.filepath)

        # Prediction on the test set AFTER fine tuning
        SBP_val_postpers, DBP_val_postpers = model.predict(rppg_test)

        results = results.append(pd.DataFrame(np.concatenate((
            i * np.ones(shape=(SBP_test.shape[0], 1)),
            np.expand_dims(SBP_test, axis=1),
            np.expand_dims(DBP_test, axis=1),
            SBP_val_prepers,
            DBP_val_prepers,
            SBP_val_postpers,
            DBP_val_postpers
        ), axis=1), columns=pd_col_names))

    results.to_csv(join(ResultsPath, experiment_name + '.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ExpName', type=str, help="Name of the training preceeded by the repsective date in the format MM-DD-YYYY")
    parser.add_argument('DataFile', type=str, help="Path to the hdf file containing rPPG signals")
    parser.add_argument('ResultsDir', type=str, help="Directory in which results are stored")
    parser.add_argument('ModelPath', type=str, help="Path where the model file used for rPPG based personalization is located")
    parser.add_argument('chkptdir', type=str, help="directory used for storing model checkpoints")
    parser.add_argument('--pers', type=int, default=0, help="If 0, performs personalizatin using data from the test subjct")
    parser.add_argument('--randompick', type=int, default=0, help="If 0, uses the first 20 %% of the test subject's data for testing, otherwise select randomly (only applies if --pers == 1)")
    args = parser.parse_args()

    DataFile = args.DataFile
    ModelPath = args.ModelPath
    experiment_name = args.ExpName
    ResultsDir = args.ResultsDir
    CheckpointDir = args.chkptdir
    PerformPers = True if args.pers == 1 else False
    RandomPick = True if args.randompick == 1 else False

    ModelFile = join(ModelPath, experiment_name + "_cb.h5")

    retrain_rppg_personalization(DataFile,
                                 ModelFile,
                                 experiment_name,
                                 CheckpointDir,
                                 ResultsDir,
                                 PerformPersonalization=PerformPers,
                                 RandomPick=RandomPick)

    #architecture = 'slapnicar'
    #date = "12-07-2021"
    #HomePath = expanduser("~")
    #experiment_name = "mimic_iii_ppg_nonmixed_pretrain"
    #ModelFile = join(HomePath, 'data', 'Sensors-Paper', 'ppg_pretrain',
    #                 date + "_" + architecture + "_" + experiment_name + '_cb.h5')
    #DataFile = join(HomePath,'data', 'rPPG-BP-UKL','rPPG-BP-UKL_rppg_7s.h5')
    #CheckpointDir = join(HomePath, 'data', 'rPPG-BP-UKL', 'checkpoints')
    #ResultsPath = join(HomePath,'Arbeit','7_Paper', '2021_Sensors_BP_ML', 'results', 'rppg_personalization')
    #retrain_rppg_personalization(DataFile,
    #                             ModelFile,
    #                             date + '_' + architecture+ '_' + experiment_name,
    #                             CheckpointDir,
    #                             ResultsPath,
    #                             PerformPersonalization=False,
    #                             RandomPick=False)
