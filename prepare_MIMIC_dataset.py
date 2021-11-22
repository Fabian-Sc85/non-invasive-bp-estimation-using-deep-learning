""" Preprocessing script for PPG data downloaded from the MIMIC-III databse

This script preprocesses the downloaded PPG data. ABP and PPG signals are divided into windows of a defined length and
overlap. Ground truth SBP and DBP values are extracted from the ABP signals. PPG signals are filtered. Different
plausibility and sanity checks are performed to exclude artifacts from the dataset. PPG signal windows and associated
SBP/DBP value pairs are stored in a .h5 file.

File: prepare_MIMIC_dataset.py
Author: Dr.-Ing. Fabian Schrumpf
E-Mail: Fabian.Schrumpf@htwk-leipzig.de
Date created: 8/5/2021
Date last modified: 8/5/2021
"""

from os.path import join, expanduser, isfile, splitext
from os import listdir, scandir
from random import shuffle
from sys import argv
import datetime
import argparse

import h5py
import numpy as np
from scipy.signal import butter, freqs, filtfilt
from sklearn.covariance import EllipticEnvelope

def CreateWindows(win_len, fs, N_samp, overlap):
    win_len = win_len * fs
    overlap = np.round(overlap*win_len)
    N_samp = N_samp - win_len + 1

    idx_start = np.round(np.arange(0,N_samp, win_len-overlap)).astype(int)
    idx_stop = np.round(idx_start + win_len - 1)

    return idx_start, idx_stop

def prepare_MIMIC_dataset(DataPath, OutputFile, NsampPerSubMax:int=None, NsampMax:int=None, win_len:int=7, win_overlap:float=0.5, savePPGData=False):

    if savePPGData == False:
        print("Saving BP data only")
    else:
        print("saving PPG data")

    RecordsFile = splitext(OutputFile)[0] + '_records.txt'
    SubjectDirs = scandir(DataPath)
    NumSubjects = sum(1 for x in SubjectDirs)
    SubjectDirs = scandir(DataPath)

    fs = 125

    SBP_min = 40;
    SBP_max = 200;
    DBP_min = 40;
    DBP_max = 120;

    # 4th order butterworth filter for PPG preprcessing
    b,a = butter(4,[0.5, 8], 'bandpass', fs=fs)

    # if output file does not exist already, create it
    if not isfile(OutputFile):
        with h5py.File(OutputFile, "a") as f:
            if savePPGData:
                f.create_dataset('ppg', (0,win_len*fs), maxshape=(None,win_len*fs), chunks=(100, win_len*fs))
            f.create_dataset('label', (0,2), maxshape=(None,2), dtype=int, chunks=(100,2))
            f.create_dataset('subject_idx', (0,1), maxshape=(None,1), dtype=int, chunks=(100,1))

        with open(RecordsFile,'w') as f:
            pass

    # loop over all subjects and their files in the source folder
    subjectID = 0
    for idx, dirs in enumerate(SubjectDirs):
        print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: Processing subject {idx+1} of {NumSubjects} ({dirs.name}): ', end='')

        PPG_RECORD = np.empty((0, win_len * fs))
        OUTPUT = np.empty((0, 2))

        DataFiles = [f for f in listdir(join(DataPath,dirs)) if isfile(join(DataPath, dirs,f)) and f.endswith('.h5')]
        shuffle(DataFiles)

        N_samp_total = 0
        for file in DataFiles:
            try:
                with h5py.File(join(DataPath, dirs, file), "r") as f:
                    data = {}
                    for key in f.keys():
                        data[key] = np.array(f[key]).transpose()
            except TypeError:
                print("could not read file. Skipping.")
            if savePPGData:
                PPG = data['val'][1, :]

            ABP = data['val'][0, :]

            if 'nB2' not in data:
                continue

            sys_p = data['nA2']


            ABP_sys_idx = data['nA3']-1
            ABP_dia_idx = data['nB3']-1

            # create start and stop indizes for time windows
            N_samples = ABP.shape[0]
            win_start, win_stop = CreateWindows(win_len, fs, N_samples, win_overlap)
            N_win = len(win_start)
            N_samp_total += N_win

            if savePPGData:
                ppg_record = np.zeros((N_win, win_len*fs))

            output = np.zeros((N_win, 2))

            # loop over windows
            for i in range(0, N_win):
                idx_start = win_start[i]
                idx_stop = win_stop[i]

                # extract peak idx of the current windows and the corresponding ABP signal values
                peak_idx = np.where(np.logical_and(sys_p >= idx_start, sys_p < idx_stop))
                sys_p_win = sys_p[peak_idx]
                N_sys_p = len(sys_p_win)

                # check if HR is in a plausible range
                if N_sys_p < (win_len/60)*40 or N_sys_p > (win_len/60)*120:
                    output[i,:] = np.nan
                    continue

                if savePPGData:
                    ppg_win = PPG[idx_start:idx_stop+1]

                # extract ABP window and fiducial points systolic and diastolic blood pressure
                abp_win = ABP[idx_start:idx_stop+1]

                # sanity check if enough peak values are present and if the number of SBP peaks matches the number of
                # DBP peaks
                ABP_sys_idx_win = ABP_sys_idx[np.logical_and(ABP_sys_idx >= idx_start, ABP_sys_idx < idx_stop)].astype(int)
                ABP_dia_idx_win = ABP_dia_idx[np.logical_and(ABP_dia_idx >= idx_start, ABP_dia_idx < idx_stop)].astype(int)

                if ABP_sys_idx_win.shape[-1] < (win_len/60)*40 or ABP_sys_idx_win.shape[-1] > (win_len/60)*120:
                    output[i, :] = np.nan
                    continue

                if ABP_dia_idx_win.shape[-1] < (win_len/60)*40 or ABP_dia_idx_win.shape[-1] > (win_len/60)*120:
                    output[i, :] = np.nan
                    continue

                if len(ABP_sys_idx_win) != len(ABP_dia_idx_win):
                    if ABP_sys_idx_win[0] > ABP_dia_idx_win[0]:
                        ABP_dia_idx_win = np.delete(ABP_dia_idx_win,0)
                    if ABP_sys_idx_win[-1] > ABP_dia_idx_win[-1]:
                        ABP_sys_idx_win = np.delete(ABP_sys_idx_win,-1)

                ABP_sys_win = ABP[ABP_sys_idx_win]
                ABP_dia_win = ABP[ABP_dia_idx_win]

                # check for NaN in ppg_win and abp_win
                if np.any(np.isnan(abp_win)):
                    output[i, :] = np.nan
                    continue

                if savePPGData:
                    if np.any(np.isnan(ppg_win)):
                        output[i, :] = np.nan
                        continue

                NN = np.diff(sys_p_win)/fs
                HR = 60/np.mean(NN)
                if HR < 50 or HR > 140:
                    output[i, :] = np.nan
                    continue

                # check for unreasonably large or small RR intervalls
                if np.any(NN < 0.3) or np.any(NN > 1.4):
                    output[i, :] = np.nan
                    continue

                # check if any of the SBP or DBP values exceed reasonable vlaues
                if np.any(np.logical_or(ABP_sys_win < SBP_min, ABP_sys_win > SBP_max)):
                    output[i, :] = np.nan
                    continue

                if np.any(np.logical_or(ABP_dia_win < DBP_min, ABP_dia_win > DBP_max)):
                    output[i, :] = np.nan
                    continue

                # check for NaN in the detected SBP and DBP peaks
                if np.any(np.isnan(ABP_sys_win)) or np.any(np.isnan(ABP_dia_win)):
                    output[i, :] = np.nan
                    continue

                # calculate the BP ground truth as the median of all SBP and DBP values in the present window
                BP_sys = np.median(ABP_sys_win).astype(int)
                BP_dia = np.median(ABP_dia_win).astype(int)

                # filter the ppg window using a 4th order Butterworth filter
                if savePPGData:
                    ppg_win = filtfilt(b,a, ppg_win)
                    ppg_win = ppg_win - np.mean(ppg_win)
                    ppg_win = ppg_win/np.std(ppg_win)
                    ppg_record[i, :] = ppg_win

                output[i,:] = [BP_sys, BP_dia]

                # if number of good samples (not NaN) exceeds maximum number of samples, stop extracting data
                N_nonNaN = np.count_nonzero(np.isnan(output[0:i+1,0]) == False)
                if NsampPerSubMax is not None:
                    if OUTPUT.shape[0] + N_nonNaN > 20*NsampPerSubMax:
                        output = np.delete(output,range(i,output.shape[0]), axis=0)

                        if savePPGData:
                            ppg_record = np.delete(ppg_record, range(i,ppg_record.shape[0]), axis=0)

                        break

            idx_nans = np.isnan(output[:,0])
            OUTPUT = np.vstack((OUTPUT, output[np.invert(idx_nans),:]))

            if savePPGData:
                PPG_RECORD = np.vstack((PPG_RECORD, ppg_record[np.invert(idx_nans),:]))

            # write record name to txt file for reproducibility
            with open(RecordsFile, 'a') as f:
                f.write(file[0:2] + "/" + file[0:-5]+"\n")

            if NsampPerSubMax is not None:
                if OUTPUT.shape[0] >= 20*NsampPerSubMax:
                    break

        if N_samp_total == 0:
            print(f'skipping')
            continue

        # save data is at least 100 good samples have been extracted
        if OUTPUT.shape[0] > 100:
            if NsampPerSubMax is not None:
                # if maximum number of samples per subject is defined, draw samples randomly
                if OUTPUT.shape[0] > NsampPerSubMax:
                    idx_select = np.random.choice(OUTPUT.shape[0]-1, size=(int(NsampPerSubMax)), replace=False)

                    if savePPGData:
                        PPG_RECORD = PPG_RECORD[idx_select,:]

                    OUTPUT = OUTPUT[idx_select,:]

            # add data to .h5 file
            with h5py.File(OutputFile, "a") as f:

                BP_dataset = f['label']
                DatasetCurrLength = BP_dataset.shape[0]
                DatasetNewLength = DatasetCurrLength + OUTPUT.shape[0]
                BP_dataset.resize(DatasetNewLength, axis=0)
                BP_dataset[-OUTPUT.shape[0]:,:] = OUTPUT

                if savePPGData:
                    ppg_dataset = f['ppg']

                    ppg_dataset.resize(DatasetNewLength, axis=0)
                    ppg_dataset[-PPG_RECORD.shape[0]:,:] = PPG_RECORD

                subject_dataset = f['subject_idx']
                subject_dataset.resize(DatasetNewLength, axis=0)
                subject_dataset[-OUTPUT.shape[0]:,:] = subjectID * np.ones((OUTPUT.shape[0], 1))

                print(f'{OUTPUT.shape[0]} samples ({DatasetNewLength} samples total)')
                if NsampMax is not None:
                    if f['label'].shape[0] > NsampMax:
                        return 0

        else:
            print(f'skipping')

        subjectID += 1

    print("script finished")

    return 0

if __name__ == "__main__":
    np.random.seed(seed=42)

    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', type=str,
                        help="Path containing data records downloaded from the MIMIC-III database")
    parser.add_argument('output', type=str, help="Target .h5 file")
    parser.add_argument('--win_len', type=int, nargs='?', default=7,
                        help="PPG window length in seconds (default: 7)")
    parser.add_argument('--win_overlap', type=float, nargs='?', default=0.5,
                        help="ammount of overlap between adjacend windows in fractions of the window length (default: 0.5)")
    parser.add_argument('--maxsampsubject', type=int, default=None, help="Maximum number of samples per subject")
    parser.add_argument('--maxsamp', type=int, default=None, help="Maximum total number os samples in the dataset")
    parser.add_argument('--save_ppg_data', type=int, default=0, help="0: save BP data only; 1: save PPG and BP data")
    args = parser.parse_args()

    DataPath = args.datapath
    OutputFile = args.output
    win_len = args.win_len
    win_overlap = args.win_overlap
    NsampPerSubMax = args.maxsampsubject
    NsampMax = args.maxsamp
    savePPGData = args.save_ppg_data
    if savePPGData == 0:
        savePPGData = False
    else:
        savePPGData = True

    prepare_MIMIC_dataset(DataPath, OutputFile, NsampPerSubMax=NsampPerSubMax, NsampMax=NsampMax,
                          savePPGData=savePPGData, win_len=win_len, win_overlap=win_overlap)