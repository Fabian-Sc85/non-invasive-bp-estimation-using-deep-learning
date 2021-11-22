
""" Download script for MIMIC-III data

In order to reproduce the results in the Sensors publication "Assessment of non-invasive blood pressure prediction from
PPG and rPPG signals using deep learning" the exact same data as used in the paper is downloaded. The record names are
provided in a text file. The scripts downloads those records, extract PPG and ABP data and performs peak detection on the
ABP (systolic and diastolic peaks of the ABP signals to generate systolic and diastolic blood pressure values as ground
truth) and PPG signals. ABP and PPG signals as well as the detected peaks are stored in .h5 files.

File: download_mimic_iii_records.py
Author: Dr.-Ing. Fabian Schrumpf
E-Mail: Fabian.Schrumpf@htwk-leipzig.de
Date created: 8/4/2021
Date last modified: 8/4/2021
"""

from os.path import expanduser, join, isdir
from os import mkdir
from sys import argv
from itertools import compress
import datetime
import argparse
#import warnings
#warnings.filterwarnings("error")

import wfdb
import numpy as np
import heartpy as hp
import h5py

# helper function to find minima between two macima
def find_minima(sig, pks, fs):
    min_pks = []
    for i in range(0,len(pks)):
        pks_curr = pks[i]
        if i == len(pks)-1:
            pks_next = len(sig)
        else:
            pks_next = pks[i+1]

        sig_win = sig[pks_curr:pks_next]
        if len(sig_win) < 1.5*fs:
            min_pks.append(np.argmin(sig_win) + pks_curr)

    return min_pks

def download_mimic_iii_records(RecordsFile, OutputPath):

    # load record names from text file
    with open(RecordsFile, 'r') as f:
        RecordFiles = f.read()
        RecordFiles = RecordFiles.split("\n")
        RecordFiles = RecordFiles[:-1]
    
    for file in RecordFiles:
        print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: Processing record {file}')

        # download record
        record = wfdb.rdrecord(file.split('/')[1], pn_dir='mimic3wdb/' + file.split('_')[0])

        # check, if ABP and PLETH are present in the record. If not, continue with next record
        if 'PLETH' in record.sig_name:
            pleth_idx = record.sig_name.index('PLETH')
            ppg = record.p_signal[:,pleth_idx]
            fs = record.fs
        else:
            continue
    
        if 'ABP' in record.sig_name:
            abp_idx = record.sig_name.index('ABP')
            abp = record.p_signal[:,abp_idx]
        else:
            continue

        # detect systolic and diastolic peaks using heartpy
        try:
            abp_FidPoints = hp.process(abp, fs)
        except hp.exceptions.BadSignalWarning:
            continue

        ValidPks = abp_FidPoints[0]['binary_peaklist']
        abp_sys_pks = abp_FidPoints[0]['peaklist']
        abp_sys_pks = list(compress(abp_sys_pks, ValidPks == 1))
        abp_dia_pks = find_minima(abp, abp_sys_pks, fs)

        try:
            ppg_FidPoints = hp.process(ppg, fs)
        except hp.exceptions.BadSignalWarning:
            continue

        ValidPks = ppg_FidPoints[0]['binary_peaklist']
        ppg_pks = ppg_FidPoints[0]['peaklist']
        ppg_pks = list(compress(ppg_pks, ValidPks == 1))
        ppg_onset_pks = find_minima(ppg, ppg_pks, fs)

        # save ABP and PPG signals as well as detected peaks in a .h5 file
        SubjectName = file.split('/')[1]
        SubjectName = SubjectName.split('_')[0]
        SubjectFolder = join(join(OutputPath, SubjectName))
        if not isdir(SubjectFolder):
            mkdir(SubjectFolder)
    
        with h5py.File(join(SubjectFolder, file.split('/')[1] + ".h5"),'w') as f:
            signals = np.concatenate((abp[:,np.newaxis],ppg[:,np.newaxis]), axis=1)
            f.create_dataset('val', signals.shape, data=signals)
            f.create_dataset('nB2', (1,len(ppg_onset_pks)), data=ppg_onset_pks)
            f.create_dataset('nA2', (1,len(ppg_pks)), data=ppg_pks)
            f.create_dataset('nB3', (1,len(abp_dia_pks)), data=abp_dia_pks)
            f.create_dataset('nA3', (1,len(abp_sys_pks)), data=abp_sys_pks)
    
    print('script finished')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str,
                        help='File containing the names of the records downloaded from the MIMIC-III DB')
    parser.add_argument('output', type=str, help='Folder for storing downloaded MIMIC-III records')

    args = parser.parse_args()

    RecordsFile = args.input
    OutputPath = args.output

    download_mimic_iii_records(RecordsFile, OutputPath)
    