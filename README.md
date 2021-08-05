# Assessment of non-invasive blood pressure prediction from PPG and rPPG signals using deep learning

## Introduction
The code contained in this repository is intended to reproduce the results of the paper "Assessment of non-invasive blood pressure prediction from PPG and rPPG signals using deep learning" [Link to Paper](). Contained herein are scripts for downloading data from the MIMC-II database, data preprocessing as well as  training neural networks for (r)PPG based blood pressure prediction.

Trainings are performed using Tensorflow 2.4.1 and Python 3.9. The scripts can be called from the command line. 

## Installation
This repository uses a python virtual environment which can be created using the command
```
python3 -m venv venv/
```
The virtual environment can be activated using the command
```angular2html
source venv/bin/activate
```
Necessary python packages can be installed using the command
```
pip install -r requirements.txt
```
## Using the scripts
### Downloading data from the MIMIC-III database
The script `download_mimic_iii_records.py` can be used to download the records used for PPG based training. The specific record names are provided in the file `MIMIC-III_ppg_dataset_records.txt`. The script can be called from the command line using the command
```
python3 download_mimic_iii_records.py [-h] input output

positional arguments:
  input       File containing the names of the records downloaded from the MIMIC-III DB
  output      Folder for storing downloaded MIMIC-III records
```
The Scripts runs a very long time and the required disc space for all records is appr. 1.5 TB

### Preparing the PPG dataset
The Script `prepare_MIMIC_dataset.py` preprocesses the data downloaded by `download_mimic_iii_records.py`. PPG and ABP signals are extracted from each record and divided into windows of a defined length and overlap. Several preprocessing steps include filtering the PPG signal. SBP/DBP values are extracted from the ABP signal using peak detection. Various heuristics exclude unsuitable BP values and their corresponding PPG signal from the dataset. Those include: check if
* SBP and DBP are within a plausible range
* ABP and PPG signals contain no missing values
* HR calculated based on ABP/PPG is within a plausible range

The maximum number os samples per subject and for the whole dataset can be defined. The dataset is saved to a .h5 file for further processing.

```
usage: prepare_MIMIC_dataset.py [-h] [--win_len WIN_LEN] [--win_overlap WIN_OVERLAP] [--maxsampsubject MAXSAMPSUBJECT]
                                [--maxsamp MAXSAMP] [--save_bp_data SAVE_BP_DATA]
                                datapath output

positional arguments:
  datapath              Path containing data records downloaded from the MIMIC-III database
  output                Target .h5 file

optional arguments:
  -h, --help            show this help message and exit
  --win_len WIN_LEN     PPG window length in seconds
  --win_overlap WIN_OVERLAP
                        ammount of overlap between adjacend windows in fractions of the window length (0...1)
  --maxsampsubject MAXSAMPSUBJECT
                        Maximum number of samples per subject
  --maxsamp MAXSAMP     Maximum total number os samples in the dataset
  --save_bp_data SAVE_BP_DATA
                        0: save BP data only; 1: save PPG and BP data
```