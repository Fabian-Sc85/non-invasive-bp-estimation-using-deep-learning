# Assessment of non-invasive blood pressure prediction from PPG and rPPG signals using deep learning

## Introduction
The code contained in this repository is intended to reproduce the results of the paper "Assessment of non-invasive blood pressure prediction from PPG and rPPG signals using deep learning" [Link to Paper](). Contained herein are scripts for downloading data from the MIMC-II database, data preprocessing as well as  training neural networks for (r)PPG based blood pressure prediction.

Trainings are performed using Tensorflow 2.4.1 and Python 3.8. The scripts can be executed from the command line. 

## Installation
To create a virtual environment using Python 3.8 as interpreter the `virtualenv` package is required. It can be installed using the command
```
pip install virtualenv
```
The virtual environment can then be created using the command
```
virtualenv --python=/usr/bin/python3.8 venv/
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
### Overview
To reproduce the paper's results, the scripts described below have to be executed in s specific order. The following table summarizes the purpose of each script.

|   |Script                             | Description                                                   |
|---|-----------------------------------|---------------------------------------------------------------|
|1  |`download_mimic_iii_records.py`    |Downloads data from the MIMIC-III database
|2  |`prepare_MIMIC_dataset.py`         |This script is used for:<ul><li>Preprocessing</li><li>dividing signals into windows</li><li>extracting ground truth SBP and DBP from signal windows</li><li>Storing singal/BP-value pairs in hdf5 format</li></ul>|
|3  |`prepare_MIMIC_dataset.py`         | divides the data into training, validation and test set and converts the data to the .tfrecord format which will be used during training|
|4  |`ppg_train_mimic_iii.py`           | trains neural networks for BP prediction using PPG data; saves the trained model for later fine tuning and personalization using (r)PPG data|
|5  |`ppg_personalization_mimic_iii.py` | Uses a pretrained neural network and fine tunes its final layers using partial data from test subjects|


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
### Creating tfrecord datasets for training
To train neural networks, the dataset created by the script `prepare_MIMIC_dataset.py` must be divided into training, validation and test set. The script `h5_to_tfrecord.py` does this by dividing the dataset based on (a) a subject based split or (b) by assigning samples randomly depending on the user's choice. The data will be stored separately for training, validation and testset in .tfrecord files which will be used during training.  
```
usage: h5_to_tfrecord.py [-h] [--ntrain NTRAIN] [--nval NVAL] [--ntest NTEST] [--divbysubj DIVBYSUBJ] input output

positional arguments:
  input                 Path to the .h5 file containing the dataset
  output                Target folder for the .tfrecord files

optional arguments:
  -h, --help            show this help message and exit
  --ntrain NTRAIN       Number of samples in the training set (default: 1e6)
  --nval NVAL           Number of samples in the validation set (default: 2.5e5)
  --ntest NTEST         Number of samples in the test set (default: 2.5e5)
  --divbysubj DIVBYSUBJ
                        Perform subject based (1) or sample based (0) division of the dataset
```
### Training neural networks using PPG signals
The script `ppg_train_mimic_iii.py` trains neural networks using tfrecord data created by script `h5_to_tfrecord.py`. Available neural architectures include AlexNet [[1]](#1), ResNet [[2]](#2), an architecture published by Slapnicar et al. [[3]](#3) and an LSTM network. The networks are trained using an early stopping strategy. The network weights that achieved the lowest validation loss are subsequently used to estimate BP values on the test set. Test results are stored in a .csv file for later analysis. Model checkpoints are also stored for later fine tuning and personalization.
```
usage: ppg_training_mimic_iii.py [-h] [--arch ARCH] [--lr LR] [--batch_size BATCH_SIZE] [--winlen WINLEN] [--epochs EPOCHS]
                                 [--gpuid GPUID]
                                 ExpName datadir resultsdir chkptdir

positional arguments:
  ExpName               unique name for the training
  datadir               folder containing the train, val and test subfolders containing tfrecord files
  resultsdir            Directory in which results are stored
  chkptdir              directory used for storing model checkpoints

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH           neural architecture used for training (alexnet (default), resnet, slapnicar, lstm)
  --lr LR               initial learning rate (default: 0.003)
  --batch_size BATCH_SIZE
                        batch size used for training (default: 32)
  --winlen WINLEN       length of the ppg windows in samples (default: 875)
  --epochs EPOCHS       maximum number of epochs for training (default: 60)
  --gpuid GPUID         GPU-ID used for training in a multi-GPU environment (default: None)
```
### Personalizing pretrained neural networks using PPG data
The script `ppg_personalization_mimic_iii.py` takes a set of test subjects and fine tunes neural network that were trained based on PPG data. The goal is to improve the MAE on those test subjects by using 20 % of each test subject's data for retraining. These 20 % can be dranwn randomly or systematically (the first 20 %). The remaining 80 % are used for validation. The script performs BP predictions using the validation data before and after personalization for comparison. Results are stored in a .csv file for later analysis. 
```
usage: ppg_personalization_mimic_iii.py [-h] [--lr LR] [--batch_size BATCH_SIZE] [--winlen WINLEN] [--epochs EPOCHS]
                                        [--nsubj NSUBJ] [--randompick RANDOMPICK]
                                        ExpName DataDir ResultsDir ModelPath chkptdir

positional arguments:
  ExpName               Name of the training preceeded by the repsective date in the format MM-DD-YYYY
  DataDir               folder containing the train, val and test subfolders containing tfrecord files
  ResultsDir            Directory in which results are stored
  ModelPath             Path where the model file used for personalization is located
  chkptdir              directory used for storing model checkpoints

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               initial learning rate (default: 0.003)
  --batch_size BATCH_SIZE
                        batch size used for training (default: 32)
  --winlen WINLEN       length of the ppg windows in samples (default: 875)
  --epochs EPOCHS       maximum number of epochs for training (default: 60)
  --nsubj NSUBJ         Number subjects used for personalization (default :20)
  --randompick RANDOMPICK
                        define wether data for personalization is drawn randomly (1) or comprises the first 20 % of the test
                        subject's data (0) (default: 0)

```

## References
<a id="1">[1]</a> A. Krizhevsky, I. Sutskever, und G. E. Hinton, „ImageNet classification with deep convolutional neural networks“,
    Commun. ACM, Bd. 60, Nr. 6, S. 84–90, Mai 2017, doi: 10.1145/3065386.

<a id="1">[2]</a> K. He, X. Zhang, S. Ren, und J. Sun, „Deep Residual Learning for Image Recognition“, in 2016 IEEE Conference on
    Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, Juni 2016, S. 770–778. doi: 10.1109/CVPR.2016.90.

<a id="1">[3]</a> G. Slapničar, N. Mlakar, und M. Luštrek, „Blood Pressure Estimation from Photoplethysmogram Using a Spectro-Temporal
    Deep Neural Network“, Sensors, Bd. 19, Nr. 15, S. 3420, Aug. 2019, doi: 10.3390/s19153420.