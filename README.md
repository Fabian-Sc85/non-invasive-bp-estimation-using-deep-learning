# Assessment of non-invasive blood pressure prediction from PPG and rPPG signals using deep learning

## Introduction
The code contained in this repository is intended to reproduce the results of the paper "Assessment of non-invasive blood pressure prediction from PPG and rPPG signals using deep learning" which can be accessed via the [Sensors Special Issue "Contactless Sensors for Healthcare](https://www.mdpi.com/1424-8220/21/18/6022) [[1]](#1). Contained herein are scripts for downloading data from the MIMC-II database, data preprocessing as well as  training neural networks for (r)PPG based blood pressure prediction.

Trainings are performed using Tensorflow 2.4.1 and Python 3.8. The scripts can be executed from the command line.

If you find this repository useful for your own research, please consider citing our paper:

```
@inproceedings{schrumpf2021assessment,
  title={Assessment of deep learning based blood pressure prediction from PPG and rPPG signals},
  author={Schrumpf, Fabian and Frenzel, Patrick and Aust, Christoph and Osterhoff, Georg and Fuchs, Mirco},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3820--3830},
  year={2021}
}
```

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
This repository contains scripts to reproduce the [paper's](https://www.mdpi.com/1424-8220/21/18/6022) result regarding the BP prediction based on the MIMIC-B dataset as well as the camera based BP prediction. Analyses based on the MIMIC-A dataset are not covered by this repository.

To reproduce the paper's results, the scripts described below have to be executed in s specific order. The following table summarizes the purpose of each script.

|   |Script                             | Description                                                   |
|---|-----------------------------------|---------------------------------------------------------------|
|1  |`download_mimic_iii_records.py`    |Downloads data from the MIMIC-III database
|2  |`prepare_MIMIC_dataset.py`         |This script is used for:<ul><li>Preprocessing</li><li>dividing signals into windows</li><li>extracting ground truth SBP and DBP from signal windows</li><li>Storing singal/BP-value pairs in hdf5 format</li></ul> Alternatively, the dataset can be downloaded from [Zenodo](https://zenodo.org/record/5590603) (32 GB)|
|3  |`h5_to_tfrecord.py`         | divides the data into training, validation and test set and converts the data to the .tfrecord format which will be used during training|
|4  |`ppg_train_mimic_iii.py`           | trains neural networks for BP prediction using PPG data; saves the trained model for later fine tuning and personalization using (r)PPG data|
|5  |`ppg_personalization_mimic_iii.py` | Uses a pretrained neural network and fine tunes its final layers using PPG data from subjects from the test set of the MIMIC-III database|
|6  |`retrain_rppg_personalization.py`  | Uses a pretrained nueral network and fine tunes it using rPPG data. |

### Datasets and trained models

The PPG dataset used for training the neural architectures and the trained models themselves can be found at [Zenodo](https://zenodo.org/record/5590603).

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

The maximum number of samples per subject and for the whole dataset can be defined. The dataset is saved to a .h5 file for further processing.

Alternatively, the dataset can be downloaded from [Zenodo](https://zenodo.org/record/5590603) (32 GB)

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
  --save_ppg_data SAVE_PPG_DATA
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
The script `ppg_train_mimic_iii.py` trains neural networks using tfrecord data created by script `h5_to_tfrecord.py`. Available neural architectures include AlexNet [[2]](#2), ResNet [[3]](#3), an architecture published by Slapnicar et al. [[4]](#4) and an LSTM network. The networks are trained using an early stopping strategy. The network weights that achieved the lowest validation loss are subsequently used to estimate BP values on the test set. Test results are stored in a .csv file for later analysis. Model checkpoints are also stored for later fine tuning and personalization. The trained models used in the paper can be found at [Zenodo](https://zenodo.org/record/5590603).
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
### rPPG based BP prediction using transfer learning

The script `retrain_rppg_personalization.py` trains a pretrained neural network (trained using the script `pg_train_mimic_iii.py`) for camera based BP prediction. The rPPG data is provided by a hdf5 file in the data subfolder. The rPPG data was collected during a study at the Leipzig University Hospital. Subjects were filmed using a standard RGB camera. rPPG signals were derived from skin regions on the subject's face using the plane-orthogonal-to-skin algorithm published by Wang et al. [[5]](#5).

If you use this data in you own research, please cite our paper:

```
@inproceedings{schrumpf2021assessment,
  title={Assessment of deep learning based blood pressure prediction from PPG and rPPG signals},
  author={Schrumpf, Fabian and Frenzel, Patrick and Aust, Christoph and Osterhoff, Georg and Fuchs, Mirco},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3820--3830},
  year={2021}
}
```

The pretrained networks are finetuned using a leave-one-subject-out cross validation scheme. Personalization can be performed by using a portion of the test subject's data for training. The networks are evaluated using the test subject's data BEFORE and AFTER fine tuning. Results are stored in a csv file for analysis.
```
usage: retrain_rppg_personalization.py [-h] [--pers PERS] [--randompick RANDOMPICK] ExpName DataFile ResultsDir ModelPath chkptdir

positional arguments:
  ExpName               Name of the training preceeded by the repsective date in the format MM-DD-YYYY
  DataFile              Path to the hdf file containing rPPG signals
  ResultsDir            Directory in which results are stored
  ModelPath             Path where the model file used for rPPG based personalization is located
  chkptdir              directory used for storing model checkpoints

optional arguments:
  -h, --help            show this help message and exit
  --pers PERS           If 0, performs personalizatin using data from the test subjct
  --randompick RANDOMPICK
                        If 0, uses the first 20 % of the test subject's data for testing, otherwise select randomly (only applies if --pers == 1)

```
The 


## Using the pretrained models
The subfolder `trained_models` contains .h5-files containing models definitions and weights. The models wer trained using a non-mixed dataset as described in [[1]](#1). To use the networks for prediction/fine-tuning, input and output data must meet the following requirements:
* input data must have a length of 875 samples (corresponds to 7 seconds using a sampling frequency of 125 Hz)
* SBP and DBP must be provided separately as there is one output node for each value

The models can be imported the following way:
```python
import tensorflow.keras as ks
from kapre import STFT, Magnitude, MagnitudeToDecibel

dependencies = {
        'ReLU': ks.layers.ReLU,
        'STFT': STFT,
        'Magnitude': Magnitude,
        'MagnitudeToDecibel': MagnitudeToDecibel

model = ks.load_model(<PathToModelFile>, custom_objects=dependencies)
```
Predictions can then be made using the `model.predict()` function. 

## References
<a id="1">[1]</a> Schrumpf, F.; Frenzel, P.; Aust, C.; Osterhoff, G.; Fuchs, M. Assessment of Non-Invasive Blood Pressure Prediction from PPG and rPPG Signals Using Deep Learning. Sensors 2021, 21, 6022. https://doi.org/10.3390/s21186022 

<a id="2">[2]</a> A. Krizhevsky, I. Sutskever, und G. E. Hinton, „ImageNet classification with deep convolutional neural networks“,
    Commun. ACM, Bd. 60, Nr. 6, S. 84–90, Mai 2017, doi: 10.1145/3065386.

<a id="3">[3]</a> K. He, X. Zhang, S. Ren, und J. Sun, „Deep Residual Learning for Image Recognition“, in 2016 IEEE Conference on
    Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, Juni 2016, S. 770–778. doi: 10.1109/CVPR.2016.90.

<a id="4">[4]</a> G. Slapničar, N. Mlakar, und M. Luštrek, „Blood Pressure Estimation from Photoplethysmogram Using a Spectro-Temporal
    Deep Neural Network“, Sensors, Bd. 19, Nr. 15, S. 3420, Aug. 2019, doi: 10.3390/s19153420.

<a id="5">[5]</a> W. Wang, A. C. den Brinker, S. Stuijk, und G. de Haan, „Algorithmic Principles of Remote PPG“, IEEE Transactions on Biomedical Engineering, Bd. 64, Nr. 7, S. 1479–1491, Juli 2017, doi: 10.1109/TBME.2016.2609282.
