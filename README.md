# README
 
This project builds a length-versatile and noise-robust LoRa radio frequency fingerprint identification (RFFI) system. The LoRa signals are collected from 10 commercial-off-the-shelf LoRa devices, with the spreading factor (SF) set to 7, 8, 9, respectively. The packet preamble part and device labels are provided.

## Citation 

If the part of the dataset/codes contributes to your project, please cite:

```
[1] G. Shen, J. Zhang, A. Marshall, M. Valkama, and J. Cavallaro.   “Towards Length-Versatile and Noise-Robust Radio Frequency Fingerprint Identification,” IEEE Trans. Inf. Forensics Security, 2023.

```

```
@article{shen2023length,
  title={Towards Length-Versatile and Noise-Robust Radio Frequency Fingerprint Identification},
  author={Shen, Guanxiong and Zhang, Junqing and Marshall, Alan and Valkama, Mikko and Cavallaro, Joseph},
  journal={IEEE Trans. Inf. Forensics Security},
  year={2023}
}
```

## Dataset Information

### Experimental Devices

**Transmitters:** device 31-35 LoPy4, device 36-40 Dragino LoRa shield.

**Receiver:** USRP N210 software-defined radio (SDR).

### Datasets

|  Name  | Number of Packets Per Device |  Spreading Factor  |
|  --- | ---  | ---  |
| sf_7_train.h5  | 2,500 | 7 |
| sf_8_train.h5 | 2,500 | 8  |
| sf_9_train.h5 | 2,500 | 9  |
| sf_7_test.h5 | 500 | 7  |
| sf_8_test.h5 | 500 | 8  |
| sf_9_test.h5 | 500 | 9  |


## Quick Start

### 1. Requirements

**a) Install Required Packages**

Please find the 'requirement.txt' file to install the required packages.

**b) Download Dataset**

Please downlaod the dataset and put it in the project folder. The download link is https://ieee-dataport.org/documents/lorarffidatasetdifferentspreadingfactors.

**c) Operating System**

This project is built entirely on the Windows operating system. There may be unexpected issues on other operating systems.


### 2. Train a Model

The function 'train()' can train a length-versatile neural network, i.e., LSTM, GRU, Transformer or 'Flatten-free CNN'. Please change the variable 'model_type' to specify the type of the trained neural network.

### 3. Inference

The function 'inference()' can evaluate the trained neural network. It returns the overall accuracy and a confusion matrix. Please change the variable 'snr_awgn' to specify the range of artificial noise added to the test data. 


## License

The dataset and code is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

## Contact

Please contact the following email addresses if you have any questions:

Guanxiong.Shen AT liverpool.ac.uk

Junqing.Zhang AT liverpool.ac.uk