# DTP-Net: Learning to Reconstruct EEG signals in Time-Frequency Domain by Multi-scale Feature Reuse

## Introduction
A deep learning model for single-channel EEG artifact removal.

Code for the model in the paper DTP-Net: Learning to Reconstruct EEG signals in Time-Frequency Domain by Multi-scale Feature Reuse.

The architecture of DTP-Net:

![DTP-Net Architecture](https://github.com/WilliamRo/EEG-Denoise/blob/main/figure/overview_architecture.png)

## Environment
The following setup has been used to reproduce this work:

* CUDA toolkit 10.1 and CuDNN 7.6.0
* Python == 3.7
* Tensorflow == 2.1.0
* Matplotlib == 3.5.3
* Numpy == 1.19.5
* Scipy == 1.4.1

## Prepare Dataset
We evaluate our DTP-Net with EEGDenoiseNet, Semi-Simulated EEG/EOG and MNE M/EEG dataset.

For EEGDenoiseNet, the dataset is publicly availabel as referenced in the article [EEGdenoiseNet: A benchmark dataset for end-to-end deep learning solutions of EEG denoising](https://github.com/ncclabsustech/EEGdenoiseNet).

For Semi-Simulated EEG/EOG, the dataset is publicly availabel as referenced in the article [Adaptive Single-Channel EEG Artifact Removal With Applications to Clinical Monitoring](https://github.com/holcman-lab/wavelet-quantile-normalization).

For MNE M/EEG dataset, the dataset is using the sample dataset provided in the article [EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces](https://github.com/vlawhern/arl-eegmodels)

For the sake of convenience, EEGDenoiseNet dataset and Semi-Simulated EEG/EOG dataset are temporarily placed in the [G-node database](https://gin.g-node.org/SummerBae/EEG-Denoise_Database) and you can run the following scripts to download datasets.
```
gin get SummerBae/EEG-Denoise_Database
```

MNE M/EEG dataset would be download automatically when running ```03-MNE_MEG_EEG_Dataset/erp.py```.

## Model Deployment I
Run this script to train a DTP-Net for model deployment I:
```
cd 01-EEGDenoiseNet_Dataset
python t1_dtpnet.py --noise_type EOG
```
```noise_type``` denotes different artifacts source, including ```EMG```, ```EOG``` and ```EMG_EOG```.

The pre-trained models are saved in ```01-EEGDenoiseNet_Dataset/checkpoints```

Run this script to evaluate the denoise performance at different SNR levels of pre-trained models:
```
cd 01-EEGDenoiseNet_Dataset
python evaluation.py --noise_type EOG --plot_type metric_report
```
![Denoise result at different SNR levels after EOG artifacts removal](https://github.com/WilliamRo/EEG-Denoise/blob/main/figure/experiment_1.png)

Run this script to get waveform results and PSD results for eliminating artifacts of pre-trained models:
```
cd 01-EEGDenoiseNet_Dataset
python evaluation.py --noise_type EOG --plot_type denoise_visualize
```
![Denoise waveform results and PSD results after EOG artifacts removal](https://github.com/WilliamRo/EEG-Denoise/blob/main/figure/experiment_1_1.png)

## Model Deployment II
Run this script to train a DTP-Net for model deployment II:
```
cd 02-Semi_Simulated_Dataset
python t2_dtpnet.py
```

The pre-trained models are saved in ```02-Semi_Simulated_Dataset/checkpoints```

Run this script to evaluate the denoise performance of pre-trained models:
```
cd 02-Semi_Simulated_Dataset
python evaluation.py --plot_type metric_report
```

Run this script to get waveform results and PSD results for eliminating artifacts of pre-trained models:
```
cd 02-Semi_Simulated_Dataset
python evaluation.py --plot_type denoise_visualize
```
![Denoise waveform results and PSD results after artifacts removal](https://github.com/WilliamRo/EEG-Denoise/blob/main/figure/experiment_2_1.png)


Run this script to get multi-channel waveform results for eliminating artifacts of pre-trained models:
```
cd 02-Semi_Simulated_Dataset
python evaluation.py --plot_type denoise_visualize_multi_channel 
```
![Multi_Channel denoise waveform results and PSD results after artifacts removal](https://github.com/WilliamRo/EEG-Denoise/blob/main/figure/experiment_2_2.png)

## Model Deployment III
Run this script to get the ERP classification performance of raw data:
```
cd 03-MNE_MEG_EEG_Dataset
python erp.py
```
Run this script to get the ERP classification performance of data denoised by DTP-Net:
```
cd 03-MNE_MEG_EEG_Dataset
python erp.py --denoise
```
## Licence
For academic and non-commercial use only.