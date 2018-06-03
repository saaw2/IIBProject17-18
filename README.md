# Implementation of deep learning framework -- Unet, using Keras

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

---

## Overview

### Directory layout

    .
    ├── DataProcessingPipeline_JupyterNotebook
        ├── CreateArraysForMulticlassImagesAndLabels.ipynb	# Python notebook to convert SAR data from tiff. into arrays
    ├── BinaryClassificationSeavsShip                   	# Using CNN for Binary Classification task: Sea vs Ships
        ├── CNNweights.hdf5
        ├── data_binaryClassification.py
        ├── trainCNN_binaryClassification.py
        ├── test_binaryClassification.py
    ├── MulticlassClassificationLandvsSeavsShip         	# Using CNN for Multiclass Classification task: Land vs Sea vs Ships
        ├── MulticlassClassifier
            ├── CNNweights.hdf5
            ├── data_multiclassClassifier.py
            ├── trainCNN_multiclassClassifier.py
            ├── test_predict_multiclassClassifier.py
        ├── OnevsAllClassifier
            ├── 0
                ├── CNNweights.hdf5
            ├── 1
                ├── CNNweights.hdf5
            ├── 2
                ├── CNNweights.hdf5
            ├── data_OnevsAll.py
            ├── trainCNN_OnevsAll.py
            ├── test_predict_OnevsAll.py
    ├── ProjectLogbook.md
    ├── ProjectReport.pdf
    └── README.md

### Data

[Provided data](https://scihub.copernicus.eu/) you can download the train and test data from this server.
The data used to obtain results presented in the report can be downloaded from [here](https://drive.google.com/drive/folders/1Hnb5jOfElWn-_n-lbNPs_TK8w3VOUu9L?usp=sharing).

### Pre-processing

The images are 3-D volume tiff, you should transfer the stacks into arrays first by running the python Notebook CreateArraysForMulticlassImagesAndLabels.ipynb.
The data for training contains 34 128*128 images, which are far not enough to feed a deep learning neural network.
To do data augumentation, an image deformation method was used, which was implemented in python using Keras.

### Model

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is a 128*128 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.

### Training

The model is trained for 10 epochs.

After 10 epochs, calculated accuracy is about 0.97.

Loss function for the training is basically just a binary crossentropy

