# Implementation of deep learning framework -- Unet, using Keras

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

---

## Overview

### Directory layout

    .
    ├── DataProcessingPipeline_JupyterNotebook	# Python notebook to convert SAR data from .tiff into arrays
        ├── CreateArraysForMulticlassImagesAndLabels.ipynb
    ├── BinaryClassificationSeavsShip		# Using CNN for Binary Classification task: Sea vs Ships
        ├── CNNweights.hdf5				# pre-trained CNN weights for sea-vs-ship classifer
        ├── data_binaryClassification.py		# data processing and augmentation
        ├── trainCNN_binaryClassification.py		# contains CNN model for training
        ├── test_binaryClassification.py		# test the performance of trained CNN model
    ├── MulticlassClassificationLandvsSeavsShip	# Using CNN for Multiclass Classification task: Land vs Sea vs Ships
        ├── MulticlassClassifier
            ├── CNNweights.hdf5				# pre-trained CNN weights for multiclass classifier
            ├── data_multiclassClassifier.py		# data processing and augmentation
            ├── trainCNN_multiclassClassifier.py	# contains CNN model for training
            ├── test_predict_multiclassClassifier.py	# test the performance of trained CNN model
        ├── OnevsAllClassifier
            ├── 0	# land class
                ├── CNNweights.hdf5 			# pre-trained CNN weights for land classifier
            ├── 1	# sea class
                ├── CNNweights.hdf5 			# pre-trained CNN weights for sea classifier
            ├── 2	# ship class
                ├── CNNweights.hdf5 			# pre-trained CNN weights for ships classifier
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

The images are 3-D volume tiff, you should transfer the stacks into arrays first by running the [python Notebook](DataProcessingPipeline_JupyterNotebook/CreateArraysForMulticlassImagesAndLabels.ipynb).

The data for training contains 34 128*128 images, which are far not enough to feed a deep learning neural network.
To do data augumentation, an image deformation method was used, which was implemented in python using Keras.

### Model

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is a 128*128 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.

