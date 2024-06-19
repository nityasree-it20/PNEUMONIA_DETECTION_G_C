

# PNEUMONIA DETECTION USING DL


## Overview

This project aims to detect pneumonia from chest X-ray images using deep learning models (VGG16 and VGG19). Additionally, a web interface is created using Streamlit to provide an easy-to-use platform for users to upload and analyze images for pneumonia detection.
## Table of Contents

    $ Introduction
    $ Models
    $ Dataset
    $ Installation
    $ Streamlit Web App
    $ Results
    $ Reference

## Introduction

### VGG16
VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford. The model achieves 90.7% top-5 test accuracy in ImageNet, a dataset of over 14 million images belonging to 1000 classes.

### VGG19
VGG19 is an extended version of VGG16 with 19 weight layers. It is also a very effective model for image classification tasks.

### Dataset
The dataset used for training and testing the models consists of labeled chest X-ray images. The images are classified into two categories:

    * Pneumonia
    * Normal
## Installation Steps

Clone the repository:

    git clone https://github.com/yourusername/pneumonia-detection.git
    cd pneumonia-detection

Install the required packages:

    pip install -r requirements.txt


## Streamlit Web App

To run the Streamlit web application:

    streamlit run app.py

## Results

The performance of the models can be evaluated using metrics such as accuracy, precision, recall, and F1-score. Below are the results for the trained models:

### EVALUATION METRICS FOR PNEUMONIA DATASET

    Epoch	Accuracy (%)	 Loss	 Val_loss	Val_Accuracy
    5	    88.09	        0.1539	 0.9102	    0.7500
    10	    88.89	        0.1678	 0.8932	    0.7850
    15	    89.01	        0.1452	 0.8925	    0.7536
    20	    88.98	        0.0984	 0.9029	    0.7475

### EVALUATION METRICS FOR NORMAL DATASET

    Epoch 	Accuracy (%)	Loss	Val_loss	Val_Accuracy
    5	    89.87	       0.1369	1.3655	    0.6250
    10	    89.91	       0.1345	0.2690	    0.8125
    15	    90.01	       0.0958	0.6528	    0.6875
    20	    00.95	       0.0950	0.4419	    0.8125
    



## Reference

    Pneumonia Detection and Medical Imaging RSNA Pneumonia Detection Challenge: 
        A Kaggle competition focused on detecting pneumonia from chest X-ray images. [Kaggle Link]()



