# TCH_Arrhythmias_F20

**<h1>Summary</h1>**

Our data comes in the form of h5 files containing data from ECG, CVP, ABP, and PPG waveforms. The goal of our project is to predict Arrhythmias in post-operative heart patients based on information from these waveforms.

**<h2>Denoising and Preprocessing</h2>**

Currently we have scripts that perform denoising and preprocessing tasks on our data:

**ECG_preprocessing.py:**

This file is responsible for preprocessing ECG signals in the raw data. It contains:

    Schmidt spike removal algorithm
    Butterworth filter
    Stationary wavelet transform
    Pan Tompkins Detector
    Segmentation

**PPG_preprocessing.py:**

This file is responsible for preprocessing PPG data. It contains the following denoising/preprocessing processes:

    Wavelet Transform
    Butterworth filter
    Notch filter 
    Linear detrending
    Wiener filter

**<h2>Feature Extraction</h2>**
    
**ECG_feature_extraction.py:**

This file is responsible for extracting features from preprocessed ECG signals during labelled events. It extracts the following features:

    Length of previous RR interval
    Length of next RR interval
    Average of past 10 RR intervals
    Wavelet coefficients
    Wavelet descriptor

**PPG_feature_extraction.py:**

This file is responsible for extracting features from preprocessed PPG signals during labelled events. It extracts the following features:

    Location of peaks in the signal
    Location of troughs in the signal
    Width of the waveform at half of waveform height
    Width of the waveform at two-thirds of waveform height

**<h2>Data Preperation</h2>**

**data_loading.py:**

This file is the one stop shop for denoising and extracting desired features from the data. It integrates the whole pipeline prior to modeling, and works through whichever patient h5py files you have locally available. It will output all labeled beats (labels are given in Labeled_Events.xlsx file - summary sheet) segmented into whichever size window is specified and each beat will have its selected features as well as a health label associated with it.

**load_cwt_features.py:**

Similarly to the data_loading file, this file runs the denoising and feature extraction for the labeled ecg and ppg data. In contrast to the data_loading file, this script extracts only continuous wavelet features, which are resized as scalogram images to be fed into a CNN (this file is used in conjunction with CNN training).

**data_generator.py:**

This file generates the training/testing data which will be used for CNN modeling as well as any stochastic information used a precursor for CNN training. 

**<h2>Modeling</h2>**

**svm_modeling.py:**

This file is responsible for conducting SVM modeling using the chosen features. The data is read it from a csv file which it was extracted too previously and can be split randomly or along patient lines for training/testing. Data is also cleaned if necesary (replacing any NaN values using imputation procedures). Results are reports in terms of accuracy, and are saved to csv files. Results can be visualized as well through PCA on the feature space.

**CNN_models.py:**

This file is responsible for creating our Convolutional Neural  Namely, our twoLayerCNN method is called by our training file. There are also methods to create a ResNet or VGG.

**train.py**

This file can be ran to create and train a CNN model using the images produced by the load_cwt_features.py file. Weights from the model will be saved into hdf5 files and accuracy/loss results will be saved in log files. In order to run the file you must set the path to the images.

**<h2>Validation</h2>**

**calculate_metrics.py:**

This file loads in the weights produced from a model created by train.py to calculate the false positive rate, total accuracy, and percentage of arrhythmias detected by the model. It calls the data_generator.py file to determine these metrics, and access to the weights is necessary to load in the model.

**cwt_plot.py:**

Visualize the resized CWT scalogram, which was fed to the model.

**beat_display.py:**

Display the beats according to classifcations. This can only be done on timestamped beats, which means it is only useful if the features were extracted in this way (which is not always the case).

**<h2>Model Specific Folders</h2>**

**svm_one_beat folder:**

Model setup and results from SVM 1 beat random split and leave-one-out training. This folder contains everything needed to run SVM modelling, and has instructions within it on how exactly this is done.

**CNN leave-one-out folder**

Model setup and results from CNN leave-one-out training. The code is very similar to the main code.

**<h2>Miscellaneous & Deprecated</h2>**

**Labelled_Events.xlsx and Labelled_Events_Summary.csv:**

Labelled_Events.xlsx lists all labelled events within the data (labelled by physician), and summarizes these events in the summary sheet. Labelled_Events_Summary.csv essentially just containts these summary results in csv format. Only the summary sheet of the xlsx files is used within the pipeline.

**svm_3beat folder:**

All SVM code for running 3-beat SVM dynamic window sampling modelling. The 3-beat results are not included in the report, but the folder does include dyanmic window sampling functionality, which is otherwise not used in the main code.

**Res_net.py:**

Resnet for CNN training. Not in use as Keras internal functions are being used in active code.

**<h1>Installation</h1>**

Before using the module, be sure to install the requirements in requirements.txt by using the following command: pip install -r requirements.txt. 

**<h1>Full Data Science Pipeline & Reproducibility</h1>**

All of our computational results can be reproduced through the following pipeline:

1. First install the necesary packages/requirements according to the installation instructions given above. You have to download the folder containing each patients folder from Rice Box and the excel file labelled by doctors.

2. To run the neural network with random split training/validation, run load_cwt_features.py in the main folder with the correct path to the raw data. Then run train.py in the main folder with correct path to the loaded images to train the model. To check the accuracy, percentage of arrhythmias detected, and false positives run the calculate_metrics.py file with the path to the weights produced by the model. deep_learning_cross_validation.jpynb provides detailed instructions. 

    a. run load_cwt_features.py. To run this file, you need to provide the path of the folder which containes each patients' folder, the path of the excel file which contains the doctors label and save path of the folder for the "cwt images" (~10GB)
    
    b. run train.py to train deep learning model and the default network is a two CNN layers architecture. To run this file, you need to provide the path of folder which you save the "cwt images". 
    
3. To run the neural network with leave-one-out method, run load_cwt_features in the CNN_leave_one_out folder with the correct path to the raw data. Then run train_leave_one_out.py with correct path to the loaded images to train the model.
4. To run SVM model with one-beat feature, run data_loading.py and data_pre.py in SVM_one_beat folder, then run svm_model.py. If anyone want to run SVM with leave-one-out method, run svm_leave_one_out.py.


**<h1>Contribute</h1>**

Source Code: https://github.com/RiceD2KLab/TCH_Arrhythmias_F20

**<h1>Support</h1>**

If you are having issues, please let us know. We can be reached at rmm13@rice.edu, or cva1@rice.edu for support.

**<h1>License</h1>**

The project is licensed under the BSD 2-Clause License.
