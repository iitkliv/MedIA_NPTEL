# Medical Image Analysis on NPTEL 
This repository contains the tutorials for the Medical Image Analysis NPTEL MOOC

### Dependencies
1. [MATLAB](https://in.mathworks.com/products/matlab.html) (Install Image Processing Toolbox, Statistics and Machine Learning Toolbox
2. PyTorch - Go to the PyTorch [website](http://pytorch.org/), choose the OS, package manager, language (Python) and CUDA version(if NVIDIA GPU is available and CUDA is installed) and run the given command to install PyTorch.

### Organisation
#### Tutorial 1: Lecture 12 - Random Forests for Segmentation and Classification  
   Steps to run the tutorial:  
   1. Download the [DRIVE](https://drive.grand-challenge.org/) dataset.
   2. Extract the dataset into a folder.
   3. In the same folder download and the file from /MedIA_NPTEL/Random_Forests_Demo/ folder.
   4. Run retinasegRF.m to create a random forst for segmenting retinal blood vessels.
#### Tutorial 2: Lecture 15 - Deep Learning for Medical Image Analysis (Contd.)
   We have provide an updated version of the tutorial which uses PyTorch (based on Python) instead of Torch7 (based on Lua).  
   Steps to run the tutorial:  
   1. Download the [ALL-IDB_2](https://homes.di.unimi.it/scotti/all/) dataset.
   2. Extarct the dataset into a folder.
   3. In the same folder download and place all the 3 files from /MedIA_NPTEL/Neural_Networks_Demo/ folder.
   4. Run the dataGenForTorch.m for generating the train,validation and test data matrices.
   5. Run the nn_demo.m for training the neural network on matlab and view the results in a GUI interface.
   6. Run the jupyter notebook nptel_media_nn.ipynb for training a neural network using PyTorch framework.

