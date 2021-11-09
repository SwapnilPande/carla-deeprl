# Code for Multitask_BEV_encoder
The code pretrains an encoder that encodes a stack of segmented BEV images.

## Envirionment
### Python
* python3.6.5

### packages
* numpy==1.19.5
* sklearn==0.0
* torch==1.9.1
* torchvision==0.10.1
* opencv-python==4.5.1.48

## Files
* main.py
	* The main file to run pretraining.
* dataset.py
	* The file defines the BEV_Dataset class that reads BEV images.
* encoder.py
	* The file defines the Encoder class for custom non-pretrained encoders.
* decoder.py
	* The file defines a Decoder class that takes a BEV embedding as input and outputs a tensor with the same spacial-temporal structure of the input.
* predictor.py
	* The file defines a Predictor class that takes a BEV embedding as input and outputs a 1D-vector that predicts various variables.
* train_fn.py
	* This file defines the function that trains the encoder.
* utils.py
	* The file defines some helper functions.

## Folder structure
A data folder named "BEV_data2" should be placed at the same level of the folder "code" to run the model.

## Arguments

## Usage

### Usage examples
