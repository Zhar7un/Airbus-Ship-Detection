# Airbus-Ship-Detection

Airbus Ship Detection is a project that builds a semantic segmentation of ships on satellite photos.

# Implementation description 

 Solution description step by step:

1. Decode segmentation masks from RLE format to PNG: `common/generate_masks.py`.
2. Split dataset into 3 parts: train(1%), validation(0.5%), test(0.5%). Numbers are selected according to available computing resources. Source: `src/split_data.py`.
3. Resize, normalize images and masks of the dataset and pack them to batches: `src/data_preprocessing.py` .
4. Build model architecture based on Unet architecture for neural network using tf.keras: `src/unet_model.py`.
5. Train model using Dice Loss cost function and Dice Score metric: `src/train_unet.py`.
6. Make model inference: `src/test_model.py`. TODO.

# Getting started

Before running code from this repo you need:
* Download dataset: https://www.kaggle.com/c/airbus-ship-detection/data (`train_v2.zip`, `train_ship_segmentations_v2.csv`).
* Put its content into `common/` directory.

To build the project run the following command from the project root: 
```
pip install -r requirements.txt
```

To extract images from archive `train_v2.zip` run following scrypt:
```
common/extract_images.py
```
To decode masks and save them as PNG run following scrypt:
```
common/generate_masks.py
```
To split dataset on train, validation, test run following scrypt:
```
src/split_data.py
```
To train model run following scrypt:
```
src/train_unet.py
```