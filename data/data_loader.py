from typing import Tuple
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
from sklearn.model_selection import train_test_split
from os.path import dirname, join as pjoin
import scipy.io as sio
import skimage.io as io
import skimage.transform as trans
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras import Model
import random
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

# def get_loader(dataset_dir, ) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
#     """This function returns 3 generators
#     train, val, test.

#     """
#     pass


def unison_shuffle(a,b):
    inx=np.random.permutation(a.shape[0])
    return a[inx],b[inx]

def get_loader(train_input_dir, train_mask_dir, test_input_dir, test_mask_dir,
                rotation_range,
                width_shift_range,
                height_shift_range,
                shear_range,
                zoom_range,
                fill_mode,
                horizontal_flip):
    
    #should there be only one direction?
    #how are we supposed to get the other 4 directions of train test and images and masks for each

    train_input_img_paths = sorted(
        [
            os.path.join(train_input_dir, fname)
            for fname in os.listdir(train_input_dir)
            if fname.endswith(".jpg")
        ]
    )
    train_input_array_list = [cv2.resize(mpimg.imread(str_index), (256,256))
                  for str_index in train_input_img_paths]

    train_mask_img_paths = sorted(
        [
            os.path.join(train_mask_dir, fname)
            for fname in os.listdir(train_mask_dir)
            if fname.endswith(".png")
        ]
    )
    train_mask_array_list = [cv2.resize(mpimg.imread(str_index), (256,256))
                  for str_index in train_mask_img_paths]


    test_input_img_paths = sorted(
        [
            os.path.join(test_input_dir, fname)
            for fname in os.listdir(test_input_dir)
            if fname.endswith(".jpg")
        ]
    )
    test_input_array_list = [cv2.resize(mpimg.imread(str_index), (256,256))
                  for str_index in test_input_img_paths]

    test_mask_img_paths = sorted(
        [
            os.path.join(test_mask_dir, fname)
            for fname in os.listdir(test_mask_dir)
            if fname.endswith(".png")
        ]
    )
    test_mask_array_list = [cv2.resize(mpimg.imread(str_index), (256,256))
                  for str_index in test_mask_img_paths]


    X_train_list, Y_train_list = train_input_array_list, train_mask_array_list
    X_test_list, Y_test_list = test_input_array_list, test_mask_array_list

    X_train = np.array(X_train_list)
    X_test = np.array(X_test_list)
    Y_train = np.array(Y_train_list)
    Y_test = np.array(Y_test_list)

    X_train = X_train/255
    X_test = X_test/255
    Y_train = Y_train
    Y_test = Y_test

    Y_train = np.expand_dims(Y_train, axis=3)
    Y_test = np.expand_dims(Y_test, axis=3)

    image_datagen = ImageDataGenerator(
    rotation_range = rotation_range,
    width_shift_range = width_shift_range,
    height_shift_range = height_shift_range,
    shear_range = shear_range,
    zoom_range = zoom_range,
    fill_mode = fill_mode,
    horizontal_flip = horizontal_flip
    )

    mask_datagen = ImageDataGenerator(
    rotation_range = rotation_range,
    width_shift_range = width_shift_range,
    height_shift_range = height_shift_range,
    shear_range = shear_range,
    zoom_range = zoom_range,
    fill_mode = fill_mode,
    horizontal_flip = horizontal_flip
    )

    seed = 909
    image_datagen.fit(X_train[:int(X_train.shape[0]*0.8)], augment=True, seed=seed)
    mask_datagen.fit(Y_train[:int(Y_train.shape[0]*0.8)], augment=True, seed=seed)

    x=image_datagen.flow(X_train[:int(X_train.shape[0]*0.8)],batch_size=32,shuffle=True, seed=seed)
    y=mask_datagen.flow(Y_train[:int(Y_train.shape[0]*0.8)],batch_size=32,shuffle=True, seed=seed)

# Creating the validation Image and Mask generator
    image_datagen_val = ImageDataGenerator()
    mask_datagen_val = ImageDataGenerator()

    image_datagen_val.fit(X_train[int(X_train.shape[0]*0.8):], augment=True, seed=seed)
    mask_datagen_val.fit(Y_train[int(Y_train.shape[0]*0.8):], augment=True, seed=seed)

    x_val=image_datagen_val.flow(X_train[int(X_train.shape[0]*0.8):],batch_size=32,shuffle=True, seed=seed)
    y_val=mask_datagen_val.flow(Y_train[int(Y_train.shape[0]*0.8):],batch_size=32,shuffle=True, seed=seed)

    Test=(X_test, Y_test)
    traingen = (pair for pair in zip(x, y))
    valgen = (pair for pair in zip(x_val, y_val))
    return traingen,valgen, Test
