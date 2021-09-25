from typing import Tuple
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
from os.path import dirname, join as pjoin
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as keras
import random
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def unison_shuffle(a, b):
    inx = np.random.permutation(a.shape[0])
    return a[inx], b[inx]


def get_loader(train_input_dir, train_mask_dir, test_input_dir, test_mask_dir,
               p_RandomRotate90=0.5,
               p_HorizontalFlip=1,
               p_VerticalFlip=0.5,
               p_CenterCrop=0.9):
    # should there be only one direction?
    # how are we supposed to get the other 4 directions of train test and images and masks for each

    train_input_img_paths = sorted(
        [
            os.path.join(train_input_dir, fname)
            for fname in os.listdir(train_input_dir)
            if fname.endswith(".jpg")
        ]
    )
    train_input_array_list = [cv2.resize(mpimg.imread(str_index), (256, 256))
                              for str_index in train_input_img_paths]

    train_mask_img_paths = sorted(
        [
            os.path.join(train_mask_dir, fname)
            for fname in os.listdir(train_mask_dir)
            if fname.endswith(".png")
        ]
    )
    train_mask_array_list = [cv2.resize(mpimg.imread(str_index), (256, 256))
                             for str_index in train_mask_img_paths]

    test_input_img_paths = sorted(
        [
            os.path.join(test_input_dir, fname)
            for fname in os.listdir(test_input_dir)
            if fname.endswith(".jpg")
        ]
    )
    test_input_array_list = [cv2.resize(mpimg.imread(str_index), (256, 256))
                             for str_index in test_input_img_paths]

    test_mask_img_paths = sorted(
        [
            os.path.join(test_mask_dir, fname)
            for fname in os.listdir(test_mask_dir)
            if fname.endswith(".png")
        ]
    )
    test_mask_array_list = [cv2.resize(mpimg.imread(str_index), (256, 256))
                            for str_index in test_mask_img_paths]

    X_train_list, Y_train_list = train_input_array_list, train_mask_array_list
    X_test_list, Y_test_list = test_input_array_list, test_mask_array_list

    image_padded = []
    mask_padded = []
    for i in range(0, len(X_train_list)):
        aug = A.Compose([
            A.RandomRotate90(p=p_RandomRotate90),
            A.VerticalFlip(p=p_VerticalFlip),
            A.HorizontalFlip(p=p_HorizontalFlip),
            A.CenterCrop(p=p_CenterCrop, height=256, width=256),
        ], p=1)
        augmented = aug(image=X_train_list[i], mask=Y_train_list[i])
        image_padded.append(augmented['image'])
        mask_padded.append(augmented['mask'])

    X_train_list = X_train_list + image_padded
    Y_train_list = Y_train_list + mask_padded

    X_train = np.array(X_train_list)
    X_test = np.array(X_test_list)
    Y_train = np.array(Y_train_list)
    Y_test = np.array(Y_test_list)

    X_train = X_train / 255
    X_test = X_test / 255
    Y_train = Y_train
    Y_test = Y_test

    print("Xtrain shape", X_train.shape)
    print("ytrain shape", Y_train.shape)
    print("Xtest shape", X_test.shape)
    print("ytest shape", Y_test.shape)
    Y_train = np.expand_dims(Y_train, axis=3)
    Y_test = np.expand_dims(Y_test, axis=3)

    trainX, valX, trainy, valy = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True)

    train = (trainX, trainy)
    validation = (valX, valy)
    test = (X_test, Y_test)
    return train, validation, test