import os
import math
import random
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import albumentations as A
import tensorflow as tf
from utils.cutmix_augmentation import CutMix
from utils.mozaic import mosaic_aug


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_list,
                 mask_list,
                 cutmix_p,
                 beta,
                 usual_aug_with_cutmix,
                 batch_size=16,
                 img_size=(256, 256),
                 img_channel=3,
                 augmentation_p: float = 0,
                 shuffle=True,
                 double_unet=False,
                 p_random_rotate_90=0.5,
                 p_horizontal_flip=1,
                 p_vertical_flip=0.5,
                 p_center_crop=0.9,
                 p_mosaic=0.25
                 ):
        self.img_paths = np.array(img_list)
        self.mask_paths = np.array(mask_list)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.usual_aug_with_cutmix = usual_aug_with_cutmix
        self.mask_channel = 2 if double_unet else 1
        self.transform = A.Compose([
            A.RandomRotate90(p=p_random_rotate_90),
            A.VerticalFlip(p=p_vertical_flip),
            A.HorizontalFlip(p=p_horizontal_flip),
            A.CenterCrop(p=p_center_crop, height=256, width=256),
        ], p=augmentation_p)
        self.img_size = img_size
        self.img_channel = img_channel
        self.cutmix_p = cutmix_p
        self.p_mosaic = p_mosaic
        self.beta = beta
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.random.permutation(len(self.img_paths)).astype(np.int)
            self.img_paths, self.mask_paths = self.img_paths[indices], self.mask_paths[indices]

    def __len__(self):
        return math.ceil(len(self.img_paths) / self.batch_size)

    def __getitem__(self, idx):
        batch_img = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_mask = self.mask_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = np.zeros((self.batch_size, *self.img_size, self.img_channel), dtype=np.uint8)
        y = np.zeros((self.batch_size, *self.img_size), dtype=np.uint8)
        rnd_p = random.random()
        if rnd_p < self.cutmix_p:
            for i, (img_path, mask_path) in enumerate(zip(batch_img, batch_mask)):
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.img_size)
                mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
                if self.usual_aug_with_cutmix:
                    img, mask = mosaic_aug(img, mask, self.img_paths, self.mask_paths, self.img_size, p=self.p_mosaic)
                    augmented = self.transform(image=img, mask=mask)
                    img, mask = augmented['image'], augmented['mask']
                x[i] = img
                y[i] = mask
            x, y = CutMix.seg_cutmix(self.beta, image_a=x, mask_a=y)
        else:
            for i, (img_path, mask_path) in enumerate(zip(batch_img, batch_mask)):
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.img_size)
                mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
                img, mask = mosaic_aug(img, mask, self.img_paths, self.mask_paths, self.img_size, p=self.p_mosaic)
                augmented = self.transform(image=img, mask=mask)
                x[i] = augmented['image']
                y[i] = augmented['mask']

        y = y.reshape((self.batch_size, *self.img_size, 1)) / 255
        y = np.concatenate([y] * self.mask_channel, axis=-1)
        return x / 255, y


def get_loader(train_input_dir,
               train_mask_dir,
               test_input_dir,
               test_mask_dir,
               model_name,
               val_size=0.2,
               batch_size=8,
               img_size=(256, 256),
               **kwargs):
    # should there be only one direction?
    # how are we supposed to get the other 4 directions of train test and images and masks for each

    train_img_paths = sorted(
        [
            os.path.join(train_input_dir, fname)
            for fname in os.listdir(train_input_dir)
            if fname.endswith(".jpg")
        ]
    )

    train_mask_paths = sorted(
        [
            os.path.join(train_mask_dir, fname)
            for fname in os.listdir(train_mask_dir)
            if fname.endswith(".png")
        ]
    )


    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(train_img_paths,
                                                                                        train_mask_paths,
                                                                                        test_size=val_size,
                                                                                        shuffle=True)
    train_img_paths = train_img_paths + train_img_paths
    train_mask_paths = train_mask_paths + train_mask_paths

    test_img_paths = sorted(
        [
            os.path.join(test_input_dir, fname)
            for fname in os.listdir(test_input_dir)
            if fname.endswith(".jpg")
        ]
    )

    test_mask_paths = sorted(
        [
            os.path.join(test_mask_dir, fname)
            for fname in os.listdir(test_mask_dir)
            if fname.endswith(".png")
        ]
    )
    double_unet = True if model_name == 'double_unet' else False
    train = DataGenerator(train_img_paths,
                          train_mask_paths,
                          batch_size=batch_size,
                          img_size=img_size,
                          augmentation_p=0.5,
                          double_unet=double_unet, **kwargs)
    val = DataGenerator(val_img_paths,
                        val_mask_paths,
                        batch_size=batch_size,
                        img_size=img_size,
                        augmentation_p=0.5,
                        double_unet=double_unet, **kwargs)
    test = DataGenerator(test_img_paths,
                         test_mask_paths,
                         batch_size=batch_size,
                         img_size=img_size,
                         augmentation_p=0,
                         shuffle=False,
                         double_unet=double_unet, **kwargs)
    return train, val, test
