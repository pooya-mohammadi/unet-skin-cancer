import os
import math
import random
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import albumentations as A
import tensorflow as tf
from utils.cutmix_augmentation import CutMix
from utils.mosaic import mosaic_aug
from utils.hair_augmentation import HairAugmentation, HairRemoval


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_list,
                 mask_list,
                 cutmix_p,
                 beta,
                 usual_aug_with_cutmix,
                 batch_size,
                 img_size,
                 img_channel,
                 augmentation_p: float,
                 p_random_rotate_90,
                 p_horizontal_flip,
                 p_vertical_flip,
                 p_center_crop,
                 p_mosaic,
                 hair_aug_p,
                 hair_rmv_p,
                 shuffle=True,
                 double_unet=False,
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
            A.CenterCrop(p=p_center_crop, height=img_size[1], width=img_size[0]),
            A.OneOf([HairAugmentation(p=hair_aug_p), HairRemoval(p=hair_rmv_p)], p=0.5),
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
                    augmented = self.transform(image=img, mask=mask)
                    img, mask = augmented['image'], augmented['mask']
                x[i] = img[..., ::-1]
                y[i] = mask
            x, y = CutMix.seg_cutmix(self.beta, image_a=x, mask_a=y)
        else:
            for i, (img_path, mask_path) in enumerate(zip(batch_img, batch_mask)):
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.img_size)
                mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
                augmented = self.transform(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']
                img, mask = mosaic_aug(img, mask, self.img_paths, self.mask_paths, self.img_size, p=self.p_mosaic)
                x[i] = img[..., ::-1]
                y[i] = mask

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
                          img_channel=kwargs.get('img_channel'),
                          augmentation_p=0.5,
                          double_unet=double_unet,
                          p_random_rotate_90=kwargs.get("p_random_rotate_90"),
                          p_horizontal_flip=kwargs.get("p_horizontal_flip"),
                          p_vertical_flip=kwargs.get("p_vertical_flip"),
                          p_center_crop=kwargs.get("p_center_crop"),
                          p_mosaic=kwargs.get("p_mosaic"),
                          hair_aug_p=kwargs.get("hair_aug_p"),
                          hair_rmv_p=kwargs.get("hair_rmv_p"),
                          cutmix_p=kwargs.get('cutmix_p'),
                          usual_aug_with_cutmix=kwargs.get('usual_aug_with_cutmix'),
                          beta=kwargs.get('beta'),
                          )
    val = DataGenerator(val_img_paths,
                        val_mask_paths,
                        batch_size=batch_size,
                        img_size=img_size,
                        img_channel=kwargs.get('img_channel'),
                        augmentation_p=0.5,
                        double_unet=double_unet,
                        cutmix_p=0,
                        beta=1,
                        usual_aug_with_cutmix=False,
                        p_random_rotate_90=kwargs.get("p_random_rotate_90"),
                        p_horizontal_flip=kwargs.get("p_horizontal_flip"),
                        p_vertical_flip=kwargs.get("p_vertical_flip"),
                        p_center_crop=kwargs.get("p_center_crop"),
                        p_mosaic=kwargs.get("p_mosaic"),
                        hair_aug_p=kwargs.get("hair_aug_p"),
                        hair_rmv_p=kwargs.get("hair_rmv_p"),
                        )
    test = DataGenerator(test_img_paths,
                         test_mask_paths,
                         batch_size=batch_size,
                         img_size=img_size,
                         img_channel=kwargs.get('img_channel'),
                         augmentation_p=0,
                         shuffle=False,
                         double_unet=double_unet,
                         cutmix_p=0,
                         beta=1,
                         usual_aug_with_cutmix=False,
                         p_random_rotate_90=kwargs.get("p_random_rotate_90"),
                         p_horizontal_flip=kwargs.get("p_horizontal_flip"),
                         p_vertical_flip=kwargs.get("p_vertical_flip"),
                         p_center_crop=kwargs.get("p_center_crop"),
                         p_mosaic=kwargs.get("p_mosaic"),
                         hair_aug_p=kwargs.get("hair_aug_p"),
                         hair_rmv_p=kwargs.get("hair_rmv_p"),
                         )
    return train, val, test
