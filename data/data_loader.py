import os
import math
import random
import cv2
from deep_utils import log_print, CutMixTF
from sklearn.model_selection import train_test_split
import numpy as np
import albumentations as A
import tensorflow as tf
from utils.mosaic import mosaic_aug
from utils.hair_augmentation import HairAugmentation, HairRemoval, Identity


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
                 random_rotate_p,
                 p_horizontal_flip,
                 p_vertical_flip,
                 p_center_crop,
                 p_mosaic,
                 hair_aug_p,
                 hair_rmv_p,
                 shuffle=True,
                 double_unet=False,
                 hue_shift_limit=0,
                 sat_shift_limit=0,
                 contrast_limit=0,
                 brightness_limit=0,
                 hue_p=0.5,
                 contrast_p=0.5,
                 brightness_p=0.5,
                 ):
        self.img_paths = np.array(img_list)
        self.mask_paths = np.array(mask_list)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.usual_aug_with_cutmix = usual_aug_with_cutmix
        self.mask_channel = 2 if double_unet else 1
        self.augmentation_p = augmentation_p
        self.transform = A.Compose([
            A.Rotate(limit=180, p=random_rotate_p),
            A.VerticalFlip(p=p_vertical_flip),
            A.HorizontalFlip(p=p_horizontal_flip),
            A.CenterCrop(p=p_center_crop, height=img_size[0], width=img_size[1]),
            A.OneOf(
                [
                    HairAugmentation(p=hair_aug_p),
                    HairRemoval(p=hair_rmv_p),
                    Identity(p=1 if hair_aug_p == 0 and hair_rmv_p == 0 else 0)],
                # apply identity if both has zero prob
                p=0.5),
            A.HueSaturationValue(hue_shift_limit=hue_shift_limit,
                                 sat_shift_limit=sat_shift_limit,
                                 val_shift_limit=0, p=hue_p),
            A.RandomContrast(limit=contrast_limit, p=contrast_p),
            A.RandomBrightness(limit=brightness_limit, p=brightness_p),
        ], p=self.augmentation_p
        )
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
        if rnd_p < self.cutmix_p * self.augmentation_p:
            for i, (img_path, mask_path) in enumerate(zip(batch_img, batch_mask)):
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.img_size)
                mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
                if self.usual_aug_with_cutmix:
                    augmented = self.transform(image=img, mask=mask)
                    img, mask = augmented['image'], augmented['mask']
                x[i] = img[..., ::-1]  # BGR2RGB
                y[i] = mask
            x, y = CutMixTF.seg_cutmix_batch(a_images=x, a_masks=y, beta=self.beta)
        else:
            for i, (img_path, mask_path) in enumerate(zip(batch_img, batch_mask)):
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.img_size)
                mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
                augmented = self.transform(image=img,
                                           mask=mask)  # at test time, self.augmentation_p is set to zero, so it doesn't make any change
                img, mask = augmented['image'], augmented['mask']
                img, mask = mosaic_aug(img, mask, self.img_paths, self.mask_paths, self.img_size,
                                       p=self.p_mosaic * self.augmentation_p)  # at test time, self.augmentation_p is set to zero, so it doesn't make any change
                x[i] = img[..., ::-1]
                y[i] = mask

        y = y.reshape((self.batch_size, *self.img_size, 1)) / 255  # normalization is done for all the samples
        y = np.concatenate([y] * self.mask_channel, axis=-1)
        return x / 255, y  # normalization is done for all the samples


def get_loader(train_input_dir,
               train_mask_dir,
               test_input_dir,
               test_mask_dir,
               model_name,
               val_size=0.2,
               batch_size=8,
               img_size=(256, 256),
               seed=1234,
               logger=None,
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
                                                                                        shuffle=True,
                                                                                        random_state=seed)
    # train_img_paths = train_img_paths + train_img_paths
    # train_mask_paths = train_mask_paths + train_mask_paths

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
                          random_rotate_p=kwargs.get("random_rotate_p"),
                          p_horizontal_flip=kwargs.get("p_horizontal_flip"),
                          p_vertical_flip=kwargs.get("p_vertical_flip"),
                          p_center_crop=kwargs.get("p_center_crop"),
                          p_mosaic=kwargs.get("p_mosaic"),
                          hair_aug_p=kwargs.get("hair_aug_p"),
                          hair_rmv_p=kwargs.get("hair_rmv_p"),
                          cutmix_p=kwargs.get('cutmix_p'),
                          usual_aug_with_cutmix=kwargs.get('usual_aug_with_cutmix'),
                          beta=kwargs.get('beta'),
                          hue_shift_limit=kwargs.get("hue_shift_limit"),
                          sat_shift_limit=kwargs.get("sat_shift_limit"),
                          contrast_limit=kwargs.get("contrast_limit"),
                          brightness_limit=kwargs.get("brightness_limit"),
                          hue_p=kwargs.get("hue_p"),
                          contrast_p=kwargs.get("contrast_p"),
                          brightness_p=kwargs.get("brightness_p")
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
                        random_rotate_p=kwargs.get("random_rotate_p"),
                        p_horizontal_flip=kwargs.get("p_horizontal_flip"),
                        p_vertical_flip=kwargs.get("p_vertical_flip"),
                        p_center_crop=kwargs.get("p_center_crop"),
                        p_mosaic=kwargs.get("p_mosaic"),
                        hair_aug_p=kwargs.get("hair_aug_p"),
                        hair_rmv_p=kwargs.get("hair_rmv_p"),
                        hue_shift_limit=kwargs.get("hue_shift_limit"),
                        sat_shift_limit=kwargs.get("sat_shift_limit"),
                        contrast_limit=kwargs.get("contrast_limit"),
                        brightness_limit=kwargs.get("brightness_limit"),
                        hue_p=kwargs.get("hue_p"),
                        contrast_p=kwargs.get("contrast_p"),
                        brightness_p=kwargs.get("brightness_p")
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
                         random_rotate_p=kwargs.get("random_rotate_p"),
                         p_horizontal_flip=kwargs.get("p_horizontal_flip"),
                         p_vertical_flip=kwargs.get("p_vertical_flip"),
                         p_center_crop=kwargs.get("p_center_crop"),
                         p_mosaic=kwargs.get("p_mosaic"),
                         hair_aug_p=kwargs.get("hair_aug_p"),
                         hair_rmv_p=kwargs.get("hair_rmv_p"),
                         hue_shift_limit=kwargs.get("hue_shift_limit"),
                         sat_shift_limit=kwargs.get("sat_shift_limit"),
                         contrast_limit=kwargs.get("contrast_limit"),
                         brightness_limit=kwargs.get("brightness_limit"),
                         hue_p=kwargs.get("hue_p"),
                         contrast_p=kwargs.get("contrast_p"),
                         brightness_p=kwargs.get("brightness_p")
                         )
    log_print(logger, "Data Loader is successfully loaded!")
    return train, val, test
