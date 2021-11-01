import cv2
import numpy as np
import random
import math

"""
   Expalanation:
   def mosaic : this function get 4 images as numpy array and return mosaic frame of them
   def mosaic_aug : this function according to p value decide whether execute mosaic augmentation or not
   if p > flag then select 3 random images from train dataset and with main image pass them to mosaic function
   for both input image and mask image do same instruction. 
"""


def mosaic(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray
           , img4: np.ndarray, w_index, h_index) -> np.ndarray:
    """

    Parameters
    ----------
    img1
    img2
    img3
    img4
    w_index
    h_index

    Returns
    -------
    new frame: Mosaic image includes of 4 above images

    """
    h = img1.shape[0]
    w = img1.shape[1]
    resize_img1 = cv2.resize(img1, (w_index, h_index), cv2.INTER_NEAREST)
    resize_img2 = cv2.resize(img2, (w_index, h - h_index), cv2.INTER_NEAREST)
    resize_img3 = cv2.resize(img3, (w - w_index, h_index), cv2.INTER_NEAREST)
    resize_img4 = cv2.resize(img4, (w - w_index, h - h_index), cv2.INTER_NEAREST)

    v1_images = np.vstack((resize_img1, resize_img2))
    v2_images = np.vstack((resize_img3, resize_img4))
    new_frame = np.hstack((v1_images, v2_images))

    return new_frame


def mosaic_aug(x: np.ndarray, y: np.ndarray,
               img_paths, mask_paths, img_size, p: float = 0.5) -> (np.ndarray, np.ndarray):
    """

    Parameters
    ----------
    x
    y
    img_paths
    mask_paths
    img_size
    p

    Returns
    x_aug , y_aug of Mozaic augmentation
    -------

    """

    flag = random.random()
    if (p > flag):
        img1_idx = random.randint(0, len(img_paths) - 1)
        img2_idx = random.randint(0, len(img_paths) - 1)
        img3_idx = random.randint(0, len(img_paths) - 1)

        x_img2 = cv2.resize(cv2.imread(img_paths[img1_idx]), img_size, cv2.INTER_NEAREST)
        y_img2 = cv2.resize(cv2.imread(mask_paths[img1_idx], cv2.IMREAD_GRAYSCALE), img_size, cv2.INTER_NEAREST)

        x_img3 = cv2.resize(cv2.imread(img_paths[img2_idx]), img_size, cv2.INTER_NEAREST)
        y_img3 = cv2.resize(cv2.imread(mask_paths[img2_idx], cv2.IMREAD_GRAYSCALE), img_size, cv2.INTER_NEAREST)

        x_img4 = cv2.resize(cv2.imread(img_paths[img3_idx]), img_size, cv2.INTER_NEAREST)
        y_img4 = cv2.resize(cv2.imread(mask_paths[img3_idx], cv2.IMREAD_GRAYSCALE), img_size, cv2.INTER_NEAREST)

        h_index = random.randint(math.floor(img_size[0] / 5), math.floor(4 * img_size[0] / 5))
        w_index = random.randint(math.floor(img_size[1] / 5), math.floor(4 * img_size[1] / 5))

        new_frame_x = mosaic(x, x_img2, x_img3, x_img4, w_index, h_index)
        new_frame_y = mosaic(y, y_img2, y_img3, y_img4, w_index, h_index)

        return new_frame_x, new_frame_y
    else:
        return x, y
