import cv2
import numpy as np

"""
Explanations:
This function get 4 images and resize to quarter of image frame and then 
combine these 4 images into one frame
"""


def combine_images(img1: np.ndarray, img2: np.ndarray,
                   img3: np.ndarray, img4: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    img1
    img2
    img3
    img4

    Returns
    -------
    new_frame : combine 4 images
    """
    resize_img1 = cv2.resize(img1, (0, 0), None, .5, .5, cv2.INTER_NEAREST)
    resize_img2 = cv2.resize(img2, (0, 0), None, .5, .5, cv2.INTER_NEAREST)
    resize_img3 = cv2.resize(img3, (0, 0), None, .5, .5, cv2.INTER_NEAREST)
    resize_img4 = cv2.resize(img4, (0, 0), None, .5, .5, cv2.INTER_NEAREST)

    v1_images = np.vstack((resize_img1, resize_img2))
    v2_images = np.vstack((resize_img3, resize_img4))
    new_frame = np.hstack((v1_images, v2_images))

    return new_frame
