import cv2

from models import load_model
from PIL import Image
import numpy as np
from utils.utils import get_gpu_grower

get_gpu_grower()


class Detect:
    def __init__(self, model_name, weight_path, **kwargs):
        self.model = load_model(model_name=model_name, **kwargs)
        self.model.load_weights(weight_path)

    def detect(self, img):
        # apply necessary preprocessing
        img = np.array(img)
        img = cv2.resize(img, (256, 256))
        img = np.expand_dims(img, axis=0)

        # predict
        result = self.model.predict(img)
        # result = result.astype(np.uint8)[0]
        # apply necessary post-processing
        result = np.array(result * 255, dtype=np.uint8).squeeze()

        # return the results
        return result

    def detect_from_path(self, img_path):
        # make necessary modifications
        img = Image.open(img_path)
        return self.detect(img)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    detect = Detect('unet_res50', './weights/unet_res50/unet_model.h5')
    results = detect.detect_from_path(
        img_path='./streamlit/files/random-images/ISIC_0010465.jpg')
    plt.imshow(results)
    plt.show()
    print(results)
