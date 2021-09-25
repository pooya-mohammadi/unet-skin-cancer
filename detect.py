from models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf
from utils.utils import get_gpu_grower

get_gpu_grower()


class Detect:
    def __init__(self, model_name, weight_path, **kwargs):
        self.model = load_model(model_name=model_name, **kwargs)
        self.model.load_weights(weight_path)

    def detect(self, img):
        # apply necessary preprocessing
        img = np.array(img)
        img_s = (np.array(img.shape) >> 4) << 4
        img = img[np.newaxis, :img_s[0], :img_s[1], :]
        img = Image.fromarray(img).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.

        # predict
        result = self.model.predict(img)

        # apply necessary post-processing
        result = np.array(result * 255, dtype=np.uint8).squeeze()

        # return the results
        return result

    def detect_from_path(self, img_path):
        # make necessary modifications
        img = Image.open(img_path)
        return self.detect(img)


if __name__ == '__main__':
    detect = Detect('unet', './weights/unet')
    results = detect.detect_from_path(
        img_path='./streamlit/files/random-images/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9.png')
    print(results)
