import cv2

from models import load_model
from PIL import Image
import numpy as np
from utils.utils import get_gpu_grower
import matplotlib.pyplot as plt

get_gpu_grower()


class Detect:
    def __init__(self, model_name, weight_path, **kwargs):
        self.model = load_model(model_name=model_name, **kwargs)
        self.model.load_weights(weight_path)

    def detect(self, img):
        # apply necessary preprocessing
        img = np.array(img)
        org_shape = img.shape
        img = cv2.resize(img, (256, 256))
        img = np.expand_dims(img, axis=0)
        img = img/255
        # predict
        result = self.model.predict(img)
        result = cv2.resize(result.squeeze(), org_shape[:2][::-1])
        result = np.array(result * 255, dtype=np.uint8)
        print("max", np.max(result))


        # return the results
        return result

    def detect_from_path(self, img_path):
        # make necessary modifications
        img = Image.open(img_path)
        return self.detect(img)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    detect = Detect('unet_res50', 'saved_models/unet_res50/unet_model.h5')
    results = detect.detect_from_path(
        img_path='./data/test/ISBI2016_ISIC_Part1_Test_Data/ISIC_0000012.jpg')
    plt.imshow(results)
    plt.show()
    print(results)
