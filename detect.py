from models import load_model
from PIL import Image


class Detect:
    def __init__(self, model_name, weight_path, **kwargs):
        self.model = load_model(model_name=model_name, **kwargs)
        self.model.load_weights(weight_path)

    def detect(self, img):
        # apply necessary preprocessing
        ...

        # predict
        result = self.model.predict(img)

        # apply necessary post-processing
        ...

        # return the results
        return result

    def detect_from_path(self, img_path):
        # make necessary modifications
        img = Image.open(img_path)
        return self.detect(img)
