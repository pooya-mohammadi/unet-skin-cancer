import os
import cv2
import random
import albumentations as A


# Hair Augmentation
class AdvancedHairAugmentation:
    def __init__(self, max_hairs: int = 4, hairs_folder: str = "./data/hair_augment_images/", p=0.5):
        self.max_hairs = max_hairs
        self.hairs = [cv2.imread(os.path.join(hairs_folder, img_path)) for img_path in os.listdir(hairs_folder) if
                      img_path.endswith('.png')]
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:

            n_hairs = random.randint(1, self.max_hairs)  # at least one needs to be chosen

            height, width, _ = img.shape  # target image width and height

            for _ in range(n_hairs):
                hair = random.choice(self.hairs).copy()
                hair = cv2.flip(hair, random.choice([-1, 0, 1]))
                hair = cv2.rotate(hair, random.choice([0, 1, 2]))

                h_height, h_width, _ = hair.shape  # hair image width and height
                roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
                roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
                roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

                img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

                dst = cv2.add(img_bg, hair_fg)
                img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst
        return img


class HairAugmentation(A.ImageOnlyTransform):
    def __init__(self, max_hairs: int = 4, hairs_folder: str = "./data/hair_augment_images/", p: float = 0.5):
        super(HairAugmentation, self).__init__(p=p)
        self.max_hairs = max_hairs
        self.hairs = [cv2.imread(os.path.join(hairs_folder, img_path)) for img_path in os.listdir(hairs_folder) if
                      img_path.endswith('.png')]

    def apply(self, img, **params):
        n_hairs = random.randint(1, self.max_hairs)

        height, width, _ = img.shape  # target image width and height

        for _ in range(n_hairs):
            hair = random.choice(self.hairs).copy()
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            hair = cv2.resize(hair, (int(h_width * 0.8), int(h_height * 0.8)))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            dst = cv2.add(img_bg, hair_fg)
            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

        return img


class HairRemoval(A.ImageOnlyTransform):
    def __init__(self, p: float = 0.5):
        super(HairRemoval, self).__init__(p=p)

    def apply(self, img, **params):
        src = img
        gray_scale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(1, (17, 17))
        black_hat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)
        ret, thresh2 = cv2.threshold(black_hat, 10, 255, cv2.THRESH_BINARY)
        dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)
        return dst
