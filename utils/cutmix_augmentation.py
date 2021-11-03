import numpy as np


class CutMix:

    @staticmethod
    def get_bbox(size, lam):
        W = size[1]
        H = size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        center_x = np.random.randint(W)
        center_y = np.random.randint(H)

        x1 = np.clip(center_x - cut_w // 2, 0, W)
        y1 = np.clip(center_y - cut_h // 2, 0, H)
        x2 = np.clip(center_x + cut_w // 2, 0, W)
        y2 = np.clip(center_y + cut_h // 2, 0, H)

        return x1, y1, x2, y2

    @staticmethod
    def get_boxes(size, lam):
        b = size[0]
        boxes = np.array([CutMix.get_bbox(size, lam) for _ in range(b)])
        return boxes

    @staticmethod
    def shuffle(array_a: np.ndarray, array_b: np.ndarray = None, copy=False):
        """

        :param array_a: The input array. It could be a batch of images or ...
        :param array_b: The labels or the masks of the input array that should be shuffled together.
        :param copy: whether generate a copy of the inputs and apply the function or not
        :return:
        returns the shuffle format of array a and b if the latter one exists.
        """
        if copy:
            array_a = array_a.copy()
        if array_b is not None:
            if copy:
                array_b = array_b.copy()
            indices = np.arange(len(array_a))
            np.random.shuffle(indices)
            array_a[:] = array_a[indices]
            array_b[:] = array_b[indices]
            return array_a, array_b
        else:
            np.random.shuffle(array_a)
            return array_a

    @staticmethod
    def seg_cutmix(beta, image_a, mask_a, image_b=None, mask_b=None, shuffle=True):
        if image_b is None:
            image_b = image_a
            mask_b = mask_a

        image_a, mask_a, image_b, mask_b = image_a.copy(), mask_a.copy(), image_b.copy(), mask_b.copy()

        if shuffle:
            CutMix.shuffle(image_a, mask_a)
            CutMix.shuffle(image_b, mask_b)

        lam = np.random.beta(beta, beta)
        boxes = CutMix.get_boxes(image_a.shape, lam)

        # x_cutmix
        x_cutmix_mask = np.ones_like(image_a)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x_cutmix_mask[i, x1:x2, y1:y2, :] = 0
        x_cutmix = (np.multiply(image_a, x_cutmix_mask) + np.multiply(image_b, (abs(1. - x_cutmix_mask)))).astype(
            np.uint8)

        # y_cutmix
        y_cutmix_mask = np.ones_like(mask_a)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            y_cutmix_mask[i, x1:x2, y1:y2] = 0
        y_cutmix = (np.multiply(mask_a, y_cutmix_mask) + np.multiply(mask_b, (abs(1. - y_cutmix_mask)))).astype(
            np.uint8)
        return x_cutmix, y_cutmix

    @staticmethod
    def cls_cutmix(beta, image_a, label_a, image_b=None, label_b=None, shuffle=True):
        if image_b is None:
            image_b = image_a
            label_b = label_a

        image_a, label_a, image_b, label_b = image_a.copy(), label_a.copy(), image_b.copy(), label_b.copy()

        if shuffle:
            CutMix.shuffle(image_a)
            CutMix.shuffle(image_b)

        lam = np.random.beta(beta, beta)
        boxes = CutMix.get_boxes(image_a.shape, lam)

        # img_cutmix
        img_cutmix_mask = np.ones_like(image_a)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            img_cutmix_mask[i, x1:x2, y1:y2, :] = 0
        img_cutmix = (np.multiply(image_a, img_cutmix_mask) + np.multiply(image_b, (abs(1. - img_cutmix_mask)))).astype(
            np.uint8)
        label_cutmix = lam * label_a + label_b * (1 - lam)
        return img_cutmix, label_cutmix
