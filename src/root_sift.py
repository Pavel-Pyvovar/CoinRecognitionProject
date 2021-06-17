import cv2 as cv
import numpy as np


class RootSift:
    """Class implementing Root-SIFT descriptor, an improvement over SIFT."""
    def __init__(self, max_features):
        self.extractor = cv.SIFT_create(max_features)

    def detect(self, img):
        return self.extractor.detect(img)

    def compute(self, img, keypoints, eps=1e-7):
        keypoints, descriptions = self.extractor.compute(img, keypoints)

        if len(keypoints) == 0:
            return [], None

        descriptions /= (descriptions.sum(axis=1, keepdims=True) + eps)
        descriptions = np.sqrt(descriptions)

        return keypoints, descriptions
