"""Generate keypoints and descriptions for template images and dump them."""
import cv2 as cv
import os
from os.path import join

from tqdm import tqdm

from src.classifier import Classifier
from src.preprocessing import prepare_image
from util.utils import CLASS_NAMES, FEATURE_DETECTORS, FEATURE_DESCRIPTORS, corner_case
from util.constants import DATA_PATH

if __name__ == '__main__':
    template_images = [(name, prepare_image(cv.imread(join(DATA_PATH, f"{name}.jpg"))))
                       for name in CLASS_NAMES]

    for detector in FEATURE_DETECTORS:
        for descriptor in FEATURE_DESCRIPTORS:
            if corner_case(detector, descriptor):
                continue
            clf = Classifier(detector, descriptor)
            for cls_name, image in tqdm(
                    template_images,
                    desc=f"Dumping features for {detector}_{descriptor}"
            ):
                clf.dump_features(cls_name, image)
