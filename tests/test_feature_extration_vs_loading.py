import pickle
from os.path import join
from time import time

import cv2 as cv

from src.detector_descriptor import DetectorDescriptor
from util.constants import DATA_PATH, FEATURES_PATH, BRISK

if __name__ == '__main__':
    test_img = cv.imread(join(DATA_PATH, "1uah_heads.jpg"))
    features_path = join(FEATURES_PATH, BRISK, "1uah_heads.pickle")

    begin = time()
    with open(features_path, "rb") as pickle_file:
        features = pickle.load(pickle_file)
    end = time()
    print("Loading features from disk", end - begin)

    detector_descriptor = DetectorDescriptor(BRISK, BRISK)
    begin = time()
    keypoints, descriptions = detector_descriptor.detect_describe(test_img)
    end = time()
    print("Extracting features", end - begin)

