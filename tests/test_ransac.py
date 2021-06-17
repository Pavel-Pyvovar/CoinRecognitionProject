from os.path import join

import cv2 as cv

from src.classifier import Classifier
from src.detector_descriptor import DetectorDescriptor
from util.constants import DATA_PATH, TEST_IMG_PATH, ORB

if __name__ == '__main__':
    template_img = cv.imread(join(DATA_PATH, "1uah_heads.jpg"))
    test_img = cv.imread(join(TEST_IMG_PATH, "1uah_heads", "top.jpeg"))

    detector_descriptor = DetectorDescriptor(ORB, ORB)

    template_keypoints, template_descriptions = detector_descriptor.detect_describe(template_img)
    test_keypoints, test_descriptions = detector_descriptor.detect_describe(test_img)

    bf_matcher = cv.BFMatcher_create(cv.NORM_HAMMING, True)

    matches = bf_matcher.match(template_descriptions, test_descriptions)

    clf = Classifier()

    print(len(matches))

    best_matches = clf.ransac_outlier_rejection(template_keypoints, test_keypoints, matches)

    print(len(best_matches))