import os
from os.path import join

import cv2 as cv

from src.detector_descriptor import RootSift
from util.constants import DATA_PATH, TEST_IMG_PATH

if __name__ == '__main__':

    template_img = cv.imread(join(DATA_PATH, "1uah_heads.jpg"))
    test_img = cv.imread(join(TEST_IMG_PATH, "1uah_heads", "bottom.jpeg"))

    root_sift = RootSift()

    template_keypoints = root_sift.detect(template_img)
    test_keypoints = root_sift.detect(test_img)

    template_keypoints, template_descriptions = root_sift.compute(template_img, template_keypoints)
    test_keypoints, test_descriptions = root_sift.compute(test_img, test_keypoints)

    matcher = cv.BFMatcher_create()

    matches = matcher.match(template_descriptions, test_descriptions)

    matches_img = cv.drawMatches(template_img, template_keypoints, test_img, test_keypoints, matches, None)

    cv.imshow("Matches", matches_img)
    cv.waitKey(0)
    cv.destroyAllWindows()