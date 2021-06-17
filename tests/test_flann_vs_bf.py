import numpy as np
from os.path import join
from time import time

import cv2 as cv
from cv2.xfeatures2d import matchGMS

from src.preprocessing import prepare_image
from util.constants import DATA_PATH, TEST_IMG_PATH
from util.utils import INDEX_PARAMS, SEARCH_PARAMS

if __name__ == '__main__':
    img1 = cv.imread(join(DATA_PATH, "1uah_heads.jpg"))
    img1 = prepare_image(img1, True)
    # img1 = cv.imread(join(DATA_PATH, "5uah_heads.jpg"))
    img2 = cv.imread(join(TEST_IMG_PATH, "1uah_heads", "bottom.jpeg"))
    img2 = prepare_image(img2, True)
    # img2 = cv.imread(join(TEST_IMG_PATH, "other", "test4.jpg"))

    detector_descriptor = cv.ORB_create(10000)

    keypoints1, descriptions1 = detector_descriptor.detectAndCompute(img1, None)
    keypoints2, descriptions2 = detector_descriptor.detectAndCompute(img2, None)

    bf = cv.BFMatcher_create(cv.NORM_HAMMING, True)
    start = time()
    bf_matches = bf.match(descriptions1, descriptions2)
    end = time()

    print(f"Brute force matching time {end - start}")

    flann = cv.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)
    start = time()
    flann_matches = flann.match(np.float32(descriptions1), np.float32(descriptions2))
    end = time()

    print(f"Flann matching time {end - start}")

    print("Number of matches by flann", len(flann_matches))
    print("Number of matches by brute force", len(bf_matches))

    best_matches_flann = matchGMS(img1.shape[:2], img2.shape[:2], keypoints1, keypoints2, flann_matches, thresholdFactor=3)
    best_matches_bf = matchGMS(img1.shape[:2], img2.shape[:2], keypoints1, keypoints2, bf_matches, thresholdFactor=3)

    print("Number of matches by flann after outlier rejection", len(best_matches_flann))
    print("Number of matches by brute force after outlier rejection", len(best_matches_bf))

    output_img_flann = cv.drawMatches(img1, keypoints1, img2, keypoints2, best_matches_flann, None)
    output_img_bf = cv.drawMatches(img1, keypoints1, img2, keypoints2, best_matches_bf, None)

    cv.imshow("Flann", output_img_flann)
    cv.imshow("BF", output_img_bf)

    cv.waitKey(0)
    cv.destroyAllWindows()
