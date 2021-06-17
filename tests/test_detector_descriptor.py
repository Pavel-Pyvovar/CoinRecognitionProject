from os.path import join

import cv2 as cv
from cv2.xfeatures2d import matchGMS, SURF_create
import numpy as np

from src.detector_descriptor import DetectorDescriptor
from src.preprocessing import prepare_image
from util.constants import DATA_PATH, TEST_IMG_PATH, SIFT, ORB

if __name__ == '__main__':

    img1 = cv.imread(join(DATA_PATH, "1uah_heads.jpg"))
    img1 = prepare_image(img1, True)
    img2 = cv.imread(join(TEST_IMG_PATH, "1uah_heads", "bottom.jpeg"))
    img2 = prepare_image(img2, True)

    detector_descriptor = DetectorDescriptor(SIFT, ORB)

    keypoints1, descriptions1 = detector_descriptor.detect_describe(img1)
    keypoints2, descriptions2 = detector_descriptor.detect_describe(img2)

    matcher = cv.BFMatcher_create()

    matches = matcher.match(descriptions1, descriptions2)
    print(f"Matcher before outlier rejection {len(matches)}")

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    des_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    _, mask = cv.findHomography(src_pts, des_pts, cv.RANSAC, 15)
    print(mask.ravel().sum())

    best_matches = matchGMS(img1.shape[:2], img2.shape[:2], keypoints1, keypoints2, matches, thresholdFactor=3)
    # best_matches = [match for match, score in zip(matches, mask.ravel()) if score == 1]

    if not best_matches:
        raise ValueError("Not enough matches left after GMS!")

    print(f"Best matches: {len(best_matches)}")

    output_img = cv.drawMatches(img1, keypoints1, img2, keypoints2, best_matches, None)

    cv.imshow("SIFT + SIFT", output_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
