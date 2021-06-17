from os.path import join

import cv2 as cv
from cv2.xfeatures2d import matchGMS, SURF_create
import numpy as np

from src.preprocessing import prepare_image
from util.utils import DATA_PATH, TEST_IMG_PATH
from zoo.sift import SiftBrisk, SiftOrb, SiftFreak, RootSiftFreak, RootSiftOrb, RootSiftBrisk

if __name__ == '__main__':
    # Hybrid detectors
    # detector_descriptor = FastFreak(20)
    # detector_descriptor = StarFreak()
    # detector_descriptor = OrbFreak()
    # detector_descriptor = BriskFreak()
    # FastSift is a total garbage.
    # detector_descriptor = FastSift()
    # SiftFreak is even worse.
    # detector_descriptor = SiftFreak()
    # detector_descriptor = RootSift(nfeatures=10000)
    # detector_descriptor = FastOrb()
    # detector_descriptor = FastBrisk()
    # detector_descriptor = FastAkaze()
    # detector_descriptor = FastRootSift()
    detector_descriptor = RootSiftBrisk()

    img1 = cv.imread(join(DATA_PATH, "1uah_heads.jpg"))
    img1 = prepare_image(img1, True)
    # img1 = cv.imread(join(DATA_PATH, "5uah_heads.jpg"))
    img2 = cv.imread(join(TEST_IMG_PATH, "1uah_heads", "bottom.jpeg"))
    img2 = prepare_image(img2, True)
    # img2 = cv.imread(join(TEST_IMG_PATH, "other", "test4.jpg"))

    keypoints1, descriptions1 = detector_descriptor.detectAndCompute(img1)
    keypoints2, descriptions2 = detector_descriptor.detectAndCompute(img2)

    matcher = cv.BFMatcher_create(cv.NORM_HAMMING, True)
    # matcher = cv.BFMatcher_create()

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

    cv.imshow("FAST + FREAK", output_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
