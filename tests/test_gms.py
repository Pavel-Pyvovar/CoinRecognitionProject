import time
import cv2
from cv2.xfeatures2d import matchGMS
from os.path import join

from src.detector_descriptor import RootSift
from src.preprocessing import prepare_image
from util.constants import DATA_PATH, TEST_IMG_PATH


if __name__ == '__main__':
    img1 = cv2.imread(join(DATA_PATH, "1uah_heads.jpg"))
    img1 = prepare_image(img1, resize=True)
    # img1 = cv2.imread(join(DATA_PATH, "5uah_heads.jpg"))
    img2 = cv2.imread(join(TEST_IMG_PATH, "1uah_heads", "bottom.jpeg"))
    img2 = prepare_image(img2, resize=True)
    # img2 = cv2.imread(join(TEST_IMG_PATH, "other", "test4.jpg"))

    # detector_descriptor = cv2.ORB_create(10000, fastThreshold=0)
    # detector_descriptor = cv2.KAZE_create()
    detector_descriptor = cv2.AKAZE_create()
    # detector_descriptor = cv2.BRISK_create()


    kp1, des1 = detector_descriptor.detectAndCompute(img1, None)
    print(f"Template image {len(kp1)} keypoints")
    kp2, des2 = detector_descriptor.detectAndCompute(img2, None)
    print(f"Input image {len(kp2)} keypoints")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    print(f"Matches {len(matches)}")

    start = time.time()
    matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches,
                           withScale=True, withRotation=True, thresholdFactor=4)
    end = time.time()

    print(f'Found GMS {len(matches_gms)} matches')
    print('GMS takes', end-start, 'seconds')

    output = cv2.drawKeypoints(img1, kp1, None)
    # output = cv2.drawMatches(img1, kp1, img2, kp2, matches_gms, None, flags=2)

    cv2.imshow("show", output)
    cv2.waitKey(0)