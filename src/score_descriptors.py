"""Script for building a comparison tables of descriptors"""
from os.path import join
from time import time

import cv2 as cv
from cv2.xfeatures2d import matchGMS
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.detector_descriptor import DetectorDescriptor
from src.preprocessing import prepare_image
from util.utils import (
    DESCRIPTOR_NAMES, DETECTOR_NAMES, CLASS_NAMES,
    SIFT, ROOT_SIFT, TEST_IMG_NAMES, AKAZE, corner_case)
from util.constants import TEST_IMG_PATH, DATA_PATH, OUTPUT_PATH, ORB, FAST

GMS_THRESHOLD = 4
TABLE_COLUMNS = [
    "Features Detected in the 1st Image", "Features Detected in the 2nd Image",
    "Features Matched", "Outliers Rejected", "Features Left After GMS",
    "1st Image Detection & Description Time", "2nd Image Detection & Description Time",
    "Feature Matching Time", "Outlier Rejection Time", "Total Image Matching Time"
]


if __name__ == '__main__':
    descriptor_results = []

    template_images = [cv.imread(join(DATA_PATH, f"{cls_name}.jpg")) for cls_name in CLASS_NAMES]
    test_images = [[cv.imread(join(TEST_IMG_PATH, cls_name, f"{test_img_name}.jpeg"))
                   for test_img_name in TEST_IMG_NAMES] for cls_name in CLASS_NAMES]

    for detector_name in DETECTOR_NAMES:
        for descriptor_name in DESCRIPTOR_NAMES:
            if corner_case(detector_name, descriptor_name):
                continue
            detector_descriptor = DetectorDescriptor(detector_name, descriptor_name)
            # matcher = cv.BFMatcher_create() if descriptor_name in [SIFT, ROOT_SIFT] \
            #     else cv.BFMatcher_create(cv.NORM_HAMMING, True)
            detector_descriptor_score = 0
            total_matching_time = []
            final_matches_cnt = []
            for cls_idx, class_tests in enumerate(test_images):
                for test_img in class_tests:
                    prepared_test_img = prepare_image(test_img.copy())
                    print(f"Test image shape: {prepared_test_img.shape}")
                    test_img_description_time_begin = time()
                    # TODO: resolve issue with SIFT_SIFT.
                    test_keypoints, test_descriptions =\
                        detector_descriptor.detect_describe(prepared_test_img)
                    test_img_description_time_end = time()
                    test_img_description_time = test_img_description_time_end - test_img_description_time_begin

                    matches_counts = []
                    for template_img in tqdm(template_images, desc="Matching images"):
                        print(detector_name, descriptor_name)
                        prepared_template_img = prepare_image(template_img.copy())
                        print(f"Template image shape: {prepared_template_img.shape}")
                        template_img_description_time_begin = time()
                        template_keypoints, template_descriptions =\
                            detector_descriptor.detect_describe(prepared_template_img)
                        template_img_description_time_end = time()
                        template_img_description_time =\
                            template_img_description_time_end - template_img_description_time_begin

                        matcher = cv.BFMatcher_create() if descriptor_name in [SIFT, ROOT_SIFT] \
                            else cv.BFMatcher_create(cv.NORM_HAMMING, True)

                        feature_matching_time_begin = time()
                        matches = matcher.match(template_descriptions, test_descriptions)
                        feature_matching_time_end = time()
                        feature_matching_time = feature_matching_time_end - feature_matching_time_begin

                        outlier_rejection_time_begin = time()
                        best_matches = matchGMS(
                            prepared_template_img.shape[:2], prepared_test_img.shape[:2], template_keypoints, test_keypoints, matches,
                            withRotation=True, withScale=True, thresholdFactor=GMS_THRESHOLD
                        )
                        outlier_rejection_time_end = time()
                        outlier_rejection_time = outlier_rejection_time_end - outlier_rejection_time_begin

                        matches_counts.append(len(best_matches))
                    if np.argmax(matches_counts) == cls_idx:
                        detector_descriptor_score += 1
                        final_matches_cnt.append(len(best_matches))
                    total_matching_time.append(
                        test_img_description_time + template_img_description_time
                        + feature_matching_time + outlier_rejection_time
                    )
            descriptor_results.append([
                f"{detector_name}_{descriptor_name}", detector_descriptor_score / 20,
                np.around(np.mean(total_matching_time), 4),
                np.mean(final_matches_cnt, dtype=np.int32)
            ])

        pd.DataFrame(descriptor_results).to_csv(join(OUTPUT_PATH, "all_dfs.csv"))
