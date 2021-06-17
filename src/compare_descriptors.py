"""Script for building a comparison tables of descriptors"""
import os
from os.path import join
from time import time

import cv2 as cv
from cv2.xfeatures2d import matchGMS
import pandas as pd
from tqdm import tqdm

from src.preprocessing import prepare_image
from util.utils import (
    CLASS_NAMES, TEST_IMG_PATH,
    SIFT, DATA_PATH, OUTPUT_PATH, BRISK,
    ORB, ROOT_SIFT, FAST_ROOT_SIFT, BRISK_FREAK)
from zoo.brisk import BriskFreak
from zoo.fast import FastRootSift
from zoo.sift import RootSift

GMS_THRESHOLD = 4
TABLE_COLUMNS = [
    "Features Detected in the 1st Image", "Features Detected in the 2nd Image",
    "Features Matched", "Outliers Rejected", "Features Left After GMS",
    "1st Image Detection & Description Time", "2nd Image Detection & Description Time",
    "Feature Matching Time", "Outlier Rejection Time", "Total Image Matching Time"
]

MAX_FEATURES_GRID = [5000, 10_000, 15_000]
# For ORB for max_features > 500 it should be 0.
FAST_THRESHOLD = 0
BRISK_THRESHOLD_GRID = [15, 30, 45]
BRISK_OCTAVES_GRID = [3, 4, 5]


DETECTOR_DESCRIPTOR_FUNC_MAP = {
    ORB: cv.ORB_create,
    BRISK: cv.BRISK_create,
    ROOT_SIFT: RootSift,
    FAST_ROOT_SIFT: FastRootSift,
    BRISK_FREAK: BriskFreak
}


def create_param_grid():

    param_grid = {}

    for detector_descriptor_name in [ROOT_SIFT]:
        for threshold in MAX_FEATURES_GRID:
            detector_descriptor = DETECTOR_DESCRIPTOR_FUNC_MAP[detector_descriptor_name](threshold)
            if detector_descriptor_name == ORB:
                detector_descriptor.setFastThreshold(0)
            param_grid.update({
                f"{detector_descriptor_name}(max_features={threshold})": detector_descriptor
            })


    # for threshold in brisk_threshold_grid:
    #     for octaves in brisk_octaves_grid:
    #         param_grid.update(
    #             {f"{BRISK}(threshold={threshold}, octaves={octaves})": cv.BRISK_create(threshold, octaves)}
    #         )

    return param_grid


if __name__ == '__main__':
    # TODO: Bring this out of the main!!!
    for cls_name in tqdm(CLASS_NAMES, desc=f"Iterating over image classes"):
        template_img = cv.imread(join(DATA_PATH, f"{cls_name}.jpg"))
        prepared_templ_img = prepare_image(template_img.copy())
        cls_test_dir = join(TEST_IMG_PATH, cls_name)
        for test_img_name in tqdm(os.listdir(cls_test_dir), desc="Iterating over test cases"):
            test_img = cv.imread(join(cls_test_dir, test_img_name))
            prepared_test_img = prepare_image(test_img.copy())
            descriptor_results = []

            # for detector_descriptor_name, detector_descriptor in tqdm(
            #         DETECTOR_DESCRIPTOR_MAP.items(), desc="Iterating over detectors and descriptros"):
            for detector_descriptor_name, detector_descriptor in tqdm(
                    create_param_grid().items(), desc="Performing grid search"):

                description_time_img1_begin = time()
                templ_kps, templ_decs = detector_descriptor.detectAndCompute(prepared_templ_img, None)
                description_time_img1_end = time()
                first_img_detection_time = description_time_img1_end - description_time_img1_begin

                description_time_img2_begin = time()
                test_kps, test_decs = detector_descriptor.detectAndCompute(prepared_test_img, None)
                description_time_img2_end = time()
                second_img_detection_time = description_time_img2_end - description_time_img2_begin

                matcher = cv.BFMatcher_create() if f"_{SIFT}" in detector_descriptor_name \
                    else cv.BFMatcher_create(cv.NORM_HAMMING, True)

                # if f"_{SIFT}" in detector_descriptor_name or f"-{SIFT}" in detector_descriptor_name:
                #

                feature_matching_time_begin = time()
                matches = matcher.match(templ_decs, test_decs)
                feature_matching_time_end = time()
                feature_matching_time = feature_matching_time_end - feature_matching_time_begin

                outlier_rejection_time_begin = time()
                best_matches = matchGMS(
                    template_img.shape[:2], test_img.shape[:2], templ_kps, test_kps, matches,
                    withRotation=True, withScale=True, thresholdFactor=GMS_THRESHOLD
                )
                outlier_rejection_time_end = time()
                outlier_rejection_time = outlier_rejection_time_end - outlier_rejection_time_begin

                descriptor_results.append([
                    detector_descriptor_name,
                    len(templ_kps), len(test_kps), len(matches), len(matches) - len(best_matches),
                    len(best_matches), first_img_detection_time, second_img_detection_time,
                    feature_matching_time, outlier_rejection_time,
                    # Total matching time
                    first_img_detection_time + second_img_detection_time + feature_matching_time
                    + outlier_rejection_time
                ])
            # Dump results
            with open(join(OUTPUT_PATH, "all_dfs.csv"), 'a') as f:
                columns = [f"{cls_name}-{test_img_name.split('.')[0]}"] + TABLE_COLUMNS
                pd.DataFrame(descriptor_results, columns=columns).round(4).to_csv(f)
                f.write("\n")
