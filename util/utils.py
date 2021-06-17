"""Utilities for scripts."""

import cv2 as cv
from cv2.xfeatures2d import (
    FREAK_create, StarDetector_create, BriefDescriptorExtractor_create)

from src.root_sift import RootSift
from util.constants import (
    ORB, BRISK, FAST, SIFT, ROOT_SIFT, AKAZE, STAR, FREAK,
    BRIEF, MAX_FEATURES, YOLO, TINY_YOLO, BRUTE_FORCE, FLANN, GMS, RANSAC)

CLASS_NAMES = ["1uah_heads", "2uah_heads", "5uah_heads", "10uah_heads"]
TEST_IMG_NAMES = ["front", "top", "bottom", "left", "right"]

FEATURE_DETECTORS = [ORB, BRISK, FAST, SIFT, ROOT_SIFT, AKAZE, STAR]

DETECTORS_MAP = {
    ORB: cv.ORB_create(MAX_FEATURES, fastThreshold=0),
    BRISK: cv.BRISK_create(),
    AKAZE: cv.AKAZE_create(),
    SIFT: cv.SIFT_create(MAX_FEATURES),
    ROOT_SIFT: RootSift(MAX_FEATURES),
    FAST: cv.FastFeatureDetector_create(),
    STAR: StarDetector_create()
}

FEATURE_DESCRIPTORS = [SIFT, ORB, BRISK, AKAZE, SIFT, ROOT_SIFT, FREAK, BRIEF]

DESCRIPTORS_MAP = {
    ORB: cv.ORB_create(MAX_FEATURES, fastThreshold=0),
    BRISK: cv.BRISK_create(),
    AKAZE: cv.AKAZE_create(),
    SIFT: cv.SIFT_create(MAX_FEATURES),
    ROOT_SIFT: RootSift(MAX_FEATURES),
    BRIEF: BriefDescriptorExtractor_create(),
    FREAK: FREAK_create()
}

FEATURE_MATCHERS = [BRUTE_FORCE, FLANN]

FLANN_INDEX_KDTREE = 1
INDEX_PARAMS = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
SEARCH_PARAMS = {"checks": 50}

OUTLIER_REJECTION_METHODS = [GMS, RANSAC]

OBJECT_DETECTORS = [YOLO, TINY_YOLO]


def sift_is_descriptor(detector_descriptor_name):
    # TODO: try to make it smarter with regexp.
    if f"_{SIFT}" in detector_descriptor_name \
            or f"-{SIFT}" in detector_descriptor_name\
            or detector_descriptor_name == SIFT:
        return True
    else:
        return False


def corner_case(detector, descriptor):
    if detector == AKAZE and descriptor != AKAZE \
            or detector != AKAZE and descriptor == AKAZE:
        return True
    if detector == ORB and descriptor in [SIFT, ROOT_SIFT]:
        return True
    if detector in [SIFT, ROOT_SIFT] and descriptor == ORB:
        return True
    if detector == FAST and descriptor in [SIFT, ROOT_SIFT]:
        return True
    return False