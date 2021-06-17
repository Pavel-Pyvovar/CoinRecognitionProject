import os
from os.path import join, exists
import pickle

import cv2 as cv
from cv2.xfeatures2d import matchGMS
import numpy as np

from src.detector_descriptor import DetectorDescriptor
from util.constants import BRISK, FEATURES_PATH, ROOT_SIFT, BRUTE_FORCE, MIN_FEATURES, UNKNOWN_CLASS, RANSAC, GMS
from util.utils import CLASS_NAMES, ORB, SIFT, INDEX_PARAMS, SEARCH_PARAMS

GMS_THRESHOLD = 4
RANSAC_THRESHOLD = 5.0
MIN_MATCHES = 10


class Classifier:
    def __init__(self, detector_name=BRISK, descriptor_name=BRISK,
                 matching_method=BRUTE_FORCE, outlier_rejection_method=GMS):
        self.detector_descriptor = DetectorDescriptor(detector_name, descriptor_name)
        self.detector_name = detector_name
        self.descriptor_name = descriptor_name
        self.features_path = join(FEATURES_PATH, f"{detector_name}_{descriptor_name}")
        self.matching_method = matching_method
        if matching_method == BRUTE_FORCE:
            self.matcher = cv.BFMatcher_create() if descriptor_name in [SIFT, ROOT_SIFT] \
                else cv.BFMatcher_create(cv.NORM_HAMMING, True)
        else:
            self.matcher = cv.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)
        self.outlier_rejection_method = outlier_rejection_method

    def classify(self, image):
        """Estimate a class of a coin on an image using keypoints and descriptions."""
        # Extracting features from input image
        input_points, input_descriptions = self.detector_descriptor.detect_describe(image)

        if input_descriptions is None or len(input_points) < MIN_FEATURES:
            return UNKNOWN_CLASS

        input_descriptions = input_descriptions if self.matching_method == BRUTE_FORCE \
            else np.float32(input_descriptions)

        scores = []
        # Searching for a match with a template
        for cls_name, (template_points, template_descriptions, template_img_shape)\
                in self.load_features().items():
            template_descriptions = template_descriptions if self.matching_method == BRUTE_FORCE \
                else np.float32(template_descriptions)
            # Feature matching
            matches = self.matcher.match(input_descriptions, template_descriptions)
            # Outlier rejection
            if self.outlier_rejection_method != "None":
                if self.outlier_rejection_method == RANSAC:
                    self.ransac_outlier_rejection(input_points, template_points, matches)
                else:
                    matches = matchGMS(image.shape[:2], template_img_shape, input_points, template_points, matches,
                                       withScale=True, withRotation=True, thresholdFactor=GMS_THRESHOLD)
            scores.append(len(matches))
        predicted_class = CLASS_NAMES[np.argmax(scores)] if max(scores) > MIN_MATCHES else UNKNOWN_CLASS
        return predicted_class


    def ransac_outlier_rejection(self, input_keypoints, template_keypoints, matches):
        keypoints1 = np.float32([input_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        keypoints2 = np.float32([template_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        _, mask = cv.findHomography(keypoints1, keypoints2, cv.RANSAC, RANSAC_THRESHOLD)
        matches_mask = mask.ravel()
        matches = np.array(matches)
        best_matches = matches[matches_mask == 1].tolist()
        return best_matches


    def load_features(self):
        class_features = {}
        for class_name_pickle in os.listdir(self.features_path):
            if class_name_pickle.rstrip(".pickle") not in CLASS_NAMES:
                continue

            with open(join(self.features_path, class_name_pickle), "rb") as pickle_file:
                keypoint_components, descriptions, shape = pickle.load(pickle_file)
            class_features.update({
                class_name_pickle.strip(".pickle"): (
                    [cv.KeyPoint(*args) for args in keypoint_components],
                    descriptions, shape
                )
            })
        return class_features

    def dump_features(self, class_name, image):
        if not exists(self.features_path):
            os.makedirs(self.features_path)

        keypoints, descriptions = self.detector_descriptor.detect_describe(image)

        def parse_keypoint(cv_keypoint):
            args = (
                *cv_keypoint.pt, cv_keypoint.size, cv_keypoint.angle,
                cv_keypoint.response, cv_keypoint.octave, cv_keypoint.class_id
            )
            return args

        keypoint_components = [parse_keypoint(point) for point in keypoints]

        with open(join(self.features_path, f"{class_name}.pickle"), 'wb') as pickle_file:
            pickle.dump((keypoint_components, descriptions, image.shape[:2]), pickle_file)

