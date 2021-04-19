import cv2 as cv
from cv2.xfeatures2d import matchGMS
from enum import Enum
import numpy as np
import os
from os.path import join
import pandas as pd
import pickle
import sys

DESCRIPTORS_MAP = {
    "BRISK": cv.BRISK_create(),
    "ORB": cv.ORB_create(nfeatures=10000, scoreType=cv.ORB_FAST_SCORE)
}
MIN_MATCHES = 10


def preprocess_image(image):
    # Resize
    # destination = np.zeros((80, 80))
    # destination = np.zeros((120, 120))
    destination = np.zeros((416, 416))
    resized = cv.resize(image, destination.shape, destination, 0, 0, cv.INTER_AREA)
    unsharped = unsharp(resized)
    return unsharped


def unsharp(img):
    blurred_img = cv.GaussianBlur(img, (3, 3), 10)
    unshapred = cv.addWeighted(img.copy(), 2, blurred_img, -1, 0, img.copy())
    return unshapred


def extract_features(image, descriptor_name="BRISK"):
    descriptor = DESCRIPTORS_MAP[descriptor_name]
    keypoints, features = descriptor.detectAndCompute(image, None)
    return keypoints, features


def dump_features(name, keypoints, features):
    if not os.path.exists("descriptions"):
        os.mkdir("descriptions")

    descriptions = []
    for point, feature in zip(keypoints, features):
        descriptions.append(
            (point.pt, point.size, point.angle, point.response,
                point.octave, point.class_id, feature))

    # Try saving with numpy
    with open(join("descriptions", f"{name}.pickle"), 'wb') as pickle_file:
        pickle.dump(descriptions, pickle_file)


def load_features(name):
    with open(join("descriptions", f"{name}.pickle"), "rb") as pickle_file:
        descriptions = pickle.load(pickle_file)

    keypoints, features = [], []
    for row in descriptions:
        point, size, angle, response, octave, class_id, feature = row
        keypoints.append(cv.KeyPoint(*point, size, angle, response, octave, class_id))
        features.append(feature)

    return keypoints, np.array(features)


def evaluate_descriptor(src_img_size, dst_img_size, src_keypoints, dst_keypoints, matches, outlier_rejection="RANSAC"):
    if outlier_rejection == "RANSAC":
        src_pts = np.float32([src_keypoints[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([dst_keypoints[m.trainIdx].pt for m in matches])
        _, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 4.0)
        outliers_cnt = len(matches) - np.sum(mask)
    else:
        gms_matches = matchGMS(src_img_size, dst_img_size, src_keypoints, dst_keypoints, matches,
                                thresholdFactor=6)
        outliers_cnt = len(matches) - len(gms_matches)
    summary_table = {
        "Features detected in the first image": len(src_keypoints),
        "Features detected in the second image": len(dst_keypoints),
        "Features matched": len(matches),
        "Outliers rejected": outliers_cnt
    }

    print(summary_table)


def gms_matches(img1, img2, img1_kps, img2_kps, matches):
    gms_matches = cv.xfeatures2d.matchGMS(img1.shape[:2], img2.shape[:2], img1_kps, img2_kps, matches,
                                          thresholdFactor=6)
    output = draw_matches_gms(img1, img2, img1_kps, img2_kps, gms_matches, DrawingType.ONLY_LINES)
    cv.imshow("GMS Matches", output)
    cv.waitKey(0)
    cv.destroyAllWindows()


def draw_matches(src_img, dst_img, src_keypoints, dst_keypoints, matches):
    src_pts = np.float32([src_keypoints[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([dst_keypoints[m.trainIdx].pt for m in matches])

    homography_matrix, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 4.0)
    mask = mask.ravel().tolist()

    height, width = src_img.shape
    points = np.float32([[0, 0], [0, height-1], [width-1, width-1], [width-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(points, homography_matrix)

    dst_img = cv.polylines(dst_img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=mask,  # draw only inliers
                       flags=2)

    img3 = cv.drawMatches(src_img, src_keypoints, dst_img, dst_keypoints, matches, None, **draw_params)
    cv.imshow("BRISK matches with RANSAC", img3)
    cv.waitKey(0)
    cv.destroyAllWindows()

class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    COLOR_CODED_POINTS_X = 3
    COLOR_CODED_POINTS_Y = 4
    COLOR_CODED_POINTS_XpY = 5


def draw_matches_gms(src1, src2, kp1, kp2, matches, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    if drawing_type == DrawingType.ONLY_LINES:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

    elif drawing_type == DrawingType.LINES_AND_POINTS:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv.applyColorMap(_1_255, cv.COLORMAP_HSV)

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))

            if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                colormap_idx = int((left[0] - src1.shape[1]*.5 + left[1] - src1.shape[0]*.5) * 256. / (src1.shape[0]*.5 + src1.shape[1]*.5))  # manhattan gradient

            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv.circle(output, tuple(map(int, left)), 1, color, 2)
            cv.circle(output, tuple(map(int, right)), 1, color, 2)
    return output


if __name__ == '__main__':
    # Load data
    template_img = cv.imread(join("data", "10uah.jpg"))
    # real_img = cv.imread(join("data", "10uah_15degrees.jpg"))
    real_img = cv.imread(join("data", "10uah_rotated.jpg"))

    # template_kps, templates_feats = load_features("BRISK_10uah")
    # real_kps, real_feats = load_features("BRISK_10uah_15degrees")

    # Preprocessing
    template_img = preprocess_image(template_img)
    real_img = preprocess_image(real_img)

    # Extract features
    template_kps_brisk, templates_feats_brisk = extract_features(template_img)
    template_kps_orb, templates_feats_orb = extract_features(template_img, "ORB")
    # dump_features("BRISK_10uah", template_kps, templates_feats)

    real_kps_brisk, real_feats_brisk = extract_features(real_img)
    real_kps_orb, real_feats_orb = extract_features(real_img, "ORB")
    # dump_features("BRISK_10uah_15degrees", real_kps, real_feats)

    # cv.imshow("Real image keypoints", cv.drawKeypoints(real_img, real_kps_brisk, None))
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Match keypoints
    matcher = cv.BFMatcher_create(cv.NORM_HAMMING, crossCheck=True)
    brisk_matches = matcher.match(templates_feats_brisk, real_feats_brisk)
    orb_matches = matcher.match(templates_feats_orb, real_feats_orb)

    # Compare descriptors
    # if len(brisk_matches) > MIN_MATCHES:
    #     evaluate_descriptor(template_kps_brisk, real_kps_brisk, brisk_matches)
    #
    # if len(orb_matches) > MIN_MATCHES:
    #     evaluate_descriptor(template_kps_orb, real_kps_orb, orb_matches)

    # evaluate_descriptor(template_img.shape, real_img.shape, template_kps_brisk, real_kps_brisk, brisk_matches,
    #                     outlier_rejection="GMS")
    # evaluate_descriptor(template_img.shape, real_img.shape, template_kps_orb, real_kps_orb, orb_matches,
    #                     outlier_rejection="GMS")

    gms_matches(template_img, real_img, template_kps_orb, real_kps_orb, orb_matches)

    # draw_matches(template_img, real_img, template_kps_brisk, real_kps_brisk, brisk_matches)

    # # Visualize results
    # cv.imshow("ORB matches", cv.drawMatches(rotated, real_kps, template_img, template_kps, matches, None))
    # cv.waitKey(0)
    # cv.destroyAllWindows()
