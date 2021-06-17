import argparse
import os.path
from time import time

import cv2 as cv
from os.path import join, isfile

from src.classifier import Classifier
from util.constants import BRISK, OUTPUT_PATH, YOLO, BRUTE_FORCE, GMS
from util.utils import (OBJECT_DETECTORS, FEATURE_DETECTORS, FEATURE_DESCRIPTORS,
                        FEATURE_MATCHERS, OUTLIER_REJECTION_METHODS)
from object_detector import Yolo
from src.preprocessing import prepare_image, extract_detection

# TODO: add logging


def _parse_args():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-i", "--image", required=False,
                           help="Path to an image to perform inference on.")
    arg_parse.add_argument("-v", "--video", required=False,
                           help="Provide real-time option if you want to perform inference in real time."
                                "Otherwise, provide path to a video to perform inference on.")
    arg_parse.add_argument("-s", "--save-output", default="no", required=False,
                           help="Flag for saving output image or video."
                                " Provide yes for saving output")
    arg_parse.add_argument("-o", "--object-detector", default=YOLO, choices=OBJECT_DETECTORS,
                           help="The object detector to use for localizing coins.")
    arg_parse.add_argument("-c", "--use-classification", default="no",
                           help="Provide yes if you want to classify coins after their detection and"
                                "any other key or word no to do that.")
    arg_parse.add_argument("-e", "--feature-detector", default=BRISK, choices=FEATURE_DETECTORS,
                           help="The name of the keypoint detector to use for feature extraction.")
    arg_parse.add_argument("-d", "--feature-descriptor", default=BRISK, choices=FEATURE_DESCRIPTORS,
                           help="The name of the keypoint descriptor to use for feature description.")
    arg_parse.add_argument("-m", "--feature-matcher", default=BRUTE_FORCE,
                           choices=FEATURE_MATCHERS,
                           help="The name of the method to use for feature matching.")
    arg_parse.add_argument("-r", "--outlier-rejection-method", default=GMS,
                           choices=OUTLIER_REJECTION_METHODS + ["None"],
                           help="Method to use for outlier rejection. Provide None if you don't want to use"
                                "outlier rejection.")
    arg_parse.add_argument("--debug", default="no",
                           help="Provide yes to run the system in debug mode.")
    # Interactive vs detailed classification
    # Enable displaying keypoints and matches (detailed classification)
    # TODO: add debug option for more detailed visualization
    return arg_parse


def draw_bboxes_with_classes(image, bboxes, classifier):

    for bbox, crop in zip(bboxes, extract_detection(image, bboxes)):
        predicted_class = classifier.classify(crop) if classifier else None
        x, y, w, h = bbox
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(image, predicted_class, (x, y),
                   fontScale=1, fontFace=cv.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), thickness=2)
    return image


if __name__ == '__main__':
    parser = _parse_args()
    args = parser.parse_args()

    if args.image is None and args.video is None:
        raise ValueError(
            "Neither image nor video arguments are provided! Please,"
            " specify either the path to an image or video or both."
        )

    # TODO: supply nms threshold and confidence for YOLO as script arguments
    object_detector = Yolo(args.object_detector)
    classifier = Classifier(args.feature_detector, args.feature_descriptor,
                            args.feature_matcher, args.outlier_rejection_method)
    if args.image:
        if not isfile(args.image):
            raise ValueError("The provided path to an image does not exist! "
                             "Please, provide a path relative to the project:) "
                             "The directory with test images is test_images/")

        image = cv.imread(args.image)
        # Image resizing is not necessary,
        # OpenCV's YOLO interface takes care of that.
        prepared_image = prepare_image(image.copy(), resize=False)
        img_for_viz = image.copy()

        object_detector.load_data(prepared_image)
        layer_outputs = object_detector.detect()
        bboxes = object_detector.process_outputs(layer_outputs)
        if args.use_classification == "yes":
            img_for_viz = draw_bboxes_with_classes(img_for_viz, bboxes, classifier)
        else:
            img_for_viz = draw_bboxes_with_classes(img_for_viz, bboxes, None)
        cv.imshow("Detections", img_for_viz)
        cv.waitKey(0)

    if args.video:
        if args.video == "real-time":
            cap = cv.VideoCapture(0)
        else:
            if not isfile(args.video):
                raise ValueError("The provided path to a video does not exist! "
                                 "Please, provide a path relative to the project:) "
                                 "The directory with test videos is test_videos/")
            cap = cv.VideoCapture(args.video)

        if args.save_output == "yes":
            fourcc = cv.VideoWriter_fourcc(*"MJPG")

            if not os.path.isdir(OUTPUT_PATH):
                os.makedirs(OUTPUT_PATH)

            writer = cv.VideoWriter(join(OUTPUT_PATH, "output.avi"), fourcc, 20.,
                                    # (frame_width, frame_height)
                                    (int(cap.get(3)), int(cap.get(4))), True)

        width, height = None, None
        prev_time = 0

        while cap.isOpened():
            grabbed, frame = cap.read()
            if not grabbed:
                break
            if width is None or height is None:
                height, width = frame.shape[:2]

            prepared_frame = prepare_image(frame.copy(), resize=False)

            object_detector.load_data(prepared_frame)
            layer_outputs = object_detector.detect()
            bboxes = object_detector.process_outputs(layer_outputs)

            if args.use_classification == "yes":
                frame_for_viz = draw_bboxes_with_classes(frame.copy(), bboxes, classifier)
            else:
                frame_for_viz = draw_bboxes_with_classes(frame.copy(), bboxes, None)

            if args.save_output == "yes":
                writer.write(frame_for_viz)

            cur_time = time()
            sec = cur_time - prev_time
            prev_time = cur_time

            fps = 1 / sec
            cv.putText(frame_for_viz, f"FPS: {fps:.4f}", (0, 100),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

            cv.imshow("Predictions", frame_for_viz)
            if cv.waitKey(1) == ord('q'):
                break
        if args.video != "real-time":
            writer.release()
        cap.release()

    cv.destroyAllWindows()


