import os

import cv2 as cv
from os.path import join

from src.classifier import Classifier
from src.inference import draw_bboxes_with_classes
from src.preprocessing import prepare_image, extract_detection, IMG_WIDTH_SIZE
from util.constants import TEST_IMG_PATH, ORB, YOLO_CONFIDENCE, NMS_THRESHOLD
from src.object_detector import Yolo

TEMPLATE_IMGS = ["1uah.jpg", "2uah.jpg", "5uah.jpg", "10uah.jpg"]

if __name__ == '__main__':
    for test_image in os.listdir(join(TEST_IMG_PATH, "5uah_heads")):
        yolo = Yolo(confidence=YOLO_CONFIDENCE, nms_threshold=NMS_THRESHOLD)
        yolo.load_model()
        image = cv.imread(join(TEST_IMG_PATH, "5uah_heads", test_image))
        prepared_image = prepare_image(image.copy())
        img_for_viz = image.copy()
        # prepared_image = image
        yolo.load_data(prepared_image)
        layer_outputs = yolo.detect()
        bboxes = yolo.process_outputs(layer_outputs)
        clf = Classifier(ORB)
        draw_bboxes_with_classes(img_for_viz, bboxes, clf)
        cv.imshow("Detections", img_for_viz)
        cv.waitKey(0)
    cv.destroyAllWindows()