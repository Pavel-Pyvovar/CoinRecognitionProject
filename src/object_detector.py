from os.path import join

import cv2 as cv
from cv2.dnn import blobFromImage, blobFromImages
import numpy as np

from util.constants import CONFIG_PATH, YOLO_CONFIDENCE, NMS_THRESHOLD

IMG_SIDE_SIZE = 416
SCALE_FACTOR = 1/255.0


class Yolo:
    """Custom class for using YOLO with OpenCV"""

    def __init__(self, model_name, confidence=YOLO_CONFIDENCE, nms_threshold=NMS_THRESHOLD):
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.model = cv.dnn.readNetFromDarknet(
            join(CONFIG_PATH, f"{model_name.lower()}.cfg"),
            join(CONFIG_PATH, f"{model_name.lower()}.weights"))
        self.output_layer_names = None
        self.height = None
        self.width = None

    def load_data(self, data):
        self.output_layer_names = self.model.getLayerNames()
        self.output_layer_names = [
            self.output_layer_names[i[0] - 1] for i in self.model.getUnconnectedOutLayers()
        ]
        if len(data.shape) == 4:
            self.height, self.width = data.shape[1:3]
            # TODO: image shape should not be hard coded.
            blob = blobFromImages(data, SCALE_FACTOR, (IMG_SIDE_SIZE, IMG_SIDE_SIZE),
                                  swapRB=True, crop=False)
            self.model.setInput(blob)
        else:
            self.height, self.width = data.shape[:2]
            blob = blobFromImage(data, SCALE_FACTOR, (IMG_SIDE_SIZE, IMG_SIDE_SIZE),
                                 swapRB=True, crop=False)
            self.model.setInput(blob)

    def detect(self):
        if self.output_layer_names:
            layer_outputs = self.model.forward(self.output_layer_names)
        else:
            raise ValueError("Invalid output layer names! Please, run Yolo.load_data first.")
        return layer_outputs

    def process_outputs(self, layer_outputs):
        bboxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence:
                    bbox = detection[:4] * np.array([self.width, self.height, self.width, self.height])
                    center_x, center_y, w, h = bbox.astype("int")

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    bboxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv.dnn.NMSBoxes(bboxes, confidences, self.confidence, self.nms_threshold)
        if len(indexes) > 0:
            best_bboxes = [bboxes[i] for i in indexes.flatten() if len(indexes) > 0]
        else:
            best_bboxes = []
        return best_bboxes
        # return bboxes
