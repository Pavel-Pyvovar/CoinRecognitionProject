from util.utils import DETECTORS_MAP, DESCRIPTORS_MAP


class DetectorDescriptor:
    """Class for combining different detectors and descriptors."""
    def __init__(self, detector_name, descriptor_name):
        """Initialize feature detector and feature descriptor."""
        self.detector = DETECTORS_MAP[detector_name]
        self.descriptor = DESCRIPTORS_MAP[descriptor_name]

    def detect_describe(self, img):
        """Convenience method to combine feature detection with feature description."""
        keypoints = self.detector.detect(img)
        _, descriptions = self.descriptor.compute(img, keypoints)
        return keypoints, descriptions
