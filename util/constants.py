"""All constants from the project."""
from os.path import normpath, join, dirname

UNKNOWN_CLASS = "unknown"

# Path constants
REPO_ROOT = normpath(join(dirname(__file__), ".."))
DATA_PATH = join(REPO_ROOT, "data")
TEST_IMG_PATH = join(REPO_ROOT, "test_images")
FEATURES_PATH = join(REPO_ROOT, "features")
OUTPUT_PATH = join(REPO_ROOT, "output")
CONFIG_PATH = join(REPO_ROOT, "config")
LABELS_PATH = join(CONFIG_PATH, "coins.names")

# Feature detectors (extractors)
ORB = "ORB"
BRISK = "BRISK"
SIFT = "SIFT"
FAST = "FAST"
AKAZE = "AKAZE"
FAST = "FAST"
ROOT_SIFT = "Root-SIFT"
STAR = "STAR"

# Feature descriptors
FREAK = "FREAK"
BRIEF = "BRIEF"

# Feature matchers
BRUTE_FORCE = "BRUTE_FORCE"
FLANN = "FLANN"

# Outlier rejection methods
GMS = "GMS"
RANSAC = "RANSAC"

# Keypoint detector's / descriptor's parameters
MAX_FEATURES = 10_000
MIN_FEATURES = 15


# Object detectors
YOLO = "YOLO"
TINY_YOLO = "TINY_YOLO"

# YOLO detector's params
YOLO_CONFIDENCE = 0.5
NMS_THRESHOLD = 0.3