import cv2 as cv
from os.path import join

from src.preprocessing import prepare_image
from util.constants import TEST_IMG_PATH

if __name__ == '__main__':
    original_image = cv.imread(join(TEST_IMG_PATH, "other", "1_2_5_10_uah_heads.jpg"))
    cv.imshow("Original image", original_image)
    processed_image = prepare_image(original_image)
    cv.imshow("Processed image", processed_image)
    cv.waitKey(0)
    cv.destroyAllWindows()