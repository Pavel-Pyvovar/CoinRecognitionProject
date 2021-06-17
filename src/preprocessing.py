import cv2 as cv

# Image shape to use for preprocessing (resizing)
IMG_WIDTH_SIZE = 416
IMG_HEIGHT_SIZE = 320
RESIZE_INTERPOLATION_METHOD = cv.INTER_AREA


def extract_detection(image, bboxes):
    object_crops = []
    padding_ratio = 0.1
    for bbox in bboxes:
        x, y, w, h = bbox
        x_padding = padding_ratio * w
        y_padding = padding_ratio * y
        padded_image = image[
           int(y-y_padding):int(y+h+y_padding), int(x-x_padding):int(x+w+x_padding)]
        object_crops.append(padded_image)
    return object_crops


def _image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # TODO: use IMG_WIDTH_SIZE constant from above
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)
    return resized


def _resize(image):
    # TODO: redo
    # If new image size is not specified, return original image.
    if IMG_WIDTH_SIZE is None and IMG_HEIGHT_SIZE is None:
        return image

    # If input image already has necessary dimensions, return it.
    (current_height, current_width) = image.shape[:2]
    if current_height == IMG_WIDTH_SIZE and current_width == IMG_HEIGHT_SIZE:
        return image

    # Define new shape of the image.
    if current_width > current_height:
        ratio = IMG_WIDTH_SIZE / float(current_width)
        new_shape = (int(ratio*current_height), IMG_WIDTH_SIZE)
    else:
        ratio = IMG_HEIGHT_SIZE / float(current_height)
        new_shape = (IMG_HEIGHT_SIZE, int(ratio*current_width))

    # Resize image
    resized = cv.resize(image, tuple(reversed(new_shape)),
                        interpolation=RESIZE_INTERPOLATION_METHOD)
    # result = np.zeros((IMG_HEIGHT_SIZE, IMG_WIDTH_SIZE, 3), dtype=np.uint8)
    # result[:resized.shape[0], :resized.shape[1]] += resized
    # return result
    return resized


def _bgr_equalization(bgr_image):
    blue, green, red = cv.split(bgr_image)
    blue = cv.equalizeHist(blue)
    green = cv.equalizeHist(green)
    red = cv.equalizeHist(red)
    equalized_bgr_image = cv.merge((blue, green, red))
    return equalized_bgr_image


def _unsharp(img):
    blurred_img = cv.GaussianBlur(img, (3, 3), 10)
    unshapred = cv.addWeighted(img.copy(), 2, blurred_img, -1, 0, img.copy())
    return unshapred


def prepare_image(image, resize=True):
    if resize:
        # image = _resize(image)
        image = _image_resize(image, height=416)
    image = _bgr_equalization(image)
    image = _unsharp(image)
    image = cv.GaussianBlur(image, (3, 3), 0)
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return image