import logging
import math

import cv2
import numpy as np

from Exceptions import TooMuchFacesException, TooMuchEyesException

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('eye_scanner')


def detect_eyes(img):
    """
    Eye detection magic happens here. We fist detect faces and than the eyes will be detected.
    :param img:
    :return:
    """
    face_cascade = cv2.CascadeClassifier(
        '/Users/maximilianandre/Software/IdeaProjects/FaceAlignment/venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        '/Users/maximilianandre/Software/IdeaProjects/FaceAlignment/venv/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces: np.ndarray = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=35, minSize=(700, 700))

    eye_x_y_list = []
    LOGGER.info(f"Found {len(faces)} faces")
    if len(faces) == 1 and faces.shape[0] == 1:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x + w, y + int(h / 2)), (122, 122, 0), 2)
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y + int(h / 2), x:x + w]
        roi_color = img[y:y + int(h / 2), x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.02, minNeighbors=25, minSize=(100, 100))
        LOGGER.info(f"Found {len(eyes)} eyes")
        if len(eyes) == 0 or eyes.shape[0] != 2:
            raise TooMuchEyesException(f"Too many Eyes where found. {len(eyes)} eyes detected")
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)
            eye_x_y_list.append((x + ex + ew * 0.5, y + ey + eh * 0.5))
    else:

        raise TooMuchFacesException(f"Too many faces where found. {len(faces)} faces detected")

    return img, eye_x_y_list


def find_left_eye(eye_coordinates):
    left_coordinate = [math.inf, 0]
    for eye_coordinate in eye_coordinates:
        left_coordinate = eye_coordinate if eye_coordinate[0] < left_coordinate[0] else left_coordinate

    return left_coordinate


def eye_translation(img, eye_coordinates, x_scaling_factor, y_scaling_factor):
    left_eye_coordinate = find_left_eye(eye_coordinates)
    should_x = 905 * x_scaling_factor
    should_y = 1350 * y_scaling_factor
    translation_x = should_x - left_eye_coordinate[0] * x_scaling_factor
    translation_y = should_y - left_eye_coordinate[1] * y_scaling_factor
    rows, cols, ch = img.shape

    M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    return cv2.warpAffine(img, M, (cols, rows))


def angle_of_points(point_1: (int, int), point_2: (int, int)) -> float:
    """
    Will calculate the sharp angle of x-achsis and the line through the two points
    :param point_1: (x,y)
    :param point_2: (x,y)
    :return: Returns the angle in radian
    """
    delta_y = (point_2[1] - point_1[1])
    delta_x = (point_2[0] - point_1[0])
    if delta_x == 0:  # for catching lines on the x-Axis
        return 0.0
    m = delta_y / delta_x
    return np.arctan(m)


def resize_image(img, x_scale, y_scale):
    """
    Scales image down so it can be better processed
    :param img:
    :param x_scale:
    :param y_scale:
    :return:
    """
    LOGGER.info(f"New X and Y Scale for Image: x: {x_scale}, y: {y_scale}")
    return cv2.resize(img, (x_scale, y_scale), interpolation=cv2.INTER_CUBIC)


def rotate_image(img, degree):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), degree, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def calculate_scaling_factor(img, x_scale, y_scale):
    """
    Will calculate how much the image has to be scaled up. Is for convinience on later images
    :param img:
    :param x_scale:
    :param y_scale:
    :return:
    """
    height, width = img.shape[:2]
    x_scaling_factor = height / x_scale
    y_scaling_factor = width / y_scale
    return x_scaling_factor, y_scaling_factor


def scale_and_extract_information_from(img):
    x_scale = 2320
    y_scale = 3088
    x_scaling_factor, y_scaling_factor = calculate_scaling_factor(img, x_scale, y_scale)

    detection_img = resize_image(img, x_scale, y_scale)
    debug_information_img, eye_list = detect_eyes(detection_img)
    return debug_information_img, eye_list, x_scaling_factor, y_scaling_factor


def transform_image(img):
    detection_img = img.copy()

    _, eye_list, x_scaling_factor, y_scaling_factor = scale_and_extract_information_from(detection_img)

    radian_of_rotation = angle_of_points(eye_list[0], eye_list[1])
    degree_of_rotation = radian_of_rotation * 180 / np.pi

    LOGGER.info(f"Will rotate image to {degree_of_rotation} degrees")
    img = rotate_image(img, degree_of_rotation)

    dst = eye_translation(img, eye_list, x_scaling_factor, y_scaling_factor)
    height, width, ch = dst.shape
    dst = dst[100:height - 350, 100:width - 300]

    return dst


if __name__ == '__main__':
    img = cv2.imread("IMG_ - 16.jpeg")

    img, _, _, _ = scale_and_extract_information_from(img)

    img = resize_image(img, 660, 808)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
