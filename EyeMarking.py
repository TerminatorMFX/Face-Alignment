import logging
import math
from operator import itemgetter
from typing import Tuple

import cv2
import numpy as np

from Exceptions import TooMuchFacesException, TooMuchEyesException

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('eye_scanner')

EYE_CASCADE = cv2.CascadeClassifier(
    '/Users/maximilianandre/Software/IdeaProjects/FaceAlignment/venv/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml')
FACE_CASCADE = cv2.CascadeClassifier(
    '/Users/maximilianandre/Software/IdeaProjects/FaceAlignment/venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')


def detect_eyes(img):
    """
    Eye detection magic happens here. We fist detect faces and than the eyes will be detected.
    :param img:
    :return:
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces: np.ndarray = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=35, minSize=(700, 700))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x + w, y + int(h / 2)), (122, 122, 0), 2)

    LOGGER.info(f"Found {len(faces)} faces")
    if len(faces) < 1:
        raise TooMuchFacesException(f"Too many faces where found. {len(faces)} faces detected")

    elif len(faces) > 1:
        gray_dimensions = (gray.shape[0], gray.shape[1])
        faces = find_best_face(faces, gray_dimensions)

    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y + int(h / 2), x:x + w]
    roi_dimensions = (roi_gray.shape[0], roi_gray.shape[1])
    eye_x_y_list = find_eyes(roi_gray, x, y, roi_dimensions)

    for x, y in eye_x_y_list:
        cv2.circle(img, (int(x), int(y)), 50, (255, 255, 0), 2)
    return img, eye_x_y_list


def find_best_face(all_face, rectangle_size):
    face_distance = map_by_distance_to_middle(all_face, rectangle_size)
    return [face_distance[0][1]]


def map_by_distance_to_middle(points, rectangle_size):
    middle = (int(rectangle_size[0] / 2), int(rectangle_size[1] / 2))
    distance_with_point = []
    for face in points:
        (x, y, w, h) = face
        distance_to_middle = calculate_distance(middle, (x, y))
        distance_with_point.append((distance_to_middle, face))
    distance_with_point = sorted(distance_with_point, key=itemgetter(0))
    return distance_with_point


def find_best_fitting(all_eye, rectangle_size):
    eye_distance = map_by_distance_to_middle(all_eye, rectangle_size)
    return eye_distance[0][1], eye_distance[1][1]


def calculate_distance(point_1: Tuple[int, int], point_2: Tuple[int, int]) -> float:
    if len(point_1) != len(point_2):
        raise ValueError(f"Points have to be in the same dimension. Point 1: {len(point_1)}, Point 2: {len(point_2)}")
    raw_dist = 0
    for index in range(len(point_1)):
        raw_dist += math.pow((point_1[index] - point_2[index]), 2)

    return math.sqrt(raw_dist)


def find_eyes(roi_gray, crop_x: int, crop_y: int, rectangle_size: Tuple[int]):
    eyes = EYE_CASCADE.detectMultiScale(roi_gray, scaleFactor=1.02, minNeighbors=25, minSize=(100, 100))
    eye_x_y_list = []
    LOGGER.info(f"Found {len(eyes)} eyes")
    if len(eyes) < 2:
        raise TooMuchEyesException(f"Too many Eyes where found. {len(eyes)} eyes detected")
    elif len(eyes) > 2:
        LOGGER.warning(f"Found {len(eyes)}, will search for the two best.")
        eyes = find_best_fitting(eyes, rectangle_size)
    for (ex, ey, ew, eh) in eyes:
        eye_x_y_list.append((crop_x + ex + ew * 0.5, crop_y + ey + eh * 0.5))
    return eye_x_y_list


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
    img = cv2.imread("IMG_ - 87.jpeg")

    img, _, _, _ = scale_and_extract_information_from(img)

    img = resize_image(img, 660, 808)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
