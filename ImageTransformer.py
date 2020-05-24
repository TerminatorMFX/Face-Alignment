import glob
import logging
import os
import shutil
import sys

import cv2

from Exceptions import TooMuchEyesException, TooMuchFacesException
from EyeMarking import transform_image

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('image_transformer')


def get_all_jpeg_files(path_to_images):
    all_jpeg = []
    for file in glob.glob(os.path.join(f"{path_to_images}", "*.jpeg")):
        all_jpeg.append(file)

    all_jpeg.sort()
    return all_jpeg


def load_and_transform(path):
    LOGGER.info(f"Transforming image {path}")
    img = cv2.imread(path)
    try:
        img = transform_image(img)
        split_path = path.rsplit("/", 1)
        new_image_path = os.path.join(split_path[0], "transformed", split_path[1])
        cv2.imwrite(new_image_path, img)
    except TooMuchEyesException as e:
        LOGGER.warning(f"Image with path: {path}, has to much eyes. {e}")
        not_transformed(path)
    except TooMuchFacesException as e:
        LOGGER.warning(f"Image with path: {path}, has to much faces. {e}")
        not_transformed(path)


def not_transformed(path_to_file):
    split_path = path_to_file.rsplit("/", 1)
    new_image_path = os.path.join(split_path[0], "not_transformed", split_path[1])
    shutil.copyfile(path_to_file, new_image_path)


def load_and_transform_all(paths):
    for path in paths:
        load_and_transform(path)


def prepare(path):
    transformed = os.path.join(path, "transformed")
    not_transformed = os.path.join(path, "not_transformed")
    if not os.path.isdir(transformed):
        os.mkdir(transformed)
    if not os.path.isdir(not_transformed):
        os.mkdir(not_transformed)


if __name__ == '__main__':
    path_to_images = sys.argv[1]
    prepare(path_to_images)

    jpegs_for_transforming = get_all_jpeg_files(path_to_images)
    load_and_transform_all(jpegs_for_transforming)
