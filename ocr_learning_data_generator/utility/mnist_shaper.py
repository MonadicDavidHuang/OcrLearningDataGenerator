"""coding: utf-8"""

import os
import os.path
import pickle
import sys

import numpy as np

from .mnist_loader import load_mnist

DATASET_DIR = '/tmp/ocr_learning_data_generator'
SAVE_DIR = DATASET_DIR + "/edgeless_mnist_dictionary.pkl"

ROW_SIZE = 28
COLUMN_SIZE = 28


def _make_white_edgeless_mnist_dictionary():
    """Load MNIST dataset and convert to dict of digit => [numpy.ndarray] for latter usage.

    Returns:
        dict -- dict of digit => [numpy.ndarray]
    """
    ret = {}

    (train_img, train_label), (test_img, test_label) = load_mnist(flatten=False)

    print('Formatting train data...')
    for i, label in enumerate(train_label):
        sys.stdout.write(str('\r') + str(i + 1) + '/' + str(len(train_label)))
        sys.stdout.flush()

        image_without_white_edges = _cut_white_edges(train_img[i, 0])
        if not label in ret:
            ret[label] = [image_without_white_edges]
        else:
            ret[label].append(image_without_white_edges)
    print('\n')

    print('Formatting test data...')
    for i, label in enumerate(test_label):
        sys.stdout.write(str('\r') + str(i + 1) + '/' + str(len(test_label)))
        sys.stdout.flush()

        image_without_white_edges = _cut_white_edges(test_img[i, 0])

        if not label in ret:
            ret[label] = [image_without_white_edges]
        else:
            ret[label].append(image_without_white_edges)
    print('\n')

    return ret


def _cut_white_edges(image):
    """Cut white space from both edges.
    Arguments:
        image {numpy.ndarray} -- with shape of (28, 28), indicates image measured by row

    Raises:
        ValueError: when condition is contradiction

    Returns:
        numpy.ndarray -- with shape of (28, ? <= 28), indicates white edgeless image measured by row
    """
    left = 0
    for i in range(ROW_SIZE):
        if _contains_no_zero_at(i, image):
            left = i
            break

    right = ROW_SIZE - 1
    for i in range(ROW_SIZE - 1, -1, -1):
        if _contains_no_zero_at(i, image):
            right = i
            break

    if left > right:
        raise ValueError("The condition is contradiction!")

    return image[:, left:right]


def _contains_no_zero_at(column, image):
    for i in range(COLUMN_SIZE):
        if abs(image[i][column]) >= np.float(1e-5):
            return True
    return False


def load_white_edgeless_mnist_dictionary():
    """Load white edgeless MNIST dictionary

    Returns:
        dict -- dict of int => [numpy.ndarray]
    """
    if not os.path.exists(SAVE_DIR):
        white_edgeless_mnist_dictionary = _make_white_edgeless_mnist_dictionary()

        print("Creating pickle file ...")
        with open(SAVE_DIR, 'wb') as out_file:
            pickle.dump(white_edgeless_mnist_dictionary, out_file, -1)
        print("Done!")
    else:
        with open(SAVE_DIR, 'rb') as out_file:
            white_edgeless_mnist_dictionary = pickle.load(out_file)

    return white_edgeless_mnist_dictionary
