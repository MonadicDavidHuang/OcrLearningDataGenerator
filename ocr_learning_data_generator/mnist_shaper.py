"""coding: utf-8"""

import os
import os.path
import pickle
import sys

import numpy as np

from mnist_loader import load_mnist

DATASET_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'
SAVE_DIR = DATASET_DIR + "/edgeless_mnist_dictionary.pkl"

ROW_SIZE = 28
COLUMN_SIZE = 28


def make_white_edgeless_mnist_dictionary():
    """Load MNIST dataset and convert to map of int => [numpy.ndarray] for latter usage.

    Returns:
        dictionary -- map of int => [numpy.ndarray]
    """
    ret = {}

    (train_img, train_label), (test_img, test_label) = load_mnist(flatten=False)

    print('Formatting train data...')
    for i, label in enumerate(train_label):
        sys.stdout.write(str('\r') + str(i + 1) + '/' + str(len(train_label)))
        sys.stdout.flush()

        image_without_white_edges = cut_white_edges(train_img[i, 0])
        if not label in ret:
            ret[label] = [image_without_white_edges]
        else:
            ret[label].append(image_without_white_edges)
    print('\n')

    print('Formatting test data...')
    for i, label in enumerate(test_label):
        sys.stdout.write(str('\r') + str(i + 1) + '/' + str(len(test_label)))
        sys.stdout.flush()

        image_without_white_edges = cut_white_edges(test_img[i, 0])

        if not label in ret:
            ret[label] = [image_without_white_edges]
        else:
            ret[label].append(image_without_white_edges)
    print('\n')

    return ret


def cut_white_edges(image):
    """Cut white space from both edges.
    Arguments:
        image {numpy.ndarray} -- with its shape of (28, 28), indicates image measured by row

    Raises:
        ValueError: when condition is contradiction

    Returns:
        numpy.ndarray -- with its shape of (28, 28), indicates white edgeless image measured by row
    """
    left = 0
    for i in range(ROW_SIZE):
        if contains_no_zero_at(i, image):
            left = i
            break

    right = ROW_SIZE - 1
    for i in range(ROW_SIZE - 1, -1, -1):
        if contains_no_zero_at(i, image):
            right = i
            break

    if left > right:
        raise ValueError("The condition is contradiction!")

    return image[:, left:right]


def contains_no_zero_at(column, image):
    """Check whther contains no zero at given column.

    Arguments:
        column {int} -- column
        image {numpy.ndarray} -- with its shape of (28, 28), indicates image measured by row

    Returns:
        [bool] -- if contains zero at given column then True else false
    """
    for i in range(COLUMN_SIZE):
        if abs(image[i][column]) >= np.float(1e-5):
            return True
    return False


def load_white_edgeless_mnist_dictionary():
    """Load white edgeless MNIST dictionary

    Returns:
        dictionary -- map of int => [numpy.ndarray]
    """
    if not os.path.exists(SAVE_DIR):
        white_edgeless_mnist_dictionary = make_white_edgeless_mnist_dictionary()

        print("Creating pickle file ...")
        with open(SAVE_DIR, 'wb') as f:
            pickle.dump(white_edgeless_mnist_dictionary, f, -1)
        print("Done!")

    with open(SAVE_DIR, 'rb') as f:
        white_edgeless_mnist_dictionary = pickle.load(f)

    return white_edgeless_mnist_dictionary
