"""coding: utf-8"""

import gzip
import os
import os.path
import pickle
from shutil import copyfileobj
from urllib.request import urlopen

import numpy as np

URL_BASE = 'http://yann.lecun.com/exdb/mnist/'
KEY_FILE = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

DATASET_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'
SAVE_FILE = DATASET_DIR + "/mnist.pkl"

TRAIN_NUM = 60000
TEST_NUM = 10000
IMAGE_DIM = (1, 28, 28)
IMAGE_SIZE = 784


def _download(file_name):
    """Download file specified with file_name.

    Arguments:
        file_name {string} -- file name of target
    """
    file_path = DATASET_DIR + "/" + file_name

    if os.path.exists(file_path):
        return

    full_path = URL_BASE + file_name

    print("Downloading " + file_name + " ... ")
    with urlopen(full_path) as in_stream, open(file_path, 'wb') as out_file:
        copyfileobj(in_stream, out_file)
    print("Done")


def _download_mnist():
    """Download all MNIST dataset.
    """
    for v in KEY_FILE.values():
        _download(v)


def _load_label(file_name):
    """Load label of from file specified with file_name.

    Arguments:
        file_name {string} -- file_name to specify target label file

    Returns:
        numpy.ndarray -- with its shape of (size,), indicates label
    """
    file_path = DATASET_DIR + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def _load_image(file_name):
    """Load image of from file specified with file_name.

    Arguments:
        file_name {string} -- file_name to specify target image file

    Returns:
        numpy.ndarray -- with its shape of (size, 28 * 28), indicates image measured by row
    """
    file_path = DATASET_DIR + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, IMAGE_SIZE)
    print("Done")

    return data


def _convert_to_numpy():
    """Convert MNITS dataset to dictionary of str => numpy.ndarray.

    Returns:
        dict -- dictionary of str => numpy.ndarray
    """
    dataset = {}

    dataset['train_img'] = _load_image(KEY_FILE['train_img'])
    dataset['train_label'] = _load_label(KEY_FILE['train_label'])
    dataset['test_img'] = _load_image(KEY_FILE['test_img'])
    dataset['test_label'] = _load_label(KEY_FILE['test_label'])

    return dataset


def init_mnist():
    """Download MNIT dataset, convert that to numpy, and make pickle of dataset dictionary.
    """
    # if dataset directory does not exist, make one
    if not os.path.exists(DATASET_DIR):
        os.mkdir(DATASET_DIR)

    _download_mnist()
    dataset = _convert_to_numpy()

    print("Creating pickle file ...")
    with open(SAVE_FILE, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(labels):
    """Change integer encoded label to one-hot encoded label.

    Arguments:
        X {numpy.ndarray} -- with its shape of (size,), array represent integer encoded label

    Returns:
        numpy.ndarray -- with its shape of (size, 10), array represent one-hot encoded label
    """
    onehot_labels = np.zeros((labels.size, 10))

    for idx, row in enumerate(onehot_labels):
        row[labels[idx]] = 1

    return onehot_labels


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """Load MNIST dataset.

    Keyword Arguments:
        normalize {bool} -- whether normalize pixel of image to [0.0, 1.0] (default: {True})
        flatten {bool} -- whether return flatten image (measured by row) (default: {True})
        one_hot_label {bool} -- whether return one-hot encoded label (default: {False})

    Returns:
        (n.n, n.n), (n.n, n.n) -- with its shape of ((60k, ?), (60k, ?)), ((10k, ?), (10k, ?))
    """
    if not os.path.exists(SAVE_FILE):
        init_mnist()

    with open(SAVE_FILE, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    train_tuple = (dataset['train_img'], dataset['train_label'])
    test_tuple = (dataset['test_img'], dataset['test_label'])

    return train_tuple, test_tuple
