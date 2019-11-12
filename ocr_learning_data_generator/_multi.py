from multiprocessing import Pool
from multiprocessing import Manager
import multiprocessing as multi

from random import choice

import numpy as np

import time


def create_digit_sequence(number, image_width, min_spacing, max_spacing):
    """ A function that create an image representing the given number, 
    with random spacing between the digits.
    Each digit is randomly sampled from the MNIST dataset.
    Returns an NumPy array representing the image.

    Parameters----------
        number: strA string representing the number, e.g. "14543"
        image_width: intThe image width (in pixel).
        min_spacing: intThe minimum spacing between digits (in pixel).
        max_spacing: intThe maximum spacing between digits (in pixel).
    """


class OCRLearningDataGeneraotr:

    def __init__(self, dictionary):

        self.dictionary = dictionary


    def create_digit_sequence(self, number, image_width, min_spacing, max_spacing):
        """ A function that create an image representing the given number, 
        with random spacing between the digits.
        Each digit is randomly sampled from the MNIST dataset.
        Returns an NumPy array representing the image.

        Parameters----------
            number: strA string representing the number, e.g. "14543"
            image_width: intThe image width (in pixel).
            min_spacing: intThe minimum spacing between digits (in pixel).
            max_spacing: intThe maximum spacing between digits (in pixel).
        """


def calc_square_usemulti(args, n_cores, dictionary):
    generator = OCRLearningDataGeneraotr(dictionary)

    n_cores = min(multi.cpu_count(), max(0, n_cores))

    p = Pool(n_cores)

    res = p.map(generator.square, args)

    # print(res)


def calc_square_usemulti_serial(args, dictionary):
    """[summary]

    Arguments:
        args {[type]} -- [description]
        dictionary {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    generator = OCRLearningDataGeneraotr(dictionary)

    res = list(map(generator.square, args))

    return res

    # print(res)


if __name__ == "__main__":
    n = 100

    dictionary = Manager().dict()

    # [0, n)
    for i in range(0, n):
        dictionary[i] = [np.random.rand(3, 10, 10) for _ in range(n)]

    args = list(range(10))

    t1 = time.time()
    calc_square_usemulti(args, 100, dictionary)
    t2 = time.time()

    elapsed_time = t2 - t1
    print(f"elapsed_time: {elapsed_time}")

    t1 = time.time()
    calc_square_usemulti_serial(args, dictionary)
    t2 = time.time()

    elapsed_time = t2 - t1
    print(f"elapsed_time: {elapsed_time}")
