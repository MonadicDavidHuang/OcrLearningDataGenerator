"""coding: utf-8"""

import math
import random
from random import choice

import numpy as np

from image_saver import save_image
from mnist_shaper import load_white_edgeless_mnist_dictionary

ROW_SIZE = 28
COLUMN_SIZE = 28


class OCRLearningDataGeneraotr:
    """OCR learning data generaotr.
    Load edgeless MNIST dictionary in its constructor.
    """

    def __init__(self):
        """Constructor, load edgeless MNIST dictionary.
        """
        self.dictionary = load_white_edgeless_mnist_dictionary()

    def create_digit_sequence(self, number, image_width=math.inf, min_spacing=0, max_spacing=0):
        """ A function that create an image representing the given number,
        with random spacing between the digits.
        Each digit is randomly sampled from the MNIST dataset.
        Returns an NumPy array representing the image.

        Arguments:
            number {str} -- strA string representing the number, e.g. "14543"

        Keyword Arguments:
            image_width {int} -- the image width (in pixel) (default: {0})
            min_spacing {int} -- the minimum spacing between digits (in pixel) (default: {0})
            max_spacing {int} -- the maximum spacing between digits (in pixel) (default: {0})

        Returns:
            numpy.ndarray -- with its shape of (28, 28), indicates image (measured by row)
        """
        avalable_max_row_size = ROW_SIZE * \
            len(number) + max_spacing * (len(number) - 1)

        if min_spacing > max_spacing:
            raise ValueError(
                'Given max spacing must be bigger than min spacing!')

        if avalable_max_row_size > image_width:
            raise ValueError('Given image width is too short!')

        list_of_array = []

        for e in [int(i) for i in number]:
            list_of_array.append(choice(self.dictionary[e]))

            space = random.randint(min_spacing, max_spacing)
            shape = (COLUMN_SIZE, space)
            list_of_array.append(np.zeros(shape))

        return np.concatenate(list_of_array[:-1], axis=1)


if __name__ == "__main__":
    generator = OCRLearningDataGeneraotr()

    hoge = generator.create_digit_sequence(
        "1145141919", min_spacing=3, max_spacing=10)

    print(hoge.shape)

    print(hoge.dtype)

    save_image(hoge)

    # (train_img, train_label), (test_img, test_label) = load_mnist(flatten=False)

    # print(type(train_img))
    # print(train_img.shape)

    # print(type(train_label))
    # print(train_label.shape)

    # print(type(test_img))
    # print(test_img.shape)

    # print(type(test_label))
    # print(test_label.shape)

    # image1 = train_img[0, 0]
    # image2 = train_img[1, 0]

    # print(image1.shape, image2.shape)

    # image3 = np.concatenate((image1, image2), axis=1)

    # save_image(image3)

    # print(image3.shape)

    # for i in range(10):
    #     save_image(train_img[i, 0], image_name=str(i))
