"""coding: utf-8"""

import argparse
import math
import random
import sys
from random import choice

import numpy as np
from PIL import Image

from .utility import load_white_edgeless_mnist_dictionary, save_images

ROW_SIZE = 28
COLUMN_SIZE = 28


class OCRLearningDataGeneraotr:
    """OCR learning data generaotr.
    """

    def __init__(self):
        """Constructor, load edgeless MNIST dictionary.
        Also calculate each digit's max size.
        """
        self.dictionary = load_white_edgeless_mnist_dictionary()

        self.digit_to_max_size = {d: 0 for d in range(0, 10)}

        for k in self.dictionary:
            for arr in self.dictionary[k]:
                self.digit_to_max_size[k] = \
                    max(self.digit_to_max_size[k], tuple(arr.shape)[1])

    def _caculate_available_max_row_size(self, number, max_spacing):
        return max_spacing * (len(number) - 1) \
            + sum([self.digit_to_max_size[d] for d in map(int, number)])

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

        Raises:
            ValueError: when given spacings have contradiction
            ValueError: when given image width is not enough

        Returns:
            numpy.ndarray -- with its shape of (28, 28), indicates image (measured by row)
        """
        if min_spacing > max_spacing:
            raise ValueError(
                'Given max spacing must be bigger than min spacing! min: {}, max: {}'
                .format(min_spacing, max_spacing)
            )

        avalable_max_row_size = \
            self._caculate_available_max_row_size(number, max_spacing)

        if avalable_max_row_size > image_width:
            raise ValueError(
                'Given image width is too short! At least, image width must be bigger than {}'
                .format(avalable_max_row_size)
            )

        list_of_array = []
        # make digit-space-digit-...-space-digit sandwith
        for digit in [int(c) for c in number]:
            list_of_array.append(choice(self.dictionary[digit]))

            space = random.randint(min_spacing, max_spacing)
            shape = (COLUMN_SIZE, space)
            list_of_array.append(np.zeros(shape))

        image = np.concatenate(list_of_array[:-1], axis=1)

        if image_width < math.inf:
            image = _padding(image, image_width)

        image = 256 * image  # broad casting

        image = image.astype(np.int32)

        return image


def _padding(image, image_width):
    # padding for remaining space
    row_size = tuple(image.shape)[1]
    remaining_space = image_width - row_size

    left_size = remaining_space // 2
    right_size = remaining_space - left_size

    left_padding = np.zeros((COLUMN_SIZE, left_size))
    right_padding = np.zeros((COLUMN_SIZE, right_size))

    image = np.concatenate(
        [left_padding, image, right_padding], axis=1)

    return image


def _crop_center(image, crop_width, crop_height):
    image_width, image_height = image.size
    return image.crop(((image_width - crop_width) // 2,
                       (image_height - crop_height) // 2,
                       (image_width + crop_width) // 2,
                       (image_height + crop_height) // 2))


def _rotate_randomly(image, args):
    max_rotation = args.max_rotation
    degree = random.randint(min(-1 * max_rotation, max_rotation),
                            max(-1 * max_rotation, max_rotation)) % 360

    image = image.rotate(degree, expand=True)

    return _crop_center(image, args.image_width, COLUMN_SIZE)


def _augmentate(array_list, args):
    image_list = map(lambda a: Image.fromarray(a).convert('P'), array_list)

    if args.max_rotation != 0:
        image_list = map(
            lambda image: _rotate_randomly(image, args),
            image_list
        )

    return list(image_list)


def _validate_arguments(args):
    digits = set(range(0, 10))  # [0, 10)

    if args.min_spacing > args.max_spacing:
        raise ValueError(
            'Given max spacing must be bigger than min spacing! min: {}, max: {}'
            .format(args.min_spacing, args.max_spacing)
        )

    if abs(args.max_rotation) > 10:  # TODO: consider threshold
        print("Warning: The rotation could be too much! Rotation Degree: {}"
              .format(args.max_rotation))

    if args.sampling_size <= 0:
        raise ValueError('Given sample size must bigger than 1!')

    if (args.image_width < 0 or args.min_spacing < 0 or args.max_spacing < 0):
        raise ValueError(
            'Some of given size related to image is smaller thant 0!'
        )

    print("All arguments looks good!")


PARSER = argparse.ArgumentParser(description='OCR learning data generaotr.')

PARSER.add_argument('--number', type=str, help='Number to render.')

PARSER.add_argument('--image_width', type=int,
                    help='Image width, must be bigger than 0.', required=True)
PARSER.add_argument('--min_spacing', type=int,
                    help='Minimum space, must be bigger than 0.', default=0)
PARSER.add_argument('--max_spacing', type=int,
                    help='Maximum space, must be bigger than 0.', default=0)
PARSER.add_argument('--max_rotation', type=int,
                    help='Maximum rotation degree.', default=0)

PARSER.add_argument('--sampling_size', type=int,
                    help='Sampling size, must be bigger than 1.', default=1)


def main():
    """Main function.
    """
    args = PARSER.parse_args()

    try:
        _validate_arguments(args)
    except ValueError as value_error:
        print(value_error)
        sys.exit(0)

    generator = OCRLearningDataGeneraotr()

    image_list = []

    for _ in range(args.sampling_size):
        try:
            image = generator.create_digit_sequence(
                number=args.number,
                image_width=args.image_width,
                min_spacing=args.min_spacing,
                max_spacing=args.max_spacing
            )
        except ValueError as value_error:
            print(value_error)
            sys.exit(0)

        Image.fromarray(image).convert('P')

        image_list.append(image)

    image_list = _augmentate(image_list, args)

    save_images(image_list, inner_directory_name=args.number)
