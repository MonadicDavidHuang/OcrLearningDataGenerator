"""coding: utf-8"""

import os

import matplotlib.pyplot as plt


def save_image(image, directory_name='figures', image_name='figure.png'):
    """Save image to given directory name by given image name.

    Arguments:
        image {numpy.ndarray} -- with its shape of (?, ?), indicates image (measured by row)

    Keyword Arguments:
        directory_name {str} -- directory to save image (default: {'figures'})
        image_name {str} -- image name to use (default: {'figure.png'})
    """

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    path = directory_name + '/' + image_name
    plt.imshow(image, cmap='gray_r')
    plt.savefig(path)
