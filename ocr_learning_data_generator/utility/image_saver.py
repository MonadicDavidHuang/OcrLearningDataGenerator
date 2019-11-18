"""coding: utf-8"""

import os

OUTPUT_DIR_NAME = './output'


def save_images(image_list, inner_directory_name='images', file_suffix_name='image.png'):
    """Save images based on given inner directory name and file suffix name.

    Arguments:
        image_list {[PIL.Image.Image]} -- images

    Keyword Arguments:
        inner_directory_name {str} -- inner directory name (default: {'images'})
        file_suffix_name {str} -- file suffix name (default: {'image.png'})
    """

    if not os.path.exists(OUTPUT_DIR_NAME):
        os.makedirs(OUTPUT_DIR_NAME)

    directory_path = OUTPUT_DIR_NAME + '/' + inner_directory_name

    for index, image in enumerate(image_list):
        cur_file_name = str(index) + file_suffix_name
        file_path = OUTPUT_DIR_NAME + '/' + inner_directory_name + '/' + cur_file_name
        _save_image(
            image,
            directory_path=directory_path,
            file_path=file_path
        )


def _save_image(image, directory_path, file_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    image.save(file_path, cmap='gray_r')
