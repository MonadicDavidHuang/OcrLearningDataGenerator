"""coding: utf-8"""

from setuptools import setup

with open('requirements.txt') as requirements_file:
    INSTALL_REQUIREMENTS = requirements_file.read().splitlines()

setup(
    name='ocr-learning-data-generator',
    version='0.0.1',

    description='OCR learning data generator.',

    author='Dongyang David Huang',
    author_email='mynameistoyoko@gmail.com',

    url='https://github.com/MonadicDavidHuang/OCR-learning-data-generator',

    install_requires=INSTALL_REQUIREMENTS,

    entry_points={
        'console_scripts': [
            'ocrldg = ocr_learning_data_generator.app:main'
        ]
    }
)
