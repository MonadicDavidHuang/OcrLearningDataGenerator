from setuptools import setup

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

setup(
    name='ocr-learning-data-generator',
    version='0.0.1',

    description='OCR learning data generator.',

    author='Dongyang David Huang',
    author_email='mynameistoyoko@gmail.com',

    url='https://github.com/MonadicDavidHuang/OCR-learning-data-generator',

    install_requires=install_requirements,

    entry_points={
        'console_scripts': [
            'oreore = ocr_learning_data_generator.app:main'
        ]
    }
)
