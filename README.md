# OCR-learning-data-generator
Learning data generator for OCR (Optical Character Recognition). 
Support rotation for data augmentation.

# Dependency
- `python` == 3.6.5+
- `numpy` == 1.17.4+
- `Pillow` == 6.2.1+

# Setup
At repository root
```
pip install -e .
```

# Usage
When set up is done, the command `ocrldg` is able under your environment. 
```
ocrldg --number 65535 --sampling_size 100 --image_width 120
```

The generated images are under `root_of_repository/output`.

For further detail, see
```
[OcrLearningDataGenerator]>>=ocrldg --help
usage: ocrldg [-h] [--number NUMBER] --image_width IMAGE_WIDTH
              [--min_spacing MIN_SPACING] [--max_spacing MAX_SPACING]
              [--max_rotation MAX_ROTATION] [--sampling_size SAMPLING_SIZE]

OCR learning data generator.

optional arguments:
  -h, --help            show this help message and exit
  --number NUMBER       Number to render.
  --image_width IMAGE_WIDTH
                        Image width, must be bigger than 0.
  --min_spacing MIN_SPACING
                        Minimum space, must be bigger than 0.
  --max_spacing MAX_SPACING
                        Maximum space, must be bigger than 0.
  --max_rotation MAX_ROTATION
                        Maximum rotation degree.
  --sampling_size SAMPLING_SIZE
                        Sampling size, must be bigger than 1.
```

# License
This software is released under the MIT License, see LICENSE.