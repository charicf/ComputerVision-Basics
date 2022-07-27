# Histogram-based Skin Color Detection
Script to apply the Color-based segmentation. The script trains two types of models: Histogram and Gaussian. Then, it uses the models to clasify the pixels within a skin color region.
To train the model, create a folder inside the project folder and put the training images set. Pass this path when running the script.
To save the images, create a folder with the name "output_images" in the same directory as the script.
The script will show five outputs: original image, skin regions mage, refine skin region image and histogram. The latter two if the respective options are enabled.

## Packages

OpenCV, numpy, argparse, os, matplotlib.pyplot.

## Usage

Arguments:

```python
"-i", "--input_image", required=True, help="path to the input image"
"-p", "--training_path", required=True, help="path to input dataset of training images"
"-m", "--model", default=0, help="Type of model. Histogram: 0. Gaussian: 1. Default value: 0"
"-r", "--refine_output", default=0, help="if refine the result is required: 1. Default value: 0"
"-k", "--kernel_width", default=2, help="size of the window for refining the result. Default value: 2"
"-t", "--threshold", default=50, help="threshold to ceterming a false positive pixel. Range: [0-255], Default value: 50"

```

Example on how to run the script:

```python
python color_segmentation.py -i test6.jpg -p training_images -r 1 -m 1
```

## Contributing
Pull requests are welcome.

## License
[MIT](https://choosealicense.com/licenses/mit/)