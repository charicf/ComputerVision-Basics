# Image classification with Bag of Words of visual features - ComputerVision
Image classification is the visual recognition’s subcategory that intents to classify images according to a set of given classes. One technique to
accomplish image classification is to use the Bag of Words approach. In a CV context, the purpose is to define the label of an image based on the most frequent features encountered in the image. These frequencies are represented by a histogram that corresponds to the image.

The first step is to find the features that describe each image. To do so, a feature descriptor such as SIFT can be used. The second step is to group similar groups of features to form the features that will represent the training set as a “visual dictionary”. A step of clustering is performed (k-means) and the expected result is a k number of representative features or “visual words”. The third step consists in using the visual dictionary to classify each training image in
accordance with the visual words. The result will be a Bag of Words vector (represented as a histogram) for each image. For classification, the k-means model is used to obtain a BOW representation for the testing images. Finally, the distance from each set of BOW from the training image to the set of BOW from the testing image are computed with a Euclidean distance. The class from the k elements with a higher frequency is selected as the predicted class.

There are two scripts, one for feature extraction and one for image classification with BOW.

## Usage
### image_classification.py

```python
python image_classification.py <args>

"-p", "--training_path", required=True, help="path to input dataset of training images"
"-v", "--validation_path", required=True, help="path to input dataset of testing images"
"-c", "--number_clusters", default=2, help="size of the number of clusters to build the dictionary. Default value: 2"
"-k", "--number_neighbors", default=3, help="size of the number of neighbors to execute knn. Default value: 3"
"-m", "--model", help="created model for the visual dictionary"

```

### features_extractor.py

It extracts features of the given images with SIFT, SURF and ORB

```python
python features_extractor.py <args>

"-i", "--image", required=True, help="path to input dataset of training images"

```

## Contributing
Pull requests are welcome.

## License
[MIT](https://choosealicense.com/licenses/mit/)