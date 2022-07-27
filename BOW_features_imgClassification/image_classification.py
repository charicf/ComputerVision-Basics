import numpy as np
import argparse
import cv2
import pdb
import os
import sys

from sklearn.cluster import KMeans, MiniBatchKMeans
import operator
import pickle

import random 

def get_images_paths(training_path):
	images_path = []

	# r=root, d=directories, f = files
	for r, d, f in os.walk(training_path):
		for file in f:
			if '.jpg' in file:
				images_path.append(os.path.join(r, file))
	return images_path

def create_visual_dictionary_kmeans(images, sift, k):

	descriptors = []
	kmeans = KMeans(n_clusters = k, n_jobs = -1)

	for image_path in images:

		image = cv2.imread(image_path)
		keypoints_sift, img_descriptors = sift.detectAndCompute(image, None)
		descriptors = descriptors + list(img_descriptors)

	descriptors = np.array(descriptors, dtype=np.float32)
	kmeans_result = kmeans.fit(descriptors)
	
	# save the model to disk
	filename = str(k) + 'k_visual_dict.sav'
	pickle.dump(kmeans, open(filename, 'wb'))

	visual_dict = kmeans_result.cluster_centers_

	return kmeans

def get_bag_of_words(images, sift, kmeans_result, k):

	bag_of_words = []

	for image_path in images:

		histogram = np.zeros(k) 

		image = cv2.imread(image_path)
		keypoints_sift, img_descriptors = sift.detectAndCompute(image, None)

		if (img_descriptors is not None):

			predicted_clusters = kmeans_result.predict(img_descriptors)
			for cluster in predicted_clusters:
				histogram[cluster] += 1

			freq_sum = np.sum(histogram)
			histogram = histogram/freq_sum
			histogram = tuple(histogram)

		count_parent_dir = len(os.path.dirname(os.path.dirname(image_path))) + 1
		label = os.path.dirname(image_path)[count_parent_dir:]
		bag_of_words.append((histogram, label))
	bag_of_words = tuple(bag_of_words)

	return bag_of_words

def euclidean_distance(training_hist, test_hist):

	distance = 0
	for idx in range(len(test_hist)):
		distance += np.square(test_hist[idx] - training_hist[idx])
	distance = np.sqrt(distance)

	return distance

def count_labels(labels):

	counter = {}
	for label in labels:
		if label in counter:
			counter[label] += 1
		else:
			counter[label] = 1

	counter = sorted(counter.items(), key=operator.itemgetter(1))
	return counter[-1]

def  K_nearest_neighbor(training_bow, validation_bow, k):

	predicted_classes = []
	for validation_histogram in validation_bow:

		distances = []
		k_neighbors = []

		for training_histogram in training_bow:

			distance = (euclidean_distance(validation_histogram[0], training_histogram[0]), training_histogram[1])
			distances.append(distance)

		distances.sort()

		k_neighbors = distances[:k]
		max_label_count = count_labels([x[1] for x in k_neighbors])
		class_label = max_label_count[0]

		predicted_classes.append(class_label)
	
	return predicted_classes

def evaluation(predicted_classes, validation_bow):

	same_labels = 0
	total_labels = len(predicted_classes)
	actual_classes = [word[1] for word in validation_bow]

	confusion_matrix = np.zeros((10, 10))

	categories = ('Coast', 'Forest', 'Highway', 'Kitchen', 'Mountain', 'Office', 'OpenCountry', 'Street', 'Suburb', 'TallBuilding')

	for idx in range(total_labels):
		if predicted_classes[idx] == actual_classes[idx]:
			same_labels += 1

		cat_idx_actual = categories.index(actual_classes[idx])
		cat_idx_predicted = categories.index(predicted_classes[idx])
		confusion_matrix[cat_idx_actual][cat_idx_predicted] += 1

	labels_counts = np.sum(confusion_matrix, axis=1)
	confusion_matrix = np.around(np.nan_to_num((confusion_matrix/labels_counts[:, None])), 4)*100


	accuracy = (same_labels*100)/total_labels

	print(confusion_matrix)

	return accuracy

def select_random_images(training_images_path):

	selected_images = []

	for n in range(0, 10):

		selected_images.append(random.choice(training_images_path))

	return selected_images

def visualize_visual_dictionary(training_images_path, training_bow):
	#print ("A random number from list is : ",end="") 
	#print (random.choice([1, 4, 8, 10, 3])) 
	  
	#print ("A random number from range is : ",end="") 
	
	print (random.randrange(20, 50, 3)) 




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--training_path", required=True, help="path to input dataset of training images")
ap.add_argument("-v", "--validation_path", required=True, help="path to input dataset of testing images")
ap.add_argument("-c", "--number_clusters", default=2, help="size of the number of clusters to build the dictionary. Default value: 2")
ap.add_argument("-k", "--number_neighbors", default=3, help="size of the number of neighbors to execute knn. Default value: 3")
ap.add_argument("-m", "--model", help="created model for the visual dictionary")
#ap.add_argument("-i", "--image", help="image to extract SIFT features")

args = vars(ap.parse_args())

training_path = args["training_path"]
validation_path = args["validation_path"]
number_clusters = int(args["number_clusters"])
number_neighbors = int(args["number_neighbors"])
model = args["model"]
#image = args["image"]

training_images_path = get_images_paths(training_path)
validation_images_path = get_images_paths(validation_path)

sift = cv2.xfeatures2d.SIFT_create()

# image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
# keypoints_sift, img_descriptors = sift.detectAndCompute(image, None)
# img = cv2.drawKeypoints(image, keypoints_sift, None)
# cv2.imshow("Image", img)

if model is None:
	kmeans = create_visual_dictionary_kmeans(training_images_path, sift, number_clusters)
else:
	number_clusters = int(model[:-17])
	kmeans = pickle.load(open(model, 'rb'))

training_bow = get_bag_of_words(training_images_path, sift, kmeans, number_clusters)

selected_validation_imgs_path = select_random_images(validation_images_path)
#pdb.set_trace()
validation_bow = get_bag_of_words(selected_validation_imgs_path, sift, kmeans, number_clusters)

predicted_classes = K_nearest_neighbor(training_bow, validation_bow, number_neighbors)

accuracy = evaluation(predicted_classes, validation_bow)

print('For clustering k = {0} and knn k = {1}: '.format(number_clusters, number_neighbors))
print('Accuracy: ', accuracy)

sys.exit(0)
 