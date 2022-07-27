import numpy as np
import argparse
import cv2
from os import listdir
from os.path import join
import matplotlib.pyplot as plt

def create_histogram_model(path, images_files):

	histogram = np.zeros([256, 180], dtype=np.uint8) # OpenCV uses H: 0-179, S: 0-255, V: 0-255

	for image_name in images_files:

		image_path = join(path, image_name)
		BGR_image = cv2.imread(image_path)
		image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2HSV)

		img_height, img_width, img_ch = image.shape

		for y in range(0, img_height):
			for x in range(0, img_width):

				# bin_num: value of the bin in x (Hue). value: value of the bin in y (Saturation).
				bin_num, value = image[y, x, 0], image[y, x, 1] #Channels: H = 0; S = 1; V = 2
				histogram[value][bin_num] += 1

	max_value = np.amax(histogram)
	histogram = histogram * (1/max_value)

	return histogram

def identify_skin_region(BGR_image, histogram):

	image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2HSV)
	img_height, img_width, img_ch = image.shape

	for y in range(0, img_height):
		for x in range(0, img_width):

			# bin_num: value of the bin in x. value: value of the bin in y.
			pixel_hue, pixel_sat = image[y, x, 0], image[y, x, 1] #Channels: H = 0; S = 1; V = 2

			if histogram[pixel_sat][pixel_hue] < 0.032 :
				image[y, x, 2] = 0

	image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
	
	return image

def refine_skin_region(BGR_image, k_width=2, threshold = 50):

	image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2HSV)
	img_height, img_width, img_ch = image.shape

	#k_width controls the window's width which determines the neighbor pixels that will be taken into account to find out the H, S, V means. 
	#k_width = 10 # Good value from tests: 2
	#threshold = 50 # Good value from tests: 50

	for y in range(k_width, img_height-k_width):
		for x in range(k_width, img_width-k_width):

			if image[y, x, 2] == 0:
				neighbors_value = [image[y, x - k_width, 2], image[y-k_width, x, 2], image[y, x+k_width, 2], image[y+k_width, x, 2]]
				neighbors_mean = np.mean(neighbors_value)

				#This threshold checks if the pixel is a false positive and needs to be corrected or not.
				if neighbors_mean > threshold:

					image[y, x, 2] = neighbors_mean
					image[y, x, 1] = np.mean([image[y, x - k_width, 1], image[y-k_width, x, 1], image[y, x+k_width, 1], image[y+k_width, x, 1]])
					image[y, x, 0] = np.mean([image[y, x - k_width, 0], image[y-k_width, x, 0], image[y, x+k_width, 0], image[y+k_width, x, 0]])

	image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
	
	return image

def create_gaussian_model(path, images_files):

	hue_values = []
	sat_values = []

	for image_name in images_files:

		image_path = join(path, image_name)
		BGR_image = cv2.imread(image_path)
		image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2HSV)

		img_height, img_width, img_ch = image.shape

		hue_values.extend(image[:, :, 0].flatten())
		sat_values.extend(image[:, :, 1].flatten())

	hue_mean = np.mean(hue_values)
	sat_mean = np.mean(sat_values)

	hue_variance = np.var(hue_values)
	sat_variance = np.var(sat_values)

	return ((hue_mean, hue_variance), (sat_mean, sat_variance))

def identify_skin_region_gauss(BGR_image, gaussian_parameters):

	image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2HSV)
	img_height, img_width, img_ch = image.shape

	hue_mean = gaussian_parameters[0][0]
	hue_variance = gaussian_parameters[0][1]

	sat_mean = gaussian_parameters[1][0]
	sat_variance = gaussian_parameters[1][1]

	for y in range(0, img_height):
		for x in range(0, img_width):

			gauss_model_hue = (1/((2*np.pi*hue_variance)**(1/2)))*(np.exp(-((image[y,x,0]-hue_mean)**2)/(2*hue_variance)))
			gauss_model_sat = (1/((2*np.pi*sat_variance)**(1/2)))*(np.exp(-((image[y,x,1]-sat_mean)**2)/(2*sat_variance)))
			gauss_model_value = gauss_model_hue * gauss_model_sat
			
			if gauss_model_value < 0.0001:

				image[y,x,2] = 0

	image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
	
	return image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--training_path", required=True, help="path to input dataset of training images")
ap.add_argument("-m", "--model", default=0, help="Type of model. Histogram: 0. Gaussian: 1. Default value: 0")
ap.add_argument("-r", "--refine_output", default=0, help="if refine the result is required: 1. Default value: 0")
ap.add_argument("-k", "--kernel_width", default=2, help="size of the window for refining the result. Default value: 2")
ap.add_argument("-t", "--threshold", default=50, help="threshold to ceterming a false positive pixel. Range: [0-255], Default value: 50")

args = vars(ap.parse_args())

image_path = args["image"]
folder_path = args["training_path"]
model = int(args["model"])
kernel_width = int(args["kernel_width"])
threshold = float(args["threshold"])
refine_output = int(args["refine_output"])

training_images = listdir(args["training_path"])

input_image = cv2.imread(image_path)

if model == 1:
	gaussian_parameters = create_gaussian_model(folder_path, training_images)
	output_image = identify_skin_region_gauss(input_image, gaussian_parameters)
else:
	histogram = create_histogram_model(folder_path, training_images)
	output_image = identify_skin_region(input_image, histogram)

	plt.imshow(histogram,interpolation='none')
	# plt.show() # Uncomment to show the histogram. If enabled, close window to continue with the results.

if refine_output == 1:
	refined_image = refine_skin_region(output_image)
	cv2.imshow("refined_image", refined_image)
	cv2.imwrite('output_images/refined_image.png', refined_image)

cv2.imshow("input_image", input_image)
cv2.imshow("skin_image", output_image)

cv2.imwrite('output_images/out_img.png', output_image)

cv2.waitKey(0)