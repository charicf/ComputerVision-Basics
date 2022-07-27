import numpy as np
import argparse
import cv2
import pdb
import os
import sys


def scale_image_height(img, new_height):

	height, width = img.shape
	ratio = width/height
	new_width = int(new_height * ratio)
	dim = (new_width, new_height)

	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

	return resized

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input dataset of training images")

args = vars(ap.parse_args())

image_path = args["image"]

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=1500)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

standard_height = 500
image = scale_image_height(image, standard_height)

keypoints_sift, img_descriptors = sift.detectAndCompute(image, None)
keypoints_surf, img_descriptors = surf.detectAndCompute(image, None)
keypoints_orb, img_descriptors = orb.detectAndCompute(image, None)
img = cv2.drawKeypoints(image, keypoints_sift, outImage=image, color = (0, 255, 255))
img2 = cv2.drawKeypoints(image, keypoints_surf, outImage=image, color = (0, 255, 255))
img3 = cv2.drawKeypoints(image, keypoints_orb, outImage=image, color = (0, 255, 255))
#cv2.imshow("Image", img)

cv2.imshow('sift.png', img)
cv2.imshow('surf.png', img2)
cv2.imshow('orb.png', img3)

cv2.waitKey(0)
#sys.exit(0)

cv2.imwrite('sift.png', img)
cv2.imwrite('surf.png', img2)
cv2.imwrite('orb.png', img3)