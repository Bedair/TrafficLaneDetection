import cv2 as cv
import numpy as np

def BGR2GrayAndHSV(img):
	# Convert the frame into gray scale
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	# Convert the image into HSV color space
	hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

	return gray, hsv


def generateMask(imgHSV, img):
	# define the limits for Yellow color
	lowerYellow = np.array([20, 100, 100], dtype = "uint8")
	upperYellow = np.array([30, 255, 255], dtype = "uint8")

	#define the limits for white color
	lowerWhite = np.array([100, 100, 100], dtype = "uint8")
	upperWhite = np.array([255, 255, 255], dtype = "uint8")

	# Generate the Mask for both colors
	yellowMask = cv.inRange(imgHSV, lowerYellow, upperYellow)
	whiteMask = cv.inRange(img, lowerWhite, upperWhite)

	# combine the two masks to get one mask
	mask = cv.bitwise_or(yellowMask, whiteMask)

	return mask

def generateMaskedImage(gray, mask):
	masked = cv.bitwise_or(gray, mask)
	return masked

def detectEdges(img):
	edge = cv.Canny(img, 100, 200)
	return edge
