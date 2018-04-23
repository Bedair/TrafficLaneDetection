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
	edge = cv.Canny(img, 250, 250)
	return edge


def getROI(img):
	# Get the number of columns and rows of the frame
	rows, columns = img.shape
	mask = np.zeros_like(img)	# make an empty image similer to the frame

	# make the coordinate for ROI (infront of the car)
	topRight = [columns/2 + columns/8, rows/2 + rows/6]
	topLeft = [columns/2 - columns/8, rows/2 + rows/6]
	bottomLeft = [columns/9, rows]
	bottomRight = [columns-columns/9, rows]
	coordinate = [np.array([bottomLeft, topLeft, topRight, bottomRight], dtype = np.int32)]

	# Fill the ROI to make a mask
	cv.fillPoly(mask, coordinate, 255)

	# Comine the mask with the frame to get the ROI
	roi = cv.bitwise_and(img, mask)
	return roi
