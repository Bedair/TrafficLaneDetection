from detect import *
import cv2 as cv
import numpy as np

def main():
	# Get a frame
	image = cv.imread('../yellow.png')

	# Get the gray, and the HSV of the image
	imgGray, imgHSV = BGR2GrayAndHSV(image)

	# make the mask for both white and yellow colors
	mask = generateMask(imgHSV, image)

	# make the masked image
	maskedImage = generateMaskedImage(imgGray, mask)

	# Detect the edges in the image
	edgedImage = detectEdges(maskedImage)

	# Remove the unimportant information of the image


	cv.imshow('yellow', edgedImage)
	cv.waitKey(0)
	cv.destroyAllWindows()

if __name__ == "__main__":
	main()
