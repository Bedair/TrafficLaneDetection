from detect import *
import cv2 as cv
import numpy as np

def main():
	# Put the path to the Video to be played
	video = cv.VideoCapture('../00.mp4')

	# Loop over all the frames in the video
	while(video.isOpened()):
		# Read frame each Time
		ret, frame = video.read()

		# Get the gray, and the HSV of the image
		imgGray, imgHSV = BGR2GrayAndHSV(frame)

		# make the mask for both white and yellow colors
		mask = generateMask(imgHSV, frame)

		# make the masked image
		maskedImage = generateMaskedImage(imgGray, mask)

		# Detect the edges in the image
		#edgedImage = cv.GaussianBlur(maskedImage, (5, 5), 0)
		edgedImage = detectEdges(maskedImage)
		edgedImage = cv.GaussianBlur(edgedImage, (5, 5), 0)

		# Remove the unimportant information of the image
		ROI = getROI(edgedImage)

		# Draw lines
		try:
			lines = cv.HoughLinesP(ROI, 2, np.pi/180, 100, np.array([]), minLineLength = 50, maxLineGap = 150)
			linedImg = np.zeros((frame.shape[0], frame.shape[1], 3), dtype = np.uint8)

			for line in lines:
				for x1, y1, x2, y2 in line:
					cv.line(linedImg, (x1, y1), (x2, y2), [255, 0, 0], 2)

			final = cv.addWeighted(frame, 0.8, linedImg, 1., 0.)
			# Display the captured frame
			cv.imshow('Video 0', final)
		except:
			continue



		# Use 'Q' to close the video
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	video.release()
	cv.destroyAllWindows()

if __name__ == "__main__":
	main()
