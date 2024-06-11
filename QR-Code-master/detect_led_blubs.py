# import the necessary packages
# from imutils import contours
from skimage import measure
import numpy as np
import cv2
# from ultralytics import YOLO

# mke LED off and run below function to aquire best thresh 
def detect_thresh_range(image):
	for min_thresh in range(0,255):
		output,mask = LED_on_detection(image ,min_thresh)
		if output == False:
			result = min_thresh 
			break
		
	return result

def LED_on_detection(image, i, number_pixels=20):
	output = False
	# load the image, convert it to grayscale, and blur it
	# image = cv2.imread(input_image)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (11, 11), 0)
	# threshold the image to reveal light regions in the
	# blurred image
	
	thresh = cv2.threshold(blurred, i, 255, cv2.THRESH_BINARY)[1]
	
	# perform a series of erosions and dilations to remove
	# any small blobs of noise from the thresholded image
	thresh = cv2.erode(thresh, None, iterations=4)
	thresh = cv2.dilate(thresh, None, iterations=4)

	# perform a connected component analysis on the thresholded
	# image, then initialize a mask to store only the "large"
	# components
	labels = measure.label(thresh, connectivity=2, background=0)
	mask = np.zeros(thresh.shape, dtype="uint8")
	# loop over the unique components
	for label in np.unique(labels):
		# if this is the background label, ignore it
		if label == 0:
			continue
		# otherwise, construct the label mask and count the
		# number of pixels 
		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		# if the number of pixels in the component is sufficiently
		# large, then add it to our mask of "large blobs"
		if numPixels > number_pixels:
			output = True
			mask = cv2.add(mask, labelMask)
			# cv2.imwrite('blob_analysis.jpg', mask)
	return output, mask, i

# result = LED_on_detection('reference_image.jpg')
# print(result)

# def LED_on_detection_yolo(LED_detection_area):
# 	led_on_detected = False
# 	LED_detection_yolo = YOLO('/home/amir/pytorch_env/WLED/models/LED_on_off_model.pt')
# 	led_detection_results = LED_detection_yolo(LED_detection_area, device=0, conf=0.5)
# 	matrix = [led.boxes.cls.cpu().numpy().astype(int).tolist() for led in led_detection_results]
# 	matrix_flatten = [item for row in matrix for item in row]
# 	matrix_bool_equal = []
# 	for val in matrix_flatten:
# 		if val==1:
# 			matrix_bool_equal.append(True)
# 		else:
# 			matrix_bool_equal.append(False)
# 	print(f"matrix_bool_equal : {matrix_bool_equal}")
# 	if True in matrix_bool_equal:
# 		led_on_detected = True
		
# 	return led_on_detected


def new_thresh(LED_detection_area, threshes):
	"""
	threshold for new camera ELP detection
	"""
	thresh = 240
	if LED_detection_area.shape[0] != 0:
		thresh = detect_thresh_range(LED_detection_area)
		threshes.append(thresh)

		if len(threshes) != 0:
			thresh = max(threshes)

	return thresh

def brightest_image_value(frame):
	return np.amax(frame)