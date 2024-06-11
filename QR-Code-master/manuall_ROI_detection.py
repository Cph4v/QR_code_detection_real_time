import cv2
import numpy as np

from camera_tunnig_helpers import *


def save_points(points, filename="ROI_points.txt"):
    with open(filename, 'w') as file:
        for point in points:
            file.write(f"{point[0]}, {point[1]}\n")
    print(f"points saved to {filename}")

def load_points(filename="ROI_points.txt"):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            x, y = line.strip().split(", ")
            points.append((int(x), int(y)))
    return points


def draw_roi(event, x, y, flags, param):
    global points, frame, overlay

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:  # Allow only 4 points
            points.append((x, y))

            # Draw the point on the overlay
            cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)

            if len(points) > 1:
                # Draw line to the previous point on the overlay
                cv2.line(overlay, points[-2], points[-1], (255, 0, 0), 2)

            if len(points) == 4:
                # Draw a line connecting the last point to the first on the overlay
                cv2.line(overlay, points[3], points[0], (255, 0, 0), 2)


def order_points(points):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    pts = np.array(eval(points), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped,rect


# Global variables
points = []  # List to store the points
draw_roi_const = False

# Initialize the video capture object
def make_ROI(device, ROI, points=points, draw_roi_const=draw_roi_const):

    cap = cv2.VideoCapture(device)

    
    cv2.namedWindow(f"{ROI}")
    cv2.setMouseCallback(f"{ROI}", draw_roi)

    set_camera_settings(cap, width=4000, height=3000, autofocus=0, focus=28, contrast=32, zoom=14700, brightness=32, fps=2)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        
        if ROI == "LED_area":
            filename = "LED_area_ROI_points.txt"
            points_loaded = load_points(filename="ROI_points.txt")
            frame,rect = four_point_transform(frame, f"{points_loaded}")

        elif ROI == "DETECTION_area":
            filename = "ROI_points.txt"
        
        
        # set_camera_settings(cap, width=cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT), \
        #                     autofocus=0, focus=cap.get(cv2.CAP_PROP_FOCUS), contrast=cap.get(cv2.CAP_PROP_CONTRAST), \
        #                     zoom=cap.get(cv2.CAP_PROP_ZOOM), brightness=cap.get(cv2.CAP_PROP_BRIGHTNESS), fps=cap.get(cv2.CAP_PROP_FPS))  

        
        key = cv2.waitKey(1) & 0xFF

        # Make a copy of the frame to use for displaying the points and lines
        if draw_roi_const == False:
            overlay = frame.copy()

            # If we have points, we want to keep the overlay with the points and lines
            if points:
                for i in range(len(points)):
                    cv2.circle(overlay, points[i], 5, (0, 0, 255), -1)
                    if i > 0:
                        cv2.line(overlay, points[i - 1], points[i], (255, 0, 0), 2)
                if len(points) == 4:
                    cv2.line(overlay, points[3], points[0], (255, 0, 0), 2)
            

            if key == ord('r'):
                points = []
                overlay = frame.copy()

            # Display the resulting frame
            cv2.imshow(f"{ROI}", overlay)

        else:
            overlay = frame.copy()
            points_loaded = load_points(filename=filename)
            overlay,rect = four_point_transform(overlay, f"{points_loaded}")
            cv2.imshow(f"{ROI}", overlay)

        # else:
        #     raise Exception(f"there is no ROI as {ROI}")
        # If the 'r' key is pressed, reset the ROI
        if key == ord('g'):
            draw_roi_const = True
            save_points(points, filename=filename)


        # If the 'q' key is pressed, break from the loop
        if key == ord('q'):
            # save_points(points)
            break

    # Usage example:
    points_loaded = load_points()
    print("Loaded points:", points_loaded)

    # When everything is done, release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()


# make_ROI("/dev/video0", 'DETECTION_area') # detection area5145
# make_ROI("/dev/video0", ROI="LED_area") # led light remover area

