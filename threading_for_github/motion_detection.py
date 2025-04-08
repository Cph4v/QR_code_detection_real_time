"""
Motion Detection Application

This module demonstrates a refactored version of the video capture and motion detection
routine. It captures frames from a camera, converts the frame to grayscale, applies
GaussianBlur, compares to a static background to detect motion via contours, and shows
the original, grayscale, and annotated frames.

Press 'q' to quit the application.
"""

import cv2
import numpy as np
from datetime import datetime
from time import sleep
import subprocess
import json
import os

# Import helper functions from your project modules (if needed)
from camera_tunnig_helpers import *
from function_for_app import *
from detect_led_blubs import LED_on_detection, detect_thresh_range
from send_request import ConnectRPI


class MotionDetectionApp:
    def __init__(self, video_source=0, min_contour_area=10000, diff_threshold=50):
        """
        Initialize the motion detection application.
        :param video_source: Camera device index or video file path.
        :param min_contour_area: Minimum area of a contour to be considered motion.
        :param diff_threshold: Pixel intensity threshold used for frame differencing.
        """
        self.video = cv2.VideoCapture(video_source)
        if not self.video.isOpened():
            raise Exception("Could not open video source.")
        self.min_contour_area = min_contour_area
        self.diff_threshold = diff_threshold
        
        # Initialize background variables and motion history list.
        self.static_back = None
        self.motion_list = [None, None]
        self.motion_times = []  # List to store timestamps when motion occurs

    def process_frame(self, frame):
        """
        Convert to grayscale and apply Gaussian blur.
        :param frame: The color frame.
        :return: The blurred grayscale frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        return gray

    def detect_motion(self, static_back, gray, frame):
        """
        Perform motion detection by comparing the static background with the current blurred frame.
        Draws rectangles on the frame if motion is detected.
        :param static_back: The background frame.
        :param gray: Current blurred grayscale frame.
        :param frame: Original color frame.
        :return: updated static_back, boolean flag for motion detection, updated motion_list, and motion flag.
        """
        diff_frame = cv2.absdiff(static_back, gray)
        thresh_frame = cv2.threshold(diff_frame, self.diff_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
        contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = 0

        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            motion = 1
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Update motion history list keeping only the last two states.
        self.motion_list.append(motion)
        self.motion_list = self.motion_list[-2:]

        # Determine if we should "draw" the motion annotation (i.e. if motion just started)
        draw = False
        if self.motion_list[-1] == 1 and self.motion_list[-2] == 0:
            draw = True

        if draw:
            cv2.putText(frame, "Motion Detected", (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return static_back, draw, self.motion_list, motion

    def run(self):
        """
        Main loop for capturing video frames, detecting motion, and displaying frames.
        Press 'q' to quit the application.
        """
        while True:
            ret, frame = self.video.read()
            if not ret:
                print("Failed to grab frame, exiting...")
                break

            # Process frame: convert to grayscale and blur.
            gray = self.process_frame(frame)

            # Set the static background on the first iteration.
            if self.static_back is None:
                self.static_back = gray
                continue

            # Detect motion comparing the static background and current gray frame.
            self.static_back, draw, self.motion_list, motion = self.detect_motion(self.static_back, gray, frame)

            # If motion detected, record the timestamp (only on transition from no motion to motion).
            if motion == 1 and (self.motion_list[-2] == 0 if self.motion_list[-2] is not None else True):
                self.motion_times.append(datetime.now())

            # Display the frames.
            cv2.imshow("Gray Frame", gray)
            cv2.imshow("Color Frame", frame)

            key = cv2.waitKey(1)
            # Exit if 'q' is pressed.
            if key == ord('q'):
                # Optionally, record end time if motion is active.
                if motion == 1:
                    self.motion_times.append(datetime.now())
                break

        # Release resources and clean up.
        self.video.release()
        cv2.destroyAllWindows()

        # Optionally, save the motion times to a file.
        print("Motion times:")
        for t in self.motion_times:
            print(t)


if __name__ == "__main__":
    try:
        app = MotionDetectionApp(video_source=0, min_contour_area=10000, diff_threshold=50)
        app.run()
    except Exception as e:
        print("An error occurred:", e)
