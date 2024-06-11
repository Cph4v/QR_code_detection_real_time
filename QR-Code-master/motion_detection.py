import cv2
import numpy as np
import copy
# from roboflow import Roboflow
import copy
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
from pyzbar.pyzbar import decode, ZBarSymbol
from tqdm import tqdm
import os
import heapq
from ultralytics import YOLO
import argparse
from sys import stderr
import multiprocessing as mp
from pathlib import Path
from time import sleep
import time
import cv2
import json

from camera_tunnig_helpers import *
from function_for_app import *
from detect_led_blubs import *
# install v4l-utils for camera
import subprocess
from send_request import ConnectRPI


# importing OpenCV, time and Pandas library 
import cv2, time, pandas 
# importing datetime class from datetime library 
from datetime import datetime 
  
# Assigning our static_back to None 
static_back = None
  
# List when any moving object appear 
motion_list = [ None, None ] 

# Time of movement 
time = [] 
  
# Initializing DataFrame, one column is start  
# time and other column is end time 
# df = pandas.DataFrame(columns = ["Start", "End"]) 
  
# Capturing video 
video = cv2.VideoCapture(0) 

draw = False
# cap = video

set_camera_settings(video, width=video.get(cv2.CAP_PROP_FRAME_WIDTH), height=video.get(cv2.CAP_PROP_FRAME_HEIGHT), \
        autofocus=0, focus=video.get(cv2.CAP_PROP_FOCUS), contrast=video.get(cv2.CAP_PROP_CONTRAST), \
        zoom=14700, brightness=video.get(cv2.CAP_PROP_BRIGHTNESS), fps=video.get(cv2.CAP_PROP_FPS))

# Infinite while loop to treat stack of image as video 
while True: 
    # Reading frame(image) from video 
    check, frame = video.read() 
    cap = frame

    out = False

    break_loop = False
    
    # Initializing motion = 0(no motion) 
    motion = 0
  
    # Converting color image to gray_scale image 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
    # Converting gray scale image to GaussianBlur  
    # so that change can be find easily 
    gray = cv2.GaussianBlur(gray, (21, 21), 0) 
  
    # In first iteration we assign the value  
    # of static_back to our first frame 

    if static_back is None: 
        static_back = gray 
        continue

    static_back,out,draw,motion_list,motion,movment = motion_detection(static_back, gray, frame, draw, motion_list, motion, out, area=20)

    # if motion_list[-1] == 0 and motion_list[-2] == 1: 

    # Displaying image in gray_scale 
    cv2.imshow("Gray Frame", gray) 
  
    # Displaying the difference in currentframe to 
    # the staticframe(very first_frame) 
    # cv2.imshow("Difference Frame", diff_frame) 
  
    # # Displaying the black and white image in which if 
    # # intensity difference greater than 30 it will appear white 
    # cv2.imshow("Threshold Frame", thresh_frame) 
  
    # Displaying color frame with contour of motion of object 
    cv2.imshow("Color Frame", frame) 
  
    key = cv2.waitKey(1) 
    # if q entered whole process will stop 
    if key == ord('q'): 
        # if something is movingthen it append the end time of movement 
        if motion == 1: 
            time.append(datetime.now()) 
        break
    print(out)
  
# Appending time of motion in DataFrame 
# for i in range(0, len(time), 2): 
#     df = df.append({"Start":time[i], "End":time[i + 1]}, ignore_index = True) 
  
# # Creating a CSV file in which time of movements will be saved 
# df.to_csv("Time_of_movements.csv") 
  
video.release() 
  
# Destroying all the windows 
cv2.destroyAllWindows() 