"""
Usage Instructions:
1. First run the process to allocate LED numbers and shelf QR codes.
2. Second run the process to allocate shelf and QR codes together.
3. Third run the real-time process for detecting laptop QR codes.
"""

import cv2
import numpy as np
import json
import subprocess
import time
from time import sleep
from pathlib import Path
from PIL import Image
from pyzbar.pyzbar import decode, ZBarSymbol
from ultralytics import YOLO
import multiprocessing as mp

# Import helper functions (assume these modules are available)
from camera_tunnig_helpers import (
    set_camera_settings, camera_current_setting,
    contrast_ctrl, brightness_ctrl, zoom_ctrl,
    focus_ctrl, exposure_ctrl, auto_focus
)
from function_for_app import (
    crop_frame, adaptive_threshold, preprocess_coordinates,
    make_detected_list, make_decoded, draw_vertical_lines_train_shelves,
    draw_vertical_lines_train_laptop, draw_horizontal_lines, allocate_data,
    brightest_image_value, qr_codes_preprocess, cv2_to_PIL,
    find_best_coordinates, choose_shelf_QR_coordinates_train_shelves,
    choose_laptop_QR_coordinates_train_laptop, find_shelf_coordinates, 
    crop_QR_for_led_light_detection
)
from detect_led_blubs import LED_on_detection, motion_detection
from send_request import ConnectRPI

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def try_cast_device(device):
    try:
        return int(device)
    except Exception:
        return device

def print_camera_properties(cap):
    print("Width =", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height =", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Framerate =", cap.get(cv2.CAP_PROP_FPS))
    print("Format =", cap.get(cv2.CAP_PROP_FORMAT))

def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError

def merge_decoded_and_detected(decoded, detected, frame_num, merged_datas=None):
    if merged_datas is None:
        merged_datas = {}
    for detected_coord in detected:
        if detected_coord not in list(decoded.values()):
            merged_datas[f"None-{frame_num}:{detected.index(detected_coord)}"] = detected_coord
    return merged_datas

def difference_each_frame(shelf_and_laptop_QR, shelf_and_laptop_QR_s):
    diffrence_each_frame = {}
    for shelf_s, laptop_s in shelf_and_laptop_QR_s.items():
        for shelf, laptop in shelf_and_laptop_QR.items():
            if shelf_s == shelf and laptop_s != laptop:
                diffrence_each_frame[shelf] = laptop_s
    return diffrence_each_frame

def diffrence_refrence(shelf_and_laptop_QR, shelf_and_laptop_QR_s):
    difference_with_allocated = {shelf: "None-" for shelf, laptop in shelf_and_laptop_QR.items()}
    for shelf, laptop in shelf_and_laptop_QR.items():
        for shelf_s, laptop_s in shelf_and_laptop_QR_s.items():
            if shelf_s == shelf and laptop_s == laptop:
                difference_with_allocated[shelf] = laptop_s
    return difference_with_allocated

# ------------------------------------------------------------------------------
# Core Components as Classes
# ------------------------------------------------------------------------------

class CameraHandler:
    """Encapsulates all camera-related logic."""
    
    def __init__(self, device, width=1920, height=1080):
        self.device = try_cast_device(device)
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print_camera_properties(self.cap)
    
    def set_settings(self):
        # Setup common settings (exposure, focus, etc.)
        set_camera_settings(self.cap)
    
    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def update_params(self, key):
        Contrast = contrast_ctrl(self.cap.get(cv2.CAP_PROP_CONTRAST), key, self.cap)
        Brightness = brightness_ctrl(self.cap.get(cv2.CAP_PROP_BRIGHTNESS), key, self.cap)
        zoom = zoom_ctrl(self.cap.get(cv2.CAP_PROP_ZOOM), key, self.cap)
        Focus = focus_ctrl(self.cap.get(cv2.CAP_PROP_FOCUS), key, self.cap)
        Exposure = exposure_ctrl(self.cap.get(cv2.CAP_PROP_EXPOSURE), key, self.cap)
        return Contrast, Brightness, zoom, Focus, Exposure

    def release(self):
        self.cap.release()


class VideoWriterHandler:
    """Handles video writing operations."""
    
    def __init__(self, filename='output55.avi', fps=30.0, resolution=(1920, 1080), codec='XVID'):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.out = cv2.VideoWriter(filename, fourcc, fps, resolution)
    
    def write_frame(self, frame):
        self.out.write(frame)
    
    def release(self):
        self.out.release()


class RPICommunicator:
    """Handles communication with the RPI."""
    
    def __init__(self, server_ip="10.42.0.1", server_port=8888):
        self.conn = ConnectRPI(server_ip=server_ip, server_port=server_port)
        # self.conn.connect()   # Connect if necessary
    
    def send(self, data):
        self.conn.send_data(data)
    
    def receive(self):
        return self.conn.receive_data()


class QRProcessor:
    """Handles QR detection, decoding, and post processing."""
    
    def __init__(self, model_path='threading_docker/new_model_QR.pt', num_rows=2, num_cols=2):
        self.model = YOLO(model_path)
        self.num_rows = num_rows
        self.num_cols = num_cols

    def process_frames(self, frames, ll, kk):
        # Apply adaptive thresholding and run the model
        thresh_frames = adaptive_threshold(frames, l=ll, k=kk)
        return self.model(thresh_frames, show=False, device=0, conf=0.55)

    def get_best_coordinates(self, frame, results, frame_dict):
        return find_best_coordinates(frame, results, frame_dict, self.num_cols, self.num_rows)

    def process_frame(self, frame, ll, kk):
        frame_dict = crop_frame(self.num_rows, self.num_cols, frame)
        frames = [img for _, img in frame_dict[0].items()]
        results = self.process_frames(frames, ll, kk)
        best_coords = self.get_best_coordinates(frame, results, frame_dict)
        centers, x_mod, y_mod, y_list, x_list = preprocess_coordinates(best_coords)
        result_rows = []
        for i in range(self.num_rows):
            start_index = i * self.num_cols
            end_index = (i + 1) * self.num_cols
            result_rows.append(np.concatenate(frames[start_index:end_index], axis=1))
        result_image = np.concatenate(result_rows, axis=0)
        return result_image, centers, x_mod, y_mod, y_list, x_list, frame_dict


class LEDProcessor:
    """Handles LED detection and thresholding."""
    
    def process_led_detection(self, LED_detection_area_list, image_brightest_value):
        threshes = {}
        masks = []
        for code, led_area in LED_detection_area_list.items():
            led_on_detected, mask, thresh = LED_on_detection(led_area, image_brightest_value - 10, 50)
            masks.append(mask)
            threshes[f"{code}"] = led_on_detected
        return threshes, masks


# ------------------------------------------------------------------------------
# Application Controllers
# ------------------------------------------------------------------------------

class AutomateDetectionApp:
    """
    Orchestrates the flow for LED number and shelf QR code allocation.
    This routine runs the first process.
    """
    
    def __init__(self, device, port):
        self.camera = CameraHandler(device)
        self.video_writer = VideoWriterHandler('output55.avi')
        self.rpi_comm = RPICommunicator(server_ip="10.42.0.1", server_port=port)
        self.qr_processor = QRProcessor()
        self.save_frames = []
        self.draw = False
        # Initialize other variables
        self.frame_number = 0
        self.frame_num = 0
        self.focus_step = 0
        self.decoded = {}
        self.detected = []
        self.all_QR_codes = {}
        self.ll = 11
        self.kk = 2

    def run(self):
        self.camera.set_settings()
        trash = []
        hold_led = False
        global response

        while True:
            self.frame_number += 1
            self.frame_num += 1
            ret, frame = self.camera.read_frame()
            if not ret:
                break

            k = cv2.waitKey(1)
            camera_current_setting(k, self.camera.cap)
            Contrast, Brightness, zoom, Focus, Exposure = self.camera.update_params(k)

            # Process the frame by splitting it into sub-frames
            result_image, centers, x_mod, y_mod, y_list, x_list, frame_dict = self.qr_processor.process_frame(frame, self.ll, self.kk)
            
            # Auto focus control logic
            if self.frame_number % 6 == 0 and self.frame_number < (14 * 6):
                self.focus_step, Focus = auto_focus(self.focus_step, self.camera.cap)
            if self.frame_number == (14 * 6 - 1):
                self.camera.cap.set(cv2.CAP_PROP_FOCUS, 29)
                Focus = 29
            if self.frame_number == (14 * 6 + 14):
                k = ord("z")
            if k == ord("m"):
                # Reset training parameters
                self.frame_number = 0
                self.focus_step = 0
            
            if k == ord('i'):
                self.kk += 1
            elif k == ord('k'):
                self.kk -= 1
            if k == ord('o'):
                self.ll += 2
            elif k == ord('l'):
                self.ll -= 2

            # Process QR regions and decode
            self.detected = make_detected_list(centers, x_mod, y_mod, self.detected)
            # Process each detected region
            for detected_xyxy in self.detected:
                diff1_x = diff2_y = diff1_y = diff2_x = 50
                color = (255, 0, 0)
                is_detected = False
                x1, y1, x2, y2 = detected_xyxy
                if x1 - diff1_x <= 0:
                    diff1_x = 0
                if y1 - diff1_y <= 0:
                    diff1_y = 0
                if x2 + diff2_x >= result_image.shape[0]:
                    diff2_x = 0
                if y2 + diff2_y >= result_image.shape[1]:
                    diff2_y = 0
                cropped_qr = result_image[(y1 - diff1_y):(y2 + diff2_y), (x1 - diff1_x):(x2 + diff2_x)].copy()
                print("QR crop shape:", cropped_qr.shape)
                if len(cropped_qr) != 0:
                    self.all_QR_codes[(x1, y1, x2, y2)] = f"None-{self.frame_num}{self.frame_num+1}"
                    cropped_qr = cv2_to_PIL(cropped_qr)
                    print("Cropped QR size:", cropped_qr.size)
                    cropped_qr = qr_codes_preprocess(cropped_qr)
                    barcodes = decode(cropped_qr)
                    for barcode in barcodes:
                        data = barcode.data.decode("utf-8")
                        print("Decoded data:", data)
                        if data:
                            is_detected = True
                            self.decoded[data] = (x1, y1, x2, y2)
                            self.all_QR_codes[(x1, y1, x2, y2)] = data
                        if barcode and data and not self.draw and is_detected:
                            color = (0, 255, 0)
                            cv2.rectangle(result_image,
                                          (x1 - diff1_x, y1 - diff1_y),
                                          (x2 + diff2_x, y2 + diff2_y),
                                          color, 2)
            
            if k == ord('z'):
                # Merge and allocate QR codes for shelf detection
                merged_data = self.decoded.copy()
                for d in self.detected:
                    if d not in list(self.decoded.values()):
                        merged_data[f"None-{self.frame_num}:{self.detected.index(d)}"] = d
                horizontal_lines_coordinates = y_list.copy()
                QR_data_final = {"y_mean": find_shelf_coordinates(horizontal_lines_coordinates, centers_neighbourhood_thrsh=60)}
                shelf_detected_QR = choose_shelf_QR_coordinates_train_shelves(merged_data, QR_data_final["y_mean"])
                with open('threading_docker/shelf_QR_data.txt', 'w') as f:
                    f.write(json.dumps(shelf_detected_QR, default=convert))
                with open('threading_docker/y_mean_finall.txt', 'w') as f:
                    f.write(json.dumps(QR_data_final, default=convert))
                self.draw = True
            elif k == ord('x'):
                self.draw = False

            # When in draw mode, perform LED detection and send results to RPI
            if self.draw:
                result_image = frame.copy()
                yc1, yc2, xc1, xc2 = 170, 1080, 0, 1920
                image_brightest_value = brightest_image_value(result_image[yc1:yc2, xc1:xc2])
                y_mean = find_shelf_coordinates(y_list, centers_neighbourhood_thrsh=60)
                shelf_detected_QR = choose_shelf_QR_coordinates_train_shelves(merged_data, y_mean)
                blub_image = frame.copy()
                _, led_area = draw_vertical_lines_train_shelves(shelf_detected_QR, result_image, y_mean, offset=0, draws=True)
                LED_detection_area_list = crop_QR_for_led_light_detection(blub_image, led_area, y_mean)
                led_processor = LEDProcessor()
                threshes, _ = led_processor.process_led_detection(LED_detection_area_list, image_brightest_value)
                if True in list(threshes.values()) and not hold_led:
                    code_to_send = [code for code, detected in threshes.items() if detected]
                    light_detected = f"{code_to_send}"
                self.rpi_comm.send(light_detected)
                print("light_detected:", light_detected)
                response = self.rpi_comm.receive()
                print("response from RPI:", response)
                if response == "END":
                    k = 27

            cv2.putText(result_image, f"decoded : {len(list(self.decoded.keys()))}", (730, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 125, 255), 2)
            cv2.putText(result_image, f"detected : {len(self.detected)}", (760, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(result_image, f"zoom: {zoom} brightness: {Brightness} focus: {self.camera.cap.get(cv2.CAP_PROP_FOCUS)} exposure: {Exposure} contrast: {Contrast}",
                        (130, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, 1)
            if k == ord('t'):
                self.save_frames.append(result_image)
                print("saving begins")
            cv2.imshow('show', result_image)

            if k == 27:
                for saved_frame in self.save_frames:
                    self.video_writer.write_frame(saved_frame)
                self.rpi_comm.send("close_connect")
                self.rpi_comm.send("close")
                break

        self.camera.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        print("Automate Detection Finished")
        print("trash:", trash)
        print("decoded:", self.decoded)
        print("All QR codes:", self.all_QR_codes)
        print("Shelf detected QR:", shelf_detected_QR)
        print("Merged Data:", merged_data)


class AllocateLaptopAndShelfApp:
    """
    Handles the allocation of laptop and shelf QR codes.
    This routine loads pre-saved shelf data and processes laptop QR codes.
    """
    
    def __init__(self, device, port):
        self.camera = CameraHandler(device)
        self.video_writer = VideoWriterHandler('output55.avi')
        self.rpi_comm = RPICommunicator(server_port=port)
        self.qr_processor = QRProcessor()
        self.save_frames = []
        self.draw = False
        self.frame_number = 0
        self.frame_num = 0
        self.decoded = {}
        self.detected = []
        self.ll = 11
        self.kk = 2

        # Load shelf data and y_mean from file
        with open('threading_docker/shelf_QR_data.txt', 'r') as f:
            self.shelf_QR_data = json.load(f)
        with open('threading_docker/y_mean_finall.txt', 'r') as f:
            self.y_mean_dict = json.load(f)
        self.y_mean = self.y_mean_dict['y_mean']

    def run(self):
        self.camera.set_settings()
        trash = []
        hold_led = False

        while self.camera.cap.isOpened():
            self.frame_number += 1
            self.frame_num += 1
            ret, frame = self.camera.read_frame()
            if not ret:
                break
            k = cv2.waitKey(1)
            camera_current_setting(k, self.camera.cap)
            Contrast, Brightness, zoom, Focus, Exposure = self.camera.update_params(k)
            frame_dict = crop_frame(self.qr_processor.num_rows, self.qr_processor.num_cols, frame)
            self.camera.cap.set(cv2.CAP_PROP_FOCUS, 29)
            if k == ord("m"):
                self.frame_number = 0
            if k == ord('i'):
                self.draw = "workTime"
            elif k == ord('k'):
                self.kk -= 1
            if k == ord('o'):
                self.ll += 2
            elif k == ord('l'):
                self.ll -= 2

            frames = [img for _, img in frame_dict[0].items()]
            if self.draw == False:
                results = self.qr_processor.process_frames(frames, self.ll, self.kk)
                best_coordinates = self.qr_processor.get_best_coordinates(frame, results, frame_dict)
                centers, x_mod, y_mod, y_list, x_list = preprocess_coordinates(best_coordinates)
                result_rows = [np.concatenate(frames[i * self.qr_processor.num_cols:(i + 1) * self.qr_processor.num_cols], axis=1)
                               for i in range(self.qr_processor.num_rows)]
                result_image = np.concatenate(result_rows, axis=0)
                image_brightest_value = brightest_image_value(result_image[400:result_image.shape[1], 0:result_image.shape[0]])
                self.detected = make_detected_list(centers, x_mod, y_mod, self.detected)
                decoded = make_decoded(self.detected, result_image, self.decoded, draw=True)
                merged_data = merge_decoded_and_detected(decoded, self.detected, self.frame_num)
            if self.draw == False:
                result_image = frame.copy()
                image_brightest_value = brightest_image_value(result_image[400:result_image.shape[1], 0:result_image.shape[0]])
                laptop_detected_QR = choose_laptop_QR_coordinates_train_laptop(merged_data, y_mean=self.y_mean)
                _, laptop_QR_data = draw_vertical_lines_train_laptop(laptop_detected_QR, result_image, self.y_mean, offset=10, draws=True)
                for xyxy in self.detected:
                    xx1, yy1, xx2, yy2 = xyxy   
                    cv2.rectangle(result_image, (xx1 - 55, yy1 - 55), (xx2 + 55, yy2 + 55), (0, 0, 255), 2)
                for data, xyxy in decoded.items():
                    xx1, yy1, xx2, yy2 = xyxy
                    cv2.rectangle(result_image, (xx1 - 50, yy1 - 50), (xx2 + 50, yy2 + 50), (0, 255, 0), 2)
                draw_horizontal_lines(result_image, y_list, x_list, self.y_mean)
                shelf_and_laptop_QR = allocate_data(laptop_QR_data, self.shelf_QR_data, thresh=70)
                self.rpi_comm.send(f"{shelf_and_laptop_QR}")
                with open('threading_docker/allocated_laptops_QRs.txt', 'w') as f:
                    f.write(json.dumps(shelf_and_laptop_QR, default=convert))
                if self.frame_num == 120:
                    k = 27
            cv2.putText(result_image, f"decoded : {len(list(decoded.keys()))}", (730, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 125, 255), 2)
            cv2.putText(result_image, f"detected : {len(self.detected)}", (760, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(result_image, f"zoom: {zoom} brightness: {Brightness} focus: {self.camera.cap.get(cv2.CAP_PROP_FOCUS)} exposure: {Exposure} contrast: {Contrast}",
                        (130, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, 1)
            if k == ord('t'):
                self.save_frames.append(result_image)
                print("saving begins")
            cv2.imshow('show', result_image)
            if k == 27:
                for f in self.save_frames:
                    self.video_writer.write_frame(f)
                self.rpi_comm.send("close")
                break
        self.camera.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        print("Allocate Laptop and Shelf Finished")
        print("Detected QR:", self.detected)
        print("Decoded Data:", self.decoded)
        print("Shelf QR Data:", self.shelf_QR_data)
        print("Merged Data:", merged_data)
        print("Shelf and Laptop QR:", shelf_and_laptop_QR)


class RealTimeApp:
    """
    Handles the real-time detection process with motion detection incorporated.
    This routine processes video frames, detects motion, and dynamically allocates laptop QR data.
    """
    
    def __init__(self, device, port):
        self.camera = CameraHandler(device)
        self.video_writer = VideoWriterHandler('output55.avi')
        self.rpi_comm = RPICommunicator(server_port=port)
        self.qr_processor = QRProcessor()
        self.save_frames = []
        self.frame_number = 0
        self.frame_num = 0
        self.decoded = {}
        self.detected = []
        self.all_QR_codes = {}
        self.ll = 11
        self.kk = 2
        self.draw = False
        self.static_back = None
        self.motion_list = [None, None]
        self.diffrence_log = []
        # Load allocated QR and shelf data
        with open('threading_docker/allocated_laptops_QRs.txt', 'r') as f:
            self.allocated_laptops_QRs = json.load(f)
        with open('threading_docker/shelf_QR_data.txt', 'r') as f:
            self.shelf_QR_data = json.load(f)
        with open('threading_docker/y_mean_finall.txt', 'r') as f:
            self.y_mean_dict = json.load(f)
        self.y_mean = self.y_mean_dict['y_mean']
        self.shelf_and_laptop_QR_update = self.allocated_laptops_QRs.copy()

    def run(self):
        self.camera.set_settings()
        trash = []
        hold_led = False
        draws = False
        while self.camera.cap.isOpened():
            ret, frame = self.camera.read_frame()
            if not ret:
                break

            # ---------------- Motion Detection ---------------
            out_motion = False
            motion = 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if self.static_back is None:
                self.static_back = gray
                continue
            self.static_back, out_motion, draws, self.motion_list, motion = motion_detection(
                self.static_back, gray, frame, draws, self.motion_list, motion, out_motion, area=50
            )
            # ---------------------------------------------------

            self.frame_number += 1
            self.frame_num += 1
            k = cv2.waitKey(1)
            camera_current_setting(k, self.camera.cap)
            Contrast, Brightness, zoom, Focus, Exposure = self.camera.update_params(k)
            frame_dict = crop_frame(self.qr_processor.num_rows, self.qr_processor.num_cols, frame)
            self.camera.cap.set(cv2.CAP_PROP_FOCUS, 29)
            if k == ord("m"):
                self.frame_number = 0
            if k == ord('i'):
                self.draw = True
            elif k == ord('k'):
                self.kk -= 1
            if k == ord('o'):
                self.ll += 2
            elif k == ord('l'):
                self.ll -= 2

            frames = [img for _, img in frame_dict[0].items()]
            if self.frame_num % 2 == 0:
                results = self.qr_processor.process_frames(frames, self.ll, self.kk)
                best_coordinates = self.qr_processor.get_best_coordinates(frame, results, frame_dict)
                centers, x_mod, y_mod, y_list, x_list = preprocess_coordinates(best_coordinates)
                result_rows = [np.concatenate(frames[i * self.qr_processor.num_cols:(i + 1) * self.qr_processor.num_cols], axis=1)
                               for i in range(self.qr_processor.num_rows)]
                result_image = np.concatenate(result_rows, axis=0)
                image_brightest_value = brightest_image_value(result_image[400:result_image.shape[1], 0:result_image.shape[0]])
                self.detected = make_detected_list(centers, x_mod, y_mod, self.detected)
                decoded = make_decoded(self.detected, result_image, self.decoded, draw=False)
                merged_data = merge_decoded_and_detected(decoded, self.detected, self.frame_num)
                laptop_detected_QR = choose_laptop_QR_coordinates_train_laptop(merged_data, y_mean=self.y_mean)
                laptop_detected_QR_s = choose_laptop_QR_coordinates_train_laptop(merged_data, y_mean=self.y_mean)
                _, laptop_QR_data = draw_vertical_lines_train_laptop(laptop_detected_QR, result_image, self.y_mean, offset=30, draws=False)
                _, laptop_QR_data_s = draw_vertical_lines_train_laptop(laptop_detected_QR_s, result_image, self.y_mean, offset=30, draws=False)
                for xyxy in self.detected:
                    xx1, yy1, xx2, yy2 = xyxy   
                    cv2.rectangle(result_image, (xx1 - 55, yy1 - 55), (xx2 + 55, yy2 + 55), (0, 0, 255), 2)
                for data, xyxy in decoded.items():
                    xx1, yy1, xx2, yy2 = xyxy
                    cv2.rectangle(result_image, (xx1 - 50, yy1 - 50), (xx2 + 50, yy2 + 50), (0, 255, 0), 2)
                for data, xyxy in laptop_QR_data.items():
                    xx1, yy1, xx2, yy2 = xyxy
                    cv2.rectangle(result_image, (xx1, self.y_mean), (xx2, result_image.shape[1]), (0, 255, 255), 2)
                draw_horizontal_lines(result_image, y_list, x_list, self.y_mean)
                shelf_and_laptop_QR = allocate_data(laptop_QR_data, self.shelf_QR_data, thresh=60)
                shelf_and_laptop_QR_s = allocate_data(laptop_QR_data_s, self.shelf_QR_data, thresh=60)
                if out_motion:
                    decoded = {}
                    self.detected = []
                    self.draw = False
                elif k == ord("v"):
                    self.draw = True

                if self.frame_num % 2 != 0:
                    diffs = difference_each_frame(shelf_and_laptop_QR, shelf_and_laptop_QR_s)
                    self.diffrence_log.append([s for s, l in diffs.items()])
                elif self.frame_num % 2 == 0:
                    self.shelf_and_laptop_QR_update = shelf_and_laptop_QR_s.copy()
                    all_shelf_QRs = []
                    for diff in self.diffrence_log:
                        for shelf in diff:
                            all_shelf_QRs.append(shelf)
                    num_of_repetiton = {QR: all_shelf_QRs.count(QR) for QR in all_shelf_QRs}
                    updated = {QR: "None-" for QR, count in num_of_repetiton.items() if count > 1}
                    self.shelf_and_laptop_QR_update.update(updated)
                    with open("threading_docker/trash.txt", 'w') as f:
                        f.write(json.dumps(updated, default=convert))
                    self.diffrence_log = []
                difference_with_allocated = diffrence_refrence(self.allocated_laptops_QRs, shelf_and_laptop_QR)
                self.rpi_comm.send(f"{difference_with_allocated}")
                cv2.putText(result_image, f"decoded : {len(list(decoded.keys()))}", (730, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 125, 255), 2)
                cv2.putText(result_image, f"detected : {len(self.detected)}", (760, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(result_image, f"zoom: {zoom} brightness: {Brightness} focus: {self.camera.cap.get(cv2.CAP_PROP_FOCUS)} exposure: {Exposure} contrast: {Contrast}",
                            (130, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, 1)
                cv2.imshow('show', result_image)
                if k == ord('t'):
                    self.save_frames.append(result_image)
                    print("saving begins")
            if k == 27:
                for f in self.save_frames:
                    self.video_writer.write_frame(f)
                self.rpi_comm.send("close")
                break
        self.camera.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        print("Real-Time Process Finished")
        print("Difference with allocated:", difference_with_allocated)

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Depending on which processing is needed, you would instantiate and run one of these apps.
    # Uncomment the desired line:
    
    # For automated detection (LED and shelf QR allocation)
    # app = AutomateDetectionApp(device=0, port=8888)
    # app.run()
    
    # For allocating laptop and shelf QR codes together
    # app = AllocateLaptopAndShelfApp(device=0, port=8888)
    # app.run()
    
    # For real-time processing (including motion detection)
    # app = RealTimeApp(device=0, port=8888)
    # app.run()
    pass
