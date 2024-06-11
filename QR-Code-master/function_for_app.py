import cv2
import numpy as np
import copy
import torch
from torchvision import transforms
# from roboflow import Roboflow
import copy
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
from pyzbar.pyzbar import decode, ZBarSymbol
from tqdm import tqdm
import os
# import heapq
# from ultralytics import YOLO
import statistics as st

from collections import defaultdict
import ast
import argparse
from sys import stderr
import multiprocessing as mp
# from pathlib import Path
from time import time

import math
# from itertools import combinations

from detect_led_blubs import LED_on_detection, detect_thresh_range
import onnxruntime as ort
import onnx
from torchvision import transforms


def reassemble_cropped_parts(cropped_parts, original_shape, num_rows, num_cols):
    original_image = np.zeros(original_shape, dtype=cropped_parts[0][0, 0].dtype)
    row_height = original_shape[0] // num_rows
    col_width = original_shape[1] // num_cols
    
    for i in range(num_rows):
        for j in range(num_cols):
            start_y = i * row_height
            end_y = (i + 1) * row_height
            start_x = j * col_width
            end_x = (j + 1) * col_width
            original_image[start_y:end_y, start_x:end_x] = cropped_parts[i*num_cols + j][0, 0]
    
    return original_image


def crop_frame(num_rows, num_cols, image):
    cropped_image_list = []
    index = {}
    for i in range(num_rows):
        for j in range(num_cols):
            start_y = i * (image.shape[0] // num_rows)
            end_y = (i + 1) * (image.shape[0] // num_rows) 

            start_x = j * (image.shape[1] // num_cols)
            end_x = (j + 1) * (image.shape[1] // num_cols)

            sub_image = image[start_y:end_y, start_x:end_x]
            index[(i,j)] = sub_image
            cropped_image_list.append(index)

    return cropped_image_list



def crop_frame_updated(num_rows, num_cols, image, stridy, stridx):
    cropped_image_list = []
    index = {}
    frames = []

    for i in range(num_rows):
        for j in range(num_cols):

            start_y = i * (image.shape[0] // num_rows - stridy)
            end_y = (i + 1) * (image.shape[0] // num_rows + stridy) 

            start_x = j * (image.shape[1] // num_cols - stridx)
            end_x = (j + 1) * (image.shape[1] // num_cols + stridx)

            if end_y > image.shape[0]:
                end_y = min(end_y, image.shape[0])
            if end_x > image.shape[1]:
                end_x = min(end_x, image.shape[1])
            
            if start_x < 0:
                start_x = max(start_x, image.shape[0])
            if start_y < 0:
                start_y = max(start_y, image.shape[1])
            
            sub_image = image[start_y:end_y, start_x:end_x]
            index[(i,j)] = sub_image
            # frames.append(sub_image)
            cropped_image_list.append(index)
    '''
    sub_image = image
    index[(1,1)] = sub_image
    frames.append(sub_image)
    cropped_image_list.append(index)
    '''


    for index,images in cropped_image_list[0].items():
        cstridx = stridx
        cstridy = stridy
        # if images.shape[1]-cstridx <= images.shape[1]:
        #     cstridx = 0
        # if images.shape[0]-cstridy <= images.shape[0]:
        #     cstridy = 0
        
        # if images.shape[1]+cstridx <= cstridy:
        #     cstridx = 0
        # if images.shape[0]+cstridy <= images.shape[0]:
        #     cstridy = 0
        # cv2.imwrite(f"test_app_long_distance/images/reconstructed_image_{index}.jpg", image)
        # frames.append(images[images.shape[0]+cstridy:images.shape[0]-cstridy, images.shape[1]+cstridx:images.shape[1]-cstridx])
        frames.append(images)
    
    return cropped_image_list, frames


# def predict_yolo(image: list, save=False, conf=0.2, device=0, iou=0.6):

#     model_QR = YOLO('/home/amir/pytorch_env/WLED/models/WLED.v2i.yolov8_QRCODE_MODEL.pt')
#     results = model_QR(image, save=save, conf=conf, device=device, iou=iou)
#     return results


def zoom_in(image, zoom_factor):
    # Resize the image to zoom in
    height, width, _ = image.shape
    new_height = int(height * zoom_factor)
    new_width = int(width * zoom_factor)
    zoomed_image = cv2.resize(image, (new_width, new_height))
    return zoomed_image



def tune_exposure_multiple_frames(images: list):
    # input multiple frames exactly from one scene with different exposure and
    # camera shutter speed

    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)

    mergeMertens = cv2.createMergeMertens(exposure_weight=0)
    exposureFusion = mergeMertens.process(images)

    res_mertens_8bit = np.clip(exposureFusion*255, 0, 255).astype('uint8')

    return res_mertens_8bit

def adjust_exposure(image, alpha, beta):
    # Apply exposure adjustment using the formula: output = alpha * input + beta
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def highlight_and_contrast(image, thresh):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray_image, (5, 5), 1)

    if type(thresh) == str:
        _, mask = cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif type(thresh) == int:
        _, mask = cv2.threshold(blured, thresh, 255, cv2.THRESH_BINARY)

    highlighted_image = cv2.bitwise_and(image, image, mask=mask)
    # highlighted_image = cv2.cvtColor(highlighted_image, cv2.COLOR_GRAY2RGB)

    return highlighted_image


def correct_uneven_lighting(image, clip_limit=2.0, tile_size=(8, 8)):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (5, 5), 1)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    corrected_image = clahe.apply(blured)

    corrected_color = cv2.cvtColor(corrected_image, cv2.COLOR_GRAY2BGR)

    return corrected_color



def adaptive_threshold(images, l=7, k=2):

    if type(images) != list:

        gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

        # adaptive_threshold_gaussian = cv2.adaptiveThreshold(PIL_to_cv2(gray), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # for l in [5,7,11,15]:
            # for k in [1,2]:
        adaptive_threshold_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, l, k)

        corrected_colors = cv2.cvtColor(adaptive_threshold_mean, cv2.COLOR_GRAY2BGR)

    else:

        corrected_colors = []

        for image in images:

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # adaptive_threshold_gaussian = cv2.adaptiveThreshold(PIL_to_cv2(gray), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            # for l in [5,7,11,15]:
                # for k in [1,2]:
            adaptive_threshold_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, l, k)

            # adaptive_threshold_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            corrected_color = cv2.cvtColor(adaptive_threshold_mean, cv2.COLOR_GRAY2BGR)

            corrected_colors.append(corrected_color)

    return corrected_colors
# def predict_roboflow(images: list, save=False, conf=30, device=0, iou=60):

#     results = {}

#     rf = Roboflow(api_key="lQ7Syo1ciDIbMhGzMf2H")
#     project = rf.workspace().project("wled")
#     model = project.version(2).model


#     i = 0
#     for image in images:
#       QR_results = model.predict(np.array(image), confidence=conf, overlap=iou)

#       QR_in_one_grid = []

#       for QR in QR_results.json()["predictions"]:
#         QR_coord = (int(QR['x']),
#                     int(QR['y']),
#                     int(QR['height']),
#                     int(QR['width']))

#         QR_in_one_grid.append(QR_coord)

#       results[f"img_{i}"] = QR_in_one_grid
#       i += 1
#     return results

def prep(image):
    image = highlight_and_contrast(image, thresh='OTSU')
    image = correct_uneven_lighting(image, clip_limit=13.0 ,tile_size=(64, 64))
    image = highlight_and_contrast(image, thresh='OTSU')
    return image


def give_max_color(pil_image, num=5):
    img = pil_image.convert('RGB')
    img = img.convert('P')
    color_counts = img.getcolors()
    output = []
    max_color = {}
    for count, color in color_counts[:]:
        max_color[color] = count

    max_value = [color for color,count in max_color.items() if count == np.max(list(max_color.values()))]

    return max_value

# def calculate_distance(point1, point2):
#     return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# def filter_points(points, threshold):
#     return list(set([tuple(p1) for p1, p2 in combinations(points, 2) if calculate_distance(p1, p2) > threshold]))


def calculate_distance(center1, center2):
    return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


def filter_duplicate_centers(centers, distance_threshold):
    filtered_centers = []
    n = len(centers)

    for i in range(n):
        keep_center = True
        for j in range(i + 1, n):
            distance = calculate_distance(centers[i], centers[j])
            if distance <= distance_threshold:
                keep_center = False
                break

        if keep_center:
            filtered_centers.append(centers[i])

    return filtered_centers


def cv2_to_PIL(image):

    return Image.fromarray(image)


def PIL_to_cv2(pil_image, gray=False):
  numpy_image = np.array(pil_image)
  if gray == True:
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
  else:
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
  return opencv_image


def qr_codes_preprocess(image, lll=17, kkk=5):

    scalar = 1.5
    # pillow image as input
    x3, y3 = image.size
 
    image_scaled = image.resize((int(x3*scalar)+1, int(y3*scalar)+1))
    # for sharpness in [0.5, 1]:

    max_value = give_max_color(image_scaled)

    image_scaled = ImageOps.autocontrast(image_scaled)
    gray = cv2.cvtColor(PIL_to_cv2(image_scaled), cv2.COLOR_BGR2GRAY)

    # adaptive_threshold_gaussian = cv2.adaptiveThreshold(PIL_to_cv2(gray), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # for l in [5,7,11,15]:
        # for k in [1,2]:
    # adaptive_threshold_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2)
    adaptive_threshold_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, lll, kkk)
    # im = PIL_to_cv2(image_scaled)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # th, im = cv2.threshold(im, max_value[0], 255, cv2.THRESH_OTSU & cv2.THRESH_BINARY)
    im = Image.fromarray(adaptive_threshold_gaussian)

    image_scaled_sharp = ImageEnhance.Sharpness(im).enhance(0.5)

    auto = PIL_to_cv2(image_scaled_sharp)

    return auto


def remove_duplicates_and_close(rectangles, threshold_distance=10):
    cleaned_rectangles = []
    
    for rect in rectangles:
        is_unique = True
        
        # Check if the current rectangle is too close to any of the previously processed rectangles
        for other_rect in cleaned_rectangles:
            if abs(rect[0] - other_rect[0]) < threshold_distance and \
               abs(rect[1] - other_rect[1]) < threshold_distance and \
               abs(rect[2] - other_rect[2]) < threshold_distance and \
               abs(rect[3] - other_rect[3]) < threshold_distance:
                is_unique = False
                break
        
        # If the rectangle is unique, add it to the cleaned list
        if is_unique:
            cleaned_rectangles.append(rect)
    
    return cleaned_rectangles


def find_best_coordinates_torch(image, results_list, cropped_image_list, num_cols, num_rows, stridx=0, stridy=0):
    
    best_coordinates = []
    QR_coordinates = []

    indices = [index for index,image in cropped_image_list[0].items()]
    for result in results_list:
        # if len(result.boxes.xyxy.cpu().numpy().astype(int)) != 0:
        if True:
            for QR in result:
                i, j = indices[results_list.index(result)]
                # coord = QR.boxes.xyxy.cpu().numpy().astype(int)
                # x1, y1, x2, y2 = coord[0]
                coord = torch.tensor(QR).cpu().numpy().astype(int)
                x1, y1, x2, y2 = coord
                diff_x = j * ((image.shape[1] // num_cols) - stridx) 
                diff_y = i * ((image.shape[0] // num_rows) - stridy)
                if (abs(x2-x1) < abs(y2-y1) + 20) and (abs(x2-x1) > abs(y2-y1) - 20):
                    QR_coordinates.append([x1+diff_x, y1+diff_y, x2+diff_x, y2+diff_y])
            best_coordinates.append(QR_coordinates)
    return best_coordinates


def find_best_coordinates(image, results_list, cropped_image_list, num_cols, num_rows, stridx=0, stridy=0):
    
    best_coordinates = []
    QR_coordinates = []

    indices = [index for index,image in cropped_image_list[0].items()]
    for result in results_list:
        if len(result.boxes.xyxy.cpu().numpy().astype(int)) != 0:
            for QR in result:
                i, j = indices[results_list.index(result)]
                coord = QR.boxes.xyxy.cpu().numpy().astype(int)
                x1, y1, x2, y2 = coord[0]
                diff_x = j * ((image.shape[1] // num_cols) - stridx) 
                diff_y = i * ((image.shape[0] // num_rows) - stridy)
                if (abs(x2-x1) < abs(y2-y1) + 20) and (abs(x2-x1) > abs(y2-y1) - 20):
                    QR_coordinates.append([x1+diff_x, y1+diff_y, x2+diff_x, y2+diff_y])
            best_coordinates.append(QR_coordinates)
    return best_coordinates


def distance(box1, box2):
  x1, y1, x2, y2 = box1
  x3, y3, x4, y4 = box2
  return np.sqrt(((x1 - x3) ** 2) + ((y1 - y3) ** 2) + ((x2 - x4) ** 2) + ((y2 - y4) ** 2))


def unique_all_QR(all_QR_codes):
  all1 = all_QR_codes.copy()
  all2 = all_QR_codes.copy()

  # Create a dictionary to store unique boxes
  unique_boxes = {}

  # Iterate through the original dictionary
  for key1, box1 in all1.items():
    for key2, box2 in all2.items():
      dist = distance(box1, box2)
      if len(key1) > 4: 
        unique_boxes[key1] = box1 
      if len(key2) > 4:
        unique_boxes[key2] = box2
      if len(key1) < 4 and len(key2) < 4 and dist < 0:
        unique_boxes[key1] = box1

  return unique_boxes 

def preprocess_coordinates_torch(best_coordinates):
    
    centers = []
    y_list = []
    x_list = []
    x_mod = 50
    y_mod = 50
    x_mean = 0
    y_mean = 0

    if len(best_coordinates) != 0:
        lengh_list = [len(x) for x in best_coordinates]
        #print(f"GOL GOL : {lengh_list}")
        for QR in best_coordinates:
            if len(QR) == max(lengh_list):
                QR_coordinates = QR.copy()
                break
        all_x_diff = []
        all_y_diff = []

        for QR_code_coords in QR_coordinates:
            #print(f"LOL LOL : {QR_code_coords}")
            x1, y1, x2, y2 = QR_code_coords
            all_x_diff.append(x2-x1)
            all_y_diff.append(y2-y1)
            center_x = (x1 + x2)//2
            center_y = (y1 + y2)//2
            centers.append([center_x, center_y])
            y_list.append(center_y)
            x_list.append(center_x)
            
            #all_x_diff = list(set(all_x_diff))
            print(f"all_x_diff : {all_x_diff}")
			#all_y_diff = list(set(all_y_diff))
        #if len(x_list) != 0 and len(y_list) != 0:
        #    y_mean = int(sum(y_list)//len(y_list))
        #    x_mean = int(sum(x_list)//len(x_list)) 
            #x_mod = int(st.mode(all_x_diff))
            #y_mod = int(st.mode(all_y_diff))
            #print(x_mod)

    decoded = {}
    undetected = []
    i = 0
    # for QR_code_coords in QR_coordinates:
    #   x1, y1, x2, y2 = QR_code_coords

    return centers, x_mod, y_mod, y_list, x_list

def preprocess_coordinates(best_coordinates):

    centers = []
    y_list = []
    x_list = []
    x_mod = 0
    y_mod = 0
    offset_y = 150
    offset_x = 80
    if len(best_coordinates) != 0:
        lengh_list = [len(x) for x in best_coordinates]
        for QR in best_coordinates:
            if len(QR) == max(lengh_list):
                QR_coordinates = QR.copy()
                break
        all_x_diff = []
        all_y_diff = []

        for QR_code_coords in QR_coordinates:
            x1, y1, x2, y2 = QR_code_coords
            all_x_diff.append(x2-x1)
            all_y_diff.append(y2-y1)
            center_x = (x1 + x2)//2
            center_y = (y1 + y2)//2
            centers.append([center_x, center_y])
            y_list.append(center_y)
            x_list.append(center_x)

        if len(x_list) != 0 and len(y_list) != 0:
            y_mean = int(np.mean(y_list))
            x_mean = int(np.mean(x_list))
            x_mod = int(st.mode(all_x_diff))
            y_mod = int(st.mode(all_y_diff))
            print(x_mod)

    decoded = {}
    undetected = []
    i = 0
    # for QR_code_coords in QR_coordinates:
    #   x1, y1, x2, y2 = QR_code_coords

    return centers, x_mod, y_mod, y_list, x_list


def union_list_dict(dict1, list1):
    final_dict = dict1.copy()

    # Add unique elements from the list to the dictionary
    for item in list1:
        # Try converting the item to a tuple efficiently using `ast.literal_eval`
        try:
            item_tuple = ast.literal_eval(item)
        except (ValueError, SyntaxError):
            # If conversion fails, assume the item is a string and create a new key
            item_tuple = item
            if item_tuple not in final_dict.values():
                    final_dict['None-' + str(len(final_dict))] = item_tuple

        # Add the unique element to the dictionary using a unique key
        if item_tuple not in final_dict.values():
            final_dict['None-' + str(len(final_dict))] = item_tuple


    return final_dict

def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def filter_coordinates(all_QR_codes):
    filtered_coords = defaultdict(list)
    non_12_digit_coords = {}

    # Separate 12-digit keys and non-12 digit keys
    for key, value in all_QR_codes.items():
        if len(key) > 5:
            filtered_coords[key] = value
        else:
            non_12_digit_coords[key] = value

    # Keep non-12 digit coordinates that are not too close to each other
    for key1, value1 in non_12_digit_coords.items():
        keep = True
        for key2, value2 in non_12_digit_coords.items():
            if key1 != key2:
                if euclidean_distance(value1, value2) < 10:  # Adjust this threshold as needed
                    keep = False
                    break
        if keep:
            filtered_coords[key1] = value1

    # Ensure that each non-12 digit key is paired with a 12 digit key
    for key, value in non_12_digit_coords.items():
        closest_12_digit_key = min(filtered_coords.keys(), key=lambda x: euclidean_distance(value, filtered_coords[x]))
        filtered_coords[closest_12_digit_key] = filtered_coords.pop(closest_12_digit_key)

    return filtered_coords

def choose_laptop_QR_coordinates(all_QR_codes, y_mean):
    # y_mean = int(np.mean(y_list))
    laptop_QRs = {}
    for code,coord in  all_QR_codes.items():
        x1,y1,x2,y2 = coord
        if y1 > y_mean:
            laptop_QRs[f"{code}"] = coord
    return laptop_QRs

def find_shelf_coordinates(horizontal_lines_coordinates, centers_neighbourhood_thrsh=50):

    mean = int(np.mean(horizontal_lines_coordinates))
    maxx = max(horizontal_lines_coordinates)
    minn = min(horizontal_lines_coordinates)
    d = abs(maxx - minn)
    if maxx - mean < centers_neighbourhood_thrsh:
        y_mean = maxx + 150
    else:
        y_mean = max(horizontal_lines_coordinates) - (d//2)
    # if d < centers_neighbourhood_thrsh:
    #     y_mean = max(horizontal_lines_coordinates) + 150
    # else:
        
    #     y_mean = max(horizontal_lines_coordinates) - (d//2) 

    return y_mean



def choose_shelf_QR_coordinates_train_shelves(all_QR_codes, y_mean):

    
    if type(all_QR_codes) == list:
        # y_mean = int(np.max(y_list) - np.min(y_list)) + 200
        shelf_QRs = []
        for coord in  all_QR_codes:
            x1,y1,x2,y2 = coord
            if y1 < y_mean:
                shelf_QRs.append(coord)
    else:
        # y_mean = int(np.max(y_list) - np.min(y_list)) + 200
        shelf_QRs = {}
        for code,coord in  all_QR_codes.items():
            x1,y1,x2,y2 = coord
            if y1 < y_mean:
                shelf_QRs[f"{code}"] = coord
    return shelf_QRs


def choose_laptop_QR_coordinates_train_laptop(all_QR_codes, y_mean):
    if type(all_QR_codes) == list:
        # y_mean = int(np.mean(y_mean))
        laptop_QRs = []
        for coord in  all_QR_codes:
            x1,y1,x2,y2 = coord
            if y1 < y_mean:
                laptop_QRs.append(coord)
    else:
        # y_mean = int(np.mean(y_mean))
        laptop_QRs = {}
        for code,coord in all_QR_codes.items():
            x1,y1,x2,y2 = coord
            if y1 > y_mean:
                laptop_QRs[f"{code}"] = coord
    return laptop_QRs


def draw_vertical_lines_train_laptop(laptop_detected_QR, result_image, y_mean, offset=80, draws=True):
    # y_mean = int(np.mean(y_list))
    if type(laptop_detected_QR) == list:
        laptop_area = []
        for xyxy in laptop_detected_QR:
            xx1, yy1, xx2, yy2 = xyxy
            if yy1 < y_mean:
                if draws:
                    cv2.line(result_image, (xx1-offset, 0), (xx1-offset, result_image.shape[0]), (255,125,60), 4) 
                    cv2.line(result_image, (xx2+offset, 0), (xx2+offset, result_image.shape[0]), (255,125,60), 4) 
                laptop_area.append((xx1-offset, xx2+offset))
    else:
        # offset = 20
        laptop_area = {}
        for data,xyxy in laptop_detected_QR.items():
            xx1, yy1, xx2, yy2 = xyxy
            if yy1 > y_mean:
                if draws:
                    cv2.line(result_image, (xx1-offset, 0), (xx1-offset, result_image.shape[0]), (255,125,60), 4) 
                    cv2.line(result_image, (xx2+offset, 0), (xx2+offset, result_image.shape[0]), (255,125,60), 4) 
                laptop_area[f"{data}"] = (xx1-offset, yy1, xx2+offset, yy2)

    return result_image, laptop_area


def draw_vertical_lines_train_shelves(shelf_detected_QR, result_image, y_mean, offset, draws=True):
    # y_mean = int(np.mean(y_list)) 

    if type(shelf_detected_QR) == list:
        led_area = []
        for xyxy in shelf_detected_QR:
            xx1, yy1, xx2, yy2 = xyxy
            if yy1 < y_mean:
                if draws:
                    cv2.line(result_image, (xx1-offset, 0), (xx1-offset, result_image.shape[0]), (255,125,60), 4) 
                    cv2.line(result_image, (xx2+offset, 0), (xx2+offset, result_image.shape[0]), (255,125,60), 4) 
                led_area.append((xx1-offset, xx2+offset))
    else:
        # offset = 20
        led_area = {}
        for data,xyxy in shelf_detected_QR.items():
            xx1, yy1, xx2, yy2 = xyxy
            if yy1 < y_mean:
                if draws:
                    cv2.line(result_image, (xx1-offset, 0), (xx1-offset, result_image.shape[0]), (255,125,60), 4) 
                    cv2.line(result_image, (xx2+offset, 0), (xx2+offset, result_image.shape[0]), (255,125,60), 4) 
                led_area[f"{data}"] = (xx1-offset, xx2+offset)

    return result_image, led_area
    

def draw_horizontal_lines(result_image, y_list, x_list, y_mean):

    if x_list != [] and y_list != []:
        cv2.line(result_image, (0, y_mean), (result_image.shape[1], y_mean), (255,255,0), 15)
    
    return result_image
    
def crop_QR_for_led_light_detection(result_image, led_area, y_mean):
    # y_mean = int(np.mean(y_list))

    y = y_mean.copy() - 150
    # LED_detection_area_list = []
    if type(led_area) == list:
        LED_detection_area_list = []

        for area in led_area:
            x1, x2 = area
            LED_detection_area_list.append(result_image[0:y, x1:x2])

    else:
        LED_detection_area_list = {}

        for data,area in led_area.items():
            x1, x2 = area
            LED_detection_area_list[f"{data}"] = (result_image[0:y, x1:x2])

    return LED_detection_area_list


def crop_frame_for_allocate_laptop(result_image, led_area, y_mean):
    # y_mean = int(np.mean(y_list))

    y = y_mean.copy() - 150
    # LED_detection_area_list = []
    if type(led_area) == list:
        undecoded_laptop_crop = []

        for area in led_area:
            x1, x2 = area
            undecoded_laptop_crop.append(result_image[0:result_image.shape[1], x1:x2])

    else:
        undecoded_laptop_crop = {}

        for data,area in led_area.items():
            x1, x2 = area
            undecoded_laptop_crop[f"{data}"] = result_image[y:result_image.shape[1], x1:x2]

    return undecoded_laptop_crop

def allocate_laptop_QR(led_area):
    alloc_data = {}
    # for data,coord in led_area.items():
    return []



# def detect_thresh_for_light(result_image, y_list, threshes, LED_detection_area):
#     # threshes = []
#     y_mean = int(np.mean(y_list))
#     thresh = 240
#     if result_image.shape[0] != 0:
        
#         for center in centers:
#             x, y = center
#             if y < y_mean:
#                 # if len(centers) == 6:
#                 if LED_detection_area.shape[0] != 0:
#                     thresh = detect_thresh_range(LED_detection_area)
#                     threshes.append(thresh)

#         if len(threshes) != 0:
#             thresh = max(threshes)

#     return thresh 

# if light detected then:

# send TRUE to RPI to store LED numbers in a list


def allocate_data(laptop_QR_data, shelf_QR_data, thresh):
    shelf_and_laptop_QR = {}
    shelf_QR = {shelf_code:"Empty-" for shelf_code,coord in shelf_QR_data.items()}
    for laptop_code, laptop_coord in laptop_QR_data.items():
        for shelf_code, shelf_coord in shelf_QR_data.items():
            x1, y1, x2, y2 = laptop_coord
            xx1, yy1, xx2, yy2 = shelf_coord
            if x1+((x2-x1)//2) < xx2+thresh and x1+((x2-x1)//2) > xx1-thresh:
                shelf_and_laptop_QR[f"{shelf_code}"] = f"{laptop_code}"
                shelf_QR[shelf_code] = f"{laptop_code}"

    return shelf_and_laptop_QR,shelf_QR


def make_detected_list(centers, x_mod, y_mod,detected):  

    for center in centers:


        xx, yy = center
        x1, y1, x2, y2 = (xx - (x_mod//2), yy - (y_mod//2), xx + (x_mod//2), yy + (y_mod//2))
        
        detected.append((x1, y1, x2, y2))
        detected = remove_duplicates_and_close(detected, threshold_distance=50)
    return detected

def make_decoded(detected, result_image, decoded, log, draw=False):

    for detected_xyxy in detected:
        diff1_x = 10
        diff2_y = 10
        diff1_y = 10
        diff2_x = 10
        color = (255,0,0)
        is_detected = False
        # xx, yy = detected_xyxy
        # x1, y1, x2, y2 = (xx - (x_mod//2), yy - (y_mod//2), xx + (x_mod//2), yy + (y_mod//2))
        x1, y1, x2, y2 = detected_xyxy
        # if x1-diff1_x <= 0:
        #     diff1_x = 0
        # if y1-diff1_y <= 0:
        #     diff1_y = 0
        # if x2+diff2_x >= result_image.shape[0]:
        #     diff2_x = 0
        # if y2+diff2_y >= result_image.shape[1]:
        #     diff2_y = 0
        
        decode_crop_p1 = time()
        cropped_qr = result_image[(y1-diff1_y):y2+diff2_y, (x1-diff1_x):x2+diff2_x].copy()
        decode_crop_p2 = time()
        log['decode_crop_p'] = decode_crop_p2 - decode_crop_p1

        if len(cropped_qr) != 0 and cropped_qr.shape[0] > 0 and cropped_qr.shape[1] > 0:
            cropped_qr = cv2_to_PIL(cropped_qr)
            
            decode_preprocess_p1 = time()
            cropped_qr = qr_codes_preprocess(cropped_qr)
            decode_preprocess_p2 = time()
            log['decode_preprocess_p'] = decode_preprocess_p2 - decode_preprocess_p1 

            decode_decode_p1 = time()
            barcodes = decode(cropped_qr)
            for barcode in barcodes:
                data = barcode.data.decode("utf-8")
                print(data)
                if len(data) > 0:
                    is_detected = True
                    decoded[data] = (x1,y1,x2,y2)
            decode_decode_p2 = time()
            log['decode_decode_p'] = decode_decode_p2 - decode_decode_p1
                # if len(barcode) != 0 and len(data) != 0 and draw == True and is_detected == True:
                    # color = (0,255,0)
                    # orginal = cv2.rectangle(result_image, (x1-diff1_x, y1-diff1_y), (x2+diff2_x, y2+diff2_y), color, 2)
                    # bre = True
                    # next_model = False

        # if is_detected == False and draw == True:
            # orginal = cv2.rectangle(result_image, (x1-diff1_x, y1-diff1_y), (x2+diff2_x, y2+diff2_y), color, 2)
    return decoded, log


def merge_decoded_and_detected(decoded, detected, frame_num, merged_data):
    # merged_data = decoded.copy()
    for detected_coord in detected:
        if detected_coord not in list(decoded.values()):
            merged_data[f"None-{frame_num}:{detected.index(detected_coord)}"] = detected_coord
    return merged_data


def motion_detection(static_back, gray, frame, draw, motion_list, motion, out, area=50):

    movment = 'None'

    # Difference between static background  
    # and current frame(which is GaussianBlur) 
    diff_frame = cv2.absdiff(static_back, gray) 

    # If change in between static background and 
    # current frame is greater than 30 it will show white color(255) 
    
    thresh_frame = cv2.threshold(diff_frame, area, 255, cv2.THRESH_BINARY)[1] 
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 

    # Finding contour of moving object 
    cnts,_ = cv2.findContours(thresh_frame.copy(),  
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    for contour in cnts: 
        if cv2.contourArea(contour) < 10000: 
            continue
        motion = 1

        (x, y, w, h) = cv2.boundingRect(contour) 
        # making green rectangle around the moving object 
        # if draw == True:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 

    # Appending status of motion 
    motion_list.append(motion) 

    motion_list = motion_list[-2:] 

    # Appending Start time of motion 
    if motion_list[-1] == 1 and motion_list[-2] == 0:
        draw = True
        movment = 'start'
        # time.append(datetime.now()) 
    
    if draw == True:
        out = True
        cv2.putText(frame, f"motion detected", (100,200), 1, 1, (0,0,255), 2)

    # Appending End time of motion 
    if motion_list[-1] == 0 and motion_list[-2] == 1: 
        draw = False
        movment = 'stop'
        # time.append(datetime.now()) 

    return static_back,out,draw,motion_list,motion,movment
    

def calculate_IOU(a,b):
    areaA = a[2] * a[3]
    areaA_bottom_right_x_coordinate = a[0] + a[2]
    areaA_bottom_right_y_coordinate = a[1] + a[3]
    
    if areaA <= 0.0:
        return 0.0

    areaB = b[2] * b[3]
    
    if areaB <= 0.0:
        return 0.0

    areaB_bottom_right_x_coordinate = b[0] + b[2]
    areaB_bottom_right_y_coordinate = b[1] + b[3]

    intersection_left_x = max(a[0], b[0])
    intersection_left_y = max(a[1], b[1])
    intersection_bottom_x = min(areaA_bottom_right_x_coordinate, areaB_bottom_right_x_coordinate)
    intersection_bottom_y = min(areaA_bottom_right_y_coordinate, areaB_bottom_right_y_coordinate)

    intersection_width = max(intersection_bottom_x - intersection_left_x, 0)
    intersection_height = max(intersection_bottom_y - intersection_left_y, 0)

    intersection_area = intersection_width * intersection_height

    return intersection_area / (areaA + areaB - intersection_area)
    



def nonMaxSuppresion(boxes,threshold):
    newBoxes = sorted(boxes,key = lambda x:x['score'],reverse=True)

    selected = []
    active = [True] * len(boxes)
    num_active = len(active)
    # print(active)

    done=False

    for i in range(len(boxes)):
        if active[i]:
            box_a = newBoxes[i]
            selected.append(box_a)

            for j in range(i+1,len(newBoxes)):
                if active[j]:
                    box_b = newBoxes[j]

                    # IOU Implementation
                    iou = calculate_IOU(box_a['bounds'],box_b['bounds'])

                    if iou > threshold:
                        active[j] = False
                        num_active -=1

                        if num_active <=0:
                            done = True
                            break
            if done:
                break
    return selected


def torch_yolo(images, orig_shape, model_path, conf=0.25,image_resize=(640,640), iou_threshold=0.45, device='cpu'):

    result_images_list = []

    for image in images:

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image
        image = cv2.resize(image, image_resize)
        
        # # Convert to tensor
        # image_tensor = transforms.ToTensor()(image)

        # # Add batch dimension and convert to float
        # image_tensor = image_tensor.unsqueeze(0)

        # # Load your model (make sure it's in eval mode)

        model = torch.load(model_path,map_location=torch.device(f"{device}"))
        model = model['model']
        model = model.eval().float()
        # # Perform inference
        # with torch.no_grad():
        #     preds = md(image_tensor)
        #     print(len(preds[0][0][0]))

                        
        # model = model['model'].float()

        convert_tensor = transforms.ToTensor()
        image_tensor = convert_tensor(image)
        if torch.cuda.is_available() and device == 'cuda:0':
            image_tensor = image_tensor.cuda()
        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            results = model(image_tensor)[0]

        # getting the predictions
        results = results[0]

        filtered_results = []
        number_of_columns = results.shape[1]
        # print('number_of-columns',number_of_columns)
        for i in range(number_of_columns):
            scores = (results[4][i].item())
            scores = "{:.5f}".format(scores)
            if(scores >= f'{conf}'):
                x1 = results[0][i].item()
                y1 = results[1][i].item()
                w = results[2][i].item()
                h = results[3][i].item()

                bounding_boxes = (x1,y1,w,h)
                result = {
                'classIndex': 'QRcode',
                'score': scores,
                'bounds': bounding_boxes,
                }
                filtered_results.append(result)
        iou_threshold = iou_threshold
        final_boxes = nonMaxSuppresion(filtered_results,iou_threshold)

        coordinates = []
        for point in final_boxes:
            xx1,yy1,ww,hh = point['bounds']
            # print(point['bounds'])

            original_height, original_width = (orig_shape[0], orig_shape[1])

            resized_height, resized_width = image_resize

            scale_x = original_width / resized_width
            scale_y = original_height / resized_height

            x11,y11,x22,y22 = (int(xx1-ww//2),int(yy1-hh//2),int(xx1+ww//2),int(yy1+hh//2))
            # Convert coordinates back to the original image size
            original_x1 = int(x11 * scale_x)
            original_y1 = int(y11 * scale_y)
            original_x2 = int(x22 * scale_x)
            original_y2 = int(y22 * scale_y)
            coordinates.append([original_x1,original_y1,original_x2,original_y2])
        
        result_images_list.append(coordinates)
    
    return result_images_list

def torch_yolo_onnx(images, orig_shape, input_name, ort_session, convert_tensor, onnx_model_path, image_resize=(640,640), iou_threshold=0.45):

    result_images_list = []

    # Initialize ONNX Runtime session
    # model = onnx.load_model(onnx_model_path)
    orig_shape = orig_shape
    # ort_session = ort.InferenceSession(onnx_model_path)
    # input_name = ort_session.get_inputs()[0].name

    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_resize)
        img = image.copy()

        # Convert to tensor and add batch dimension
        # convert_tensor = transforms.ToTensor()
        image_tensor = convert_tensor(image)
        image_tensor = image_tensor.unsqueeze(0).numpy()

        # Perform inference
        ort_inputs = {input_name: image_tensor}
        results = ort_session.run(None, ort_inputs)

        # Assuming results[0] contains the detection output
        results = results[0][0]

        filtered_results = []
        number_of_columns = results.shape[1]
        # print('number_of-columns',number_of_columns)
        for i in range(number_of_columns):
            scores = (results[4][i].item())
            scores = "{:.5f}".format(scores)
            if(scores >= f'{0.45}'):
                x1 = results[0][i].item()
                y1 = results[1][i].item()
                w = results[2][i].item()
                h = results[3][i].item()

                bounding_boxes = (x1,y1,w,h)
                result = {
                'classIndex': 'QRcode',
                'score': scores,
                'bounds': bounding_boxes,
                }
                filtered_results.append(result)
        iou_threshold = iou_threshold
        final_boxes = nonMaxSuppresion(filtered_results,iou_threshold)

        coordinates = []
        for point in final_boxes:
            xx1,yy1,ww,hh = point['bounds']
            # print(point['bounds'])

            original_height, original_width = (orig_shape[0], orig_shape[1])

            resized_height, resized_width = image_resize

            scale_x = original_width / resized_width
            scale_y = original_height / resized_height
            # scale_x = 1
            # scale_y = 1

            x11,y11,x22,y22 = (int(xx1-ww//2),int(yy1-hh//2),int(xx1+ww//2),int(yy1+hh//2))
            # Convert coordinates back to the original image size
            original_x1 = int(x11 * scale_x)
            original_y1 = int(y11 * scale_y)
            original_x2 = int(x22 * scale_x)
            original_y2 = int(y22 * scale_y)
            coordinates.append([original_x1,original_y1,original_x2,original_y2])
        
        result_images_list.append(coordinates)
        # print("ONNX model version:", model.ir_version)
        # print("ONNX model producer name:", model.producer_name)
        # print("ONNX model producer version:", model.producer_version)

        # # To get more detailed information about the model, including the opset version used
        # for opset in model.opset_import:
        #     print("Opset domain:", opset.domain if opset.domain else "ai.onnx")
        #     print("Opset version:", opset.version)
        
    return result_images_list


import paramiko

def run_command_on_rpi(host, port, username, password, command):
    # Create an SSH client
    client = paramiko.SSHClient()
    
    # Automatically add the host key (do not use in production)
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Connect to the Raspberry Pi
        client.connect(host, port=port, username=username, password=password)
        
        # Run the command
        stdin, stdout, stderr = client.exec_command(command)
        
        # Read the command output
        output = stdout.read().decode()
        error = stderr.read().decode()
        
        # Print output and error
        print("Output:", output)
        if error:
            print("Error:", error)
    
    finally:
        # Close the connection
        client.close()

