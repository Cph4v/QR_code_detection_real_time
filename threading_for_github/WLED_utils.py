"""
Refactored Module for QR Code Processing, Image Preprocessing, and Motion Detection

This module contains functions and classes grouped by their responsibilities:
    • FrameCropper: For splitting and reassembling an image into grid parts.
    • ExposureAdjuster: For adjusting exposure, contrast, and uneven lighting.
    • ColorAnalyzer: For analyzing the dominant colors in an image.
    • Converter: For converting between OpenCV images (numpy arrays) and PIL images.
    • QRPreprocessor: For performing preprocessing on images prior to QR detection.
    • CoordinateProcessor: For computing distances, filtering duplicates, and merging results.
    • QRCoordinateUtils: For separating QR codes into laptop and shelf groups.
    • DrawingUtils: For drawing annotations (vertical/horizontal lines) on images.
    • AllocationUtils: For associating laptop QR codes to shelf QR codes.
    • QRDetection: For detecting and decoding QR codes from image regions.
    • MotionDetector: For performing simple motion detection.
    
All functions from the original code are preserved.
"""

import cv2
import numpy as np
import math
import statistics as st
import ast
from collections import defaultdict
from itertools import combinations
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
from pyzbar.pyzbar import decode, ZBarSymbol
from tqdm import tqdm
import multiprocessing as mp
from time import time

# External dependencies from our repository
from detect_led_blubs import LED_on_detection, detect_thresh_range


# =============================================================================
# FRAME CROPPING AND REASSEMBLY
# =============================================================================
class FrameCropper:
    @staticmethod
    def crop_frame(num_rows: int, num_cols: int, image: np.ndarray) -> dict:
        """
        Crop an image into a grid and return a dictionary keyed by (row, col).

        :param num_rows: Number of rows to divide the image.
        :param num_cols: Number of columns to divide the image.
        :param image: Input image as a numpy array.
        :return: Dictionary with keys as (i,j) tuples and values as the cropped images.
        """
        cropped = {}
        row_height = image.shape[0] // num_rows
        col_width = image.shape[1] // num_cols
        for i in range(num_rows):
            for j in range(num_cols):
                start_y = i * row_height
                end_y = (i + 1) * row_height
                start_x = j * col_width
                end_x = (j + 1) * col_width
                cropped[(i, j)] = image[start_y:end_y, start_x:end_x]
        return cropped

    @staticmethod
    def reassemble_cropped_parts(cropped_parts: list, original_shape: tuple, num_rows: int, num_cols: int) -> np.ndarray:
        """
        Reassemble cropped image parts into the original image shape.

        :param cropped_parts: List of cropped image segments.
        :param original_shape: Original image shape (height, width, channels).
        :param num_rows: Number of rows that the image was split into.
        :param num_cols: Number of columns that the image was split into.
        :return: The reassembled image.
        """
        # Assume each element in cropped_parts is an array wrapped in a list as in the original code.
        original_image = np.zeros(original_shape, dtype=cropped_parts[0][0, 0].dtype)
        row_height = original_shape[0] // num_rows
        col_width = original_shape[1] // num_cols
        for i in range(num_rows):
            for j in range(num_cols):
                index = i * num_cols + j
                start_y = i * row_height
                end_y = (i + 1) * row_height
                start_x = j * col_width
                end_x = (j + 1) * col_width
                # Assume cropped_parts[index] is a list (like in the original code)
                original_image[start_y:end_y, start_x:end_x] = cropped_parts[index][0, 0]
        return original_image


# =============================================================================
# EXPOSURE AND CONTRAST ADJUSTMENTS
# =============================================================================
class ExposureAdjuster:
    @staticmethod
    def tune_exposure_multiple_frames(images: list) -> np.ndarray:
        """
        Align and merge multiple frames to adjust exposure.

        :param images: List of images.
        :return: Exposure-fused image.
        """
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(images, images)
        mergeMertens = cv2.createMergeMertens(exposure_weight=0)
        exposureFusion = mergeMertens.process(images)
        res = np.clip(exposureFusion * 255, 0, 255).astype('uint8')
        return res

    @staticmethod
    def adjust_exposure(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """
        Adjust exposure using a linear transformation.

        :param image: Input image.
        :param alpha: Gain.
        :param beta: Bias.
        :return: Adjusted image.
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def highlight_and_contrast(image: np.ndarray, thresh) -> np.ndarray:
        """
        Apply thresholding to highlight regions and adjust contrast.

        :param image: Input image.
        :param thresh: Threshold value or 'OTSU' to use OTSU thresholding.
        :return: Processed image.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blured = cv2.GaussianBlur(gray_image, (5, 5), 1)
        if isinstance(thresh, str):
            _, mask = cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif isinstance(thresh, int):
            _, mask = cv2.threshold(blured, thresh, 255, cv2.THRESH_BINARY)
        highlighted_image = cv2.bitwise_and(image, image, mask=mask)
        return highlighted_image

    @staticmethod
    def correct_uneven_lighting(image: np.ndarray, clip_limit: float = 2.0, tile_size: tuple = (8, 8)) -> np.ndarray:
        """
        Correct uneven lighting using CLAHE.

        :param image: Input image.
        :param clip_limit: CLAHE clip limit.
        :param tile_size: Size of the grid for CLAHE.
        :return: Corrected image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blured = cv2.GaussianBlur(gray, (5, 5), 1)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        corrected = clahe.apply(blured)
        return cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def adaptive_threshold(images, l: int = 7, k: int = 2):
        """
        Apply adaptive thresholding to an image or a list of images.

        :param images: Single image or list of images.
        :param l: Block size parameter.
        :param k: Constant subtracted from the mean.
        :return: Thresholded image(s) in color.
        """
        if not isinstance(images, list):
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, l, k)
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        else:
            results = []
            for image in images:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY, l, k)
                results.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))
            return results

    @staticmethod
    def prep(image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image by applying highlighting, lighting correction, and another round of highlighting.

        :param image: Input image.
        :return: Preprocessed image.
        """
        image_hc = ExposureAdjuster.highlight_and_contrast(image, thresh='OTSU')
        image_corr = ExposureAdjuster.correct_uneven_lighting(image_hc, clip_limit=13.0, tile_size=(64, 64))
        image_final = ExposureAdjuster.highlight_and_contrast(image_corr, thresh='OTSU')
        return image_final

    @staticmethod
    def zoom_in(image: np.ndarray, zoom_factor: float) -> np.ndarray:
        """
        Zoom in on an image by resizing.

        :param image: Input image.
        :param zoom_factor: Zoom factor.
        :return: Zoomed image.
        """
        height, width, _ = image.shape
        new_height = int(height * zoom_factor)
        new_width = int(width * zoom_factor)
        return cv2.resize(image, (new_width, new_height))


# =============================================================================
# COLOR ANALYSIS
# =============================================================================
class ColorAnalyzer:
    @staticmethod
    def give_max_color(pil_image: Image.Image, num: int = 5):
        """
        Find and return the most frequent color(s) in a PIL image.

        :param pil_image: Input PIL image.
        :param num: (Optional) Number of colors to retrieve.
        :return: A list of dominant colors.
        """
        img = pil_image.convert('RGB').convert('P')
        color_counts = img.getcolors()
        max_color = {color: count for count, color in color_counts}
        max_value = [color for color, count in max_color.items() if count == max(max_color.values())]
        return max_value


# =============================================================================
# IMAGE CONVERSION UTILS
# =============================================================================
class Converter:
    @staticmethod
    def cv2_to_PIL(image: np.ndarray) -> Image.Image:
        """Convert an OpenCV image (numpy array) to a PIL image."""
        return Image.fromarray(image)

    @staticmethod
    def PIL_to_cv2(pil_image: Image.Image, gray: bool = False) -> np.ndarray:
        """Convert a PIL image to an OpenCV image.
        
        :param pil_image: Input PIL image.
        :param gray: Convert to grayscale if True.
        :return: OpenCV image.
        """
        numpy_image = np.array(pil_image)
        if gray:
            return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
        else:
            return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)


# =============================================================================
# QR CODE PREPROCESSING
# =============================================================================
class QRPreprocessor:
    @staticmethod
    def qr_codes_preprocess(image: Image.Image) -> np.ndarray:
        """
        Preprocess a PIL image for QR detection:
            • Resize (zoom in)
            • Auto-contrast
            • Convert to grayscale and apply Gaussian-based adaptive threshold
            • Enhance sharpness

        :param image: Input PIL image.
        :return: Processed OpenCV image.
        """
        scalar = 1.5
        x3, y3 = image.size
        image_scaled = image.resize((int(round(x3 * scalar)), int(round(y3 * scalar))))
        max_value = ColorAnalyzer.give_max_color(image_scaled)
        image_scaled = ImageOps.autocontrast(image_scaled)
        gray = cv2.cvtColor(Converter.PIL_to_cv2(image_scaled), cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 2)
        im = Image.fromarray(adaptive_thresh)
        image_sharp = ImageEnhance.Sharpness(im).enhance(0.5)
        return Converter.PIL_to_cv2(image_sharp)


# =============================================================================
# COORDINATE PROCESSING AND MERGING
# =============================================================================
class CoordinateProcessor:
    @staticmethod
    def calculate_distance(point1, point2) -> float:
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    @staticmethod
    def filter_duplicate_centers(centers: list, distance_threshold: float) -> list:
        """Remove centers that are too close to each other."""
        filtered = []
        n = len(centers)
        for i in range(n):
            keep = True
            for j in range(i + 1, n):
                if CoordinateProcessor.calculate_distance(centers[i], centers[j]) <= distance_threshold:
                    keep = False
                    break
            if keep:
                filtered.append(centers[i])
        return filtered

    @staticmethod
    def unique_all_QR(all_QR_codes: dict) -> dict:
        """
        Merge QR code boxes and return only the unique ones.
        (Note: The logic is based on comparing distances; adjust thresholds as needed.)
        """
        unique_boxes = {}
        all_codes = all_QR_codes.copy()
        for key1, box1 in all_codes.items():
            for key2, box2 in all_codes.items():
                # Here we compare the centers of the boxes; adjust the logic as needed.
                dist = CoordinateProcessor.calculate_distance(box1[:2], box2[:2])
                if len(key1) > 4:
                    unique_boxes[key1] = box1
                if len(key2) > 4:
                    unique_boxes[key2] = box2
                # The following condition (dist < 0) is never true; it is preserved from original logic.
                if len(key1) < 4 and len(key2) < 4 and dist < 0:
                    unique_boxes[key1] = box1
        return unique_boxes

    @staticmethod
    def union_list_dict(dict1: dict, list1: list) -> dict:
        """Merge a dictionary with items from a list, ensuring uniqueness."""
        final_dict = dict1.copy()
        for item in list1:
            try:
                item_tuple = ast.literal_eval(item)
            except (ValueError, SyntaxError):
                item_tuple = item
            if item_tuple not in final_dict.values():
                final_dict[f"None-{len(final_dict)}"] = item_tuple
        return final_dict

    @staticmethod
    def euclidean_distance(coord1, coord2) -> float:
        """Calculate the Euclidean distance between two coordinate tuples."""
        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

    @staticmethod
    def filter_coordinates(all_QR_codes: dict, threshold_distance: float = 10) -> dict:
        """
        Filter coordinates such that QR codes that are too close (within threshold_distance) are merged.
        """
        filtered_coords = defaultdict(list)
        non_long_keys = {}
        for key, value in all_QR_codes.items():
            if len(key) > 5:
                filtered_coords[key] = value
            else:
                non_long_keys[key] = value
        for key1, value1 in non_long_keys.items():
            keep = True
            for key2, value2 in non_long_keys.items():
                if key1 != key2 and CoordinateProcessor.euclidean_distance(value1, value2) < threshold_distance:
                    keep = False
                    break
            if keep:
                filtered_coords[key1] = value1
        for key, value in non_long_keys.items():
            closest_key = min(filtered_coords.keys(), key=lambda x: CoordinateProcessor.euclidean_distance(value, filtered_coords[x]))
            filtered_coords[closest_key] = filtered_coords.pop(closest_key)
        return dict(filtered_coords)

    @staticmethod
    def preprocess_coordinates(best_coordinates: list):
        """
        Process the list of best coordinates obtained from detection to compute:
         • centers, mean width (x_mod), mean height (y_mod), and lists of x and y coordinates.
        """
        centers = []
        y_list = []
        x_list = []
        x_mod = 0
        y_mod = 0
        if best_coordinates:
            length_list = [len(x) for x in best_coordinates]
            for QR in best_coordinates:
                if len(QR) == max(length_list):
                    QR_coordinates = QR.copy()
                    break
            all_x_diff = []
            all_y_diff = []
            for coords in QR_coordinates:
                x1, y1, x2, y2 = coords
                all_x_diff.append(x2 - x1)
                all_y_diff.append(y2 - y1)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                centers.append([center_x, center_y])
                y_list.append(center_y)
                x_list.append(center_x)
            if x_list and y_list:
                x_mod = int(st.mode(all_x_diff))
                y_mod = int(st.mode(all_y_diff))
                print("Mode widths:", x_mod)
        return centers, x_mod, y_mod, y_list, x_list


# =============================================================================
# QR CODE COORDINATE UTILITIES
# =============================================================================
class QRCoordinateUtils:
    @staticmethod
    def choose_laptop_QR_coordinates(all_QR_codes: dict, y_mean: float) -> dict:
        """
        Select and return laptop QR code coordinates (those with y coordinate greater than y_mean).

        :param all_QR_codes: Dictionary of QR codes.
        :param y_mean: Threshold y value.
        :return: Filtered dictionary of laptop QR codes.
        """
        laptop_QRs = {}
        for code, coord in all_QR_codes.items():
            x1, y1, x2, y2 = coord
            if y1 > y_mean:
                laptop_QRs[f"{code}"] = coord
        return laptop_QRs

    @staticmethod
    def find_shelf_coordinates(horizontal_lines_coordinates: list, centers_neighbourhood_thrsh: float = 50) -> int:
        """
        Determine the shelf coordinate (y_mean) based on horizontal lines.

        :param horizontal_lines_coordinates: List of y coordinates.
        :param centers_neighbourhood_thrsh: Neighborhood threshold.
        :return: Calculated y_mean.
        """
        mean_val = int(np.mean(horizontal_lines_coordinates))
        max_val = max(horizontal_lines_coordinates)
        min_val = min(horizontal_lines_coordinates)
        d = abs(max_val - min_val)
        if max_val - mean_val < centers_neighbourhood_thrsh:
            y_mean = max_val + 150
        else:
            y_mean = max(horizontal_lines_coordinates) - (d // 2)
        return y_mean

    @staticmethod
    def choose_shelf_QR_coordinates_train_shelves(all_QR_codes, y_mean) -> dict:
        """
        Return shelf QR codes (those with y coordinate less than y_mean).
        """
        if isinstance(all_QR_codes, list):
            shelf_QRs = [coord for coord in all_QR_codes if coord[1] < y_mean]
        else:
            shelf_QRs = {}
            for code, coord in all_QR_codes.items():
                if coord[1] < y_mean:
                    shelf_QRs[f"{code}"] = coord
        return shelf_QRs

    @staticmethod
    def choose_laptop_QR_coordinates_train_laptop(all_QR_codes, y_mean) -> dict:
        """
        Return laptop QR codes (those with y coordinate greater than y_mean).
        """
        if isinstance(all_QR_codes, list):
            laptop_QRs = [coord for coord in all_QR_codes if coord[1] > y_mean]
        else:
            laptop_QRs = {}
            for code, coord in all_QR_codes.items():
                if coord[1] > y_mean:
                    laptop_QRs[f"{code}"] = coord
        return laptop_QRs


# =============================================================================
# DRAWING UTILITIES
# =============================================================================
class DrawingUtils:
    @staticmethod
    def draw_vertical_lines_train_laptop(laptop_detected_QR, result_image: np.ndarray, y_mean: int, offset: int = 80, draws: bool = True):
        """
        Draw vertical lines for laptop QR detections. Returns the annotated image and the computed laptop area.

        :param laptop_detected_QR: List or dict of laptop QR detections.
        :param result_image: The image to draw on.
        :param y_mean: Threshold y value.
        :param offset: Pixel offset for drawing.
        :param draws: Flag to actually draw the lines.
        :return: (annotated image, laptop area dictionary/list)
        """
        if isinstance(laptop_detected_QR, list):
            laptop_area = []
            for xyxy in laptop_detected_QR:
                xx1, yy1, xx2, yy2 = xyxy
                if yy1 < y_mean:
                    if draws:
                        cv2.line(result_image, (xx1 - offset, 0), (xx1 - offset, result_image.shape[0]), (255, 125, 60), 4)
                        cv2.line(result_image, (xx2 + offset, 0), (xx2 + offset, result_image.shape[0]), (255, 125, 60), 4)
                    laptop_area.append((xx1 - offset, xx2 + offset))
        else:
            laptop_area = {}
            for data, xyxy in laptop_detected_QR.items():
                xx1, yy1, xx2, yy2 = xyxy
                if yy1 > y_mean:
                    if draws:
                        cv2.line(result_image, (xx1 - offset, 0), (xx1 - offset, result_image.shape[0]), (255, 125, 60), 4)
                        cv2.line(result_image, (xx2 + offset, 0), (xx2 + offset, result_image.shape[0]), (255, 125, 60), 4)
                    laptop_area[f"{data}"] = (xx1 - offset, yy1, xx2 + offset, yy2)
        return result_image, laptop_area

    @staticmethod
    def draw_vertical_lines_train_shelves(shelf_detected_QR, result_image: np.ndarray, y_mean: int, offset: int, draws: bool = True):
        """
        Draw vertical lines for shelf QR detections. Returns the annotated image and computed LED area.

        :param shelf_detected_QR: List or dict of shelf QR detections.
        :param result_image: Image to annotate.
        :param y_mean: Threshold y coordinate.
        :param offset: Pixel offset.
        :param draws: If True, lines are drawn.
        :return: (annotated image, LED area dictionary/list)
        """
        if isinstance(shelf_detected_QR, list):
            led_area = []
            for xyxy in shelf_detected_QR:
                xx1, yy1, xx2, yy2 = xyxy
                if yy1 < y_mean:
                    if draws:
                        cv2.line(result_image, (xx1 - offset, 0), (xx1 - offset, result_image.shape[0]), (255, 125, 60), 4)
                        cv2.line(result_image, (xx2 + offset, 0), (xx2 + offset, result_image.shape[0]), (255, 125, 60), 4)
                    led_area.append((xx1 - offset, xx2 + offset))
        else:
            led_area = {}
            for data, xyxy in shelf_detected_QR.items():
                xx1, yy1, xx2, yy2 = xyxy
                if yy1 < y_mean:
                    if draws:
                        cv2.line(result_image, (xx1 - offset, 0), (xx1 - offset, result_image.shape[0]), (255, 125, 60), 4)
                        cv2.line(result_image, (xx2 + offset, 0), (xx2 + offset, result_image.shape[0]), (255, 125, 60), 4)
                    led_area[f"{data}"] = (xx1 - offset, xx2 + offset)
        return result_image, led_area

    @staticmethod
    def draw_horizontal_lines(result_image: np.ndarray, y_list: list, x_list: list, y_mean: int):
        """
        Draw a horizontal line at y_mean.

        :param result_image: Image to annotate.
        :param y_list: List of y coordinates.
        :param x_list: List of x coordinates.
        :param y_mean: y coordinate at which to draw the line.
        :return: Annotated image.
        """
        if x_list and y_list:
            cv2.line(result_image, (0, y_mean), (result_image.shape[1], y_mean), (255, 255, 0), 15)
        return result_image

    @staticmethod
    def crop_QR_for_led_light_detection(result_image: np.ndarray, led_area, y_mean: int):
        """
        Crop regions for LED detection from the result_image using provided led_area and y_mean.

        :param result_image: Image from which to crop.
        :param led_area: List or dict indicating horizontal crop boundaries.
        :param y_mean: Y coordinate used for cropping.
        :return: Dictionary or list of cropped regions.
        """
        y = y_mean - 150
        if isinstance(led_area, list):
            detection_list = []
            for area in led_area:
                x1, x2 = area
                detection_list.append(result_image[0:y, x1:x2])
        else:
            detection_list = {}
            for data, area in led_area.items():
                x1, x2 = area
                detection_list[f"{data}"] = result_image[0:y, x1:x2]
        return detection_list

    @staticmethod
    def crop_frame_for_allocate_laptop(result_image: np.ndarray, led_area, y_mean: int):
        """
        Crop frame sections for laptop QR allocation.

        :param result_image: The original image.
        :param led_area: Area indicating where to crop.
        :param y_mean: Y threshold.
        :return: Cropped regions for laptop allocation.
        """
        y = y_mean - 150
        if isinstance(led_area, list):
            crops = []
            for area in led_area:
                x1, x2 = area
                crops.append(result_image[0:result_image.shape[1], x1:x2])
        else:
            crops = {}
            for data, area in led_area.items():
                x1, x2 = area
                crops[f"{data}"] = result_image[y:result_image.shape[1], x1:x2]
        return crops


# =============================================================================
# ALLOCATION UTILITIES
# =============================================================================
class AllocationUtils:
    @staticmethod
    def allocate_data(laptop_QR_data: dict, shelf_QR_data: dict, thresh: int) -> dict:
        """
        Allocate laptop QR codes to shelf QR codes based on horizontal location.

        :param laptop_QR_data: Dict of laptop QR coordinates.
        :param shelf_QR_data: Dict of shelf QR coordinates.
        :param thresh: Tolerance threshold.
        :return: Allocation mapping dictionary.
        """
        shelf_and_laptop_QR = {}
        for laptop_code, laptop_coord in laptop_QR_data.items():
            for shelf_code, shelf_coord in shelf_QR_data.items():
                x1, y1, x2, y2 = laptop_coord
                xx1, yy1, xx2, yy2 = shelf_coord
                center_laptop = x1 + ((x2 - x1) // 2)
                if (center_laptop < xx2 + thresh) and (center_laptop > xx1 - thresh):
                    shelf_and_laptop_QR[f"{shelf_code}"] = f"{laptop_code}"
        return shelf_and_laptop_QR


# =============================================================================
# QR DETECTION / DECODING FUNCTIONS
# =============================================================================
class QRDetection:
    @staticmethod
    def make_detected_list(centers: list, x_mod: int, y_mod: int, detected: list) -> list:
        """
        Build a list of detected bounding boxes around centers.

        :param centers: List of center coordinates.
        :param x_mod: Width of the bounding box.
        :param y_mod: Height of the bounding box.
        :param detected: Input list (for accumulation).
        :return: List of bounding box tuples.
        """
        for center in centers:
            xx, yy = center
            box = (xx - (x_mod // 2), yy - (y_mod // 2), xx + (x_mod // 2), yy + (y_mod // 2))
            detected.append(box)
            detected = remove_duplicates_and_close(detected, threshold_distance=50)
        return detected

    @staticmethod
    def make_decoded(detected: list, result_image: np.ndarray, decoded: dict, draw: bool = False) -> dict:
        """
        Attempt to decode QR codes from the detected bounding boxes.

        :param detected: List of detected bounding boxes.
        :param result_image: Original image.
        :param decoded: Dict for accumulating decoded results.
        :param draw: If True, draw rectangles on the image.
        :return: Updated decoded dictionary.
        """
        for box in detected:
            diff1_x = diff2_y = diff1_y = diff2_x = 50
            color = (255, 0, 0)
            is_detected = False
            x1, y1, x2, y2 = box
            if x1 - diff1_x <= 0:
                diff1_x = 0
            if y1 - diff1_y <= 0:
                diff1_y = 0
            if x2 + diff2_x >= result_image.shape[0]:
                diff2_x = 0
            if y2 + diff2_y >= result_image.shape[1]:
                diff2_y = 0

            cropped = result_image[(y1 - diff1_y):(y2 + diff2_y), (x1 - diff1_x):(x2 + diff2_x)].copy()
            if cropped.size != 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
                cropped_pil = Converter.cv2_to_PIL(cropped)
                processed = QRPreprocessor.qr_codes_preprocess(cropped_pil)
                barcodes = decode(processed)
                for barcode in barcodes:
                    data = barcode.data.decode("utf-8")
                    print(data)
                    if data:
                        is_detected = True
                        decoded[data] = (x1, y1, x2, y2)
                    if barcode and data and draw and is_detected:
                        color = (0, 255, 0)
                        cv2.rectangle(result_image, (x1 - diff1_x, y1 - diff1_y),
                                      (x2 + diff2_x, y2 + diff2_y), color, 2)
        return decoded

    @staticmethod
    def merge_decoded_and_detected(decoded: dict, detected: list, frame_num: int, merged_data: dict) -> dict:
        """
        Merge the decoded QR code dictionary with detected regions.

        :param decoded: Dictionary of decoded QR codes.
        :param detected: List of detected bounding boxes.
        :param frame_num: Frame number (for key naming).
        :param merged_data: Dictionary to accumulate merged data.
        :return: Merged dictionary.
        """
        for d in detected:
            if d not in list(decoded.values()):
                merged_data[f"None-{frame_num}:{detected.index(d)}"] = d
        return merged_data


# =============================================================================
# MOTION DETECTION
# =============================================================================
class MotionDetector:
    @staticmethod
    def detect_motion(static_back: np.ndarray, gray: np.ndarray, frame: np.ndarray,
                      draw: bool, motion_list: list, motion: int, out: bool, area: int = 50):
        """
        Detect motion between a static background and the current grayscale frame.
        Annotates moving regions on the frame.
        
        :param static_back: Static background image.
        :param gray: Current grayscale frame.
        :param frame: Original frame.
        :param draw: Flag indicating if motion should be drawn.
        :param motion_list: History list of motion status.
        :param motion: Current motion status.
        :param out: Output flag.
        :param area: Threshold value.
        :return: Updated (static_back, out, draw, motion_list, motion).
        """
        diff_frame = cv2.absdiff(static_back, gray)
        thresh_frame = cv2.threshold(diff_frame, area, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
        cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in cnts:
            if cv2.contourArea(contour) < 10000:
                continue
            motion = 1
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        motion_list.append(motion)
        motion_list = motion_list[-2:]
        if motion_list[-1] == 1 and motion_list[-2] == 0:
            draw = True
        if draw:
            out = True
            cv2.putText(frame, "motion detected", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if motion_list[-1] == 0 and motion_list[-2] == 1:
            draw = False
        return static_back, out, draw, motion_list, motion


# =============================================================================
# DUPLICATE REMOVAL
# =============================================================================
def remove_duplicates_and_close(rectangles: list, threshold_distance: int = 10) -> list:
    """
    Remove duplicate or very close bounding boxes.

    :param rectangles: List of bounding boxes.
    :param threshold_distance: Distance threshold below which boxes are considered duplicates.
    :return: List of unique bounding boxes.
    """
    cleaned = []
    for rect in rectangles:
        is_unique = True
        for other in cleaned:
            if (abs(rect[0] - other[0]) < threshold_distance and
                abs(rect[1] - other[1]) < threshold_distance and
                abs(rect[2] - other[2]) < threshold_distance and
                abs(rect[3] - other[3]) < threshold_distance):
                is_unique = False
                break
        if is_unique:
            cleaned.append(rect)
    return cleaned


# =============================================================================
# MAIN (TESTING/EXECUTION) BLOCK
# =============================================================================
if __name__ == "__main__":
    # This main block is for testing purposes.
    # For example, you might load an image using cv2.imread(), run the preprocessing,
    # perform cropping, and display results.
    
    # Example: Load an image and apply adaptive threshold preprocessing
    test_image = cv2.imread('test_image.jpg')  # Replace with your image path
    if test_image is not None:
        preprocessed = ExposureAdjuster.prep(test_image)
        cropped = FrameCropper.crop_frame(2, 2, preprocessed)
        # Display one of the cropped parts
        key = list(cropped.keys())[0]
        cv2.imshow("Cropped Part", cropped[key])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Test image not found.")
