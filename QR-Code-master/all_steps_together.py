
"""
first run automate_detection.py for allocating LED numbers and shelf QR codes.
second run allocate_laptop_and_shelf.py for allocating shelf and QR codes together.
third run realtime_process.py for real time detecting laptop QR codes.
"""
import cv2
import numpy as np

import copy
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
from pyzbar.pyzbar import decode, ZBarSymbol
from ultralytics import YOLO

from time import sleep,time
import time
import json

from camera_tunnig_helpers import *
from function_for_app import *
from detect_led_blubs import *
from manuall_ROI_detection import load_points,four_point_transform

from send_request import ConnectRPI




def automate_detection(server_ip, device, port,  grid, headless, camera_settings):

    (width, height, autofocus,
    focus, contrast, zoom,
    brightness, fps) = eval(camera_settings)

    grid = eval(grid)

    num_rows = grid[0]
    num_cols = grid[1]
    stridy = grid[2]
    stridx = grid[3]

    try:
        if type(int(device)) == int:
            device = int(device)
    except:
        device=device

    cap = cv2.VideoCapture(device, cv2.CAP_V4L)

    if not (cap.isOpened()):
        print("Could not open video device")


    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Change codec accordingly
    out = cv2.VideoWriter('output55.avi', fourcc, 30.0, (1920,1080))

    print("Width = ",cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height = ",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Framerate = ",cap.get(cv2.CAP_PROP_FPS))
    print("Format = ",cap.get(cv2.CAP_PROP_FORMAT))

    zoom = cap.get(cv2.CAP_PROP_ZOOM)

    # subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=auto_exposure=3".split(' '))
    # subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=auto_exposure=1".split(' ')) # this is for exposure

    # subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=1".split(' ')) # this must be run to set camera at 60HZ power line
    # subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=0".split(' ')) # this must be run to set camera at 60HZ power line
    # subprocess.run("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=2".split(' ')) # this must be run to set camera at 60HZ power line

    Brightness=cap.get(cv2.CAP_PROP_BRIGHTNESS)

    Contrast=cap.get(cv2.CAP_PROP_CONTRAST)
    # Saturation=cap.get(cv2.CAP_PROP_SATURATION)
    # Gain=cap.get(cv2.CAP_PROP_GAIN)
    # Hue=cap.get(cv2.CAP_PROP_HUE)
    Exposure=cap.get(cv2.CAP_PROP_EXPOSURE)
    Focus=cap.get(cv2.CAP_PROP_FOCUS)
    # Auto_Focus = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    # Auto_Exposure = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    # Brightnesss = 10


    diff1_x = 50
    diff2_y = 50
    diff1_y = 50
    diff2_x = 50


    # cap.set(cv2.CAP_PROP_ZOOM, cv2.CAP_PROP_ZOOM//2)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


    save = False

    decoded = {}
    detected = []


    train_again = False

    all_QR_codes = {}

    centers = []

    save_frames = []



    ll = 11
    kk = 35

    """
    # TODO : 
        connect to RPI through wifi and run a script in RPI for listenning to port 8888
        and waiting to recieving commands for training or light up LED's after
    """
    draw = False

    onnx_model_path = 'new_model_QR_opset14.onnx'

    ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name

    convert_tensor = transforms.ToTensor()

    set_camera_settings(cap, width, height, autofocus, focus, contrast, zoom, brightness, fps)

    set_camera_settings(cap, width=cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT), \
                autofocus=0, focus=cap.get(cv2.CAP_PROP_FOCUS), contrast=cap.get(cv2.CAP_PROP_CONTRAST), \
                zoom=cap.get(cv2.CAP_PROP_ZOOM), brightness=cap.get(cv2.CAP_PROP_BRIGHTNESS), fps=cap.get(cv2.CAP_PROP_FPS))

    connect = ConnectRPI(server_ip, server_port=port)


    # p = []
    trash = []
    frame_number = 0
    frame_num = 0
    focus_step = 0

    hold_led = False

    global response

    while cap.isOpened():

        light_detected = False

        frame_number += 1
        frame_num += 1

        ret, frame = cap.read()

        #------------manuall ROI area------------------------
        points_loaded = load_points(filename="ROI_points.txt")
        (xf1, yf1), (xf2, yf2) = points_loaded[0],points_loaded[2]
        # frame = frame[yf1:yf2, xf1:xf2]
        frame,rect = four_point_transform(frame, f"{points_loaded}")
        #------------manuall ROI area------------------------

        # p.append(frame)

        # if ret:
        k = cv2.waitKey(1)

        # tune_exposure_multiple_frames(p)

        
        camera_current_setting(k, cap)
            
        #-------------------------BRIGHTNESS-----------------
        # y & h
        Contrast = contrast_ctrl(Contrast, k, cap)

        #-------------------------BRIGHTNESS-----------------
        # e & d
        Brightness = brightness_ctrl(Brightness, k, cap)

        #-------------------------ZOOM-----------------
        # w & s
        zoom = zoom_ctrl(zoom, k, cap)

        #-------------------------FOCUS-----------------
        # r & f
        Focus = focus_ctrl(Focus, k, cap)

        #------------------------Exposure---------------
        # q & a
        Exposure = exposure_ctrl(Exposure, k, cap)

        frame_dict, frames = crop_frame_updated(num_rows, num_cols, frame, stridy, stridx)

        # -------------auto focus-----------------------
        if frame_number % 6 == 0 and frame_number < 15*6:
            # focus_step += 1
            focus_step, Focus = auto_focus(focus_step, cap)
        if frame_number == 15*6-1:
            cap.set(cv2.CAP_PROP_FOCUS, 39)
            Focus = 39
        # -------------auto focus-----------------------
        
        # automate send to RPI4
        if frame_number == 15*6+9:
            k = ord("z")


        # ------------train again decoded and detected-------
        if k == ord("m"):
            train_again = True

            frame_number = 0
            focus_step = 0

            # -------------auto focus-----------------------

                
        # ------------train again decoded and detected-------

        if k == ord('i'):
            kk += 1
        elif k == ord('k'):
            kk -= 1
        if k == ord('o'):
            ll += 2
        elif k == ord('l'):
            ll -= 2 
                
        if draw == False:

            tt1 = time()
            model_QR = YOLO('new_model_QR.pt')
            results = model_QR(adaptive_threshold(frames, l=ll, k=kk), show=False, device=0, conf=0.2)
            tt2 = time()



            t1 = time()
            best_coordinates = find_best_coordinates(frame, results, frame_dict, num_cols, num_rows, stridx, stridy)
            t2 = time()

            centers, x_mod, y_mod, y_list, x_list = preprocess_coordinates(best_coordinates)

            result_image = frame.copy()

            detected = make_detected_list(centers, x_mod, y_mod, detected)

            for detected_xyxy in detected:

                diff1_x = 20
                diff2_y = 20
                diff1_y = 20
                diff2_x = 20


                color = (255,0,0)
                is_detected = False


                x1, y1, x2, y2 = detected_xyxy
                if x1-diff1_x <= 0:
                    diff1_x = 0
                if y1-diff1_y <= 0:
                    diff1_y = 0
                if x2+diff2_x >= result_image.shape[0]:
                    diff2_x = 0
                if y2+diff2_y >= result_image.shape[1]:
                    diff2_y = 0

                
                cropped_qr = result_image[(y1-diff1_y):(y2+diff2_y), (x1-diff1_x):x2+diff2_x].copy()
                print(cropped_qr.shape)
                if len(cropped_qr) != 0:

                    all_QR_codes[(x1, y1, x2, y2)] = f"None-{frame_num}{frame_num+1}" 

                    cropped_qr = cv2_to_PIL(cropped_qr)
                    print(cropped_qr.size)
                    cropped_qr = qr_codes_preprocess(cropped_qr)
                    
                    barcodes = decode(cropped_qr)
                    for barcode in barcodes:
                        data = barcode.data.decode("utf-8")
                        print(data)
                        if len(data) > 0:
                            is_detected = True
                            decoded[data] = (x1,y1,x2,y2)
                            all_QR_codes[(x1, y1, x2, y2)] = f"{data}" 
                        if len(barcode) != 0 and len(data) != 0 and draw == False and is_detected == True:
                            color = (0,255,0)
                            orginal = cv2.rectangle(result_image, (x1-diff1_x, y1-diff1_y), (x2+diff2_x, y2+diff2_y), color, 2)
                            bre = True
                            next_model = False
                if is_detected == False and draw == False:
                    orginal = cv2.rectangle(result_image, (x1-diff1_x, y1-diff1_y), (x2+diff2_x, y2+diff2_y), color, 2)

        if k == ord('z'):
            # all_QR_codes = union_list_dict(decoded, detected)
            vertical_lines_coordinates = list(decoded.values()).copy()
            horizontal_lines_coordinates = y_list.copy()
            merged_data = decoded.copy()
            for detected_coord in detected:
                if detected_coord not in list(decoded.values()):
                    merged_data[f"None-{frame_num}:{detected.index(detected_coord)}"] = detected_coord

            def convert(o):
                if isinstance(o, np.int64):
                    return int(o)
                raise TypeError
            
            QR_data_finall = {}
            y_mean_finall = find_shelf_coordinates(horizontal_lines_coordinates, centers_neighbourhood_thrsh=60)
            QR_data_finall['y_mean'] = y_mean_finall

            shelf_detected_QR = choose_shelf_QR_coordinates_train_shelves(merged_data, y_mean_finall)

            with open('shelf_QR_data.txt', 'w') as shelf_QR_dataset:
                shelf_QR_dataset.write(json.dumps(shelf_detected_QR, default=convert))
            
            with open('y_mean_finall.txt', 'w') as finall_Y_MEAN:
                finall_Y_MEAN.write(json.dumps(QR_data_finall, default=convert))


            # laptop_detected_QR = choose_laptop_QR_coordinates(all_QR_codes, y_list)
            draw = True

        elif k == ord('x'):
            draw = False

        if draw == True:

            result_image = frame.copy()

            #------------manuall LED ROI area------------------------
            points_loaded = load_points(filename="LED_area_ROI_points.txt")
            result_image_LED_area,rect = four_point_transform(result_image, f"{points_loaded}")
            image_brightest_value = brightest_image_value(result_image_LED_area)
            # cv2.imshow('fff', result_image_LED_area)
            #------------manuall LED ROI area------------------------
            

            y_mean = find_shelf_coordinates(horizontal_lines_coordinates, centers_neighbourhood_thrsh=60)

            shelf_detected_QR = choose_shelf_QR_coordinates_train_shelves(merged_data, y_mean)
            
            blub_image = frame.copy()
            
            _, led_area = draw_vertical_lines_train_shelves(shelf_detected_QR, result_image, y_mean, offset=-15, draws=True)
            
            LED_detection_area_list = crop_QR_for_led_light_detection(blub_image, led_area, y_mean)
            

            for xyxy in detected:  
                xx1, yy1, xx2, yy2 = xyxy   
                orginal = cv2.rectangle(result_image, (xx1-10, yy1-10), (xx2+10, yy2+10), (0,0,255), 2)

            for data, xyxy in decoded.items():  
                xx1, yy1, xx2, yy2 = xyxy   
                orginal = cv2.rectangle(result_image, (xx1-10, yy1-10), (xx2+10, yy2+10), (0,255,0), 2)
            
            threshes = {}
            masks = []

            for code,LED_detection_area in LED_detection_area_list.items():
     
                led_on_detected, mask, thresh = LED_on_detection(LED_detection_area, image_brightest_value-10, 50)
                masks.append(mask)
                threshes[f"{code}"] = led_on_detected

            

            if True in list(threshes.values()) and hold_led == False:
                code_to_send = [code for code,led_detected_or_not in threshes.items() if led_detected_or_not]
                light_detected = f"{code_to_send}"

            connect.send_data(f"{light_detected}")
            print(f"light_detected : {light_detected}")

            response = connect.receive_data()
            print(f"response : {response}")
            if response == "END":
                k = 27

        
        cv2.putText(result_image, f"decoded : {len(list(decoded.keys()))}", (730,20), 1, 2, (125,125,255), 2)
        cv2.putText(result_image, f"detected : {len(detected)}", (760,50), 1, 2, (0,0,255), 2)

        cv2.putText(result_image, f"zoom: {zoom} brightness: {Brightness} focus: {cap.get(cv2.CAP_PROP_FOCUS)} exposure: {Exposure} contrast: {Contrast}", (130,230), color=(0,255,255), fontFace=2, fontScale=1, thickness=3, lineType=1)

        if k == ord('t'):
            save = True
            print("saving begins")

        if save == True:
            cv2.putText(result_image, "save image begins", (0,100), color=(0,255,0), fontFace=2, fontScale=2, thickness=4, lineType=1)
            save_frames.append(result_image)

        if eval(headless) == False:
            cv2.imshow('show',result_image)
            # cv2.imshow('fff', result_image_LED_area)


        if (k == 27): #Esc key to quite the application 
            if len(save_frames) != 0:
                for saved_frame in save_frames:
                    out.write(saved_frame)

            connect.send_data("close_connect")
            connect.send_data("close")
            break
    
    # response = connect.recieve_data()
    # print(f"LAST response : {response}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"trash : {trash}")
    print(f"detected : {detected}")
    print('-'*100)
    print(f"decoded : {decoded}")
    print(len(decoded.keys()))
    print(f"all_QR_codes : {all_QR_codes}")
    print(f"shelf_detected_QR : {shelf_detected_QR}")
    print(f"merged_data : {merged_data}")
    print(f"y_min : {int(np.mean(horizontal_lines_coordinates))}")

    # return response

def allocate_laptop_and_shelf(server_ip, device, port ,grid, headless, camera_settings):

    (width, height, autofocus,
    focus, contrast, zoom,
    brightness, fps) = eval(camera_settings)

    grid = eval(grid)

    num_rows = grid[0]
    num_cols = grid[1]
    stridy = grid[2]
    stridx = grid[3]

    try:
        if type(int(device)) == int:
            device = int(device)
    except:
        device=device

    cap = cv2.VideoCapture(device, cv2.CAP_V4L)

    if not (cap.isOpened()):
        print("Could not open video device")


    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Change codec accordingly
    out = cv2.VideoWriter('output55.avi', fourcc, 30.0, (1920,1080))

    print("Width = ",cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height = ",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Framerate = ",cap.get(cv2.CAP_PROP_FPS))
    print("Format = ",cap.get(cv2.CAP_PROP_FORMAT))

    zoom = cap.get(cv2.CAP_PROP_ZOOM)

    Brightness=cap.get(cv2.CAP_PROP_BRIGHTNESS)

    Contrast=cap.get(cv2.CAP_PROP_CONTRAST)

    Exposure=cap.get(cv2.CAP_PROP_EXPOSURE)
    Focus=cap.get(cv2.CAP_PROP_FOCUS)



    diff1_x = 50
    diff2_y = 50
    diff1_y = 50
    diff2_x = 50


    # cap.set(cv2.CAP_PROP_ZOOM, cv2.CAP_PROP_ZOOM//2)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    y_mean_averag = []

    save = False

    decoded = {}
    detected = []


    train_again = False

    all_QR_codes = {}

    centers = []

    save_frames = []



    ll = 11
    kk = 35

    """
    # TODO : 
        connect to RPI through wifi and run a script in RPI for listenning to port 8888
        and waiting to recieving commands for training or light up LED's after
    """
    draw = False

    # set_camera_settings(cap)
    set_camera_settings(cap, width, height, autofocus, focus, contrast, zoom, brightness, fps)

    set_camera_settings(cap, width=cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT), \
            autofocus=0, focus=cap.get(cv2.CAP_PROP_FOCUS), contrast=cap.get(cv2.CAP_PROP_CONTRAST), \
            zoom=cap.get(cv2.CAP_PROP_ZOOM), brightness=cap.get(cv2.CAP_PROP_BRIGHTNESS), fps=cap.get(cv2.CAP_PROP_FPS))


    connect = ConnectRPI(server_ip, server_port=port)

    # p = []
    trash = []
    frame_number = 0
    frame_num = 0
    focus_step = 0

    hold_led = False

    onnx_model_path = 'new_model_QR_opset14.onnx'
    # model = onnx.load_model(onnx_model_path)
    ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name

    convert_tensor = transforms.ToTensor()

    with open('shelf_QR_data.txt', 'r') as file:
        shelf_QR_data = json.load(file)

    with open('y_mean_finall.txt', 'r') as file:
        y_mean_dict = json.load(file)

    y_mean = y_mean_dict['y_mean']

    while cap.isOpened():

        decoded_s = {}
        detected_s = []

        light_detected = False

        frame_number += 1
        frame_num += 1

        ret, frame = cap.read()

        #------------manuall ROI area------------------------
        points_loaded = load_points(filename="ROI_points.txt")
        # frame,rect = four_point_transform(frame, f"{points_loaded}")
        (xf1, yf1), (xf2, yf2) = points_loaded[0],points_loaded[2]
        frame = frame[yf1:yf2, xf1:xf2]
        #------------manuall ROI area------------------------

        # p.append(frame)

        # if ret:
        k = cv2.waitKey(1)

        # tune_exposure_multiple_frames(p)

        
        camera_current_setting(k, cap)
            
        #-------------------------BRIGHTNESS-----------------
        # y & h
        Contrast = contrast_ctrl(Contrast, k, cap)

        #-------------------------BRIGHTNESS-----------------
        # e & d
        Brightness = brightness_ctrl(Brightness, k, cap)

        #-------------------------ZOOM-----------------
        # w & s
        zoom = zoom_ctrl(zoom, k, cap)

        #-------------------------FOCUS-----------------
        # r & f
        Focus = focus_ctrl(Focus, k, cap)

        #------------------------Exposure---------------
        # q & a
        Exposure = exposure_ctrl(Exposure, k, cap)

        # frame_dict = crop_frame(num_rows,num_cols,frame)
        frame_dict, frames = crop_frame_updated(num_rows, num_cols, frame, stridy, stridx)
        # frame_dict = crop_frame_with_stride(frames, num_rows, num_cols, row_stride, col_stride)

        # -------------auto focus-----------------------
        if frame_number % 6 == 0 and frame_number < 15*6:
            # focus_step += 1
            focus_step, Focus = auto_focus(focus_step, cap)
        if frame_number == 15*6-1:
            Focus = 39
            k = ord("z")
            cap.set(cv2.CAP_PROP_FOCUS, 39)
        # -------------auto focus-----------------------


        # ------------train again decoded and detected-------
        if k == ord("m"):
            train_again = True
            # decoded = {}
            # detected = []
            frame_number = 0
            focus_step = 0
            # all_QR_codes = {}
            # centers = []
            # threshes = {}
            # all_unique_data = {}
            # -------------auto focus-----------------------

                
        # ------------train again decoded and detected-------

        if k == ord('i'):
            draw="workTime"
        elif k == ord('k'):
            kk -= 1
        if k == ord('o'):
            ll += 2
        elif k == ord('l'):
            ll -= 2 
        
        if draw == False:

            model_QR = YOLO('new_model_QR.pt')
            results = model_QR(adaptive_threshold(frames, l=ll, k=kk), show=False, device=0, conf=0.15)
            best_coordinates = find_best_coordinates(frame, results, frame_dict, num_cols, num_rows, stridx, stridy)

            # centers, x_mod, y_mod, y_list, x_list = preprocess_coordinates_torch(best_coordinates)
            centers, x_mod, y_mod, y_list, x_list = preprocess_coordinates(best_coordinates)

            result_image = frame.copy()

            # find brightest pixel value
            image_brightest_value = brightest_image_value(result_image[400:result_image.shape[1], 0:result_image.shape[0]]) # avoid calculating led light as max value of frames.
            # print(image_brightest_value)

            detected = make_detected_list(centers, x_mod, y_mod, detected)
            detected_s = make_detected_list(centers, x_mod, y_mod, detected_s)

            detected = make_detected_list(centers, x_mod, y_mod, detected)
            detected_s = make_detected_list(centers, x_mod, y_mod, detected_s)
            log_time = {}
            decoded, _ = make_decoded(detected, result_image, decoded, log_time, draw=True)
            decoded_s,_ = make_decoded(detected_s, result_image, decoded_s, log_time, draw=False)
                

            def merge_decoded_and_detected(decoded, detected, frame_num, merged_datas):
                for detected_coord in detected:
                    if detected_coord not in list(decoded.values()):
                        merged_datas[f"None-{frame_num}:{detected.index(detected_coord)}"] = detected_coord
                return merged_datas
            merged_datas = decoded.copy()
            merged_datas_s = decoded_s.copy()
            
        merged_data = merge_decoded_and_detected(decoded, detected, frame_num, merged_datas)
        merged_data_s = merge_decoded_and_detected(decoded_s, detected_s, frame_num, merged_datas_s)



        if draw == False:

            result_image = frame.copy()

            image_brightest_value = brightest_image_value(result_image[400:result_image.shape[1], 0:result_image.shape[0]]) # avoid calculating led light as max value of frames.
            

            laptop_detected_QR = choose_laptop_QR_coordinates_train_laptop(merged_data, y_mean=y_mean)
            

            _, laptop_QR_data = draw_vertical_lines_train_laptop(laptop_detected_QR, result_image, y_mean, offset=5, draws=True)
            

            for xyxy in detected:  
                xx1, yy1, xx2, yy2 = xyxy   
                orginal = cv2.rectangle(result_image, (xx1-10, yy1-10), (xx2+10, yy2+10), (0,0,255), 2)

            for data, xyxy in decoded.items():  
                xx1, yy1, xx2, yy2 = xyxy   

                orginal = cv2.rectangle(result_image, (xx1-10, yy1-10), (xx2+10, yy2+10), (0,255,0), 2)
            
            for data, xyxy in laptop_QR_data.items():  
                xx1, yy1, xx2, yy2 = xyxy   
                orginal = cv2.rectangle(result_image, (xx1, y_mean), (xx2, result_image.shape[1]), (0,255,255), 2)
                
            draw_horizontal_lines(result_image, y_list, x_list, y_mean)

            shelf_and_laptop_QR,_ = allocate_data(laptop_QR_data, shelf_QR_data, thresh=5)

            connect.send_data(f"{shelf_and_laptop_QR}")
            
            
            def convert(o):
                if isinstance(o, np.int64):
                    return int(o)
                raise TypeError

            with open('allocated_laptops_QRs.txt', 'w') as file:
                file.write(json.dumps(shelf_and_laptop_QR, default=convert))
        
        cv2.putText(result_image, f"decoded : {len(list(decoded.keys()))}", (730,20), 1, 2, (125,125,255), 2)
        cv2.putText(result_image, f"detected : {len(detected)}", (760,50), 1, 2, (0,0,255), 2)


        cv2.putText(result_image, f"zoom: {zoom} brightness: {Brightness} focus: {cap.get(cv2.CAP_PROP_FOCUS)} exposure: {Exposure} contrast: {Contrast}", (130,230), color=(0,255,255), fontFace=2, fontScale=1, thickness=3, lineType=1)

        if k == ord('t'):
            save = True
            print("saving begins")

        if save == True:
            cv2.putText(result_image, "save image begins", (0,100), color=(0,255,0), fontFace=2, fontScale=2, thickness=4, lineType=1)
            save_frames.append(result_image)

        if eval(headless) == False:
            cv2.imshow('show',result_image)

        if (k == 27): #Esc key to quite the application 
            if len(save_frames) != 0:
                for saved_frame in save_frames:
                    out.write(saved_frame)
            connect.send_data("close")
            
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()


    print(f"trash : {trash}")
    print(f"detected : {detected}")
    print('-'*100)
    print(f"decoded : {decoded}")
    print(len(decoded.keys()))
    print(f"shelf_QR_data : {shelf_QR_data}")
    print(f"laptop_QR_data : {laptop_QR_data}")
    print(f"merged_data : {merged_data}")
    print(f"shelf_and_laptop_QR : {shelf_and_laptop_QR}")



def real_time(server_ip, device, port, grid, headless, camera_settings):

    log_time = {}

    (width, height, autofocus,
    focus, contrast, zoom,
    brightness, fps) = eval(camera_settings)

    grid = eval(grid)

    num_rows = grid[0]
    num_cols = grid[1]
    stridy = grid[2]
    stridx = grid[3]


    try:
        if type(int(device)) == int:
            device = int(device)
    except:
        device=device

    cap = cv2.VideoCapture(device, cv2.CAP_V4L)

    if not (cap.isOpened()):
        print("Could not open video device")


    def convert(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError


    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Change codec accordingly
    output_video = cv2.VideoWriter('output55.avi', fourcc, 30.0, (1920,1080))

    print("Width = ",cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height = ",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Framerate = ",cap.get(cv2.CAP_PROP_FPS))
    print("Format = ",cap.get(cv2.CAP_PROP_FORMAT))

    zoom = cap.get(cv2.CAP_PROP_ZOOM)

    Brightness=cap.get(cv2.CAP_PROP_BRIGHTNESS)

    Contrast=cap.get(cv2.CAP_PROP_CONTRAST)
    Exposure=cap.get(cv2.CAP_PROP_EXPOSURE)
    Focus=cap.get(cv2.CAP_PROP_FOCUS)



    diff1_x = 50
    diff2_y = 50
    diff1_y = 50
    diff2_x = 50


    y_mean_averag = []

    save = False

    decoded = {}
    detected = []


    train_again = False

    all_QR_codes = {}

    centers = []

    save_frames = []

    diffrence_log = []


    ll = 11
    kk = 35

    """
    # TODO : 
        connect to RPI through wifi and run a script in RPI for listenning to port 8888
        and waiting to recieving commands for training or light up LED's after
    """
    draw = False

    # set_camera_settings(cap)
    set_camera_settings(cap, width, height, autofocus, focus, contrast, zoom, brightness, fps)
    
    set_camera_settings(cap, width=cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT), \
                autofocus=0, focus=cap.get(cv2.CAP_PROP_FOCUS), contrast=cap.get(cv2.CAP_PROP_CONTRAST), \
                zoom=cap.get(cv2.CAP_PROP_ZOOM), brightness=cap.get(cv2.CAP_PROP_BRIGHTNESS), fps=cap.get(cv2.CAP_PROP_FPS))
    
    cap.set(cv2.CAP_PROP_ZOOM, 14700)


    connect = ConnectRPI(server_ip, server_port=port)

    # p = []
    trash = []
    frame_number = 0
    frame_num = 0
    focus_step = 0
    train_time = 0

    hold_led = False

    #############################
    static_back = None
   
    motion_list = [ None, None ] 

    draws = False

    detect_motion = False
    #############################


    load_part1 = time()
    with open('allocated_laptops_QRs.txt', 'r') as file:
        allocated_laptops_QRs = json.load(file)
        print(f"can_read this : {allocated_laptops_QRs}")

    with open('shelf_QR_data.txt', 'r') as file:
        shelf_QR_data = json.load(file)

    with open('y_mean_finall.txt', 'r') as file:
        y_mean_dict = json.load(file)
    load_part2 = time()
    log_time['load_part'] = load_part2 - load_part1

    y_mean = y_mean_dict['y_mean']

    shelf_and_laptop_QR_update = allocated_laptops_QRs.copy() 
    
    onnx_model_path = 'new_model_QR_opset14.onnx'
    # model = onnx.load_model(onnx_model_path)
    ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name

    convert_tensor = transforms.ToTensor()


    main_part1 = time()
    while cap.isOpened():

        each_fram_time_part1 = time()
        ret, frame = cap.read()

        with open('allocated_laptops_QRs.txt', 'r') as file:
            allocated_laptops_QRs = json.load(file)

        ROI_part1 = time()
        #------------manuall ROI area------------------------
        points_loaded = load_points(filename="ROI_points.txt")
        frame,rect = four_point_transform(frame, f"{points_loaded}")
        #------------manuall ROI area------------------------
        ROI_part2 = time()
        log_time[f'ROI_part '] = ROI_part2 - ROI_part1
        #########################--motion_detection--#################################

        out = False
        
        motion = 0
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
        gray = cv2.GaussianBlur(gray, (21, 21), 0) 

        if static_back is None: 
            static_back = gray 
            continue

        static_back,out,draws,motion_list,motion,movment = motion_detection(static_back, gray, frame, draws, motion_list, motion, out, area=80)


        #########################--motion_detection--################################



        decoded_s = {}
        detected_s = []

        light_detected = False

        frame_number += 1
        frame_num += 1
        # train_time += 1

        k = cv2.waitKey(1)
        
        camera_current_setting(k, cap)
            
        #-------------------------BRIGHTNESS-----------------
        # y & h
        Contrast = contrast_ctrl(Contrast, k, cap)

        #-------------------------BRIGHTNESS-----------------
        # e & d
        Brightness = brightness_ctrl(Brightness, k, cap)

        #-------------------------ZOOM-----------------
        # w & s
        zoom = zoom_ctrl(zoom, k, cap)

        #-------------------------FOCUS-----------------
        # r & f
        Focus = focus_ctrl(Focus, k, cap)

        #------------------------Exposure---------------
        # q & a
        Exposure = exposure_ctrl(Exposure, k, cap)

        # frame_dict = crop_frame(num_rows,num_cols,frame)
        crop_part1 = time()
        frame_dict, frames = crop_frame_updated(num_rows, num_cols, frame, stridy, stridx)
        crop_part2 = time()
        log_time[f'crop_part '] = crop_part2 - crop_part1
        # frame_dict = crop_frame_with_stride(frames, num_rows, num_cols, row_stride, col_stride)

        # -------------auto focus-----------------------
        if frame_number % 6 == 0 and frame_number < 14*6:
            # focus_step += 1
            # focus_step, Focus = auto_focus(focus_step, cap)
            Focus = 47
            focus_step += 1 
            Focus = Focus-focus_step
            cap.set(cv2.CAP_PROP_FOCUS, Focus)

        if frame_number == 14*6-1:
            cap.set(cv2.CAP_PROP_FOCUS, 47)
            Focus = 47
            cap.set(cv2.CAP_PROP_FOCUS, 47)
        # -------------auto focus-----------------------


        # ------------train again decoded and detected-------
        if k == ord("m"):
            train_again = True
            frame_number = 0
            focus_step = 0
            # -------------auto focus-----------------------

                
        # ------------train again decoded and detected-------

        if k == ord('i'):
            draw=True
        elif k == ord('k'):
            kk -= 1
        if k == ord('o'):
            ll += 2
        elif k == ord('l'):
            ll -= 2 
        
        if movment == 'stop':
            train_again = True
            frame_number = 0
            focus_step = 0
            # cap.set(cv2.CAP_PROP_FPS, 5)
        if movment == 'start':
            train_again = False
            # cap.set(cv2.CAP_PROP_FPS, 5)
            
        if train_again == True:
            train_time += 1
        if train_time == 14*6-1:
            train_again = False
            train_time = 0
            # cap.set(cv2.CAP_PROP_FPS, 5)
        print(f"fps : {cap.get(cv2.CAP_PROP_FPS)}")



        frames = [image for index,image in frame_dict[0].items()]
        if frame_num < 14*6-1 or (frame_num > 14*6-1 and train_again == True):
            each_fram_time_train_mode_part1 = time()
            # if draw == False:

            model_QR = YOLO('new_model_QR.pt')
            results = model_QR(adaptive_threshold(frames, l=ll, k=kk), show=False, device=0, conf=0.2)
            model_part1 = time()
            model_part2 = time()
            log_time[f'model_part '] = model_part2 - model_part1
            
            find_best_thresh_part1 = time()
            best_coordinates = find_best_coordinates(frame, results, frame_dict, num_cols, num_rows, stridx, stridy)
            print(f"sh-i : {len(best_coordinates)}")
            find_best_thresh_part2 = time()
            log_time[f'find_best_thresh_part '] = find_best_thresh_part2 - find_best_thresh_part1

            preprocess_coordinates_torch_part1 = time()
            centers, x_mod, y_mod, y_list, x_list = preprocess_coordinates(best_coordinates)
            preprocess_coordinates_torch_part2 = time()
            log_time[f'preprocess_coordinates_torch_part '] = preprocess_coordinates_torch_part2 - preprocess_coordinates_torch_part1

            result_image = frame.copy()


            make_detected_list1 = time()
            detected = make_detected_list(centers, x_mod, y_mod, detected)
            make_detected_list2 = time()
            log_time[f'make_detected_list '] = make_detected_list2 - make_detected_list1

            
            make_decoded_list_part1 = time()
            decoded, log_time = make_decoded(detected, result_image, decoded, log_time, draw=False)
            make_decoded_list_part2 = time()
            log_time[f'make_decoded_list_part '] = make_decoded_list_part2 - make_decoded_list_part1


            def merge_decoded_and_detected(decoded, detected, frame_num, merged_datas):
                for detected_coord in detected:
                    if detected_coord not in list(decoded.values()):
                        merged_datas[f"None-{frame_num}:{detected.index(detected_coord)}"] = detected_coord
                return merged_datas
            
            merged_datas = decoded.copy()
            #merged_datas_s = decoded_s.copy()
            
            merge_part1 = time()
            merged_data = merge_decoded_and_detected(decoded, detected, frame_num, merged_datas)
            merge_part2 = time()
            log_time[f'merge_part '] = merge_part2 - merge_part1


            choose_laptop_QR_coordinates_train_laptop_part1 = time()
            laptop_detected_QR = choose_laptop_QR_coordinates_train_laptop(merged_data, y_mean=y_mean)
            choose_laptop_QR_coordinates_train_laptop_part2 = time()
            log_time[f'choose_laptop_QR_coordinates_train_laptop_part '] = choose_laptop_QR_coordinates_train_laptop_part2 - choose_laptop_QR_coordinates_train_laptop_part1


            draw_vertical_lines_train_laptop_part1 = time()
            _, laptop_QR_data = draw_vertical_lines_train_laptop(laptop_detected_QR, result_image, y_mean, offset=5, draws=False)
            draw_vertical_lines_train_laptop_part2 = time()
            log_time[f'draw_vertical_lines_train_laptop_part '] = draw_vertical_lines_train_laptop_part2 - draw_vertical_lines_train_laptop_part1
           

            draw_horizontal_lines(result_image, y_list, x_list, y_mean)
            # draw_horizontal_lines_part2 = time()
            # log_time[f'draw_horizontal_lines_part '] = draw_horizontal_lines_part2 - draw_horizontal_lines_part1
            allocate_time_part1 = time()
            shelf_and_laptop_QR, shelf_QR_data_empty = allocate_data(laptop_QR_data, shelf_QR_data, thresh=5)
            allocate_time_part2 = time()
            log_time[f'allocate_time_part '] = allocate_time_part2 - allocate_time_part1

            #shelf_and_laptop_QR_s, shelf_QR_data_empty_s = allocate_data(laptop_QR_data_s, shelf_QR_data, thresh=80)
            print(f"shelf_QR_data_empty : {shelf_QR_data_empty}")

            each_fram_time_train_mode_part2 = time()
            log_time[f'each_fram_time_train_mode_part '] = each_fram_time_train_mode_part2 - each_fram_time_train_mode_part1
        if len(detected) != 0:

            for xyxy in detected:  
                xx1, yy1, xx2, yy2 = xyxy   
                orginal = cv2.rectangle(result_image, (xx1-10, yy1-10), (xx2+10, yy2+10), (0,0,255), 2)

            for data, xyxy in decoded.items():  
                xx1, yy1, xx2, yy2 = xyxy   
                # orginal = cv2.putText(result_image, f"pyzbar {data}", (xx2+10,yy2), 1, 1, (0,0,255), 2)
                orginal = cv2.rectangle(result_image, (xx1-10, yy1-10), (xx2+10, yy2+10), (0,255,0), 2)
            
            for data, xyxy in laptop_QR_data.items():  
                xx1, yy1, xx2, yy2 = xyxy   
                # orginal = cv2.putText(result_image, f"pyzbar {data}", (xx2+10,yy2), 1, 1, (0,0,255), 2)
                orginal = cv2.rectangle(result_image, (xx1, y_mean), (xx2, result_image.shape[1]), (0,255,255), 2)
            

            if out == True:
                # shelf_and_laptop_QR = {}
                decoded = {}
                detected = []
                draw = False
            elif k == ord("v"):
                draw = True
                

            def difference_each_frame(shelf_and_laptop_QR, shelf_and_laptop_QR_s):
                diffrence_each_frame = {}
                for shelf_s,laptop_s in shelf_and_laptop_QR_s.items():
                    for shelf,laptop in shelf_and_laptop_QR.items():
                        if shelf_s == shelf and laptop_s != laptop:
                            diffrence_each_frame[shelf] = laptop_s
                return diffrence_each_frame

            def allocated_laptop_QR_cloner(allocated_laptops_QRs, shelf_QR_data_empty):

                shelf_QR_data_empty_clone = {shelf:'Empty-' for shelf,QR in shelf_QR_data_empty.items()}

                for sh,QR in shelf_QR_data_empty_clone.items():
                    for shelf,code in allocated_laptops_QRs.items():
                        if sh == shelf:
                            shelf_QR_data_empty_clone[sh] = code

                return shelf_QR_data_empty_clone

            allocated_laptop_QR_cloner_part1 = time()
            # {'12345678912 : 'Empty-','12345678918 : '45123698789'}
            allocated_laptop_QR_clone = allocated_laptop_QR_cloner(allocated_laptops_QRs, shelf_QR_data_empty) # reference allocated data
            allocated_laptop_QR_cloner_part2 = time()
            log_time[f'allocated_laptop_QR_cloner_part '] = allocated_laptop_QR_cloner_part2 - allocated_laptop_QR_cloner_part1


            print(f"allocated_laptop_QR_clone : {allocated_laptop_QR_clone}")
            print(f"shelf_QR_data_empty : {shelf_QR_data_empty}")

            def update_allocated_laptop_QR(allocated_laptops_QRs, shelf):

                del allocated_laptops_QRs[shelf]
                def convert(o):
                    if isinstance(o, np.int64):
                        return int(o)
                    raise TypeError
        
                with open('allocated_laptops_QRs.txt', 'w') as file:
                    file.write(json.dumps(allocated_laptops_QRs, default=convert))


                return allocated_laptops_QRs

                    # print("allocated_laptops_QRs.txt update sucessfully!")

            def diffrence_refrence(allocated_laptop_QR_clone, shelf_QR_data_empty, allocated_laptops_QRs):
                
                difference_with_allocated = {shelf:"None-" for shelf,laptop in allocated_laptop_QR_clone.items()}

                for shelf,laptop in allocated_laptop_QR_clone.items():
                    for shelf_empty,laptop_empty in shelf_QR_data_empty.items():

                        if shelf_empty == shelf and laptop_empty == laptop:
                            difference_with_allocated[shelf] = laptop_empty

                        elif shelf_empty == shelf and laptop_empty != laptop:
                            if laptop.split('-')[0] != "Green":
                                difference_with_allocated[shelf] = "None-"
                            elif laptop.split('-')[0] == "Green":
                                difference_with_allocated[shelf] = "Green"
                            if laptop_empty == "Empty-" and laptop.split('-')[0] == "Green":
                                update_allocated_laptop_QR(allocated_laptops_QRs, shelf)

                return difference_with_allocated
        
            
            diffrence_refrence_part1 = time()
            difference_with_allocated = diffrence_refrence(allocated_laptop_QR_clone, shelf_QR_data_empty, allocated_laptops_QRs)
            diffrence_refrence_part2 = time()
            log_time[f'diffrence_refrence_part '] = diffrence_refrence_part2 - diffrence_refrence_part1

            print(f"difference_with_allocated : {difference_with_allocated}")

            connect_part1 = time()
            connect.send_data(f"{difference_with_allocated}")
            connect_part2 = time()
            log_time[f'connect_part '] = connect_part2 - connect_part1

            show_part1 = time()
            cv2.putText(result_image, f"zoom: {zoom} brightness: {Brightness} focus: {cap.get(cv2.CAP_PROP_FOCUS)} exposure: {Exposure} contrast: {Contrast}", (130,230), color=(0,255,255), fontFace=2, fontScale=1, thickness=3, lineType=1)
            if eval(headless) == False:
                cv2.imshow('show',result_image)

            if k == ord('t'):
                save = True
                print("saving begins")

            if save == True:
                cv2.putText(result_image, "save image begins", (0,100), color=(0,255,0), fontFace=2, fontScale=2, thickness=4, lineType=1)
                save_frames.append(result_image)
            show_part2 = time()
            log_time[f'show_part '] = show_part2 - show_part1
        
        each_fram_time_part2 = time()
        log_time[f'each_fram_time_part '] = each_fram_time_part2 - each_fram_time_part1
        if (k == 27): #Esc key to quite the application 
            if len(save_frames) != 0:
                for saved_frame in save_frames:
                    output_video.write(saved_frame)
            connect.send_data("close")
            break



    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
    main_part2 = time()
    log_time['main_time'] = main_part2 - main_part1

    last_load_part1 = time()
    with open('allocated_laptops_QRs.txt', 'r') as file:
        allocated_laptops_QRsss = json.load(file)
    last_load_part2 = time()
    log_time[f'last_load_part '] = last_load_part2 - last_load_part1 
    
    with open('log_time_file.txt', 'w') as file:
        file.write(json.dumps(log_time, default=convert))
     
    print(f"allocated_laptops_QRs : {allocated_laptops_QRsss}")
    print(f"difference_with_allocated : {difference_with_allocated}")
