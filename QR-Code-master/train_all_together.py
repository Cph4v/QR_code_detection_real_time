from all_steps_together import *
from send_request import ConnectRPI
import time

from all_steps_together import *
from manuall_ROI_detection import *


import argparse


# Use argparse to handle command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a specific function.")
    parser.add_argument("function_name", help="The name of the function to run")
    parser.add_argument("--port", type=int, default=8070, help="The port number (default: 8070)")
    parser.add_argument("--device", type=str, default="/dev/ELP1", help="The camera address (default is 0)")
    parser.add_argument("--server_ip", type=str, default="10.42.0.1", help="The camera address (default is 10.42.0.1)")
    parser.add_argument("--grid", type=str, default="[1,1,0,0]", help="The grid of frames default is [2,2,50,50]")
    parser.add_argument("--headless", type=str, default="False", help="show result or not, make headless = True to prevent show windows!")
    parser.add_argument("--camera_settings", type=str, default="[4000, 3000, 0, 39, 32, 0, 32, 12]")
    parser.add_argument("--ROI", type=str, default="DETECTION_area", help="make ROI area for LED's and perspective detection area")
    args = parser.parse_args()

    # Call the specified function
    if args.function_name == "allocate_laptop_and_shelf":
        allocate_laptop_and_shelf(server_ip=args.server_ip ,device=args.device, port=args.port, grid=args.grid, headless=args.headless, camera_settings=args.camera_settings)
    elif args.function_name == "real_time":
        real_time(server_ip=args.server_ip, device=args.device, port=args.port, grid=args.grid, headless=args.headless, camera_settings=args.camera_settings)
    elif args.function_name == "automate_detection":
        automate_detection(server_ip=args.server_ip, device=args.device, port=args.port, grid=args.grid, headless=args.headless, camera_settings=args.camera_settings)
    elif args.function_name == "make_ROI":
        make_ROI(device=args.device, ROI=args.ROI)
    else:
        print("Invalid function name. Please choose 'allocate_laptop_and_shelf', 'real_time', or 'automate_detection' or 'make_ROI'")


