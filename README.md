# QR-Code Project

This repository is a collection of scripts demonstrating my capabilities in programming and deploying solutions. The project showcases various functionalities related to QR code processing, camera tuning, motion detection, and more. It is intended as a portfolio piece rather than a deployable application. This code was similar to a part of my project in Rapid Solutions International.

## Project Overview

### Purpose
The project is designed to control and manage laptop allocation on warehouse shelves. It ensures that workers pick the correct laptops by providing visual cues and alerts.

### Functionality
1. **Error Prevention**: If a laptop is picked incorrectly, a red LED lights up above its position, indicating an error.
2. **Guided Retrieval**: When the main software sends a JSON request to the local warehouse software, a green LED lights up above the designated laptop's shelf, guiding workers to pick the correct laptop without triggering the red LED alarm.
3. **Inventory Management**: Managers can add new laptops to the inventory by placing them on empty shelves and manually updating the system. The local software then assigns the laptops to shelf QR codes, updates the local database, and synchronizes this information with the main software.

## Project Structure

- **all_steps_together.py**: Integrates all major steps of the project in one script.
- **camera_tuning_helpers.py**: Contains helper functions for tuning camera settings.
- **client.py**: Client-side script for handling specific tasks related to QR code processing.
- **detect_led_blubs.py**: Script for detecting LED bulbs within an image.
- **function_for_app.py**: Contains various utility functions used across the application.
- **manual_ROI_detection.py**: Script for manually detecting regions of interest (ROIs) in images.
- **motion_detection.py**: Implements motion detection functionality.
- **train_all_together.py**: Script to train models or processes required for the application.
- **Dockerfile**: Docker configuration for deployment on Jetson Nano and potentially the cloud.
- **environment.yml**: Conda environment configuration file for setting up dependencies.

## Features

- **Comprehensive QR Code Processing**: Includes all necessary steps from detection to processing.
- **Camera Tuning**: Tools and functions for adjusting camera settings to optimize QR code detection.
- **Motion Detection**: Implements algorithms to detect movement within a frame.
- **LED Detection**: Identifies and processes images containing LED bulbs.
- **Utility Functions**: A set of helper functions to support various tasks within the project.
- **Docker Deployment**: Configuration for running the project on a Jetson Nano device, with potential for cloud deployment.

## Usage

Since this project is primarily for showcasing my skills, there are no installation or usage instructions. However, you can explore the scripts to understand the implementation details and how they interact.
<p align="center">
  <img src="https://github.com/Cph4v/QR_code_detection_real_time/blob/main/photo_2024-06-09_15-04-02.jpg" width="400"/>
  <img src="https://github.com/Cph4v/QR_code_detection_real_time/blob/main/edited2.png" width="400"/>
</p>


