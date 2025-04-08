# QR-Code Project

Welcome to the **QR-Code Project** repository! This collection of scripts demonstrates advanced programming capabilities in the fields of QR code processing, camera tuning, motion detection, and more. The repository serves as a portfolio piece, highlighting a part of a project previously developed for Rapid Solutions International (source code has been removed due to a Non-Disclosure Agreement).

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Usage](#usage)
- [Demo](#demo)

## Project Overview

### Purpose
This project is designed to control and manage laptop allocation on warehouse shelves. Its primary goal is to ensure that workers pick the correct laptops by providing visual cues and alerts.

### Functionality
- **Error Prevention**  
  If a laptop is picked incorrectly, a red LED lights up above its position as an error indicator.

- **Guided Retrieval**  
  When the main software sends a JSON request to the local warehouse system, a green LED lights up above the correct laptop’s shelf—guiding workers without triggering the red LED alarm.

- **Inventory Management**  
  Managers can add new laptops by placing them on empty shelves and manually updating the system. The local software assigns these laptops to shelf QR codes, updates the local database, and synchronizes with the main software.

> **Note:** Source code for this project was removed due to an NDA between me and Rapid Solutions International.

## Project Structure

The repository contains the following key files and folders:

- **all_steps_together.py**:  
  Integrates all major steps of the project into one script.

- **camera_tuning_helpers.py**:  
  Contains helper functions for tuning camera settings.

- **client.py**:  
  Client-side script for handling specific tasks related to QR code processing.

- **detect_led_blubs.py**:  
  Script for detecting LED bulbs within an image.

- **function_for_app.py**:  
  Houses various utility functions used across the application.

- **manual_ROI_detection.py**:  
  Script for manually detecting regions of interest (ROIs) in images.

- **motion_detection.py**:  
  Implements motion detection functionality.

- **train_all_together.py**:  
  Script to train models or run processes required for the application.

- **Dockerfile**:  
  Docker configuration for deploying the application on Jetson Nano and potential cloud environments.

- **environment.yml**:  
  Conda environment configuration for setting up dependencies.

## Features

- **Comprehensive QR Code Processing**  
  All necessary steps from detection to processing are included.

- **Camera Tuning Tools**  
  A variety of functions to adjust camera settings and optimize QR code detection.

- **Motion Detection Algorithms**  
  Implements techniques to detect movement within a video frame.

- **LED Detection**  
  Capable of identifying and processing images containing LED bulbs.

- **Utility Functions**  
  A robust set of helper functions to support numerous project tasks.

- **Docker Deployment**  
  Includes configuration to deploy on Jetson Nano with potential cloud deployment options.

## Usage

This project is primarily intended as a demonstration of my technical skills rather than for production use. While there are no explicit installation or usage instructions, you can review the source scripts to gain insights into:

- How QR codes are detected and processed.
- The methods used for camera tuning and motion detection.
- How LED detection is implemented and integrated.

Feel free to explore the repository and adapt the code for your own research or learning purposes.

## Demo

Check out the video demo to see the system in action:  
**[After Detecting Motion: Software starts checking all QR codes and lights up LED’s upon lost QR code and laptop](./after%20detecting%20a%20motion%20software%20start%20to%20check%20all%20QR%20codes%20and%20light%20up%20LED%E2%80%99s%20upon%20lost%20QR%20code%20and%20laptop.mp4)**

<p align="center">
  <img src="https://github.com/Cph4v/QR_code_detection_real_time/blob/main/photo_2024-06-09_15-04-02.jpg" width="400" alt="Demo Photo 1"/>
  <img src="https://github.com/Cph4v/QR_code_detection_real_time/blob/main/edited2.png" width="400" alt="Demo Photo 2"/>
</p>
