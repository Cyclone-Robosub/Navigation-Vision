# Navigation Vision

This repository contains a set of Python scripts for object detection and 3D localization using a monocular camera. The project uses a YOLO model (for example, `my_coin.pt`) along with additional modules to:
- Detect objects and draw bounding boxes.
- Estimate distance (range) and horizontal bearing (angle) from the camera to the object using the pinhole camera model.
- (Optionally) Compute a 3D coordinate estimate by combining YOLO detections with a monocular depth estimation model (MiDaS).
- Record the detection video including overlays such as bounding boxes, distance, bearing, and 3D coordinates.

> **Note:**  
> - The distance estimation requires calibration of camera parameters (e.g., focal length, known object width).  
> - The MiDaS depth output is provided in relative units. For accurate depth (e.g., in centimeters), you may need to calibrate the output.
> - Large files (such as demo videos) are not included in the Git history.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Run](#run)
- [Project Structure](#project-structure)

## Prerequisites

- **Python 3.7+**  
- **pip**  
- **Git**

You will also need to install the following Python packages:

- `torch`
- `opencv-python`
- `numpy`
- `Pillow`
- `timm`
- `ultralytics`

## Installation
 ```bash
  pip install torch opencv-python numpy Pillow timm ultralytics
  pip install timm
```
1. **Clone the Repository**

```bash
   git clone https://github.com/Cyclone-Robosub/Navigation-Vision.git
   cd Navigation-Vision
```
## Run
```bash
python my_coin_script.py --model "my_coin.pt" --source "usb0" --resolution "640x480" --record
```


## Project Structure

- `my_coin_script.py`
Main script for object detection and 3D localization. It overlays bounding boxes, distance, bearing, and estimated 3D coordinates on the camera feed and can record the output video when the --record flag is used.

- `distance_estimator.py`
Contains functions for distance and bearing estimation using the pinhole camera model.

- `midas_depth_estimator.py`
Loads and runs the MiDaS model to generate a depth map from an image frame.

- `video_recorder.py`
A standalone script to record raw video from your camera.

- `my_coin.pt`
Your trained YOLO model file.

Other files: Readme.txt, sample images, videos, etc.
