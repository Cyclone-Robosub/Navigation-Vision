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
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Running the Detection & 3D Localization Script](#running-the-detection--3d-localization-script)
  - [Recording the Output](#recording-the-output)
  - [Using the Standalone Video Recorder](#using-the-standalone-video-recorder)
- [Configuration and Calibration](#configuration-and-calibration)
- [Git LFS and Large Files](#git-lfs-and-large-files)
- [License](#license)

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
## To run it then you do
```bash
python my_coin_script.py --model "my_coin.pt" --source "usb0" --resolution "640x480" --record
```
