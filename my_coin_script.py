import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO
import distance_estimator
import torch
import midas_depth_estimator

# -------------------------
# Parse command-line arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (e.g., "runs/detect/train/weights/best.pt")', required=True)
parser.add_argument('--source', help='Image source: image file ("test.jpg"), folder ("test_dir"), video file ("testvid.mp4"), or USB camera ("usb0")', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold (e.g., "0.4")', default=0.5)
parser.add_argument('--resolution', help='Display resolution in WxH (e.g., "640x480"), otherwise matches source', default=None)
parser.add_argument('--record', help='Record the video with all overlays. (Requires --resolution)', action='store_true')
args = parser.parse_args()

# -------------------------
# User inputs
# -------------------------
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model not found.')
    sys.exit(0)

# -------------------------
# Load YOLO model
# -------------------------
model = YOLO(model_path, task='detect')
labels = model.names

# -------------------------
# Determine source type
# -------------------------
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']
if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])  #"usb0" => 0
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid.')
    sys.exit(0)

# -------------------------
# Handle display resolution
# -------------------------
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# -------------------------
# Set up recording if requested
# -------------------------
if record:
    if source_type not in ['video', 'usb']:
        print('Recording only works for video and USB sources.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution for recording.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 240
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))
    print(f"Recording will be saved to {record_name}")

# -------------------------
# Initialize image/video source
# -------------------------
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [file for file in glob.glob(os.path.join(img_source, '*')) if os.path.splitext(file)[1] in img_ext_list]
elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# -------------------------
# Load MiDaS model for depth estimation
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas, midas_transform = midas_depth_estimator.load_midas_model(device)

# -------------------------
# Define camera intrinsic parameters
# -------------------------
# For simplicity, if resolution is set we assume the principal point is at half the resolution.
if resize:
    CX, CY = resW / 2.0, resH / 2.0
else:
    CX, CY = None, None  # this will be updated on first frame
FX = distance_estimator.FOCAL_LENGTH
FY = distance_estimator.FOCAL_LENGTH

# -------------------------
# Colors for drawing (Tableau 10)
# -------------------------
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133),
               (88,159,106), (96,202,231), (159,124,168), (169,162,241),
               (98,118,150), (172,176,184)]

# -------------------------
# FPS tracking
# -------------------------
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

while True:
    t_start = time.perf_counter()

    # Acquire frame from source
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images processed. Exiting.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of video. Exiting.')
            break
    elif source_type == 'usb':
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Unable to read from USB camera. Exiting.')
            break
    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(frame_bgra.copy(), cv2.COLOR_BGRA2BGR)
        if frame is None:
            print('Unable to capture from Picamera. Exiting.')
            break

    # Resize frame if we need
    if resize:
        frame = cv2.resize(frame, (resW, resH))
    else:
        if CX is None or CY is None:
            h, w = frame.shape[:2]
            CX, CY = w / 2.0, h / 2.0

    # -------------------------
    # Compute depth map using MiDaS
    # -------------------------
    depth_map = midas_depth_estimator.get_depth_map(frame, midas, midas_transform, device)

    # -------------------------
    # Run YOLO detection
    # -------------------------
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    for i in range(len(detections)):

        # we need to extract bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        # Get detection class and confidence
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

            # --- Existing Range and Bearing (2D) Estimation ---
            range_est, bearing = distance_estimator.estimate_range_and_bearing(
                xmin, xmax, frame.shape[1],
                distance_estimator.KNOWN_COIN_WIDTH,
                distance_estimator.FOCAL_LENGTH
            )

            # --- 3D Coordinate Estimation using Depth Map ---
            u = (xmin + xmax) / 2.0  # horizontal center of bounding box
            v = (ymin + ymax) / 2.0  # vertical center

            # We extract depth values within the bounding box
            u_min, u_max = max(0, xmin), min(frame.shape[1], xmax)
            v_min, v_max = max(0, ymin), min(frame.shape[0], ymax)
            box_depth = depth_map[v_min:v_max, u_min:u_max]
            if box_depth.size == 0:
                mean_depth = 0
            else:
                mean_depth = np.mean(box_depth)
            # For demonstration, we assume mean_depth correlates with the real distance.
            # (Calibration is needed to convert relative depth to real units.)
            X_coord = (u - CX) * mean_depth / FX
            Y_coord = (v - CY) * mean_depth / FY
            Z_coord = mean_depth

            # Display the distance, bearing, and 3D coordinate
            cv2.putText(frame, f'Dist: {range_est:.2f} cm, Bear: {bearing:.2f}°',
                        (xmin, label_ymin - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'3D: ({X_coord:.2f}, {Y_coord:.2f}, {Z_coord:.2f})',
                        (xmin, label_ymin - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # -------------------------
    # Draw FPS and object count
    # -------------------------
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # -------------------------
    # Show frame and record if enabled
    # -------------------------
    cv2.imshow('Detection and 3D Localization', frame)
    if record:
        recorder.write(frame)

    # -------------------------
    # Key handling and FPS calculation
    # -------------------------
    key = cv2.waitKey(5) if source_type in ['video', 'usb', 'picamera'] else cv2.waitKey()
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', frame)

    t_stop = time.perf_counter()
    frame_rate_calc = 1 / (t_stop - t_start)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)


print(f'Average FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
