import torch
import cv2
import numpy as np
from PIL import Image

def load_midas_model(device):

    # Choose model type: "MiDaS_small" is faster, "DPT_Large" is more accurate.
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "MiDaS_small":
        transform = midas_transforms.small_transform
    else:
        transform = midas_transforms.dpt_transform
    return midas, transform

def get_depth_map(frame, midas, transform, device):
    """ we are given an input frame (BGR image), compute and return the depth map.
    This updated version converts the frame to a PIL image and then to a numpy
    array before applying the transform."""

    # Convert from BGR to RGB and create a PIL Image
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Convert the PIL image to a NumPy array of type float32
    img_np = np.array(img_pil).astype(np.float32)
    
    # Apply the MiDaS transform using the numpy array rather than the PIL image
    input_batch = transform(img_np).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    # We could normalize depth_map for visualization if needed.
    return depth_map
