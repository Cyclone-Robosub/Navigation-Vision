import math

def estimate_distance(pixel_width, known_width, focal_length):
    if pixel_width <= 0:
        return None
    return (known_width * focal_length) / pixel_width

def estimate_range_and_bearing(xmin, xmax, image_width, known_width, focal_length):
    pixel_width = xmax - xmin
    if pixel_width <= 0:
        return None, None

    range_est = (known_width * focal_length) / pixel_width

    object_center_x = (xmin + xmax) / 2.0
    image_center_x = image_width / 2.0
    dx = object_center_x - image_center_x
    bearing_rad = math.atan(dx / focal_length)
    bearing_deg = math.degrees(bearing_rad)

    return range_est, bearing_deg

KNOWN_COIN_WIDTH = 2.5  # in centimeters (adjust to match your object’s real width)
FOCAL_LENGTH = 700      # in pixels (should be determined by calibration)
