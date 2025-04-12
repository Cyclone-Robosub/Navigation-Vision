import cv2
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Record video from a camera and save it to a file.")
    parser.add_argument("--source", type=str, required=True,
                        help='Camera source index (e.g., "0" or "usb0" to indicate camera 0).')
    parser.add_argument("--resolution", type=str, default="640x480",
                        help='Capture resolution in WxH format (e.g., "640x480").')
    parser.add_argument("--output", type=str, default="recorded_video.avi",
                        help='Output video file name (e.g., recorded_video.avi).')
    args = parser.parse_args()
    try:
        if args.source.startswith("usb"):
            cam_idx = int(args.source[3:])
        else:
            cam_idx = int(args.source)
    except ValueError:
        print("Invalid source. Please provide a numeric index (e.g., 0 or usb0).")
        sys.exit(1)
    try:
        width, height = map(int, args.resolution.split("x"))
    except Exception as e:
        print("Invalid resolution format. Expected format is WxH, for example, 640x480.")
        sys.exit(1)

    # Open the camera using OpenCV
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Unable to open camera source with index {cam_idx}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    output_filename = args.output
    fps = 240.0
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    recorder = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print(f"Recording from camera index {cam_idx} at {width}x{height}.")
    print("Press 'q' in the window to stop recording.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab a frame. Exiting.")
            break

        # Write the frame to the output file
        recorder.write(frame)

        # Display the frame in a window (optional)
        cv2.imshow("Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up and release resources
    cap.release()
    recorder.release()
    cv2.destroyAllWindows()
    print(f"Recording saved to {output_filename}")

if __name__ == "__main__":
    main()
