'''
This script is used to capture images and depth maps from a RealSense camera and save them to a timestamp folder.
The images and depth maps are saved in the GraspNet compatible format.

bash command:
python photo.py --camera realsense  
'''

import cv2
import numpy as np
import pyrealsense2 as rs
import os
import time
from datetime import datetime


def create_timestamp_folder(base_dir="captures"):
    """create timestamp folder for saving images"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    folder_path = os.path.join(base_dir, timestamp)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_graspnet_depth(depth_data, path):
    """save graspnet compatible depth map (uint16 PNG, unit: mm)"""
    # convert to mm and ensure in uint16 range
    depth_mm = np.clip(depth_data * 1000, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, depth_mm)


def display_and_capture():
    """ display and capture images from RealSense camera and save them to a timestamp folder """
    # initialize RealSense
    pipeline = rs.pipeline()
    config = rs.config()

    # configure camera stream (consistent with GraspNet training size)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    try:
        pipeline.start(config)
        align = rs.align(rs.stream.color)  # align depth to color frame
        print("Press ENTER to capture current frame, Q to exit...")

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # convert image data
            color_image = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())  # original uint16 data

            # get actual physical unit data (meters)
            depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
            depth_meters = depth_data.astype(np.float32) * depth_scale

            # display processing
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_data, alpha=0.03),  # only for display
                cv2.COLORMAP_JET
            )

            # display image
            cv2.imshow("RGB Preview", color_image)
            cv2.imshow("Depth Preview", depth_colormap)

            key = cv2.waitKey(1)

            if key == 13:  # ENTER key
                try:
                    save_folder = create_timestamp_folder()

                    # generate filename for saving
                    timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
                    prefix = os.path.join(save_folder, f"capture_{timestamp}")

                    # save original data 
                    cv2.imwrite(f"{prefix}_color.png", color_image)
                    np.save(f"{prefix}_depth_meters.npy", depth_meters)

                    # save GraspNet compatible format
                    save_graspnet_depth(depth_meters, f"{prefix}_depth_graspnet.png")

                    print(f"Captured and saved to: {save_folder}")

                except Exception as e:
                    print(f"Save failed: {str(e)}")

            elif key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # create root storage directory for saving images
    os.makedirs("captures", exist_ok=True)
    display_and_capture()