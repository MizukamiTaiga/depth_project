import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os

def export_for_yolo(bag_file, output_dir, interval=30):
    """
    Exports frames from a bag file for YOLO annotation.
    
    Args:
        bag_file (str): Path to the .bag file.
        output_dir (str): Directory to save images.
        interval (int): Frame interval to save (e.g., every 30 frames).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file)
    config.enable_stream(rs.stream.color)

    pipeline.start(config)
    
    frame_count = 0
    saved_count = 0
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
                
            if frame_count % interval == 0:
                color_image = np.asanyarray(color_frame.get_data())
                filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(filename, color_image)
                saved_count += 1
                
            frame_count += 1
            
    except RuntimeError:
        pass
    finally:
        pipeline.stop()
        
    print(f"Export Complete. Saved {saved_count} images to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Frames for YOLO Annotation")
    parser.add_argument("bag_file", help="Path to input .bag file")
    parser.add_argument("--output", default="yolo_dataset", help="Output directory")
    parser.add_argument("--interval", type=int, default=30, help="Frame interval to save")
    args = parser.parse_args()
    
    export_for_yolo(args.bag_file, args.output, args.interval)
