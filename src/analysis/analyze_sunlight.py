import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os

def analyze_sunlight(bag_file, output_dir):
    """
    Analyzes a bag file for sunlight/lens flare impact.
    
    Args:
        bag_file (str): Path to the .bag file.
        output_dir (str): Directory to save analysis results.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file)
    
    # Configure streams (assuming recorded at standard resolution)
    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    frame_count = 0
    total_high_intensity_pixels = 0
    total_invalid_depth_in_high_intensity = 0
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
                
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # 1. Detect High Intensity Areas (Potential Flare/Sunlight)
            # Convert to grayscale
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            # Threshold for very bright pixels (e.g., > 250)
            _, high_intensity_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
            
            high_intensity_count = cv2.countNonZero(high_intensity_mask)
            
            if high_intensity_count > 0:
                # 2. Check Depth Validity in these areas
                # Depth value of 0 means invalid/no data
                depth_in_high_intensity = cv2.bitwise_and(depth_image, depth_image, mask=high_intensity_mask)
                invalid_depth_mask = (depth_in_high_intensity == 0) & (high_intensity_mask > 0)
                invalid_count = cv2.countNonZero(invalid_depth_mask.astype(np.uint8))
                
                total_high_intensity_pixels += high_intensity_count
                total_invalid_depth_in_high_intensity += invalid_count
                
                # Visualization for first few frames or significant events
                if frame_count % 30 == 0:
                    # Create visualization: Red = High Intensity, Blue = Invalid Depth
                    vis = color_image.copy()
                    vis[high_intensity_mask > 0] = [0, 0, 255] # Red for bright
                    vis[invalid_depth_mask > 0] = [255, 0, 0] # Blue for invalid in bright
                    
                    cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:04d}_analysis.png"), vis)

            frame_count += 1
            
    except RuntimeError:
        pass # End of bag file
    finally:
        pipeline.stop()
        
    print(f"Analysis Complete for {bag_file}")
    print(f"Total Frames: {frame_count}")
    if total_high_intensity_pixels > 0:
        ratio = (total_invalid_depth_in_high_intensity / total_high_intensity_pixels) * 100
        print(f"Invalid Depth Ratio in High Intensity Areas: {ratio:.2f}%")
        print("High ratio indicates sunlight/flare is likely causing depth dropouts.")
    else:
        print("No significant high intensity areas detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Sunlight/Flare Impact in Bag File")
    parser.add_argument("bag_file", help="Path to input .bag file")
    parser.add_argument("--output", default="analysis_results", help="Output directory")
    args = parser.parse_args()
    
    analyze_sunlight(args.bag_file, args.output)
