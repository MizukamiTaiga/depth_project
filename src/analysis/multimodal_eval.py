import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os
from ultralytics import YOLO

def multimodal_eval(bag_file, model_path, output_dir):
    """
    Evaluates Single Modal (RGB, Depth) vs Multi-modal Fusion.
    
    Args:
        bag_file (str): Path to the .bag file.
        model_path (str): Path to YOLO model.
        output_dir (str): Directory to save analysis results.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load Model
    model = YOLO(model_path)
    
    # Setup RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file)
    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)
    
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    frame_count = 0
    
    # Metrics
    rgb_detections = 0
    fusion_confirmations = 0
    depth_only_candidates = 0 # Hypothetical (objects with depth but no RGB class)
    
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
            
            # --- 1. RGB Single Modal Eval ---
            results = model(color_image, verbose=False)
            
            current_frame_detections = []
            
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    rgb_detections += 1
                    current_frame_detections.append((x1, y1, x2, y2, conf, model.names[cls_id]))
            
            # --- 2. Multi-modal Fusion Eval (RGB + Depth) ---
            # Check if detected objects have valid depth
            valid_fusion_count = 0
            for (x1, y1, x2, y2, conf, label) in current_frame_detections:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                dist = depth_frame.get_distance(cx, cy)
                
                if dist > 0.1 and dist < 5.0: # Valid range
                    valid_fusion_count += 1
                    fusion_confirmations += 1
                else:
                    # RGB detected something, but Depth says it's invalid or too far/close
                    # This could be a "Ghost" detection (e.g. poster) or just sensor noise
                    pass

            # --- 3. Depth Single Modal Eval (Hypothetical) ---
            # Simple obstacle detection: count pixels in close range
            # This is just a proxy metric for "Depth sees something"
            depth_mask = (depth_image > 100) & (depth_image < 2000) # 10cm to 2m
            depth_pixel_count = np.count_nonzero(depth_mask)
            
            if depth_pixel_count > 10000: # Arbitrary threshold for "Obstacle Present"
                depth_only_candidates += 1
            
            # Visualization
            if frame_count % 30 == 0:
                vis = color_image.copy()
                cv2.putText(vis, f"RGB Dets: {len(current_frame_detections)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis, f"Fusion Valid: {valid_fusion_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.imwrite(os.path.join(output_dir, f"eval_{frame_count:04d}.jpg"), vis)

            frame_count += 1
            
    except RuntimeError:
        pass
    finally:
        pipeline.stop()
        
    print("=== Evaluation Report ===")
    print(f"Total Frames: {frame_count}")
    print(f"RGB Detections: {rgb_detections}")
    print(f"Fusion Confirmations (Valid Depth): {fusion_confirmations}")
    if rgb_detections > 0:
        print(f"Fusion Confirmation Rate: {(fusion_confirmations/rgb_detections)*100:.2f}%")
    print(f"Frames with Depth Obstacles: {depth_only_candidates}")
    print("Conclusion: Fusion filters out objects with invalid depth (potential false positives or out of range).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Evaluation")
    parser.add_argument("bag_file", help="Path to input .bag file")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--output", default="eval_results", help="Output directory")
    args = parser.parse_args()
    
    multimodal_eval(args.bag_file, args.model, args.output)
