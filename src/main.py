import cv2
import time
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sensors.realsense_driver import RealSenseDriver
from src.sensors.respeaker_driver import RespeakerDriver
from src.vision.landmark_detector import LandmarkDetector
from src.map.map_manager import MapManager
from src.navigation.localizer import Localizer

def main():
    parser = argparse.ArgumentParser(description="Multimodal Navigation System")
    parser.add_argument("--map", type=str, default="map.json", help="Path to the map file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    args = parser.parse_args()

    # Initialize components
    rs_driver = RealSenseDriver()
    audio_driver = RespeakerDriver()
    map_manager = MapManager(args.map)
    detector = LandmarkDetector(args.model)
    localizer = Localizer(map_manager)

    try:
        rs_driver.start()
        audio_driver.start()
        
        intrinsics = rs_driver.get_intrinsics()
        
        print("System started. Press 'q' to exit.")
        
        while True:
            # 1. Capture Sensors
            color, depth, depth_frame = rs_driver.get_frames()
            if color is None:
                continue
                
            doa = audio_driver.get_direction()
            
            # 2. Detect Landmarks
            landmarks = detector.detect(color, depth_frame, intrinsics)
            
            # 3. Update Localization
            current_pos = localizer.update(landmarks)
            
            # 4. Visualization / Feedback
            # Draw landmarks on color image
            for lm in landmarks:
                x1, y1, x2, y2 = lm['bbox']
                cv2.rectangle(color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color, f"{lm['class']} {lm['position'][2]:.2f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display Status
            cv2.putText(color, f"Pos: {current_pos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(color, f"DOA: {doa}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow("Navigation System", color)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rs_driver.stop()
        audio_driver.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
