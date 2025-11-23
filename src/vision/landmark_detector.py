from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs

class LandmarkDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.classes = self.model.names

    def detect(self, color_image, depth_frame, intrinsics):
        """
        Detects landmarks in the image and calculates their 3D positions.
        
        Args:
            color_image (numpy.ndarray): RGB image.
            depth_frame (rs.frame): Raw depth frame (for distance queries).
            intrinsics (rs.intrinsics): Camera intrinsics.
            
        Returns:
            list: List of detected landmarks with format:
                  {'class': str, 'confidence': float, 'bbox': [x1, y1, x2, y2], 'position': [x, y, z]}
        """
        results = self.model(color_image, verbose=False)
        landmarks = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.classes[cls_id]

                # Calculate center of the bbox
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Get distance at the center point
                # Note: In a real application, we might want to take the median of a small region
                dist = depth_frame.get_distance(cx, cy)

                if dist > 0:
                    # Deproject pixel to 3D point
                    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], dist)
                    
                    landmarks.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'position': point_3d
                    })

        return landmarks
