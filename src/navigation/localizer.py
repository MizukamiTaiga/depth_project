import numpy as np

class Localizer:
    def __init__(self, map_manager):
        self.map_manager = map_manager
        self.current_position = np.array([0.0, 0.0, 0.0]) # Initial guess
        self.current_orientation = 0.0 # Yaw angle in degrees

    def update(self, detected_landmarks):
        """
        Updates the current position based on detected landmarks.
        
        Args:
            detected_landmarks (list): List of landmarks detected by the vision system.
                                       Each item has 'class' and 'position' (relative to camera).
        """
        if not detected_landmarks:
            return self.current_position

        # Simple approach: Find the nearest matching landmark in the map and assume we are close to it
        # minus the relative position.
        # Ideally, we would use a particle filter or least squares optimization for multiple landmarks.
        
        for det_lm in detected_landmarks:
            # Find corresponding landmark in the map by class
            # This is a naive matching. In reality, we need data association (ID matching).
            # For now, we assume unique classes or nearest neighbor.
            
            # Let's try to match by class first
            candidates = [lm for lm in self.map_manager.landmarks if lm['class'] == det_lm['class']]
            
            if candidates:
                # Find the closest candidate to our current estimated position
                # But wait, det_lm['position'] is relative to the CAMERA.
                # Global_Pos = Current_Pos + Rotation * Relative_Pos
                # So, Current_Pos = Global_Pos - Rotation * Relative_Pos
                
                # For this prototype, let's assume the camera is facing North (0 degrees) for simplicity
                # or that we just want to know WHICH landmark we are seeing.
                
                # Let's take the first candidate for now (simplification)
                target_lm = candidates[0]
                target_pos = np.array(target_lm['position'])
                relative_pos = np.array(det_lm['position'])
                
                # Estimate position
                # Assuming no rotation for this step 0
                estimated_pos = target_pos - relative_pos
                
                # Simple smoothing
                alpha = 0.2
                self.current_position = (1 - alpha) * self.current_position + alpha * estimated_pos
                
                # print(f"Matched {det_lm['class']}. Est Pos: {self.current_position}")
                
        return self.current_position

    def get_position(self):
        return self.current_position
