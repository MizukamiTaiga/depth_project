import json
import os
import numpy as np

class MapManager:
    def __init__(self, map_path=None):
        self.landmarks = []
        self.map_path = map_path
        if map_path and os.path.exists(map_path):
            self.load_map(map_path)

    def load_map(self, path):
        """Loads landmarks from a JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.landmarks = data.get('landmarks', [])
            print(f"Map loaded from {path} with {len(self.landmarks)} landmarks.")
        except Exception as e:
            print(f"Error loading map: {e}")

    def save_map(self, path):
        """Saves landmarks to a JSON file."""
        data = {'landmarks': self.landmarks}
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Map saved to {path}.")
        except Exception as e:
            print(f"Error saving map: {e}")

    def add_landmark(self, class_name, position, audio_signature=None):
        """Adds a new landmark to the map.
        
        Args:
            class_name (str): The label of the landmark (e.g., 'vending_machine').
            position (list): [x, y, z] coordinates.
            audio_signature (dict, optional): Audio features or metadata.
        """
        new_id = len(self.landmarks) + 1
        landmark = {
            "id": new_id,
            "class": class_name,
            "position": position,
            "audio_signature": audio_signature
        }
        self.landmarks.append(landmark)

    def find_nearest_landmark(self, position):
        """Finds the nearest landmark to a given position."""
        if not self.landmarks:
            return None, float('inf')
        
        pos_np = np.array(position)
        min_dist = float('inf')
        nearest = None
        
        for lm in self.landmarks:
            lm_pos = np.array(lm['position'])
            dist = np.linalg.norm(pos_np - lm_pos)
            if dist < min_dist:
                min_dist = dist
                nearest = lm
                
        return nearest, min_dist
