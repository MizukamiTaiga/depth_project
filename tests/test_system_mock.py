import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock numpy before importing modules that use it
mock_np = MagicMock()
# Setup basic numpy behavior needed for the code
def mock_array(x):
    return MagicMock(side_effect=lambda: x) # Simplified
mock_np.array.side_effect = lambda x: x # Just return the list for simplicity in this mock
mock_np.linalg.norm.return_value = 1.0
sys.modules['numpy'] = mock_np
sys.modules['pyrealsense2'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['cv2'] = MagicMock()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Now import the modules
from src.map.map_manager import MapManager
# We need to patch Localizer because it imports numpy at top level, which is now mocked.
# But our mock_np is a MagicMock, so it should be fine.
from src.navigation.localizer import Localizer
from src.vision.landmark_detector import LandmarkDetector

class TestNavigationSystem(unittest.TestCase):
    def setUp(self):
        # Mock MapManager
        self.map_manager = MapManager()
        self.map_manager.landmarks = [
            {"id": 1, "class": "test_obj", "position": [10.0, 0.0, 5.0]}
        ]
        # We need to adjust Localizer to handle our mocked numpy or lists
        # Since we mocked numpy.array to return the list itself (lambda x: x), 
        # we can do list arithmetic if we are careful, OR we patch the update method logic.
        
        # Actually, let's just test the flow and MapManager, 
        # and for Localizer, we verify it calls the right things.
        self.localizer = Localizer(self.map_manager)
        # Reset current_position to a list for our mock math
        self.localizer.current_position = [0.0, 0.0, 0.0]

    def test_map_manager(self):
        self.assertEqual(len(self.map_manager.landmarks), 1)
        self.assertEqual(self.map_manager.landmarks[0]['class'], 'test_obj')

    def test_localizer_logic(self):
        # Since we mocked numpy, the vector math in Localizer will fail if we don't mock it precisely.
        # Instead of complex numpy mocking, let's verify the logic by patching the math or 
        # just checking if it finds the correct landmark.
        
        detected_landmarks = [
            {
                'class': 'test_obj',
                'confidence': 0.9,
                'bbox': [0, 0, 100, 100],
                'position': [0.0, 0.0, 2.0]
            }
        ]
        
        # We want to verify that Localizer finds the landmark and updates position.
        # Let's patch np.array to return a class that supports subtraction
        
        class MockArray(list):
            def __sub__(self, other):
                return MockArray([a - b for a, b in zip(self, other)])
            def __add__(self, other):
                return MockArray([a + b for a, b in zip(self, other)])
            def __mul__(self, other):
                if isinstance(other, (int, float)):
                    return MockArray([a * other for a in self])
                return MockArray([a * b for a, b in zip(self, other)])
            def __rmul__(self, other):
                return self.__mul__(other)
                
        mock_np.array.side_effect = lambda x: MockArray(x)
        
        # Re-init localizer to use MockArray
        self.localizer.current_position = MockArray([0.0, 0.0, 0.0])
        
        new_pos = self.localizer.update(detected_landmarks)
        
        # Target: [10, 0, 5]
        # Relative: [0, 0, 2]
        # Est: [10, 0, 3]
        # Update: 0.8*0 + 0.2*10 = 2.0
        #         0.8*0 + 0.2*0 = 0.0
        #         0.8*0 + 0.2*3 = 0.6
        
        self.assertAlmostEqual(new_pos[0], 2.0)
        self.assertAlmostEqual(new_pos[2], 0.6)
        print(f"Mock Test Passed. Updated Pos: {new_pos}")

if __name__ == '__main__':
    unittest.main()
