import pyrealsense2 as rs
import numpy as np

class RealSenseDriver:
    def __init__(self, width=1280, height=800, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.profile = None

    def start(self):
        """Starts the RealSense pipeline with aligned streams."""
        # Enforce the project standard: Color 1280x800, Depth 1280x720 -> Aligned to Color
        self.config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, self.fps)

        self.profile = self.pipeline.start(self.config)
        
        # Create alignment object (align to color)
        self.align = rs.align(rs.stream.color)
        print("RealSense pipeline started. Aligned to Color (1280x800).")

    def get_frames(self):
        """Returns aligned color and depth frames."""
        if not self.pipeline:
            return None, None

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image, depth_frame # Return raw depth frame for distance queries

    def get_intrinsics(self):
        """Returns the intrinsics of the color stream (which depth is aligned to)."""
        if self.profile:
            return self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return None

    def stop(self):
        """Stops the pipeline."""
        if self.pipeline:
            self.pipeline.stop()
            print("RealSense pipeline stopped.")
