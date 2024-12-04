import cv2
import numpy as np
import tempfile
import os

def save_frame_as_video(frame, fps=25):
    """
    Convert a single frame to a video file suitable for VSR processing.
    The VSR system expects a video file with proper frame rate.
    
    Args:
        frame: numpy array of shape (H, W, C)
        fps: frame rate to use (default: 25 as expected by VSR)
        
    Returns:
        path to temporary video file
    """
    # Ensure frame is in correct format
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        # Write frame multiple times to create a short video
        # VSR system needs some frames to work with
        for _ in range(int(fps)):  # 1 second of video
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
        writer.release()
        return temp_path
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

def cleanup_temp_file(filepath):
    """Safely remove temporary file"""
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Error cleaning up temporary file: {e}")
