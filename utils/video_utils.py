import cv2
import numpy as np
import logging
import threading
from queue import Queue
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class VideoManager:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.video_queue = Queue()
        self.is_capturing = False
        self.save_video = False
        self.output_dir = "recordings"
        
        # Create recordings directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def start_capture(self, camera_index=0, save_video=False):
        """Start video capture"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
            
            self.is_capturing = True
            self.save_video = save_video
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            # Start saving thread if needed
            if save_video:
                self.save_thread = threading.Thread(target=self._save_video_loop)
                self.save_thread.daemon = True
                self.save_thread.start()
            
            logger.info("Video capture started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start video capture: {str(e)}")
            return False

    def stop_capture(self):
        """Stop video capture"""
        try:
            self.is_capturing = False
            
            if hasattr(self, 'capture_thread'):
                self.capture_thread.join(timeout=1.0)
            
            if self.save_video and hasattr(self, 'save_thread'):
                self.save_thread.join(timeout=1.0)
            
            if hasattr(self, 'cap'):
                self.cap.release()
            
            if hasattr(self, 'video_writer'):
                self.video_writer.release()
            
            logger.info("Video capture stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop video capture: {str(e)}")
            return False

    def _capture_loop(self):
        """Main video capture loop"""
        while self.is_capturing:
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.video_queue.put(frame)
                else:
                    logger.warning("Failed to capture frame")
                    break
            except Exception as e:
                logger.error(f"Frame capture error: {str(e)}")
                break

    def _save_video_loop(self):
        """Save video frames to file"""
        try:
            filename = os.path.join(
                self.output_dir,
                f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                filename, 
                fourcc, 
                self.fps,
                (self.width, self.height)
            )
            
            while self.is_capturing:
                if not self.video_queue.empty():
                    frame = self.video_queue.get()
                    self.video_writer.write(frame)
            
            logger.info(f"Video saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save video: {str(e)}")

    def get_frame(self):
        """Get frame from queue"""
        if not self.video_queue.empty():
            return self.video_queue.get()
        return None

    def clear_queue(self):
        """Clear video queue"""
        while not self.video_queue.empty():
            self.video_queue.get()

class VideoProcessor:
    @staticmethod
    def resize_frame(frame, width=None, height=None):
        """Resize frame while maintaining aspect ratio"""
        if width is None and height is None:
            return frame
            
        h, w = frame.shape[:2]
        if width is None:
            aspect = height / h
            width = int(w * aspect)
        elif height is None:
            aspect = width / w
            height = int(h * aspect)
            
        return cv2.resize(frame, (width, height))

    @staticmethod
    def normalize_frame(frame):
        """Normalize frame values to [0, 1]"""
        return frame.astype(np.float32) / 255.0

    @staticmethod
    def denormalize_frame(frame):
        """Convert normalized frame back to uint8"""
        return (frame * 255).astype(np.uint8)

    @staticmethod
    def apply_brightness_contrast(frame, brightness=0, contrast=0):
        """Adjust brightness and contrast of frame"""
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow
            frame = cv2.addWeighted(frame, alpha_b, frame, 0, gamma_b)

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            frame = cv2.addWeighted(frame, alpha_c, frame, 0, gamma_c)

        return frame

    @staticmethod
    def draw_face_landmarks(frame, landmarks, color=(0, 255, 0), thickness=1):
        """Draw facial landmarks on frame"""
        if landmarks is None:
            return frame
            
        frame_copy = frame.copy()
        for point in landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame_copy, (x, y), 2, color, thickness)
        return frame_copy

    @staticmethod
    def draw_bounding_box(frame, bbox, label=None, color=(0, 255, 0), thickness=2):
        """Draw bounding box and optional label on frame"""
        if bbox is None:
            return frame
            
        frame_copy = frame.copy()
        x, y, w, h = bbox
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)
        
        if label:
            cv2.putText(
                frame_copy,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness
            )
            
        return frame_copy
