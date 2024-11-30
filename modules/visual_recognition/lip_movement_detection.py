import cv2
import numpy as np
import logging
import threading
from queue import Queue
import mediapipe as mp
from datetime import datetime

class LipDetection:
    def __init__(self):
        """Initialize lip detection system"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.video_queue = Queue()
        self.is_running = False
        
        # MediaPipe Face Mesh initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Lip landmarks indices
        self.LIPS_INDICES = [
            # Outer lip landmarks
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
            # Inner lip landmarks
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
        ]
        
        # Movement detection parameters
        self.prev_lip_distances = {}
        self.movement_threshold = 0.01
        self.speaking_threshold = 0.5
        self.frame_window = 10
        self.movement_history = {}

    def initialize(self):
        """Initialize lip detection system"""
        try:
            self.logger.info("Initializing lip detection system...")
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.is_running = True
            self.processing_thread.start()
            
            self.logger.info("Lip detection system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize lip detection: {str(e)}")
            return False

    def _processing_loop(self):
        """Main video processing loop"""
        while self.is_running:
            try:
                if not self.video_queue.empty():
                    frame = self.video_queue.get()
                    self._process_frame(frame)
            except Exception as e:
                self.logger.error(f"Processing loop error: {str(e)}")

    def _process_frame(self, frame):
        """Process a video frame"""
        try:
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get face mesh results
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            frame_height, frame_width = frame.shape[:2]
            lip_data = []
            
            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                # Extract lip landmarks
                lip_landmarks = self._extract_lip_landmarks(
                    face_landmarks, 
                    frame_width, 
                    frame_height
                )
                
                # Calculate lip movements
                movement_data = self._analyze_lip_movement(face_idx, lip_landmarks)
                
                # Determine if speaking
                is_speaking = self._detect_speaking(face_idx, movement_data['movement'])
                
                lip_data.append({
                    'face_id': face_idx,
                    'timestamp': datetime.now().isoformat(),
                    'lip_landmarks': lip_landmarks,
                    'movement': movement_data['movement'],
                    'movement_direction': movement_data['direction'],
                    'is_speaking': is_speaking,
                    'confidence': movement_data['confidence']
                })
            
            return lip_data
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {str(e)}")
            return None

    def _extract_lip_landmarks(self, face_landmarks, frame_width, frame_height):
        """Extract lip landmarks from face landmarks"""
        lip_points = []
        for idx in self.LIPS_INDICES:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            z = landmark.z
            lip_points.append((x, y, z))
        return lip_points

    def _analyze_lip_movement(self, face_id, lip_landmarks):
        """Analyze lip movement between frames"""
        try:
            if face_id not in self.prev_lip_distances:
                self.prev_lip_distances[face_id] = self._calculate_lip_distances(lip_landmarks)
                return {
                    'movement': 0.0,
                    'direction': 'neutral',
                    'confidence': 0.0
                }
            
            current_distances = self._calculate_lip_distances(lip_landmarks)
            
            # Calculate movement
            movement = np.mean(np.abs(
                np.array(current_distances) - 
                np.array(self.prev_lip_distances[face_id])
            ))
            
            # Determine movement direction
            avg_diff = np.mean(
                np.array(current_distances) - 
                np.array(self.prev_lip_distances[face_id])
            )
            direction = 'opening' if avg_diff > 0 else 'closing' if avg_diff < 0 else 'neutral'
            
            # Calculate confidence based on movement magnitude
            confidence = min(1.0, movement / self.movement_threshold)
            
            # Update previous distances
            self.prev_lip_distances[face_id] = current_distances
            
            # Update movement history
            if face_id not in self.movement_history:
                self.movement_history[face_id] = []
            self.movement_history[face_id].append(movement)
            if len(self.movement_history[face_id]) > self.frame_window:
                self.movement_history[face_id].pop(0)
            
            return {
                'movement': movement,
                'direction': direction,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Movement analysis error: {str(e)}")
            return {
                'movement': 0.0,
                'direction': 'neutral',
                'confidence': 0.0
            }

    def _calculate_lip_distances(self, lip_landmarks):
        """Calculate various distances between lip landmarks"""
        distances = []
        
        # Calculate vertical distances
        for i in range(len(lip_landmarks)//2):
            dist = np.sqrt(sum((np.array(lip_landmarks[i]) - 
                              np.array(lip_landmarks[-i-1]))**2))
            distances.append(dist)
        
        return distances

    def _detect_speaking(self, face_id, movement):
        """Detect if a person is speaking based on lip movement history"""
        if face_id not in self.movement_history:
            return False
        
        # Calculate the average movement over the window
        avg_movement = np.mean(self.movement_history[face_id])
        
        # Determine if speaking based on movement threshold
        return avg_movement > self.speaking_threshold

    def process_frame(self, frame):
        """Process a video frame and return lip movement information"""
        try:
            self.video_queue.put(frame)
            return self._process_frame(frame)
        except Exception as e:
            self.logger.error(f"Frame processing error: {str(e)}")
            return None

    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up lip detection system...")
        try:
            self.is_running = False
            if hasattr(self, 'processing_thread'):
                self.processing_thread.join(timeout=1.0)
            
            # Clear queue
            while not self.video_queue.empty():
                self.video_queue.get()
            
            # Clear movement history
            self.prev_lip_distances.clear()
            self.movement_history.clear()
            
            # Release MediaPipe resources
            self.face_mesh.close()
            
            self.logger.info("Lip detection cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
