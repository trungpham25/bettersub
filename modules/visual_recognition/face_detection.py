import cv2
import numpy as np
import logging
import threading
from queue import Queue
import mediapipe as mp
import face_recognition
from datetime import datetime

class FaceRecognition:
    def __init__(self):
        """Initialize face recognition system"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.video_queue = Queue()
        self.is_running = False
        self.known_face_encodings = []
        self.known_face_names = []
        self.current_face_id = 0
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe Face Mesh for more detailed facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def initialize(self):
        """Initialize face recognition system"""
        try:
            self.logger.info("Initializing face recognition system...")
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.is_running = True
            self.processing_thread.start()
            
            self.logger.info("Face recognition system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize face recognition: {str(e)}")
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
            # Only process every other frame to save processing time
            if self.process_this_frame:
                # Convert the BGR image to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get face mesh landmarks
                mesh_results = self.face_mesh.process(rgb_frame)
                
                # Process face recognition
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                face_data = []
                
                for face_location, face_encoding in zip(face_locations, face_encodings):
                    # Try to recognize the face
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, 
                        face_encoding,
                        tolerance=0.6
                    )
                    
                    face_id = "Unknown"
                    confidence = 0.0
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        face_id = self.known_face_names[first_match_index]
                        # Calculate confidence based on face distance
                        face_distances = face_recognition.face_distance(
                            [self.known_face_encodings[first_match_index]], 
                            face_encoding
                        )
                        confidence = 1 - min(face_distances)
                    else:
                        # Register new face
                        face_id = f"Person_{self.current_face_id}"
                        self.current_face_id += 1
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(face_id)
                        confidence = 1.0
                    
                    # Get facial landmarks if available
                    landmarks = None
                    if mesh_results.multi_face_landmarks:
                        for face_landmarks in mesh_results.multi_face_landmarks:
                            landmarks = [(point.x, point.y, point.z) 
                                       for point in face_landmarks.landmark]
                    
                    face_data.append({
                        'face_id': face_id,
                        'confidence': confidence,
                        'location': face_location,
                        'landmarks': landmarks,
                        'timestamp': datetime.now().isoformat()
                    })
                
                return face_data
            
            self.process_this_frame = not self.process_this_frame
            return None
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {str(e)}")
            return None

    def process_frame(self, frame):
        """Process a video frame and return face information"""
        try:
            self.video_queue.put(frame)
            return self._process_frame(frame)
        except Exception as e:
            self.logger.error(f"Frame processing error: {str(e)}")
            return None

    def get_face_landmarks(self, frame):
        """Get facial landmarks using MediaPipe Face Mesh"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks_data = []
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = [(point.x, point.y, point.z) 
                               for point in face_landmarks.landmark]
                    landmarks_data.append(landmarks)
                return landmarks_data
            return None
            
        except Exception as e:
            self.logger.error(f"Landmark detection error: {str(e)}")
            return None

    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up face recognition system...")
        try:
            self.is_running = False
            if hasattr(self, 'processing_thread'):
                self.processing_thread.join(timeout=1.0)
            
            # Clear queue
            while not self.video_queue.empty():
                self.video_queue.get()
            
            # Clear face data
            self.known_face_encodings = []
            self.known_face_names = []
            self.face_locations = []
            self.face_encodings = []
            self.face_names = []
            
            # Release MediaPipe resources
            self.face_mesh.close()
            
            self.logger.info("Face recognition cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
