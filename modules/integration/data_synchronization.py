import logging
import threading
from queue import Queue
from datetime import datetime, timedelta
import numpy as np

class DataSynchronizer:
    def __init__(self):
        """Initialize data synchronization system"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_running = False
        
        # Queues for different data streams
        self.transcription_queue = Queue()
        self.audio_diarization_queue = Queue()
        self.face_recognition_queue = Queue()
        self.lip_movement_queue = Queue()
        
        # Synchronization parameters
        self.sync_window = timedelta(milliseconds=500)  # Time window for syncing data
        self.max_queue_size = 100
        
        # Buffers for synchronized data
        self.synchronized_data = []
        self.buffer_duration = timedelta(seconds=5)

    def initialize(self):
        """Initialize synchronization system"""
        try:
            self.logger.info("Initializing data synchronization system...")
            
            # Start synchronization thread
            self.sync_thread = threading.Thread(target=self._synchronization_loop)
            self.sync_thread.daemon = True
            self.is_running = True
            self.sync_thread.start()
            
            self.logger.info("Data synchronization system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data synchronization: {str(e)}")
            return False

    def _synchronization_loop(self):
        """Main synchronization loop"""
        while self.is_running:
            try:
                # Get current timestamp
                current_time = datetime.now()
                
                # Collect data from all queues within sync window
                transcription_data = self._get_data_in_window(
                    self.transcription_queue, 
                    current_time
                )
                audio_diarization_data = self._get_data_in_window(
                    self.audio_diarization_queue, 
                    current_time
                )
                face_recognition_data = self._get_data_in_window(
                    self.face_recognition_queue, 
                    current_time
                )
                lip_movement_data = self._get_data_in_window(
                    self.lip_movement_queue, 
                    current_time
                )
                
                # Synchronize data
                if any([transcription_data, audio_diarization_data, 
                       face_recognition_data, lip_movement_data]):
                    synchronized_entry = self._synchronize_data(
                        transcription_data,
                        audio_diarization_data,
                        face_recognition_data,
                        lip_movement_data,
                        current_time
                    )
                    
                    if synchronized_entry:
                        self.synchronized_data.append(synchronized_entry)
                        
                        # Remove old data
                        self._cleanup_old_data(current_time)
                
            except Exception as e:
                self.logger.error(f"Synchronization error: {str(e)}")

    def _get_data_in_window(self, queue, current_time):
        """Get all data from queue within sync window"""
        data = []
        while not queue.empty():
            item = queue.get()
            item_time = datetime.fromisoformat(item['timestamp'])
            if current_time - item_time <= self.sync_window:
                data.append(item)
            else:
                # Put back items outside window
                queue.put(item)
                break
        return data

    def _synchronize_data(self, transcription_data, audio_data, face_data, lip_data, timestamp):
        """Synchronize data from different sources"""
        try:
            # Create base synchronized entry
            sync_entry = {
                'timestamp': timestamp.isoformat(),
                'transcription': None,
                'speaker_info': None,
                'face_info': None,
                'lip_movement': None
            }
            
            # Add transcription if available
            if transcription_data:
                latest_transcription = transcription_data[-1]
                sync_entry['transcription'] = {
                    'text': latest_transcription.get('text', ''),
                    'segment': latest_transcription.get('segment', '')
                }
            
            # Add speaker information if available
            if audio_data:
                latest_audio = audio_data[-1]
                sync_entry['speaker_info'] = {
                    'speaker_id': latest_audio.get('speaker_id'),
                    'confidence': latest_audio.get('confidence', 0.0)
                }
            
            # Add face recognition data if available
            if face_data:
                latest_face = face_data[-1]
                sync_entry['face_info'] = {
                    'face_id': latest_face.get('face_id'),
                    'confidence': latest_face.get('confidence', 0.0),
                    'location': latest_face.get('location')
                }
            
            # Add lip movement data if available
            if lip_data:
                latest_lip = lip_data[-1]
                sync_entry['lip_movement'] = {
                    'is_speaking': latest_lip.get('is_speaking', False),
                    'confidence': latest_lip.get('confidence', 0.0),
                    'movement': latest_lip.get('movement', 0.0)
                }
            
            # Verify data consistency
            if sync_entry['speaker_info'] and sync_entry['face_info']:
                sync_entry['matched_identity'] = self._match_speaker_to_face(
                    sync_entry['speaker_info'],
                    sync_entry['face_info'],
                    sync_entry['lip_movement']
                )
            
            return sync_entry
            
        except Exception as e:
            self.logger.error(f"Data synchronization error: {str(e)}")
            return None

    def _match_speaker_to_face(self, speaker_info, face_info, lip_info):
        """Match speaker identity with face recognition data"""
        try:
            # Calculate confidence scores
            audio_confidence = speaker_info.get('confidence', 0.0)
            face_confidence = face_info.get('confidence', 0.0)
            
            # Consider lip movement if available
            lip_speaking = lip_info.get('is_speaking', False) if lip_info else False
            lip_confidence = lip_info.get('confidence', 0.0) if lip_info else 0.0
            
            # Weighted confidence calculation
            combined_confidence = (
                0.4 * audio_confidence +
                0.4 * face_confidence +
                0.2 * float(lip_speaking) * lip_confidence
            )
            
            return {
                'speaker_id': speaker_info['speaker_id'],
                'face_id': face_info['face_id'],
                'confidence': combined_confidence,
                'verified_by_lip_movement': lip_speaking
            }
            
        except Exception as e:
            self.logger.error(f"Identity matching error: {str(e)}")
            return None

    def _cleanup_old_data(self, current_time):
        """Remove data older than buffer duration"""
        cutoff_time = current_time - self.buffer_duration
        self.synchronized_data = [
            entry for entry in self.synchronized_data
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]

    def add_transcription_data(self, data):
        """Add transcription data to queue"""
        if len(data) > 0:
            self.transcription_queue.put(data)

    def add_audio_data(self, data):
        """Add audio diarization data to queue"""
        if len(data) > 0:
            self.audio_diarization_queue.put(data)

    def add_face_data(self, data):
        """Add face recognition data to queue"""
        if len(data) > 0:
            self.face_recognition_queue.put(data)

    def add_lip_data(self, data):
        """Add lip movement data to queue"""
        if len(data) > 0:
            self.lip_movement_queue.put(data)

    def get_synchronized_data(self):
        """Get latest synchronized data"""
        if self.synchronized_data:
            return self.synchronized_data[-1]
        return None

    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up data synchronization system...")
        try:
            self.is_running = False
            if hasattr(self, 'sync_thread'):
                self.sync_thread.join(timeout=1.0)
            
            # Clear queues
            for queue in [self.transcription_queue, self.audio_diarization_queue,
                         self.face_recognition_queue, self.lip_movement_queue]:
                while not queue.empty():
                    queue.get()
            
            # Clear synchronized data
            self.synchronized_data.clear()
            
            self.logger.info("Data synchronization cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
