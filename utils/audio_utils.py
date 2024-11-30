import numpy as np
import sounddevice as sd
import logging
from queue import Queue
import threading
import wave
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class AudioManager:
    def __init__(self, sample_rate=16000, channels=1, dtype=np.int16):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.audio_queue = Queue()
        self.is_recording = False
        self.save_audio = False
        self.output_dir = "recordings"
        
        # Create recordings directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def start_recording(self, save_audio=False):
        """Start audio recording"""
        try:
            self.is_recording = True
            self.save_audio = save_audio
            
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio recording status: {status}")
                self.audio_queue.put(indata.copy())

            # Start the recording stream
            self.stream = sd.InputStream(
                callback=audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=self.dtype
            )
            self.stream.start()
            
            # Start saving thread if needed
            if save_audio:
                self.save_thread = threading.Thread(target=self._save_audio_loop)
                self.save_thread.daemon = True
                self.save_thread.start()
            
            logger.info("Audio recording started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio recording: {str(e)}")
            return False

    def stop_recording(self):
        """Stop audio recording"""
        try:
            self.is_recording = False
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
            
            if self.save_audio and hasattr(self, 'save_thread'):
                self.save_thread.join(timeout=1.0)
            
            logger.info("Audio recording stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop audio recording: {str(e)}")
            return False

    def _save_audio_loop(self):
        """Save audio data to file"""
        try:
            filename = os.path.join(
                self.output_dir,
                f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(np.dtype(self.dtype).itemsize)
                wf.setframerate(self.sample_rate)
                
                while self.is_recording:
                    if not self.audio_queue.empty():
                        audio_data = self.audio_queue.get()
                        wf.writeframes(audio_data.tobytes())
            
            logger.info(f"Audio saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save audio: {str(e)}")

    def get_audio_data(self):
        """Get audio data from queue"""
        if not self.audio_queue.empty():
            return self.audio_queue.get()
        return None

    def clear_queue(self):
        """Clear audio queue"""
        while not self.audio_queue.empty():
            self.audio_queue.get()

class AudioProcessor:
    @staticmethod
    def normalize_audio(audio_data):
        """Normalize audio data"""
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            return np.clip(audio_data, -1.0, 1.0)
        return audio_data

    @staticmethod
    def convert_format(audio_data, source_dtype, target_dtype):
        """Convert audio data between different formats"""
        if source_dtype == target_dtype:
            return audio_data
            
        # Convert to float32 as intermediate format
        if source_dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            if source_dtype == np.int16:
                audio_data /= 32768.0
            elif source_dtype == np.int32:
                audio_data /= 2147483648.0
        
        # Convert to target format
        if target_dtype == np.int16:
            audio_data = (audio_data * 32768.0).astype(np.int16)
        elif target_dtype == np.int32:
            audio_data = (audio_data * 2147483648.0).astype(np.int32)
        elif target_dtype == np.float32:
            audio_data = audio_data.astype(np.float32)
        
        return audio_data

    @staticmethod
    def resample(audio_data, source_rate, target_rate):
        """Resample audio data"""
        if source_rate == target_rate:
            return audio_data
            
        # Calculate resampling ratio
        ratio = target_rate / source_rate
        output_length = int(len(audio_data) * ratio)
        
        # Use linear interpolation for simple resampling
        x_old = np.linspace(0, 1, len(audio_data))
        x_new = np.linspace(0, 1, output_length)
        
        return np.interp(x_new, x_old, audio_data)
