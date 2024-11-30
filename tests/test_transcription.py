import sys
import os
import unittest
import logging
from datetime import datetime
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.transcription.realtime_transcription import TranscriptionEngine
from utils.audio_utils import AudioManager, AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestTranscription(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.transcription_engine = TranscriptionEngine()
        self.audio_manager = AudioManager()
        
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'transcription_engine'):
            self.transcription_engine.cleanup()
        if hasattr(self, 'audio_manager'):
            self.audio_manager.stop_recording()

    def test_initialization(self):
        """Test transcription engine initialization"""
        logger.info("Testing transcription engine initialization...")
        result = self.transcription_engine.initialize()
        self.assertTrue(result, "Transcription engine initialization failed")

    def test_audio_capture(self):
        """Test audio capture functionality"""
        logger.info("Testing audio capture...")
        
        # Start audio recording
        result = self.audio_manager.start_recording()
        self.assertTrue(result, "Audio recording failed to start")
        
        # Wait for some audio data
        import time
        time.sleep(2)
        
        # Get audio data
        audio_data = self.audio_manager.get_audio_data()
        self.assertIsNotNone(audio_data, "No audio data captured")
        
        # Stop recording
        self.audio_manager.stop_recording()

    def test_audio_processing(self):
        """Test audio processing functionality"""
        logger.info("Testing audio processing...")
        
        # Create sample audio data
        sample_rate = 16000
        duration = 2  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Process audio
        processed_audio = AudioProcessor.normalize_audio(audio_data)
        self.assertIsNotNone(processed_audio, "Audio processing failed")
        self.assertTrue(np.all(processed_audio >= -1.0) and np.all(processed_audio <= 1.0),
                       "Audio normalization failed")

    def test_transcription(self):
        """Test transcription functionality"""
        logger.info("Testing transcription...")
        
        # Initialize transcription engine
        self.assertTrue(self.transcription_engine.initialize(),
                       "Transcription engine initialization failed")
        
        # Create sample audio data
        sample_rate = 16000
        duration = 2  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Process audio through transcription engine
        result = self.transcription_engine.process_audio((sample_rate, audio_data))
        
        # Note: We don't check the actual transcription text since it's a sine wave
        # Just verify we get a result in the expected format
        self.assertIsNotNone(result, "Transcription failed")
        self.assertIn('timestamp', result, "Transcription result missing timestamp")
        self.assertIn('text', result, "Transcription result missing text")
        self.assertIn('segment', result, "Transcription result missing segment")

def run_tests():
    """Run the test suite"""
    logger.info("Starting transcription tests...")
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == '__main__':
    run_tests()
