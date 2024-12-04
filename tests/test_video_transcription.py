import unittest
import os
import tempfile
import shutil
from modules.transcription.video_transcription import VideoTranscriptionEngine

class TestVideoTranscription(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.engine = VideoTranscriptionEngine()
        self.test_dir = tempfile.mkdtemp()
        
        # Create a small test video file using FFmpeg
        self.test_video = os.path.join(self.test_dir, "test_video.mp4")
        self._create_test_video()

    def tearDown(self):
        """Clean up test environment"""
        self.engine.cleanup()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_test_video(self):
        """Create a test video file with audio"""
        try:
            # Generate a 5-second test video with a test tone
            command = [
                'ffmpeg',
                '-f', 'lavfi',  # Use lavfi input format
                '-i', 'sine=frequency=1000:duration=5',  # Generate a 1kHz tone for 5 seconds
                '-f', 'lavfi',
                '-i', 'color=c=blue:s=1280x720:d=5',  # Blue screen for 5 seconds
                '-shortest',
                '-y',
                self.test_video
            ]
            os.system(' '.join(command))
            
            self.assertTrue(os.path.exists(self.test_video), "Test video creation failed")
        except Exception as e:
            self.fail(f"Failed to create test video: {str(e)}")

    def test_initialization(self):
        """Test model initialization"""
        self.assertTrue(self.engine.initialize(), "Model initialization failed")

    def test_audio_extraction(self):
        """Test audio extraction from video"""
        self.engine.initialize()
        audio_path = self.engine.extract_audio(self.test_video)
        
        self.assertIsNotNone(audio_path, "Audio extraction failed")
        self.assertTrue(os.path.exists(audio_path), "Extracted audio file doesn't exist")
        
        # Clean up extracted audio
        if os.path.exists(audio_path):
            os.remove(audio_path)

    def test_video_transcription(self):
        """Test video transcription with timestamps"""
        self.engine.initialize()
        result = self.engine.transcribe_video(self.test_video)
        
        self.assertEqual(result['status'], 'success', "Transcription failed")
        self.assertIn('segments', result, "No segments in transcription result")
        self.assertIsInstance(result['segments'], list, "Segments should be a list")
        
        # Check segment format
        if result['segments']:
            segment = result['segments'][0]
            self.assertIn('timestamp', segment, "No timestamp in segment")
            self.assertIn('text', segment, "No text in segment")
            self.assertEqual(len(segment['timestamp']), 2, "Timestamp should have start and end")

if __name__ == '__main__':
    unittest.main()
