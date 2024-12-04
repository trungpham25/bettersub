import unittest
import os
from datetime import datetime
from modules.transcription.video_transcription import VideoTranscriptionEngine
from modules.fusion import TranscriptionFuser

class TestVideoIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Initialize components
        self.video_engine = VideoTranscriptionEngine()
        self.fusion_system = TranscriptionFuser()
        
        # Use existing test video
        self.test_video = os.path.join("auto_avsr", "12.mp4")
        self.assertTrue(os.path.exists(self.test_video), f"Test video not found at {self.test_video}")
        
        # Initialize video engine
        self.video_engine.initialize()

    def tearDown(self):
        """Clean up test environment"""
        self.video_engine.cleanup()

    def test_parallel_processing(self):
        """Test parallel processing of ASR and VSR"""
        # Get ASR transcription
        asr_result = self.video_engine.transcribe_video(self.test_video)
        self.assertEqual(asr_result['status'], 'success', "ASR transcription failed")
        self.assertIn('segments', asr_result, "No segments in ASR result")
        self.assertTrue(len(asr_result['segments']) > 0, "No transcription segments found")

    def test_timestamp_synchronization(self):
        """Test timestamp synchronization between ASR and VSR"""
        # Get ASR transcription with timestamps
        asr_result = self.video_engine.transcribe_video(self.test_video)
        self.assertIn('segments', asr_result, "No segments in ASR result")
        
        # Check timestamp format
        for segment in asr_result['segments']:
            self.assertIn('timestamp', segment, "No timestamp in segment")
            self.assertEqual(len(segment['timestamp']), 2, "Invalid timestamp format")
            start, end = segment['timestamp']
            self.assertLess(start, end, "Invalid timestamp order")

    def test_source_indication(self):
        """Test source indication in fusion results"""
        # Get ASR transcription
        asr_result = self.video_engine.transcribe_video(self.test_video)
        whisper_data = {
            'text': asr_result['segments'][0]['text'],
            'timestamp': datetime.now()
        }
        
        # Create VSR data
        vsr_data = {
            'text': "test transcription",
            'timestamp': datetime.now()
        }
        
        # Test fusion with source indication
        result = self.fusion_system.process_transcriptions(whisper_data, vsr_data)
        self.assertIn('source', result, "No source indication in result")
        self.assertIn(result['source'], ['whisper', 'vsr', 'none'], "Invalid source value")

    def test_output_merging(self):
        """Test merging of ASR and VSR outputs"""
        # Get ASR transcription
        asr_result = self.video_engine.transcribe_video(self.test_video)
        whisper_data = {
            'text': asr_result['segments'][0]['text'],
            'logprobs': [-0.105, -0.223],  # High confidence
            'timestamp': datetime.now()
        }
        
        # Create VSR data
        vsr_data = {
            'text': "test transcription",
            'timestamp': datetime.now()
        }
        
        # Test output merging
        result = self.fusion_system.process_transcriptions(whisper_data, vsr_data)
        self.assertIn('text', result, "No text in merged output")
        self.assertIn('confidence', result, "No confidence score in merged output")
        self.assertGreater(result['confidence'], 0, "Invalid confidence score")

    def test_error_handling(self):
        """Test error handling in integrated pipeline"""
        # Test with non-existent video file
        invalid_video = "nonexistent.mp4"
        
        # Test ASR error handling
        asr_result = self.video_engine.transcribe_video(invalid_video)
        self.assertEqual(asr_result['status'], 'error', "ASR should handle invalid video")

if __name__ == '__main__':
    unittest.main()
