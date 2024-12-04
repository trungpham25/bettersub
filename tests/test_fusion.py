import unittest
from datetime import datetime
import numpy as np
from modules.fusion import TranscriptionFuser, ConfidenceCalculator

class TestConfidenceCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = ConfidenceCalculator()

    def test_whisper_confidence_with_logprobs(self):
        """Test Whisper confidence calculation with log probabilities"""
        prediction = {
            'logprobs': [-0.105, -0.223, -0.357]  # High confidence (exp(-0.105) ≈ 0.9, etc.)
        }
        confidence = self.calculator.get_whisper_confidence(prediction)
        self.assertGreater(confidence, 0.7)  # Should be high confidence

    def test_whisper_confidence_with_token_probs(self):
        """Test Whisper confidence calculation with token probabilities"""
        prediction = {
            'token_probs': [0.9, 0.8, 0.7]  # High confidence
        }
        confidence = self.calculator.get_whisper_confidence(prediction)
        self.assertGreater(confidence, 0.7)

    def test_whisper_confidence_empty(self):
        """Test Whisper confidence calculation with no probability data"""
        prediction = {}
        confidence = self.calculator.get_whisper_confidence(prediction)
        self.assertEqual(confidence, 0.0)

    def test_vsr_confidence_single_word(self):
        """Test VSR confidence calculation with single word"""
        confidence = self.calculator.get_vsr_confidence("hello")
        self.assertLess(confidence, 0.5)  # Should be low confidence for single word

    def test_vsr_confidence_multiple_words(self):
        """Test VSR confidence calculation with multiple words"""
        confidence = self.calculator.get_vsr_confidence("hello world today")
        self.assertGreater(confidence, 0.6)  # Should be higher confidence

    def test_vsr_confidence_with_consistency(self):
        """Test VSR confidence calculation with previous transcript"""
        prev = "hello world"
        current = "hello there world"
        confidence = self.calculator.get_vsr_confidence(current, prev)
        self.assertGreater(confidence, 0.5)  # Should show some consistency

class TestTranscriptionFuser(unittest.TestCase):
    def setUp(self):
        self.fuser = TranscriptionFuser()

    def test_whisper_high_confidence(self):
        """Test fusion with high Whisper confidence"""
        whisper_data = {
            'text': 'hello world',
            'logprobs': [-0.105, -0.223, -0.357]  # High confidence
        }
        vsr_data = {
            'text': 'hello there'
        }
        
        result = self.fuser.process_transcriptions(whisper_data, vsr_data)
        self.assertEqual(result['source'], 'whisper')
        self.assertEqual(result['text'], 'hello world')
        self.assertGreater(result['confidence'], 0.4)

    def test_whisper_low_vsr_high(self):
        """Test fusion with low Whisper confidence and high VSR confidence"""
        whisper_data = {
            'text': 'hello world',
            'logprobs': [-1.204, -1.609, -1.204]  # Low confidence
        }
        vsr_data = {
            'text': 'hello there world today'  # Longer text = higher confidence
        }
        
        result = self.fuser.process_transcriptions(whisper_data, vsr_data)
        self.assertEqual(result['source'], 'vsr')
        self.assertEqual(result['text'], 'hello there world today')
        self.assertGreater(result['confidence'], 0.6)

    def test_both_low_confidence(self):
        """Test fusion when both models have low confidence but provide text"""
        whisper_data = {
            'text': 'hi',
            'logprobs': [-1.609, -1.204]  # Low confidence
        }
        vsr_data = {
            'text': 'hello'  # Single word = low confidence
        }
        
        result = self.fuser.process_transcriptions(whisper_data, vsr_data)
        # Should use Whisper text with low confidence rather than [inaudible]
        self.assertEqual(result['source'], 'whisper')
        self.assertEqual(result['text'], 'hi')
        self.assertLess(result['confidence'], 0.4)  # Confidence should be low

    def test_both_empty(self):
        """Test fusion when both models return empty text"""
        whisper_data = {
            'text': '',
            'logprobs': []
        }
        vsr_data = {
            'text': ''
        }
        
        result = self.fuser.process_transcriptions(whisper_data, vsr_data)
        self.assertEqual(result['source'], 'none')
        self.assertEqual(result['text'], '[inaudible]')
        self.assertEqual(result['confidence'], 0.0)

    def test_error_handling(self):
        """Test fusion error handling"""
        whisper_data = None
        vsr_data = None
        
        result = self.fuser.process_transcriptions(whisper_data, vsr_data)
        self.assertEqual(result['source'], 'none')
        self.assertEqual(result['text'], '[error]')
        self.assertEqual(result['confidence'], 0.0)

    def test_latency_tracking(self):
        """Test latency tracking functionality"""
        whisper_data = {'text': 'hello world'}
        vsr_data = {'text': 'hello there'}
        
        self.fuser.process_transcriptions(whisper_data, vsr_data)
        stats = self.fuser.get_latency_stats()
        
        self.assertIsNotNone(stats)
        self.assertIn('mean', stats)
        self.assertIn('max', stats)

if __name__ == '__main__':
    unittest.main()
