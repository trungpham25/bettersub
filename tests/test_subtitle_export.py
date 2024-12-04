import unittest
import os
import shutil
from modules.export.subtitle_export import (
    create_srt,
    create_vtt,
    export_subtitles,
    validate_subtitle_file
)

class TestSubtitleExport(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_segments = [
            {
                'timestamp': [0.0, 2.5],
                'text': "Hello world"
            },
            {
                'timestamp': [2.5, 5.0],
                'text': "This is a test"
            }
        ]
        self.test_dir = "test_exports"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_srt_format(self):
        """Test SRT format creation"""
        srt_content = create_srt(self.test_segments)
        expected_content = (
            "1\n"
            "00:00:00,000 --> 00:00:02,500\n"
            "Hello world\n"
            "\n"
            "2\n"
            "00:00:02,500 --> 00:00:05,000\n"
            "This is a test\n"
            "\n"
        )
        self.assertEqual(srt_content, expected_content)

    def test_vtt_format(self):
        """Test WebVTT format creation"""
        vtt_content = create_vtt(self.test_segments)
        expected_content = (
            "WEBVTT\n"
            "\n"
            "00:00:00.000 --> 00:00:02.500\n"
            "Hello world\n"
            "\n"
            "00:00:02.500 --> 00:00:05.000\n"
            "This is a test\n"
            "\n"
        )
        self.assertEqual(vtt_content, expected_content)

    def test_srt_export(self):
        """Test SRT file export"""
        success, message, filepath = export_subtitles(
            self.test_segments,
            "srt",
            self.test_dir
        )
        self.assertTrue(success)
        self.assertTrue(os.path.exists(filepath))
        
        # Validate exported file
        is_valid, error = validate_subtitle_file(filepath)
        self.assertTrue(is_valid, error)

    def test_vtt_export(self):
        """Test WebVTT file export"""
        success, message, filepath = export_subtitles(
            self.test_segments,
            "vtt",
            self.test_dir
        )
        self.assertTrue(success)
        self.assertTrue(os.path.exists(filepath))
        
        # Validate exported file
        is_valid, error = validate_subtitle_file(filepath)
        self.assertTrue(is_valid, error)

    def test_invalid_format(self):
        """Test export with invalid format"""
        success, message, filepath = export_subtitles(
            self.test_segments,
            "invalid",
            self.test_dir
        )
        self.assertFalse(success)
        self.assertIn("Unsupported format", message)

    def test_empty_segments(self):
        """Test export with empty segments"""
        success, message, filepath = export_subtitles(
            [],
            "srt",
            self.test_dir
        )
        self.assertTrue(success)  # Should create empty file
        self.assertTrue(os.path.exists(filepath))

    def test_validation(self):
        """Test subtitle file validation"""
        # Test invalid VTT file
        invalid_vtt = os.path.join(self.test_dir, "invalid.vtt")
        with open(invalid_vtt, 'w') as f:
            f.write("Invalid VTT content")
        is_valid, error = validate_subtitle_file(invalid_vtt)
        self.assertFalse(is_valid)
        self.assertIn("Missing WEBVTT header", error)

        # Test empty SRT file
        empty_srt = os.path.join(self.test_dir, "empty.srt")
        with open(empty_srt, 'w') as f:
            f.write("")
        is_valid, error = validate_subtitle_file(empty_srt)
        self.assertFalse(is_valid)
        self.assertIn("Empty file", error)

if __name__ == '__main__':
    unittest.main()
