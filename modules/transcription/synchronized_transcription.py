import os
import logging
import subprocess
import numpy as np
import torch
from datetime import datetime
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperTokenizer,
    pipeline
)
from auto_avsr.lightning_vsr import VSRModelModule
from preparation.detectors.mediapipe.detector_v2 import EnhancedLandmarksDetector
from preparation.detectors.mediapipe.video_process_v2 import EnhancedVideoProcess
import torchvision

logger = logging.getLogger(__name__)

class SynchronizedTranscriptionEngine:
    def __init__(self):
        """Initialize synchronized transcription engine"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model_name = "openai/whisper-large-v3-turbo"
        self.sample_rate = 16000
        self.temp_dir = "temp_audio"
        self.chunk_size = 30  # 30 seconds
        self.stride_size = 5   # 5 seconds overlap
        
        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def initialize(self, vsr_config=None):
        """Initialize both Whisper and VSR models"""
        try:
            logger.info("Initializing transcription models...")
            
            # Initialize Whisper
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(self.device)

            processor = AutoProcessor.from_pretrained(self.model_name)
            tokenizer = WhisperTokenizer.from_pretrained(self.model_name, language="en")

            self.whisper_pipe = pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=self.chunk_size,
                stride_length_s=self.stride_size,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            
            # Initialize VSR components
            if vsr_config:
                self.vsr_model = VSRModelModule(vsr_config)
                self.vsr_model.model.load_state_dict(
                    torch.load(vsr_config.pretrained_model_path, 
                             map_location=lambda storage, loc: storage)
                )
                self.vsr_model.eval()
                
                self.landmarks_detector = EnhancedLandmarksDetector(min_detection_confidence=0.3)
                self.video_process = EnhancedVideoProcess(convert_gray=False)
                
                logger.info("VSR model initialized successfully")
            
            logger.info("Transcription engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize transcription engine: {str(e)}")
            return False

    def extract_audio(self, video_path):
        """Extract audio from video file"""
        try:
            audio_filename = os.path.join(
                self.temp_dir,
                f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
            
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate),
                '-ac', '1',
                '-y',
                audio_filename
            ]
            
            subprocess.run(command, check=True)
            logger.info(f"Audio extracted to {audio_filename}")
            
            return audio_filename
            
        except Exception as e:
            logger.error(f"Failed to extract audio: {str(e)}")
            return None

    def process_video_chunk(self, video_frames, start_time, end_time):
        """Process a chunk of video frames with VSR"""
        try:
            # Detect and process landmarks
            landmarks = self.landmarks_detector(video_frames)
            video = self.video_process(video_frames, landmarks)
            
            if video is None:
                return None
                
            # Convert to tensor and process
            video = torch.tensor(video)
            video = video.permute((0, 3, 1, 2))
            
            # Generate transcript
            with torch.no_grad():
                transcript = self.vsr_model(video)
                
            if transcript and transcript.strip():
                return {
                    'text': transcript,
                    'timestamp': [start_time, end_time],
                    'source': 'vsr'
                }
            return None
            
        except Exception as e:
            logger.error(f"Error processing video chunk: {str(e)}")
            return None

    def transcribe_video(self, video_path):
        """Transcribe video with synchronized Whisper and VSR"""
        try:
            # Extract audio for Whisper
            audio_path = self.extract_audio(video_path)
            if not audio_path:
                raise Exception("Failed to extract audio from video")

            try:
                # Get Whisper transcription
                logger.info("Starting Whisper transcription...")
                whisper_result = self.whisper_pipe(
                    audio_path,
                    return_timestamps=True
                )

                # Load video for VSR
                video = torchvision.io.read_video(video_path, pts_unit="sec")[0].numpy()
                fps = 30  # Assuming 30fps
                
                # Process each chunk
                segments = []
                for chunk in whisper_result["chunks"]:
                    start_time = chunk['timestamp'][0]
                    end_time = chunk['timestamp'][1]
                    whisper_text = chunk['text'].strip()
                    
                    # Calculate frame indices
                    start_frame = int(start_time * fps)
                    end_frame = int(end_time * fps)
                    
                    # Extract video chunk
                    video_chunk = video[start_frame:end_frame]
                    
                    if whisper_text:
                        # If Whisper produced output, use it
                        segments.append({
                            'timestamp': [start_time, end_time],
                            'text': whisper_text,
                            'source': 'whisper'
                        })
                    else:
                        # If Whisper failed, try VSR
                        vsr_result = self.process_video_chunk(
                            video_chunk, start_time, end_time
                        )
                        if vsr_result:
                            segments.append(vsr_result)
                            logger.info(f"VSR output for {start_time:.1f}s - {end_time:.1f}s: {vsr_result['text']}")

                return {
                    'status': 'success',
                    'segments': segments,
                    'video_path': video_path
                }

            finally:
                # Clean up temporary audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'video_path': video_path
            }

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up transcription engine...")
        try:
            self.whisper_pipe = None
            self.vsr_model = None
            
            if os.path.exists(self.temp_dir) and not os.listdir(self.temp_dir):
                os.rmdir(self.temp_dir)
            
            logger.info("Transcription engine cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
