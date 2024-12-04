from modules.transcription.video_transcription import VideoTranscriptionEngine
from modules.fusion import TranscriptionFuser
from auto_avsr.demo_default import InferencePipeline
from datetime import datetime
import os
import hydra
from hydra.core.global_hydra import GlobalHydra
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        print("\nStarting transcription process...")
        
        # Initialize components
        print("Initializing components...")
        engine = VideoTranscriptionEngine()
        fusion_system = TranscriptionFuser()
        
        print("Initializing Whisper model...")
        if not engine.initialize():
            print("Failed to initialize Whisper model")
            return
        
        # Path to test video
        video_path = os.path.join("auto_avsr", "44.mp4")
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return
            
        print(f"\nProcessing video: {video_path}")
        print("=" * 50)
        
        try:
            # Initialize VSR pipeline
            print("\nInitializing VSR pipeline...")
            GlobalHydra.instance().clear()
            hydra.initialize(version_base="1.3", config_path="auto_avsr/configs")
            cfg = hydra.compose(config_name="config")
            cfg.data.modality = "video"
            cfg.pretrained_model_path = os.path.join("auto_avsr", "vsr_trlrwlrs2lrs3vox2avsp_base.pth")
            
            if os.path.exists(cfg.pretrained_model_path):
                print("Running VSR transcription...")
                vsr_pipeline = InferencePipeline(cfg)
                vsr_text = vsr_pipeline(video_path)
                print(f"\nVSR Result: {vsr_text}\n")
            else:
                print(f"VSR model not found at {cfg.pretrained_model_path}")
                vsr_text = ""
                
        except Exception as e:
            print(f"Error in VSR processing: {str(e)}")
            traceback.print_exc()
            vsr_text = ""
        
        # Get ASR transcription
        print("\nRunning ASR transcription...")
        asr_result = engine.transcribe_video(video_path)
        
        if asr_result['status'] == 'success':
            print("\nTranscription Results:")
            print("-" * 20)
            
            for segment in asr_result['segments']:
                start, end = segment['timestamp']
                text = segment['text'].strip()
                print(f"\nTimestamp [{start:.1f}s - {end:.1f}s]")
                print(f"ASR: {text}")
                
                # Create fusion input
                whisper_data = {
                    'text': text,
                    'logprobs': [-0.105, -0.223],  # Example confidence
                    'timestamp': datetime.now()
                }
                
                # VSR data
                vsr_data = {
                    'text': vsr_text,
                    'timestamp': datetime.now()
                }
                
                # Get fusion result
                fusion_result = fusion_system.process_transcriptions(whisper_data, vsr_data)
                
                print(f"Fusion: [{fusion_result['source']} {fusion_result['confidence']:.2f}] {fusion_result['text']}")
        else:
            print(f"ASR Error: {asr_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error in transcription process: {str(e)}")
        traceback.print_exc()
        
    finally:
        print("\nCleaning up...")
        engine.cleanup()
        print("Process completed")

if __name__ == "__main__":
    main()
