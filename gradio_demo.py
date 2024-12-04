import gradio as gr
import os
from modules.transcription.video_transcription import VideoTranscriptionEngine
from modules.fusion import TranscriptionFuser
from auto_avsr.demo_default import InferencePipeline
from datetime import datetime
import hydra
from hydra.core.global_hydra import GlobalHydra

def initialize_vsr():
    """Initialize VSR pipeline"""
    GlobalHydra.instance().clear()
    hydra.initialize(version_base="1.3", config_path="auto_avsr/configs")
    cfg = hydra.compose(config_name="config")
    cfg.data.modality = "video"
    cfg.pretrained_model_path = os.path.join("auto_avsr", "vsr_trlrwlrs2lrs3vox2avsp_base.pth")
    
    if os.path.exists(cfg.pretrained_model_path):
        return InferencePipeline(cfg)
    return None

def process_video(video_path):
    """Process video with ASR, VSR, and fusion"""
    if not video_path:
        return "Please upload a video file."
        
    try:
        # Initialize components
        engine = VideoTranscriptionEngine()
        fusion_system = TranscriptionFuser()
        engine.initialize()
        
        # Initialize VSR
        vsr_pipeline = initialize_vsr()
        
        results = []
        results.append("Processing video...\n")
        
        # Run VSR if available
        if vsr_pipeline:
            results.append("Running VSR transcription...")
            vsr_text = vsr_pipeline(video_path)
            results.append(f"VSR Result: {vsr_text}\n")
        else:
            results.append("VSR model not available")
            vsr_text = ""
        
        # Run ASR transcription
        results.append("Running ASR transcription...")
        asr_result = engine.transcribe_video(video_path)
        
        if asr_result['status'] == 'success':
            results.append("\nTranscription Results:")
            results.append("-" * 20)
            
            for segment in asr_result['segments']:
                start, end = segment['timestamp']
                text = segment['text'].strip()
                
                # Format timestamp and text
                results.append(f"\nTimestamp [{start:.1f}s - {end:.1f}s]")
                results.append(f"ASR: {text}")
                
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
                results.append(f"Fusion: [{fusion_result['source']} {fusion_result['confidence']:.2f}] {fusion_result['text']}")
        else:
            results.append(f"Error: {asr_result.get('error', 'Unknown error')}")
            
        return "\n".join(results)
        
    except Exception as e:
        return f"Error processing video: {str(e)}"
    finally:
        engine.cleanup()

# Create Gradio interface
demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Textbox(label="Transcription Results", lines=10),
    title="Bettersub 3.0 - Video Transcription Demo",
    description="""
    Upload a video to get transcription results from:
    - ASR (Audio Speech Recognition)
    - VSR (Visual Speech Recognition)
    - Fusion System (Combined Results)
    
    Example videos are provided below.
    """,
    examples=[
        [os.path.join("auto_avsr", "44.mp4")],
        [os.path.join("auto_avsr", "12.mp4")],
        [os.path.join("auto_avsr", "22.mp4")],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=False)
