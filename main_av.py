import torch
import gradio as gr
import scipy.io.wavfile
import uuid
import os
import time
import numpy as np
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperTokenizer,
    pipeline
)
import hydra
from omegaconf import OmegaConf
from auto_avsr.demo_vsr import InferencePipeline
from utils.av_utils import save_frame_as_video, cleanup_temp_file

# Initialize Whisper
MODEL_NAME = "openai/whisper-large-v3-turbo"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading Whisper model on {device}...")
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
whisper_model.to(device)

processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="en")

whisper_pipe = pipeline(
    task="automatic-speech-recognition",
    model=whisper_model,
    tokenizer=tokenizer,
    feature_extractor=processor.feature_extractor,
    generate_kwargs={
        "max_new_tokens": 25,
        "task": "transcribe"
    },
    torch_dtype=torch_dtype,
    device=device,
)

# Initialize VSR
print("Loading VSR model...")
cfg = OmegaConf.load('auto_avsr/configs/config.yaml')
cfg.data.modality = "video"  # Set to video-only mode initially
vsr_pipeline = InferencePipeline(cfg)

def process_frame(frame, previous_transcription):
    """Process video frame for lip reading"""
    if frame is None:
        return previous_transcription
    
    try:
        # Convert frame to video file
        temp_video = save_frame_as_video(frame)
        
        try:
            # Get VSR transcription
            vsr_transcription = vsr_pipeline(temp_video)
            
            if vsr_transcription:
                # Update transcription
                lines = previous_transcription.split('\n') if previous_transcription else []
                lines.append(f"[VSR] {vsr_transcription}")
                if len(lines) > 2:
                    lines = lines[-2:]
                return '\n'.join(lines)
            return previous_transcription
            
        finally:
            cleanup_temp_file(temp_video)
                
    except Exception as e:
        print(f"Error during VSR: {e}")
        return previous_transcription

def process_audio(inputs, previous_transcription):
    """Process audio input for speech recognition"""
    try:
        if inputs is None:
            return previous_transcription

        # Save audio to temporary file
        filename = f"{uuid.uuid4().hex}.wav"
        sample_rate, audio_data = inputs
        scipy.io.wavfile.write(filename, sample_rate, audio_data)

        try:
            # Get Whisper transcription
            transcription = whisper_pipe(filename)["text"].strip()
            
            if transcription:
                # Update transcription
                lines = previous_transcription.split('\n') if previous_transcription else []
                lines.append(f"[ASR] {transcription}")
                if len(lines) > 2:
                    lines = lines[-2:]
                return '\n'.join(lines)
            return previous_transcription
            
        finally:
            cleanup_temp_file(filename)
        
    except Exception as e:
        print(f"Error during ASR: {e}")
        return previous_transcription

def clear():
    """Clear the transcription"""
    return ""

print("Creating UI...")
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .video-container {
        position: relative;
        width: 100%;
        max-width: 960px;
        margin: 0 auto;
        background: #000;
        border-radius: 12px;
        overflow: hidden;
        aspect-ratio: 16/9;
    }
    .webcam-feed {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .caption-container {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(0, 0, 0, 0.7);
        padding: 20px;
        z-index: 10;
    }
    .caption-text {
        color: white !important;
        font-family: Arial, sans-serif !important;
        font-size: 24px !important;
        line-height: 1.4 !important;
        text-align: center !important;
        text-shadow: 2px 2px 2px rgba(0, 0, 0, 0.5) !important;
        min-height: 80px !important;
    }
    .caption-text textarea {
        color: white !important;
        background: transparent !important;
        border: none !important;
        text-align: center !important;
        font-size: 24px !important;
    }
    .controls {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 20px;
        padding: 10px;
    }
    .mode-switch {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 20px;
    }
    """
) as demo:
    with gr.Column(elem_classes="container"):
        # Mode selection
        with gr.Row(elem_classes="mode-switch"):
            asr_enabled = gr.Checkbox(label="Enable ASR", value=True)
            vsr_enabled = gr.Checkbox(label="Enable VSR", value=True)
            
        with gr.Column(elem_classes="video-container"):
            # Webcam feed
            webcam = gr.Image(
                label="",
                sources="webcam",
                streaming=True,
                elem_classes="webcam-feed",
                width=960,
                height=540  # 16:9 aspect ratio
            )
            
            with gr.Column(elem_classes="caption-container"):
                # Captions
                output = gr.Textbox(
                    label="",
                    value="",
                    lines=2,
                    max_lines=2,
                    autoscroll=True,
                    show_label=False,
                    elem_classes=["caption-text"]
                )
        
        with gr.Row(elem_classes="controls"):
            # Audio input
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="Microphone Input",
                scale=2
            )
            
            # Clear button
            clear_button = gr.Button(
                "Clear Captions",
                variant="secondary",
                scale=1
            )
        
        # Event handlers
        def conditional_process_audio(inputs, previous_transcription, asr_enabled):
            if not asr_enabled:
                return previous_transcription
            return process_audio(inputs, previous_transcription)
            
        def conditional_process_frame(frame, previous_transcription, vsr_enabled):
            if not vsr_enabled:
                return previous_transcription
            return process_frame(frame, previous_transcription)
        
        audio_input.stream(
            conditional_process_audio,
            inputs=[audio_input, output, asr_enabled],
            outputs=[output],
            show_progress=False,
            batch=False
        )
        
        webcam.stream(
            conditional_process_frame,
            inputs=[webcam, output, vsr_enabled],
            outputs=[output],
            show_progress=False,
            batch=False
        )
        
        clear_button.click(
            clear,
            outputs=[output]
        )

if __name__ == "__main__":
    print("Starting server...")
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
