import torch
import gradio as gr
import scipy.io.wavfile
import uuid
import os
import time
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperTokenizer,
    pipeline
)
from modules.transcription.video_transcription import VideoTranscriptionEngine

MODEL_NAME = "openai/whisper-large-v3-turbo"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading Whisper model on {device}...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="en")

pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=processor.feature_extractor,
    generate_kwargs={
        "max_new_tokens": 25,
        "task": "transcribe"
    },
    torch_dtype=torch_dtype,
    device=device,
)

# Initialize video transcription engine
video_engine = VideoTranscriptionEngine()
video_engine.initialize()

def transcribe(inputs, previous_transcription):
    """Process audio input and return transcription"""
    try:
        if inputs is None:
            return previous_transcription

        # Save audio to temporary file
        filename = f"{uuid.uuid4().hex}.wav"
        sample_rate, audio_data = inputs
        scipy.io.wavfile.write(filename, sample_rate, audio_data)

        try:
            # Get transcription
            transcription = pipe(filename)["text"].strip()
            
            # Update transcription
            if transcription:
                # Keep only the last few lines for closed caption style
                lines = previous_transcription.split('\n') if previous_transcription else []
                lines.append(transcription)
                if len(lines) > 2:  # Keep only last 2 lines
                    lines = lines[-2:]
                return '\n'.join(lines)
            return previous_transcription
            
        finally:
            # Cleanup temporary file
            if os.path.exists(filename):
                os.remove(filename)
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return previous_transcription

def transcribe_file(file_path):
    """Process video/audio file and return transcription"""
    try:
        if file_path is None:
            return "", None
            
        # Check if it's an audio file
        if isinstance(file_path, str) and file_path.lower().endswith(('.wav', '.mp3')):
            result = pipe(file_path)
            return result["text"].strip(), None
            
        # Otherwise treat as video
        result = video_engine.transcribe_video(file_path)
        if result['status'] == 'success':
            # Format transcription with timestamps
            lines = []
            for segment in result['segments']:
                start = segment['timestamp'][0]
                end = segment['timestamp'][1]
                text = segment['text']
                lines.append(f"[{start:.2f}s - {end:.2f}s] {text}")
            return '\n'.join(lines), file_path
        else:
            return f"Error: {result.get('error', 'Unknown error')}", None
    except Exception as e:
        return f"Error processing file: {str(e)}", None

def clear():
    """Clear the transcription"""
    return ""

print("Creating UI...")
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
    .header { text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white; }
    .header h1 { margin: 0; font-size: 2.5em; font-weight: 700; }
    .header p { margin: 10px 0 0; opacity: 0.9; }
    .video-container { position: relative; width: 100%; max-width: 960px; margin: 0 auto; background: #000; border-radius: 12px; overflow: hidden; aspect-ratio: 16/9; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    .webcam-feed { width: 100%; height: 100%; object-fit: cover; }
    .caption-container { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0, 0, 0, 0.7); padding: 20px; z-index: 10; }
    .caption-text { color: white !important; font-family: Arial, sans-serif !important; font-size: 24px !important; line-height: 1.4 !important; text-align: center !important; text-shadow: 2px 2px 2px rgba(0, 0, 0, 0.5) !important; min-height: 80px !important; }
    .caption-text textarea { color: white !important; background: transparent !important; border: none !important; text-align: center !important; font-size: 24px !important; }
    .controls { display: flex; justify-content: center; gap: 10px; margin-top: 20px; padding: 10px; }
    .mode-select { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 12px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); }
    .file-mode { background: #f8f9fa; padding: 30px; border-radius: 12px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); }
    .file-output { font-family: 'Fira Code', monospace; font-size: 14px; line-height: 1.6; white-space: pre-wrap; background: #fff; padding: 20px; border-radius: 8px; border: 1px solid #e9ecef; margin-top: 20px; }
    .video-preview { width: 100%; max-width: 960px; margin: 20px auto; border-radius: 12px; overflow: hidden; background: #000; }
    .video-preview video { width: 100%; height: auto; display: block; }
    """
) as demo:
    with gr.Column(elem_classes="container"):
        # Header
        with gr.Column(elem_classes="header"):
            gr.Markdown("# Bettersub Transcription")
            gr.Markdown("Real-time and file-based transcription with audio-visual processing")
        
        # Mode selection
        with gr.Column(elem_classes="mode-select"):
            mode = gr.Radio(
                choices=["Real-time", "File Upload"],
                value="Real-time",
                label="Transcription Mode",
                info="Choose between real-time transcription or file upload"
            )
        
        # Real-time mode components
        with gr.Column(visible=True) as realtime_mode:
            with gr.Column(elem_classes="video-container"):
                webcam = gr.Image(
                    label="",
                    sources="webcam",
                    streaming=True,
                    elem_classes="webcam-feed",
                    width=960,
                    height=540
                )
                
                with gr.Column(elem_classes="caption-container"):
                    realtime_output = gr.Textbox(
                        label="",
                        value="",
                        lines=2,
                        max_lines=2,
                        autoscroll=True,
                        show_label=False,
                        elem_classes=["caption-text"]
                    )
            
            with gr.Row(elem_classes="controls"):
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                    label="Microphone Input",
                    scale=2
                )
                
                clear_button = gr.Button(
                    "Clear Captions",
                    variant="secondary",
                    scale=1
                )
        
        # File upload mode components
        with gr.Column(visible=False, elem_classes="file-mode") as file_mode:
            gr.Markdown("### File Transcription")
            with gr.Tabs():
                with gr.TabItem("Video Upload"):
                    with gr.Column():
                        video_input = gr.File(
                            label="Upload Video File",
                            file_types=["video"],
                            elem_classes=["video-preview"]
                        )
                        video_player = gr.Video(
                            label="Video Preview",
                            visible=False,
                            elem_classes=["video-preview"],
                            height=540
                        )
                
                with gr.TabItem("Audio Upload"):
                    audio_file = gr.Audio(
                        label="Upload Audio File",
                        sources=["upload"],
                        type="filepath"
                    )
            
            file_output = gr.Textbox(
                label="Transcription Output",
                value="",
                lines=15,
                max_lines=30,
                show_label=True,
                elem_classes=["file-output"]
            )
        
        # Event handlers
        def update_mode(choice):
            if choice == "Real-time":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        
        mode.change(
            update_mode,
            inputs=[mode],
            outputs=[realtime_mode, file_mode]
        )
        
        audio_input.stream(
            transcribe,
            inputs=[audio_input, realtime_output],
            outputs=[realtime_output],
            show_progress=False,
            batch=False
        )
        
        clear_button.click(
            clear,
            outputs=[realtime_output]
        )
        
        def handle_video(file):
            if file is None:
                return "", gr.update(value=None, visible=False)
            transcription, video_path = transcribe_file(file.name)
            return transcription, gr.update(value=file.name, visible=True)
        
        video_input.change(
            handle_video,
            inputs=[video_input],
            outputs=[file_output, video_player]
        )
        
        audio_file.change(
            lambda x: transcribe_file(x)[0],  # Only return transcription, not video path
            inputs=[audio_file],
            outputs=[file_output]
        )

if __name__ == "__main__":
    print("Starting server...")
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
