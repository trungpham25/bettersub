# BetterSub 

An advanced subtitle generation system that combines Visual Speech Recognition (VSR) with traditional audio-based transcription, featuring real-time capabilities.

## Features

- Real-time transcription with webcam and microphone support
- Audio, Visual, and Audio-Visual processing modes
- Editable transcription output
- Export to SRT and VTT subtitle formats
- Advanced fusion of audio and visual recognition results


## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the VSR model:
   - Download the VSR model file `vsr_trlrwlrs2lrs3vox2avsp_base.pth` (955 MB)
   - Place it in the `auto_avsr` directory
   

## Usage

Run the main application:
```bash
python auto_avsr/demo_gradio_editable.py
```

The application will start at `http://127.0.0.1:7861` with two main modes:

### Real-time Transcription
- Access your webcam and microphone
- View live transcription results
- Clear captions as needed

### File Processing
- Upload video or audio files
- Choose between audio, visual, or audiovisual processing
- Edit transcription output directly
- Export subtitles in SRT or VTT format

## Project Structure

- `auto_avsr/`: Core AVSR implementation
  - `demo_gradio_editable.py`: Main application entry point
  - `configs/`: Model and training configurations
  - `preparation/`: Video preprocessing tools
  - `espnet/`: Neural network implementations
- `modules/`: Core functionality
  - `export/`: Subtitle export utilities
  - `fusion/`: Audio-visual fusion algorithms
  - `transcription/`: Transcription engines
- `utils/`: Helper functions
- `tests/`: Unit and integration tests

## Testing

Run the test suite:
```bash
pytest
```

## Requirements

See `requirements.txt` for full list of dependencies. Key requirements:
- Python 3.8+
- PyTorch 2.0+
- Gradio 4.0+
- Transformers (Whisper model)
- OpenCV
- MediaPipe

## Important Note

The VSR model file (`vsr_trlrwlrs2lrs3vox2avsp_base.pth`) is required but not included in the repository due to its size (955 MB). 
Download link: https://drive.google.com/file/d/19GA5SqDjAkI5S88Jt5neJRG-q5RUi5wi/view




