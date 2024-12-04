import os
from datetime import timedelta

def format_timestamp_srt(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    td = timedelta(seconds=float(seconds))
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def format_timestamp_vtt(seconds):
    """Convert seconds to WebVTT timestamp format (HH:MM:SS.mmm)"""
    return format_timestamp_srt(seconds).replace(',', '.')

def create_srt(segments):
    """
    Create SRT format subtitles from transcription segments
    
    Args:
        segments: List of dictionaries containing:
            - timestamp: [start, end] in seconds
            - text: transcription text
            
    Returns:
        String containing SRT formatted subtitles
    """
    srt_lines = []
    for i, segment in enumerate(segments, 1):
        start = format_timestamp_srt(segment['timestamp'][0])
        end = format_timestamp_srt(segment['timestamp'][1])
        srt_lines.extend([
            str(i),
            f"{start} --> {end}",
            segment['text'].strip(),
            ""  # Empty line between entries
        ])
    if segments:  # Add final newline only if there are segments
        srt_lines.append("")
    return "\n".join(srt_lines)

def create_vtt(segments):
    """
    Create WebVTT format subtitles from transcription segments
    
    Args:
        segments: List of dictionaries containing:
            - timestamp: [start, end] in seconds
            - text: transcription text
            
    Returns:
        String containing WebVTT formatted subtitles
    """
    vtt_lines = ["WEBVTT", ""]  # WebVTT header with blank line
    for segment in segments:
        start = format_timestamp_vtt(segment['timestamp'][0])
        end = format_timestamp_vtt(segment['timestamp'][1])
        vtt_lines.extend([
            f"{start} --> {end}",
            segment['text'].strip(),
            ""  # Empty line between entries
        ])
    if segments:  # Add final newline only if there are segments
        vtt_lines.append("")
    return "\n".join(vtt_lines)

def export_subtitles(segments, format_type, output_dir="exports", filename_prefix="transcription"):
    """
    Export transcription segments as subtitle file
    
    Args:
        segments: List of transcription segments
        format_type: "srt" or "vtt"
        output_dir: Directory to save subtitle files
        filename_prefix: Prefix for the output filename
        
    Returns:
        Tuple of (success: bool, message: str, filepath: str)
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate subtitle content
        if format_type.lower() == "srt":
            content = create_srt(segments)
            ext = "srt"
        elif format_type.lower() == "vtt":
            content = create_vtt(segments)
            ext = "vtt"
        else:
            return False, f"Unsupported format: {format_type}", None
            
        # Create output filepath
        filepath = os.path.join(output_dir, f"{filename_prefix}.{ext}")
        
        # Write subtitle file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return True, f"Successfully exported subtitles to {filepath}", filepath
        
    except Exception as e:
        return False, f"Error exporting subtitles: {str(e)}", None

def validate_subtitle_file(filepath):
    """
    Validate exported subtitle file
    
    Args:
        filepath: Path to subtitle file
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    try:
        if not os.path.exists(filepath):
            return False, "File does not exist"
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Basic validation
        if filepath.endswith('.vtt'):
            if not content.startswith('WEBVTT'):
                return False, "Invalid VTT file: Missing WEBVTT header"
        elif filepath.endswith('.srt'):
            if not content.strip():
                return False, "Invalid SRT file: Empty file"
                
        return True, "Subtitle file is valid"
        
    except Exception as e:
        return False, f"Error validating subtitle file: {str(e)}"
