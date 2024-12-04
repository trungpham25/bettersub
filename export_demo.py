from modules.transcription.video_transcription import VideoTranscriptionEngine
from modules.export.subtitle_export import export_subtitles
import os

def main():
    print("\nBettersub 3.0 - Subtitle Export Demo")
    print("=" * 50)
    
    try:
        # Initialize transcription engine
        print("\nInitializing transcription engine...")
        engine = VideoTranscriptionEngine()
        engine.initialize()
        
        # Process video file
        video_path = os.path.join("auto_avsr", "44.mp4")
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return
            
        print(f"\nProcessing video: {video_path}")
        result = engine.transcribe_video(video_path)
        
        if result['status'] == 'success':
            print("\nTranscription successful!")
            print("\nExporting subtitles...")
            
            # Export SRT
            srt_success, srt_msg, srt_path = export_subtitles(
                result['segments'],
                "srt",
                output_dir="exports",
                filename_prefix="transcription_44"
            )
            print(f"\nSRT Export: {srt_msg}")
            
            # Export VTT
            vtt_success, vtt_msg, vtt_path = export_subtitles(
                result['segments'],
                "vtt",
                output_dir="exports",
                filename_prefix="transcription_44"
            )
            print(f"VTT Export: {vtt_msg}")
            
            if srt_success and vtt_success:
                print("\nSubtitle files created successfully:")
                print(f"- SRT: {srt_path}")
                print(f"- VTT: {vtt_path}")
        else:
            print(f"\nError: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        
    finally:
        print("\nCleaning up...")
        engine.cleanup()
        print("\nProcess completed")

if __name__ == "__main__":
    main()
