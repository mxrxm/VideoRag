import os
import subprocess
from pathlib import Path

class AudioExtractor:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.audio_dir = self.project_root / "extracted" / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
    
    def extract(self, video_path, sample_rate=16000, channels=1):

        video_path = Path(video_path)     
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        video_name = video_path.stem
        audio_output = self.audio_dir / f"{video_name}.wav"
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",                      # Disable video
            "-acodec", "pcm_s16le",     # WAV 16-bit PCM
            "-ar", str(sample_rate),    # Sample rate
            "-ac", str(channels),       # Audio channels
            "-y",                       # Overwrite if exists
            str(audio_output)
        ]
        
        print(f" Extracting audio from {video_path.name}...")
        
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            print(f" Audio saved: {audio_output.relative_to(self.project_root)}")
            return audio_output
            
        except subprocess.CalledProcessError as e:
            print(f" Error extracting audio: {e.stderr.decode()}")
            raise
       
     