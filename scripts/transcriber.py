import whisper
import os
from pathlib import Path
import json

class AudioTranscriber:
    def __init__(self, model_name="base"):
        print(f"Loading Whisper {model_name} model...")
        self.model = whisper.load_model(model_name)
        print("Model loaded successfully!")
    
    def transcribe(self, audio_path, language=None, task="transcribe"):
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es'). None for auto-detect
            task: 'transcribe' or 'translate' (translate to English)
            
        Returns:
            Dictionary with transcription results
        """
        print(f"Transcribing: {audio_path}")
        
        options = {
            "task": task,
            "verbose": None
        }
        
        if language:
            options["language"] = language
        

        result = self.model.transcribe(audio_path, **options)
        
        return result
    
    def transcribe_with_timestamps(self, audio_path, language=None):
        """
        Transcribe with word-level timestamps
        
        Args:
            audio_path: Path to audio file
            language: Language code or None for auto-detect
            
        Returns:
            Dictionary with detailed segments and timestamps
        """
        result = self.transcribe(audio_path, language=language)
        
        # Extract segments with timestamps
        segments = []
        for segment in result["segments"]:
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "id": segment["id"]
            })
        
        return {
            "text": result["text"],
            "language": result["language"],
            "segments": segments
        }
    
    def save_transcription(self, result, output_path):
        """
        Save transcription to file
        
        Args:
            result: Transcription result dictionary
            output_path: Output file path
            format: json
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Transcription saved to: {output_path}")
    


