import json
from typing import List, Dict, Any
from ..core.Models import ASRSegment


def load_asr_ocr_segments_from_json(
    path: str,
    ocr_key: str = "ocr_text",
) -> List[ASRSegment]:
    """
    Load Whisper+OCR style JSON and convert to ASRSegment objects.

    Expected JSON structure:
    {
      "full_text": "...",
      "language": "en",
      "segments": [
        {
          "id": 0,
          "start": 0.0,
          "end": 25.0,
          "text": "ASR text here",
          "ocr_text": "OCR text here"   # optional
        },
        ...
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    segments_data = data.get("segments", [])
    segments: List[ASRSegment] = []

    for seg in segments_data:
        asr_text = seg.get("text", "") or ""
        ocr_text = seg.get(ocr_key, "") or ""

        # Combined text that will actually be embedded
        if ocr_text:
            combined_text = f"{asr_text} [SCREEN_TEXT] {ocr_text}"
        else:
            combined_text = asr_text

        metadata = {
            "id": seg.get("id"),
            "language": data.get("language"),
            "raw_asr_text": asr_text,
        }
        if ocr_text:
            metadata["ocr_text"] = ocr_text

        segments.append(
            ASRSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=combined_text,
                metadata=metadata,
            )
        )

    return segments
