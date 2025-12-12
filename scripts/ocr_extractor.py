import json
from pathlib import Path
from typing import List, Dict, Optional
import cv2
import numpy as np
import easyocr
import sys
sys.stdout.reconfigure(encoding='utf-8')

class OCRExtractor:
    def __init__(self, languages=['en'], project_root="."):
        """
        Initialize OCR extractor
        
        Args:
            languages: List of language codes (e.g., ['en', 'ar', 'fr'])
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.languages = languages
        
        # Create OCR results directory
        self.ocr_dir = self.project_root / "extracted" / "ocr"
        self.ocr_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OCR engine
        try:
            self.reader = easyocr.Reader(self.languages, gpu=True, verbose=False)
            print("OCR engine ready!\n")
        except Exception as e:
            print("Failed to initialize easyOCR with GPU:")
            print("\nTrying CPU mode...")
            try:
                self.reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
                print("OCR engine (CPU mode) ready!\n")
            except Exception as e2:
                raise RuntimeError(f"OCR initialization failed")
    
    def preprocess_image(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        thresh = cv2.adaptiveThreshold(
            sharpened, 
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # C constant
        )
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_morph, iterations=1)
        denoised = cv2.fastNlMeansDenoising(opening, h=10)
        
        return denoised
    
    def extract_text(self, image_path, preprocess=True, confidence_threshold=0.5):
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Preprocess if requested
        if preprocess:
            img = self.preprocess_image(image_path)
        else:
            img = cv2.imread(str(image_path))
        
        results = self.reader.readtext(img)
        
        # Format results - convert numpy types to native Python types
        text_blocks = []
        for bbox, text, conf in results:
            if conf >= confidence_threshold:
                # Convert bbox coordinates to native Python floats
                clean_bbox = [[float(x), float(y)] for x, y in bbox]
                
                text_blocks.append({
                    "text": str(text),
                    "confidence": float(conf),
                    "bbox": clean_bbox
                })
        
        full_text = " ".join([block["text"] for block in text_blocks])
        
        return {
            "image_path": str(image_path),
            "full_text": full_text,
            "text_blocks": text_blocks,
            "num_blocks": int(len(text_blocks))  # Convert to native int
        }
    
    def extract_from_frames(self, frames_list, video_name, preprocess=True, confidence_threshold=0.5):
        """
        Extract text from multiple frames
        
        Args:
            frames_list: List of frame paths
            video_name: Name of the video (for saving results)
            preprocess: Whether to preprocess images
            confidence_threshold: Minimum confidence score
            
        Returns:
            Dictionary with OCR results for all frames
        """
        print(f"Running OCR on {len(frames_list)} frames...")
        
        ocr_results = []
        
        for i, frame_path in enumerate(frames_list): 
            try:
                result = self.extract_text(
                    frame_path,
                    preprocess=preprocess,
                    confidence_threshold=confidence_threshold
                )
                
                # Add frame info - ensure all values are JSON serializable
                result["frame_id"] = int(i)
                result["frame_number"] = int(i + 1)
                
                ocr_results.append(result)
                
            except Exception as e:
                print(f"Failed to process {frame_path}: {e}")
                ocr_results.append({
                    "frame_id": int(i),
                    "frame_number": int(i + 1),
                    "image_path": str(frame_path),
                    "full_text": "",
                    "text_blocks": [],
                    "num_blocks": 0,
                    "error": str(e)
                })
        
        print(f"OCR complete! Processed {len(ocr_results)} frames\n")
        
        # Save results - all values are now JSON serializable
        output_data = {
            "video_name": video_name,
            "total_frames": int(len(frames_list)),
            "languages": self.languages,
            "confidence_threshold": float(confidence_threshold),
            "results": ocr_results
        }
        
        # Save to JSON
        output_path = self.ocr_dir / f"{video_name}_ocr.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"OCR results saved: {output_path.relative_to(self.project_root)}")
        
        # Save text-only version
        # text_output_path = self.ocr_dir / f"{video_name}_ocr.txt"
        # with open(text_output_path, "w", encoding="utf-8") as f:
        #     for result in ocr_results:
        #         if result.get("full_text"):
        #             f.write(f"Frame {result['frame_number']}: {result['full_text']}\n")
        
        # print(f"Text-only saved: {text_output_path.relative_to(self.project_root)}\n")
        
        return output_data
    
