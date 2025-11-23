"""
Text Extraction Service
Handles OCR text extraction from images using olmOCR model.
Separated from UI concerns for better maintainability.
"""
import base64
import json
import os
import re
from io import BytesIO
from typing import Dict, Tuple, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt


class TextExtractionService:
    """
    Service class for extracting text from images using olmOCR model.
    Handles model initialization, image processing, and result formatting.
    """
    
    def __init__(self, model_name: str = "allenai/olmOCR-2-7B-1025", 
                 processor_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """
        Initialize the text extraction service with model and processor.
        
        Args:
            model_name: Name of the olmOCR model to use
            processor_name: Name of the processor to use
        """
        self.model_name = model_name
        self.processor_name = processor_name
        self.model = None
        self.processor = None
        self.device = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model and processor, set up device."""
        # Initialize model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype=torch.bfloat16
        ).eval()
        
        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(self.processor_name)
        
        # Determine device (CUDA, MPS for Mac, or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Move model to device
        self.model.to(self.device)
    
    def _parse_ocr_output(self, raw_text: str) -> Tuple[Dict, str]:
        """
        Parse OCR output that contains YAML frontmatter and extract metadata and text separately.
        
        Args:
            raw_text: Raw output from OCR model
            
        Returns:
            Tuple of (metadata_dict, extracted_text)
        """
        # Split by YAML delimiters
        parts = raw_text.split("---")
        
        metadata = {}
        extracted_text = ""
        
        if len(parts) >= 3:
            # Extract metadata from between first two --- markers
            yaml_content = parts[1].strip()
            # Extract text after second --- marker
            extracted_text = parts[2].strip()
            
            # Parse YAML-like key-value pairs
            for line in yaml_content.split("\n"):
                line = line.strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert string booleans and numbers
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif re.match(r"^-?\d+\.\d+$", value):
                        value = float(value)
                    
                    metadata[key] = value
        else:
            # No YAML frontmatter found, use entire text
            extracted_text = raw_text.strip()
        
        return metadata, extracted_text
    
    def extract_text_from_image(self, image: Image.Image, 
                                max_new_tokens: int = 2048) -> Dict:
        """
        Extract text from a PIL Image object.
        
        Args:
            image: PIL Image object to extract text from
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Build the full prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_no_anchoring_v4_yaml_prompt()},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]
        
        # Apply the chat template and processor
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for (key, value) in inputs.items()}
        
        # Generate the output
        output = self.model.generate(
            **inputs,
            temperature=0.1,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
        )
        
        # Decode the output
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        text_output = self.processor.tokenizer.batch_decode(
            new_tokens, 
            skip_special_tokens=True
        )
        
        # Extract the text content
        raw_output = text_output[0] if text_output else ""
        
        # Parse the output
        metadata, extracted_text = self._parse_ocr_output(raw_output)
        
        # Prepare result data structure
        result_data = {
            "extracted_text": extracted_text,
            "primary_language": metadata.get("primary_language", None),
            "is_rotation_valid": metadata.get("is_rotation_valid", None),
            "rotation_correction": metadata.get("rotation_correction", None),
            "is_table": metadata.get("is_table", None),
            "is_diagram": metadata.get("is_diagram", None),
            "model": self.model_name,
            "processor": self.processor_name
        }
        
        return result_data
    
    def save_result_to_json(self, result_data: Dict, output_path: str, 
                           source_image_name: Optional[str] = None):
        """
        Save extraction result to JSON file.
        
        Args:
            result_data: Dictionary containing extraction results
            output_path: Path where to save the JSON file
            source_image_name: Optional name of the source image
        """
        # Add source image name if provided
        if source_image_name:
            result_data["source_image"] = source_image_name
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save to JSON file
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(result_data, json_file, ensure_ascii=False, indent=2)

