import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from service.text_extraction_service import TextExtractionService

@pytest.fixture
def mock_service(mocker):
    """Fixture to create a TextExtractionService with mocked model and processor."""
    with patch("service.text_extraction_service.Qwen2_5_VLForConditionalGeneration") as mock_model_cls, \
         patch("service.text_extraction_service.AutoProcessor") as mock_processor_cls, \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False):
        
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        
        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        
        service = TextExtractionService()
        return service, mock_model, mock_processor

def test_parse_ocr_output_with_yaml(mock_service):
    service, _, _ = mock_service
    
    raw_text = """Some prefix text
---
primary_language: English
is_rotation_valid: true
rotation_correction: 0
is_table: false
---
This is the extracted text content.
It has multiple lines.
"""
    metadata, text = service._parse_ocr_output(raw_text)
    
    assert metadata["primary_language"] == "English"
    assert metadata["is_rotation_valid"] is True
    assert metadata["rotation_correction"] == 0
    assert metadata["is_table"] is False
    assert text == "This is the extracted text content.\nIt has multiple lines."

def test_parse_ocr_output_without_yaml(mock_service):
    service, _, _ = mock_service
    
    raw_text = "Just some plain text without any YAML frontmatter."
    metadata, text = service._parse_ocr_output(raw_text)
    
    assert metadata == {}
    assert text == "Just some plain text without any YAML frontmatter."

def test_parse_ocr_output_malformed_yaml(mock_service):
    service, _, _ = mock_service
    
    # Missing the second separator
    raw_text = """---
key: value
This should probably fail to parse as YAML but return text.
"""
    metadata, text = service._parse_ocr_output(raw_text)
    
    # Based on current implementation logic:
    # split('---') will return ['', '\nkey: value\nThis should...', ''] if it ends with ---
    # or just 2 parts if it starts with --- but doesn't end.
    # The implementation checks if len(parts) >= 3.
    
    # If there are only 2 parts (one separator), it falls back to returning everything as text.
    assert metadata == {}
    assert "key: value" in text

def test_extract_text_from_image(mock_service):
    service, mock_model, mock_processor = mock_service
    
    # Mock image
    image = Image.new('RGB', (100, 100), color='red')
    
    # Mock processor output
    mock_processor.apply_chat_template.return_value = "mock_prompt"
    mock_processor.return_value = {"input_ids": MagicMock(), "pixel_values": MagicMock()}
    mock_processor.return_value["input_ids"].shape = [1, 10] # Mock shape
    
    # Mock tokenizer decode
    mock_processor.tokenizer.batch_decode.return_value = ["""---
primary_language: English
---
Extracted Text"""]
    
    # Mock model generate
    mock_model.generate.return_value = MagicMock() # Return value doesn't matter much as we mock batch_decode
    
    result = service.extract_text_from_image(image)
    
    assert result["extracted_text"] == "Extracted Text"
    assert result["primary_language"] == "English"
    assert result["model"] == service.model_name

def test_initialization_device_selection():
    """Test that the correct device is selected based on availability."""
    with patch("service.text_extraction_service.Qwen2_5_VLForConditionalGeneration"), \
         patch("service.text_extraction_service.AutoProcessor"):
             
        # Test CPU
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            service = TextExtractionService()
            assert service.device.type == "cpu"

        # Test CUDA
        with patch("torch.cuda.is_available", return_value=True):
            service = TextExtractionService()
            assert service.device.type == "cuda"
