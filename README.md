# Image Text Extractor

This project is a Streamlit application that uses the `olmOCR` model (based on Qwen2.5-VL) to extract text from images. It provides a user-friendly interface to upload images and view the extracted text along with metadata.

## Features

-   **Image Upload**: Support for PNG, JPG, and JPEG formats.
-   **Text Extraction**: Uses state-of-the-art Vision-Language Models for accurate OCR.
-   **Metadata Extraction**: Extracts additional information like primary language, rotation, and content type (table, diagram).
-   **JSON Export**: Download extraction results as JSON files.
-   **Configurable**: Adjust maximum token generation for longer documents.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd image-text-extractor
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit app**:
    ```bash
    streamlit run streamlit_app.py
    ```

2.  **Open your browser**:
    The app should automatically open in your default browser at `http://localhost:8501`.

## Testing

This project uses `pytest` for unit testing.

1.  **Run tests**:
    ```bash
    pytest tests/
    ```

## Project Structure

-   `streamlit_app.py`: The main entry point for the Streamlit application.
-   `service/`: Contains the backend logic for text extraction.
    -   `text_extraction_service.py`: The core service class handling model interaction.
-   `tests/`: Unit tests for the application.
-   `requirements.txt`: Python dependencies.

## License

[Add License Here]
