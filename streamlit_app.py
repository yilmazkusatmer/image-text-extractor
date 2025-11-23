"""
Streamlit App for Text Extraction from Images
UI layer for the text extraction service.
"""
import html
import json
from pathlib import Path

import streamlit as st
from PIL import Image

from service.text_extraction_service import TextExtractionService


# Page configuration
st.set_page_config(
    page_title="Text Extraction from Images",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "extraction_service" not in st.session_state:
    st.session_state.extraction_service = None
if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None


@st.cache_resource
def get_extraction_service():
    """
    Get or create the text extraction service instance.
    Cached to avoid reloading the model on every interaction.
    """
    if st.session_state.extraction_service is None:
        with st.spinner("Loading OCR model... This may take a moment."):
            service = TextExtractionService()
            st.session_state.extraction_service = service
    return st.session_state.extraction_service


def main():
    """Main application function."""
    st.title("📄 Text Extraction from Images")
    st.markdown("Upload an image to extract text using olmOCR model.")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        max_tokens = st.slider(
            "Max Tokens",
            min_value=512,
            max_value=4096,
            value=2048,
            step=256,
            help="Maximum number of tokens to generate. Higher values allow longer text extraction."
        )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg"],
        help="Upload an image file (PNG, JPG, JPEG)"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        st.subheader("📷 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image)
        st.caption(f"File: {uploaded_file.name}")
        
        st.divider()
        
        # Extract button
        st.subheader("📝 Text Extraction")
        if st.button("🔍 Extract Text", type="primary"):
            try:
                # Get extraction service
                service = get_extraction_service()
                
                # Extract text
                with st.spinner("Extracting text from image... This may take a while."):
                    result = service.extract_text_from_image(
                        image, 
                        max_new_tokens=max_tokens
                    )
                
                # Store result in session state
                st.session_state.extraction_result = result
                st.session_state.extraction_result["source_image"] = uploaded_file.name
                
            except Exception as e:
                st.error(f"❌ Error during extraction: {str(e)}")
                st.exception(e)
        
        # Display results if available
        if st.session_state.extraction_result:
            st.divider()
            result = st.session_state.extraction_result
            
            st.subheader("📄 Extracted Text")
            # Display extracted text with black color
            extracted_text = result.get("extracted_text", "")
            # Escape HTML to prevent injection and ensure proper display
            escaped_text = html.escape(extracted_text)
            st.markdown(
                f'<div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; max-height: 300px; overflow-y: auto; color: #000000; white-space: pre-wrap; font-family: monospace;">{escaped_text}</div>',
                unsafe_allow_html=True
            )
            
            # Display metadata (full JSON)
            with st.expander("📊 Full JSON Metadata"):
                st.json(result)
            
            # Download JSON button
            json_str = json.dumps(result, ensure_ascii=False, indent=2)
            st.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name=f"{Path(uploaded_file.name).stem}.json",
                mime="application/json"
            )
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload an image file to get started.")
        
        # Example section
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. **Upload an image**: Click on the file uploader and select an image file (PNG, JPG, JPEG)
            2. **Adjust settings** (optional): Use the sidebar to adjust max tokens if needed
            3. **Extract text**: Click the "Extract Text" button
            4. **View results**: The extracted text and metadata will be displayed
            5. **Download**: Download the results as JSON if needed
            
            **Note**: The first extraction may take longer as the model needs to be loaded.
            Subsequent extractions will be faster.
            """)


if __name__ == "__main__":
    main()

