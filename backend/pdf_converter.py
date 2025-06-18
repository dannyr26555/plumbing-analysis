"""
PDF Processing Module
Converts PDF documents into individual PNG images for each page using PyMuPDF
"""

from PIL import Image
import os
import logging
import tempfile
import shutil
from datetime import datetime
import fitz  # PyMuPDF
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFConverter:
    """Handles PDF document processing and page extraction using PyMuPDF"""
    
    def __init__(self):
        """Initialize PDF converter"""
        # Create a unique directory for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_dir = os.path.join(tempfile.gettempdir(), f"plumbing_analysis_{timestamp}")
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"Created temporary directory: {self.temp_dir}")
    
    def pdf_to_images(self, pdf_content: bytes, dpi: int = 300) -> List[Dict[str, str]]:
        """
        Convert PDF pages to PNG images using PyMuPDF
        
        Args:
            pdf_content: PDF file content as bytes
            dpi: Resolution for image conversion (default: 300 for better quality)
            
        Returns:
            List of dictionaries containing page information and image paths
        """
        logger.info(f"Processing PDF content with {dpi} DPI")
        logger.info(f"PDF content size: {len(pdf_content)} bytes")
        
        try:
            return self._convert_pdf(pdf_content, dpi)
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise

    def _convert_pdf(self, pdf_content: bytes, dpi: int) -> List[Dict[str, str]]:
        """Convert PDF to images using PyMuPDF"""
        logger.info("Starting PDF to image conversion with PyMuPDF")
        
        # Save PDF content to temporary file
        pdf_path = os.path.join(self.temp_dir, "temp.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)
        
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        pages_info = []
        
        for idx, page in enumerate(doc):
            logger.info(f"Processing page {idx + 1}/{len(doc)}")
            
            # Calculate zoom factor based on DPI
            zoom = dpi / 72  # PyMuPDF uses 72 DPI as base
            matrix = fitz.Matrix(zoom, zoom)
            
            # Get page pixmap
            pix = page.get_pixmap(matrix=matrix)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Save image to file
            image_filename = f"page_{idx + 1:03d}.png"
            image_path = os.path.join(self.temp_dir, image_filename)
            
            logger.info(f"Saving image to: {image_path}")
            img.save(image_path, "PNG", quality=100)
            
            # Verify saved file
            if os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
                logger.info(f"Successfully saved image, size: {file_size} bytes")
            else:
                logger.error(f"Failed to save image: {image_path}")
            
            # Add page info to results
            pages_info.append({
                "page_number": idx + 1,
                "total_pages": len(doc),
                "image_path": image_path,
                "width": img.width,
                "height": img.height
            })
        
        doc.close()
        logger.info(f"PDF processing complete. Generated {len(pages_info)} images.")
        return pages_info
    
    def cleanup(self):
        """Remove all temporary files and directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Removed temporary directory: {self.temp_dir}") 