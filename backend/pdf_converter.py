"""
PDF Processing Module
Converts PDF documents into individual PNG images for each page using PyMuPDF
Extracts text layers, sheet metadata, and generates high-quality images

Environment Variables:
- PDF_RENDER_DPI: Override default DPI for PDF rendering (default: 400 - conservative balance)
- PDF_MAX_DPI: Maximum allowed DPI to prevent memory issues (default: 900 - conservative limit)
- PDF_ENABLE_SHARPENING: Enable/disable image sharpening (default: true)
"""

from PIL import Image
import os
import logging
import tempfile
import shutil
from datetime import datetime
import fitz  # PyMuPDF
import json
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class PDFConverter:
    """Handles PDF document processing and page extraction using PyMuPDF"""
    
    def __init__(self, enable_sharpening: bool = None, max_dpi: int = None):
        """
        Initialize PDF converter with conservative quality options
        
        Args:
            enable_sharpening: Whether to apply sharpening filter for symbol clarity 
                              (default: from env var PDF_ENABLE_SHARPENING or True)
            max_dpi: Maximum DPI to prevent excessive memory usage
                    (default: from env var PDF_MAX_DPI or 900 - conservative setting)
        """
        # Create a unique directory for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_dir = os.path.join(tempfile.gettempdir(), f"plumbing_analysis_{timestamp}")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Conservative quality settings with environment variable defaults
        if enable_sharpening is None:
            enable_sharpening = os.getenv('PDF_ENABLE_SHARPENING', 'true').lower() in ('true', '1', 'yes', 'on')
        if max_dpi is None:
            max_dpi = int(os.getenv('PDF_MAX_DPI', 900))  # Conservative default
            
        self.enable_sharpening = enable_sharpening
        self.max_dpi = max_dpi
        
        logger.info("Created temporary directory for PDF processing")
        logger.info(f"Conservative quality settings: sharpening={enable_sharpening}, max_dpi={max_dpi}")
        logger.info("PDF processing configured with environment overrides")
    
    def pdf_to_images(self, pdf_content: bytes, dpi: int = None) -> List[Dict]:
        """
        Convert PDF pages to PNG images and extract text/metadata using PyMuPDF
        
        Args:
            pdf_content: PDF file content as bytes
            dpi: Resolution for image conversion (default: from env var PDF_RENDER_DPI or 400 - conservative balance)
            
        Returns:
            List of dictionaries containing page information, image paths, text, and metadata
        """
        # Use environment variable for default DPI if not specified
        if dpi is None:
            dpi = int(os.getenv('PDF_RENDER_DPI', 400))  # Conservative default
        
        logger.info(f"Processing PDF content with {dpi} DPI (conservative high-resolution mode for construction plans)")
        logger.info(f"PDF content size: {len(pdf_content)} bytes")
        
        try:
            return self._convert_pdf_enhanced(pdf_content, dpi)
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise

    def _convert_pdf_enhanced(self, pdf_content: bytes, dpi: int) -> List[Dict]:
        """Convert PDF to images and extract text/metadata using PyMuPDF"""
        logger.info("Starting enhanced PDF processing with text extraction")
        
        # Save PDF content to temporary file
        pdf_path = os.path.join(self.temp_dir, "temp.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)
        
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        pages_info = []
        
        for idx, page in enumerate(doc):
            logger.info(f"Processing page {idx + 1}/{len(doc)}")
            
            # Extract text and metadata
            text_data = self._extract_text_and_metadata(page)
            
            # Generate high-quality image
            image_info = self._generate_high_quality_image(page, idx + 1, dpi)
            
            # Save context data as JSON
            context_data = {
                "sheet_metadata": text_data["metadata"],
                "legend": text_data["legend"],
                "text_blocks": text_data["text_blocks"],
                "notes": text_data["notes"]
            }
            
            context_filename = f"page_{idx + 1:03d}.context.json"
            context_path = os.path.join(self.temp_dir, context_filename)
            with open(context_path, "w", encoding='utf-8') as f:
                json.dump(context_data, f, indent=2, ensure_ascii=False)
            
            # Combine all information
            page_info = {
                "page_number": idx + 1,
                "total_pages": len(doc),
                "image_path": image_info["image_path"],
                "context_path": context_path,
                "width": image_info["width"],
                "height": image_info["height"],
                "sheet_metadata": text_data["metadata"],
                "legend": text_data["legend"],
                "text_blocks": text_data["text_blocks"],
                "notes": text_data["notes"],
                "raw_text": text_data["raw_text"]
            }
            
            pages_info.append(page_info)
        
        doc.close()
        logger.info(f"Enhanced PDF processing complete. Generated {len(pages_info)} pages with text and metadata.")
        return pages_info
    
    def _extract_text_and_metadata(self, page) -> Dict:
        """Extract text, metadata, and structured content from a PDF page"""
        # Get text with positioning information
        text_dict = page.get_text("dict")
        raw_text = page.get_text()
        
        # Initialize extraction results
        metadata = {}
        legend = []
        text_blocks = []
        notes = []
        
        # Extract sheet metadata
        metadata = self._extract_sheet_metadata(raw_text, text_dict)
        
        # Extract legend/symbol information
        legend = self._extract_legend_data(text_dict)
        
        # Extract structured text blocks
        text_blocks = self._extract_text_blocks(text_dict)
        
        # Extract notes and instructions
        notes = self._extract_notes(raw_text)
        
        return {
            "metadata": metadata,
            "legend": legend,
            "text_blocks": text_blocks,
            "notes": notes,
            "raw_text": raw_text
        }
    
    def _generate_high_quality_image(self, page, page_num: int, dpi: int) -> Dict:
        """Generate maximum quality PNG image from PDF page with enhanced settings"""
        # Clamp DPI to maximum to prevent excessive memory usage
        actual_dpi = min(dpi, self.max_dpi)
        if actual_dpi != dpi:
            logger.warning(f"DPI clamped from {dpi} to {actual_dpi} to prevent excessive memory usage")
        
        # Calculate zoom factor based on DPI (higher zoom for better clarity)
        zoom = actual_dpi / 72  # PyMuPDF uses 72 DPI as base
        matrix = fitz.Matrix(zoom, zoom)
        
        # Get page pixmap with maximum quality settings
        # alpha=False for RGB (no transparency overhead)
        # colorspace=fitz.csRGB for accurate color representation
        pix = page.get_pixmap(
            matrix=matrix, 
            alpha=False,
            colorspace=fitz.csRGB
        )
        
        # Convert to PIL Image with explicit RGB mode
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Apply optional sharpening to enhance symbol clarity
        if self.enable_sharpening:
            try:
                from PIL import ImageFilter
                # UnsharpMask with conservative settings to enhance fine details
                # without creating artifacts in construction drawings
                img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=110, threshold=2))
                logger.debug(f"Applied sharpening filter to page {page_num}")
            except ImportError:
                logger.warning("PIL ImageFilter not available, skipping sharpening")
            except Exception as e:
                logger.warning(f"Failed to apply sharpening filter: {e}")
        
        # Generate output filename
        image_filename = f"page_{page_num:03d}.png"
        image_path = os.path.join(self.temp_dir, image_filename)
        
        # Save with maximum quality PNG settings (completely lossless)
        # compress_level=1 provides good file size with no quality loss
        # optimize=False to avoid any potential quality degradation
        # format="PNG" explicitly specifies lossless PNG format
        try:
            img.save(
                image_path, 
                format="PNG",
                compress_level=1,  # Light compression for smaller files, no quality loss
                optimize=False,    # Disable optimization to prevent any quality changes
                pnginfo=None       # No metadata to keep files clean
            )
        except Exception as e:
            # Fallback to basic PNG save if advanced options fail
            logger.warning(f"Advanced PNG save failed, using basic mode: {e}")
            img.save(image_path, format="PNG")
        
        # Clean up PyMuPDF pixmap to free memory
        pix = None
        
        # Calculate actual file size for logging
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        
        logger.info("Generated maximum quality image")
        logger.info(f"Image dimensions: {img.width}x{img.height} pixels at {actual_dpi} DPI")
        logger.info(f"File size: {file_size_mb:.1f} MB")
        
        return {
            "image_path": image_path,
            "width": img.width,
            "height": img.height,
            "dpi": actual_dpi,
            "file_size_mb": file_size_mb
        }
    
    def _extract_sheet_metadata(self, raw_text: str, text_dict: Dict) -> Dict:
        """Extract sheet metadata like sheet ID, discipline, floor, etc."""
        metadata = {}
        
        # Extract sheet ID patterns (P-102, M-201, etc.)
        sheet_id_pattern = r'([PMEAC]-\d+[A-Z]?)'
        sheet_matches = re.findall(sheet_id_pattern, raw_text)
        if sheet_matches:
            metadata["sheet_id"] = sheet_matches[0]
            
            # Determine discipline from sheet prefix
            prefix = sheet_matches[0][0]
            discipline_map = {
                'P': 'Plumbing',
                'M': 'Mechanical', 
                'E': 'Electrical',
                'A': 'Architectural',
                'C': 'Civil'
            }
            metadata["discipline"] = discipline_map.get(prefix, "Unknown")
        
        # Extract floor/level information
        floor_patterns = [
            r'(LEVEL\s+\d+)',
            r'(FLOOR\s+\d+)',
            r'(\d+(?:ST|ND|RD|TH)\s+FLOOR)',
            r'(BASEMENT)',
            r'(ROOF)',
            r'(PENTHOUSE)'
        ]
        
        for pattern in floor_patterns:
            matches = re.findall(pattern, raw_text, re.IGNORECASE)
            if matches:
                metadata["floor"] = matches[0]
                break
        
        # Extract drawing title/name
        title_lines = raw_text.split('\n')[:5]  # Check first few lines
        for line in title_lines:
            if len(line.strip()) > 10 and not re.match(r'^[A-Z]-\d+', line):
                metadata["title"] = line.strip()
                break
        
        return metadata
    
    def _extract_legend_data(self, text_dict: Dict) -> List[Dict]:
        """Extract legend/symbol information from text dictionary with enhanced column detection"""
        legend = []
        seen_entries = set()  # Avoid duplicates
        
        # Debug: log text structure for troubleshooting
        total_blocks = len(text_dict.get("blocks", []))
        total_lines = sum(len(block.get("lines", [])) for block in text_dict.get("blocks", []))
        logger.debug(f"Text extraction debug: {total_blocks} blocks, {total_lines} lines total")
        
        # Sample some text content for debugging
        sample_texts = []
        for block in text_dict.get("blocks", [])[:3]:  # First 3 blocks
            for line in block.get("lines", [])[:2]:  # First 2 lines per block
                line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                if line_text.strip():
                    sample_texts.append(line_text.strip()[:50])  # First 50 chars
        
        if sample_texts:
            logger.debug(f"Sample extracted text: {sample_texts[:5]}")
        else:
            logger.warning("No text content found in PDF - may be image-based or poorly encoded")
        
        # Look for legend tables and symbol definitions
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    legend_entry = None
                    
                    # Method 1: Look for explicit separators (existing logic)
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        
                        if "=" in text or "-" in text:
                            parts = re.split(r'[=-]', text, 1)
                            if len(parts) == 2:
                                symbol = parts[0].strip()
                                description = parts[1].strip()
                                if symbol and description and len(symbol) < 10:
                                    legend_entry = {
                                        "symbol": symbol,
                                        "description": description,
                                        "bbox": span.get("bbox", [])
                                    }
                                    break
                    
                    # Method 2: Look for space-separated columns (enhanced heuristic)
                    if not legend_entry and len(line.get("spans", [])) >= 2:
                        spans = line["spans"]
                        
                        # Check if this looks like a legend entry
                        first_span_text = spans[0]["text"].strip()
                        
                        # Only consider if first span looks like a symbol (short text, possibly with special chars)
                        if (first_span_text and 
                            len(first_span_text) < 15 and  # Reasonable symbol length
                            not first_span_text.isdigit() and  # Not just a number
                            not first_span_text.lower() in ['the', 'and', 'or', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'as']):  # Not common words
                            
                            # Collect description from remaining spans
                            description_parts = []
                            for span in spans[1:]:
                                span_text = span["text"].strip()
                                if span_text:
                                    description_parts.append(span_text)
                            
                            description = " ".join(description_parts).strip()
                            
                            # Additional validation for description
                            if (description and 
                                len(description) > 2 and  # Description should be meaningful
                                not description.isdigit()):  # Not just numbers
                                
                                legend_entry = {
                                    "symbol": first_span_text,
                                    "description": description,
                                    "bbox": spans[0].get("bbox", [])
                                }
                    
                    # Method 3: Look for tabular layouts with positioning
                    if not legend_entry and len(line.get("spans", [])) >= 2:
                        spans = line["spans"]
                        
                        # Check for horizontal separation between spans (typical in tables)
                        first_span = spans[0]
                        second_span = spans[1]
                        
                        first_bbox = first_span.get("bbox", [])
                        second_bbox = second_span.get("bbox", [])
                        
                        # If we have valid bboxes, check for reasonable horizontal separation
                        if (len(first_bbox) == 4 and len(second_bbox) == 4):
                            # Horizontal gap between spans
                            gap = second_bbox[0] - first_bbox[2]  # left edge of second - right edge of first
                            
                            # If there's a reasonable gap (suggests columnar layout)
                            if gap > 10:  # pixels - adjust as needed
                                symbol = first_span["text"].strip()
                                description_parts = [span["text"].strip() for span in spans[1:] if span["text"].strip()]
                                description = " ".join(description_parts)
                                
                                if (symbol and description and 
                                    len(symbol) < 15 and 
                                    len(description) > 2 and
                                    not symbol.lower() in ['page', 'sheet', 'drawing', 'plan', 'scale', 'date', 'by', 'checked', 'approved']):
                                    
                                    legend_entry = {
                                        "symbol": symbol,
                                        "description": description,
                                        "bbox": first_bbox
                                    }
                    
                    # Method 4: Look for indented entries (common in legend formatting)
                    if not legend_entry and len(line.get("spans", [])) == 1:
                        span = line["spans"][0]
                        text = span["text"].strip()
                        bbox = span.get("bbox", [])
                        
                        # Check for patterns like "Symbol: Description" or "Symbol - Description"
                        colon_pattern = r'^([A-Z0-9&/\-\+\*\#\@\$\%\^\&\(\)]{1,10})\s*[:]\s*(.+)$'
                        dash_pattern = r'^([A-Z0-9&/\-\+\*\#\@\$\%\^\&\(\)]{1,10})\s+[-–—]\s+(.+)$'
                        
                        for pattern in [colon_pattern, dash_pattern]:
                            match = re.match(pattern, text, re.IGNORECASE)
                            if match:
                                symbol = match.group(1).strip()
                                description = match.group(2).strip()
                                
                                if symbol and description and len(description) > 2:
                                    legend_entry = {
                                        "symbol": symbol,
                                        "description": description,
                                        "bbox": bbox
                                    }
                                    break
                    
                    # Method 5: Flexible pattern matching for poor text extraction
                    if not legend_entry:
                        # Combine all span text in the line
                        full_line_text = ""
                        line_bbox = []
                        for span in line.get("spans", []):
                            full_line_text += span.get("text", "") + " "
                            if not line_bbox and span.get("bbox"):
                                line_bbox = span.get("bbox", [])
                        
                        full_line_text = full_line_text.strip()
                        
                        # Look for various legend patterns in the full line
                        legend_patterns = [
                            r'^(\w{1,8})\s+[-–—]\s*(.{3,50})$',  # Symbol - Description
                            r'^(\w{1,8})\s*[:]\s*(.{3,50})$',   # Symbol: Description  
                            r'^(\w{1,8})\s*[=]\s*(.{3,50})$',   # Symbol = Description
                            r'^(\w{1,8})\s+(.{10,50})$',        # Symbol Description (space separated)
                            r'^([A-Z]{1,5})\s*-\s*(.+)$',       # Caps - Description
                            r'^([A-Z0-9/]{1,8})\s+(.+)$'        # Caps/numbers Description
                        ]
                        
                        for pattern in legend_patterns:
                            match = re.match(pattern, full_line_text, re.IGNORECASE)
                            if match:
                                symbol = match.group(1).strip()
                                description = match.group(2).strip()
                                
                                # Additional validation
                                if (symbol and description and 
                                    len(symbol) <= 10 and len(description) >= 3 and
                                    symbol.upper() not in ['PAGE', 'SHEET', 'SCALE', 'DATE', 'BY', 'DRAWN', 'CHECKED', 'REV', 'REVISION'] and
                                    not description.lower().startswith(('page', 'sheet', 'scale', 'date', 'drawn', 'checked'))):
                                    
                                    legend_entry = {
                                        "symbol": symbol,
                                        "description": description,
                                        "bbox": line_bbox
                                    }
                                    logger.debug(f"Method 5 found: {symbol} = {description}")
                                    break
                    
                    # Add to legend if we found a valid entry and haven't seen it before
                    if legend_entry:
                        entry_key = f"{legend_entry['symbol']}:{legend_entry['description']}"
                        if entry_key not in seen_entries:
                            seen_entries.add(entry_key)
                            legend.append(legend_entry)
                            logger.debug(f"Found legend entry: {legend_entry['symbol']} = {legend_entry['description']}")
        
        logger.info(f"Extracted {len(legend)} legend entries using enhanced detection")
        return legend
    
    def _extract_text_blocks(self, text_dict: Dict) -> List[Dict]:
        """Extract structured text blocks with positioning and quality metadata"""
        text_blocks = []
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                block_text = ""
                bbox = block.get("bbox", [])
                
                # Calculate confidence based on text extraction quality
                confidence = self._calculate_text_confidence(block)
                
                for line in block["lines"]:
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    block_text += line_text + "\n"
                
                if block_text.strip():
                    text_blocks.append({
                        "text": block_text.strip(),
                        "bbox": bbox,
                        "confidence": confidence,
                        "source": "pdf_text"  # PyMuPDF direct text extraction
                    })
        
        return text_blocks
    
    def _calculate_text_confidence(self, block: Dict) -> float:
        """Calculate confidence score for text block based on extraction quality"""
        try:
            confidence = 0.8  # Base confidence for PDF text extraction
            
            # Check for text clarity indicators
            total_chars = 0
            suspicious_chars = 0
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    total_chars += len(text)
                    
                    # Count suspicious characters that might indicate poor extraction
                    suspicious_chars += sum(1 for char in text if char in '?□▯▪▫■□')
                    
                    # Check font size (very small text might be less reliable)
                    font_size = span.get("size", 12)
                    if font_size < 8:
                        confidence -= 0.1
                    elif font_size > 20:
                        confidence += 0.1
            
            # Adjust confidence based on suspicious character ratio
            if total_chars > 0:
                suspicious_ratio = suspicious_chars / total_chars
                confidence -= suspicious_ratio * 0.5
            
            # Ensure confidence stays within bounds
            confidence = max(0.1, min(1.0, confidence))
            
            return round(confidence, 3)
            
        except Exception as e:
            logger.warning(f"Error calculating text confidence: {e}")
            return 0.5  # Default medium confidence
    
    def _extract_notes(self, raw_text: str) -> List[str]:
        """Extract notes and general instructions"""
        notes = []
        lines = raw_text.split('\n')
        
        # Look for common note patterns
        note_patterns = [
            r'NOTE:(.+)',
            r'NOTES:(.+)',
            r'GENERAL:(.+)', 
            r'ALL\s+(.+)\s+SHALL',
            r'PROVIDE\s+(.+)',
            r'INSTALL\s+(.+)'
        ]
        
        for line in lines:
            for pattern in note_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                if matches:
                    notes.extend(matches)
        
        return notes
    
    def _convert_pdf(self, pdf_content: bytes, dpi: int) -> List[Dict[str, str]]:
        """Legacy method for backwards compatibility"""
        return self._convert_pdf_enhanced(pdf_content, dpi)
    
    def cleanup(self):
        """Remove all temporary files and directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info("Removed temporary directory") 