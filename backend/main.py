from fastapi import FastAPI, UploadFile, HTTPException, Depends, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import os
import fitz  # PyMuPDF
from PIL import Image
import logging
import tempfile
import shutil
from datetime import datetime
from pdf_converter import PDFConverter
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import Image as AutoGenImage, CancellationToken
from autogen_agentchat.messages import MultiModalMessage
import json
from pathlib import Path
import asyncio
import uuid
import base64
from typing import List, Dict, Any, Optional, Tuple
from pydantic import ValidationError
import re
import difflib

# Import refactored components
from config.app_config import AppConfig
from config.prompt_manager import PromptManager
from utils.agent_factory import AgentFactory
from utils.response_parser import AgentResponseParser
from utils.image_processor import ImageProcessor

# Import legacy config for backward compatibility (will be removed)
from config.agent_config import SYSTEM_MESSAGES, AGENT_CONFIG, JSON_RESPONSE_FORMAT

from models import (
    AgentInput, MaterialItem, ContextOutput, PlumbingOutput, 
    AnalysisResult, ProcessingStatus, 
    LegendEntry, TextBlock, SheetMetadata, BoundingBox, ProcessingError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store for progress updates
progress_store = {}
# Store for analysis results
result_store = {}

# Initialize centralized configuration
try:
    AppConfig.initialize()
    AppConfig.validate_required()
    logger.info(f"Configuration loaded: {AppConfig.get_config_summary()}")
except Exception as e:
    logger.error(f"Failed to initialize application configuration: {str(e)}")
    raise RuntimeError("Application configuration initialization failed")

# Initialize FastAPI app
app = FastAPI()

# CORS middleware - restrict to necessary methods and headers only
app.add_middleware(
    CORSMiddleware,
    allow_origins=[AppConfig.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"]
)

# Initialize agents with factory pattern
try:
    # Initialize prompt manager and load system messages
    prompt_manager = PromptManager()
    system_messages = prompt_manager.get_system_messages()
    
    # Initialize agent factory
    agent_factory = AgentFactory()
    
    # Create all agents at once
    agents = agent_factory.create_standard_agents(system_messages)
    
    # Extract agents for backward compatibility
    context_agent = agents["context"]
    plumbing_agent = agents["plumbing"]
    
    logger.info(f"Successfully initialized all agents: {list(agents.keys())}")
    logger.info(f"Agent factory summary: {agent_factory.get_agent_summary()}")
    
except Exception as e:
    logger.error(f"Failed to initialize agents: {str(e)}")
    raise RuntimeError(f"Agent initialization error: {str(e)}")

# Initialize image processor
try:
    image_processor = ImageProcessor()
    logger.info(f"Image processor initialized: max_size={image_processor.max_image_size}, sharpening={image_processor.enable_sharpening}")
except Exception as e:
    logger.error(f"Failed to initialize image processor: {str(e)}")
    raise RuntimeError(f"Image processor initialization error: {str(e)}")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal, injection, and remove sensitive information."""
    import re
    
    # Get basename to prevent path traversal
    safe_name = os.path.basename(filename)
    
    # Remove any potentially dangerous characters
    # Keep only alphanumeric, dots, dashes, underscores
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', safe_name)
    
    # Prevent hidden files and ensure it ends with .pdf
    if safe_name.startswith('.'):
        safe_name = 'file_' + safe_name[1:]
    
    # Ensure it ends with .pdf and isn't too long
    if not safe_name.lower().endswith('.pdf'):
        safe_name += '.pdf'
    
    # Limit length to prevent issues
    if len(safe_name) > 100:
        safe_name = safe_name[:96] + '.pdf'
    
    return safe_name

def sanitize_prompt_text(text: str) -> str:
    """Sanitize text before including in prompts to prevent injection."""
    if not text:
        return "Unknown"
    
    # Remove potential prompt injection patterns
    sanitized = text.replace('\n', ' ').replace('\r', ' ')
    sanitized = sanitized.replace('"""', '').replace("'''", '')
    sanitized = sanitized.replace('SYSTEM:', '').replace('USER:', '').replace('ASSISTANT:', '')
    sanitized = sanitized.strip()
    
    # Limit length and ensure safe content
    if len(sanitized) > 50:
        sanitized = sanitized[:47] + "..."
    
    return sanitized if sanitized else "Unknown"

def update_progress(task_id: str, stage: str, current: int, total: int, message: str):
    """Update progress for a specific task"""
    progress_store[task_id] = {
        "stage": stage,
        "current": current,
        "total": total,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }

def parse_json_response(response_text: str, expected_model=None, agent_name: str = "unknown", sheet_id: str = None, processing_errors: List[ProcessingError] = None) -> Dict[str, Any]:
    """Parse JSON response from agent and validate against model if provided"""
    try:
        # Use the new AgentResponseParser for basic JSON cleaning and parsing
        cleaned_text = AgentResponseParser.clean_json_response(response_text)
        parsed_data = AgentResponseParser.parse_json_safely(cleaned_text)
        
        # Convert bbox lists to BoundingBox objects for validation
        def convert_bbox_in_data(data):
            if isinstance(data, dict):
                # Convert bbox lists/tuples to BoundingBox dict format
                if 'bbox' in data and isinstance(data['bbox'], (list, tuple)) and len(data['bbox']) == 4:
                    data['bbox'] = {
                        'x0': float(data['bbox'][0]),
                        'y0': float(data['bbox'][1]), 
                        'x1': float(data['bbox'][2]),
                        'y1': float(data['bbox'][3])
                    }
                
                # Recursively convert nested structures
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        convert_bbox_in_data(value)
            elif isinstance(data, list):
                for item in data:
                    convert_bbox_in_data(item)
        
        # Clean material data issues
        def clean_material_data(data):
            if isinstance(data, dict):
                # Clean materials list if present
                if 'materials' in data and isinstance(data['materials'], list):
                    for material in data['materials']:
                        if isinstance(material, dict):
                            # Fix clearly invalid quantities (only convert valid strings to numbers)
                            if material.get('quantity') is not None:
                                if isinstance(material.get('quantity'), str):
                                    quantity_str = material['quantity'].strip()
                                    # Only convert if it's clearly a number
                                    if quantity_str and (quantity_str.replace('.', '').replace(',', '').isdigit() or 
                                                        quantity_str.replace('.', '').isdigit()):
                                        try:
                                            # Remove commas and convert
                                            clean_qty = quantity_str.replace(',', '')
                                            if '.' in clean_qty:
                                                material['quantity'] = float(clean_qty)
                                            else:
                                                material['quantity'] = int(clean_qty)
                                        except (ValueError, TypeError):
                                            # If conversion fails, leave as None (don't guess)
                                            material['quantity'] = None
                                    else:
                                        # Non-numeric string - don't guess, set to None
                                        material['quantity'] = None
                            
                            # Only provide default unit if there's clear quantity information
                            # Don't guess units when there's no quantity context
                            if not material.get('unit') and material.get('quantity') is not None:
                                # Only default to EA if we have a valid quantity and no unit
                                material['unit'] = 'EA'
                            elif not material.get('unit'):
                                # No quantity or unclear context - don't guess unit
                                material['unit'] = None
                            
                            # Clean up confidence scores
                            if material.get('confidence') is not None:
                                try:
                                    confidence = float(material['confidence'])
                                    # Clamp to 0-1 range
                                    material['confidence'] = max(0.0, min(1.0, confidence))
                                except (ValueError, TypeError):
                                    material['confidence'] = None
                
                # Recursively clean nested structures
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        clean_material_data(value)
            elif isinstance(data, list):
                for item in data:
                    clean_material_data(item)
        
        # Apply bbox conversion to parsed data
        convert_bbox_in_data(parsed_data)
        
        # Apply material data cleaning
        clean_material_data(parsed_data)
        
        # Normalize field names before validation if model is provided
        if expected_model:
            parsed_data = AgentResponseParser.normalize_field_names(parsed_data, expected_model)
        
        # Validate against model if provided
        if expected_model:
            validated_data = expected_model(**parsed_data)
            return validated_data.model_dump()
        
        return parsed_data
        
    except json.JSONDecodeError as e:
        error_msg = f"JSON parsing error: {str(e)}"
        logger.error(error_msg)
        logger.error("Raw response content could not be parsed as valid JSON")
        
        # Add to structured error tracking if provided
        if processing_errors is not None:
            add_error_to_status(processing_errors, agent_name, sheet_id, "parsing", error_msg)
            
        raise ValueError(f"Invalid JSON response: {str(e)}")
    except ValidationError as e:
        error_msg = f"Model validation error: {str(e)}"
        logger.error(error_msg)
        logger.error("Parsed data failed model validation")
        
        # Add to structured error tracking if provided
        if processing_errors is not None:
            add_error_to_status(processing_errors, agent_name, sheet_id, "validation", error_msg)
            
        raise ValueError(f"Response validation failed: {str(e)}")

def create_agent_input(page_info: Dict, context_data: Optional[Dict] = None) -> AgentInput:
    """Create structured agent input from page information"""
    
    # Convert extracted data to models with proper bbox handling
    legend_entries = []
    for legend_item in page_info.get("legend", []):
        # Convert bbox list/tuple to BoundingBox object if present
        if legend_item.get("bbox") and isinstance(legend_item["bbox"], (list, tuple)) and len(legend_item["bbox"]) == 4:
            legend_entry_data = {**legend_item, "bbox": BoundingBox(
                x0=float(legend_item["bbox"][0]),
                y0=float(legend_item["bbox"][1]), 
                x1=float(legend_item["bbox"][2]),
                y1=float(legend_item["bbox"][3])
            )}
        else:
            legend_entry_data = legend_item
        legend_entries.append(LegendEntry(**legend_entry_data))
    
    text_blocks = []
    for text_item in page_info.get("text_blocks", []):
        # Convert bbox list/tuple to BoundingBox object if present
        if text_item.get("bbox") and isinstance(text_item["bbox"], (list, tuple)) and len(text_item["bbox"]) == 4:
            text_block_data = {**text_item, "bbox": BoundingBox(
                x0=float(text_item["bbox"][0]),
                y0=float(text_item["bbox"][1]),
                x1=float(text_item["bbox"][2]), 
                y1=float(text_item["bbox"][3])
            )}
        else:
            text_block_data = text_item
        text_blocks.append(TextBlock(**text_block_data))
    
    sheet_metadata = SheetMetadata(**page_info.get("sheet_metadata", {}))
    
    # Create agent input
    agent_input = AgentInput(
        sheet_id=sheet_metadata.sheet_id,
        image_path=page_info.get("image_path"),
        legend=legend_entries,
        text_blocks=text_blocks,
        sheet_metadata=sheet_metadata,
        notes=page_info.get("notes", []),
        context_from_previous=context_data
    )
    
    return agent_input

def save_intermediate_result(task_id: str, sheet_id: str, agent_name: str, result_data: Dict):
    """Save intermediate analysis results for persistence"""
    results_dir = os.path.join(tempfile.gettempdir(), f"analysis_results_{task_id}")
    os.makedirs(results_dir, exist_ok=True)
    
    filename = f"{sheet_id}_{agent_name}_result.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, "w", encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved intermediate result: {filepath}")
    return filepath

def create_processing_error(agent: str, sheet_id: Optional[str], error_type: str, error_message: str) -> ProcessingError:
    """Create a structured processing error"""
    return ProcessingError(
        agent=agent,
        sheet_id=sheet_id,
        error_type=error_type,
        error_message=error_message
    )

def add_error_to_status(processing_errors: List[ProcessingError], agent: str, sheet_id: Optional[str], error_type: str, error_message: str):
    """Add a structured error and update tracking"""
    error = create_processing_error(agent, sheet_id, error_type, error_message)
    processing_errors.append(error)
    return error

def create_processing_status_with_errors(stage: str, progress: float, message: str, current_sheet: Optional[str], errors: List[ProcessingError]) -> ProcessingStatus:
    """Create processing status with structured error summaries"""
    
    # Calculate summaries
    total_errors = len(errors)
    agent_summary = {}
    sheet_summary = {}
    
    for error in errors:
        # Count by agent
        agent_summary[error.agent] = agent_summary.get(error.agent, 0) + 1
        
        # Count by sheet (if sheet_id exists)
        if error.sheet_id:
            sheet_summary[error.sheet_id] = sheet_summary.get(error.sheet_id, 0) + 1
    
    return ProcessingStatus(
        stage=stage,
        progress=progress,
        message=message,
        current_sheet=current_sheet,
        errors=errors,
        total_errors=total_errors,
        agent_error_summary=agent_summary,
        sheet_error_summary=sheet_summary
    )

def normalize_material_name(name: str) -> str:
    """Normalize material names for comparison by removing formatting variations"""
    if not name:
        return ""
    
    # Convert to lowercase and strip whitespace
    normalized = name.lower().strip()
    
    # Standardize inch notation first (before other replacements)
    normalized = re.sub(r'(\d+(?:\.\d+)?)\s*[""′″\'\-]?\s*inch(?:es)?', r'\1 inch', normalized)
    normalized = re.sub(r'(\d+(?:\.\d+)?)\s*[""′″\']', r'\1 inch', normalized)
    
    # Replace various separators with consistent spacing
    normalized = re.sub(r'\s*[\-–—]\s*', ' ', normalized)  # Replace dashes with spaces
    normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
    
    # Remove common filler words that don't affect material identity
    filler_words = ['the', 'a', 'an', 'type', 'grade', 'standard', 'each']
    words = normalized.split()
    words = [w for w in words if w not in filler_words]
    
    return ' '.join(words)

def extract_size_from_name(name: str) -> Optional[str]:
    """Extract size information from material name"""
    if not name:
        return None
    
    # Look for common size patterns (more comprehensive)
    size_patterns = [
        r'(\d+(?:\.\d+)?)\s*[\"″′]',  # inch sizes: 6", 2.5"
        r'(\d+(?:\.\d+)?)\s*[\-]?\s*inch(?:es)?',   # inch sizes: 6 inch, 6-inch
        r'(\d+(?:\.\d+)?)\s*mm',     # metric sizes: 150mm
        r'(\d+(?:\.\d+)?)\s*cm',     # metric sizes: 15cm
        r'(\d+(?:\.\d+)?)\s*ft',     # foot sizes: 10ft
        r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)',  # dimensions: 2x4
    ]
    
    for pattern in size_patterns:
        match = re.search(pattern, name.lower())
        if match:
            # Normalize the size format
            size_text = match.group(0).strip()
            # Convert to standard format: "8 inch"
            if '"' in size_text or '″' in size_text or '′' in size_text:
                num = re.search(r'(\d+(?:\.\d+)?)', size_text).group(1)
                return f"{num} inch"
            elif 'inch' in size_text:
                num = re.search(r'(\d+(?:\.\d+)?)', size_text).group(1)
                return f"{num} inch"
            else:
                return size_text
    
    return None

def materials_are_similar(material1: Dict, material2: Dict, similarity_threshold: float = 0.75) -> bool:
    """Determine if two materials represent the same item with intelligent comparison"""
    
    # Get normalized names
    name1 = normalize_material_name(material1.get("item_name", ""))
    name2 = normalize_material_name(material2.get("item_name", ""))
    
    if not name1 or not name2:
        return False
    
    # Check for exact match after normalization
    if name1 == name2:
        return True
    
    # Check string similarity with lower threshold for more lenient matching
    similarity = difflib.SequenceMatcher(None, name1, name2).ratio()
    
    # For very short names, require higher similarity
    min_length = min(len(name1), len(name2))
    if min_length < 10:
        similarity_threshold = 0.85
    
    if similarity < similarity_threshold:
        return False
    
    # Additional checks for material characteristics
    size1 = material1.get("size") or extract_size_from_name(material1.get("item_name", ""))
    size2 = material2.get("size") or extract_size_from_name(material2.get("item_name", ""))
    
    # If both have sizes, they must match exactly (sizes are critical for material identity)
    if size1 and size2:
        # Normalize sizes to compare properly
        if size1 != size2:
            # Try to extract just the numeric part for comparison
            num1 = re.search(r'(\d+(?:\.\d+)?)', size1)
            num2 = re.search(r'(\d+(?:\.\d+)?)', size2)
            if num1 and num2:
                if float(num1.group(1)) != float(num2.group(1)):
                    return False
            else:
                return False
    elif size1 and not size2:
        # One has size, other doesn't - check if the base material type is similar
        # Remove size from the name with size and compare base types
        name_without_size = re.sub(r'\d+(?:\.\d+)?\s*[\-]?\s*inch(?:es)?|\d+(?:\.\d+)?\s*[\"″′]', '', name1).strip()
        name_without_size = re.sub(r'\s+', ' ', name_without_size)
        base_similarity = difflib.SequenceMatcher(None, name_without_size, name2).ratio()
        if base_similarity < 0.8:
            return False
    elif size2 and not size1:
        # Other has size, first doesn't - check if the base material type is similar
        name_without_size = re.sub(r'\d+(?:\.\d+)?\s*[\-]?\s*inch(?:es)?|\d+(?:\.\d+)?\s*[\"″′]', '', name2).strip()
        name_without_size = re.sub(r'\s+', ' ', name_without_size)
        base_similarity = difflib.SequenceMatcher(None, name1, name_without_size).ratio()
        if base_similarity < 0.8:
            return False
    
    # Check specifications if available
    spec1 = material1.get("specification", "")
    spec2 = material2.get("specification", "")
    if spec1 and spec2 and spec1 != spec2:
        # Allow some variation in specifications
        spec_similarity = difflib.SequenceMatcher(None, spec1.lower(), spec2.lower()).ratio()
        if spec_similarity < 0.6:
            return False
    
    # Check units compatibility if both have quantities
    unit1 = material1.get("unit")
    unit2 = material2.get("unit")
    if (unit1 and unit2 and unit1 != unit2 and 
        material1.get("quantity") is not None and material2.get("quantity") is not None):
        # Don't combine materials with incompatible units
        return False
    
    return True

def combine_materials(material1: Dict, material2: Dict) -> Dict:
    """Combine two similar materials into one, merging quantities and metadata"""
    
    # Start with the material that has more complete information
    base_material = material1 if len(str(material1)) > len(str(material2)) else material2
    other_material = material2 if base_material == material1 else material1
    
    combined = base_material.copy()
    
    # Combine quantities if both have them
    qty1 = material1.get("quantity")
    qty2 = material2.get("quantity")
    
    if qty1 is not None and qty2 is not None:
        try:
            combined["quantity"] = float(qty1) + float(qty2)
        except (ValueError, TypeError):
            # If conversion fails, keep the first valid quantity
            combined["quantity"] = qty1 if qty1 is not None else qty2
    elif qty1 is not None:
        combined["quantity"] = qty1
    elif qty2 is not None:
        combined["quantity"] = qty2
    
    # Use the unit from whichever material has a quantity
    if qty1 is not None and material1.get("unit"):
        combined["unit"] = material1.get("unit")
    elif qty2 is not None and material2.get("unit"):
        combined["unit"] = material2.get("unit")
    
    # Combine reference sheets
    ref1 = material1.get("reference_sheet", "")
    ref2 = material2.get("reference_sheet", "")
    refs = []
    if ref1: refs.append(ref1)
    if ref2: refs.append(ref2)
    
    if refs:
        # Remove duplicates and sort
        unique_refs = sorted(list(set(refs)))
        combined["reference_sheet"] = ", ".join(unique_refs)
    
    # Combine zones
    zone1 = material1.get("zone", "")
    zone2 = material2.get("zone", "")
    zones = []
    if zone1: zones.append(zone1)
    if zone2: zones.append(zone2)
    
    if zones:
        unique_zones = sorted(list(set(zones)))
        combined["zone"] = ", ".join(unique_zones)
    
    # Use the more specific/complete information
    for field in ["size", "specification"]:
        val1 = material1.get(field, "")
        val2 = material2.get(field, "")
        
        if val1 and not val2:
            combined[field] = val1
        elif val2 and not val1:
            combined[field] = val2
        elif val1 and val2:
            # Use the longer, more descriptive one
            combined[field] = val1 if len(val1) > len(val2) else val2
    
    # Average confidence scores
    conf1 = material1.get("confidence")
    conf2 = material2.get("confidence")
    if conf1 is not None and conf2 is not None:
        combined["confidence"] = (float(conf1) + float(conf2)) / 2.0
    elif conf1 is not None:
        combined["confidence"] = conf1
    elif conf2 is not None:
        combined["confidence"] = conf2
    
    # Combine notes
    notes1 = material1.get("notes", "")
    notes2 = material2.get("notes", "")
    notes = []
    if notes1: notes.append(notes1)
    if notes2: notes.append(notes2)
    
    if notes:
        # Remove duplicates
        unique_notes = []
        for note in notes:
            if note not in unique_notes:
                unique_notes.append(note)
        combined["notes"] = "; ".join(unique_notes)
    
    return combined

def deduplicate_materials(materials: List[Dict]) -> List[Dict]:
    """Deduplicate a list of materials by combining similar items"""
    
    if not materials:
        return []
    
    deduplicated = []
    
    for material in materials:
        # Look for similar materials in our deduplicated list
        found_similar = False
        
        for i, existing_material in enumerate(deduplicated):
            if materials_are_similar(material, existing_material):
                # Combine the materials
                logger.debug(f"Combining materials: '{existing_material.get('item_name')}' + '{material.get('item_name')}'")
                deduplicated[i] = combine_materials(existing_material, material)
                found_similar = True
                break
        
        if not found_similar:
            # Add as new material
            deduplicated.append(material.copy())
    
    # Sort by item name for consistent output
    deduplicated.sort(key=lambda x: x.get("item_name", "").lower())
    
    return deduplicated

def validate_material_quantities(materials: List[Dict]) -> List[Dict]:
    """
    Validate material quantities and units for accuracy and consistency
    
    Args:
        materials: List of material dictionaries from agent analysis
        
    Returns:
        List of validated materials with corrected or flagged issues
    """
    validated_materials = []
    
    for material in materials:
        validated_material = material.copy()
        
        # Extract material information
        item_name = material.get("item_name", "").lower()
        quantity = material.get("quantity")
        unit = material.get("unit")
        confidence = material.get("confidence", 0.5)
        notes = material.get("notes", "")
        
        validation_flags = []
        
        # Check for extremely high quantities without explicit documentation
        if quantity is not None and quantity >= 1000:
            # Look for explicit labeling evidence in notes
            explicit_keywords = ["schedule", "dimension label", "quantity callout", "measured from", "detail specifies"]
            has_explicit_evidence = any(keyword in notes.lower() for keyword in explicit_keywords)
            
            if not has_explicit_evidence:
                validation_flags.append("Large quantity (≥1000) without explicit documentation")
                validated_material["confidence"] = min(confidence, 0.4)  # Cap confidence for suspicious quantities
                logger.warning(f"Flagged large quantity without evidence: {item_name} - {quantity} {unit}")
        
        # Validate unit compatibility with item type
        unit_validation_passed = True
        if unit and quantity is not None:
            # Define expected units for different material types
            pipe_keywords = ["pipe", "main", "line", "conduit", "tubing"]
            fitting_keywords = ["valve", "elbow", "tee", "coupling", "union", "adapter", "fitting", "hydrant", "meter"]
            area_keywords = ["insulation", "coating", "lining", "membrane"]
            volume_keywords = ["tank", "reservoir", "basin"]
            
            expected_linear_units = ["LF"]
            expected_each_units = ["EA"]
            expected_area_units = ["SF"]
            expected_volume_units = ["CF", "GAL"]
            
            # Check pipe materials should use linear units
            if any(keyword in item_name for keyword in pipe_keywords):
                if unit not in expected_linear_units:
                    validation_flags.append(f"Pipe material using non-linear unit: {unit}")
                    unit_validation_passed = False
            
            # Check fittings/equipment should use each units
            elif any(keyword in item_name for keyword in fitting_keywords):
                if unit not in expected_each_units:
                    validation_flags.append(f"Fitting/equipment using non-each unit: {unit}")
                    unit_validation_passed = False
            
            # Check area materials should use area units
            elif any(keyword in item_name for keyword in area_keywords):
                if unit not in expected_area_units:
                    validation_flags.append(f"Area material using non-area unit: {unit}")
                    unit_validation_passed = False
            
            # Check volume materials should use volume units
            elif any(keyword in item_name for keyword in volume_keywords):
                if unit not in expected_volume_units:
                    validation_flags.append(f"Volume material using non-volume unit: {unit}")
                    unit_validation_passed = False
        
        # Cross-check confidence vs visibility vs type
        confidence_issues = []
        
        # High confidence should have supporting evidence
        if confidence > 0.8:
            high_confidence_keywords = ["measured from", "dimension label", "quantity callout", "schedule", "detail"]
            has_high_confidence_evidence = any(keyword in notes.lower() for keyword in high_confidence_keywords)
            if not has_high_confidence_evidence:
                confidence_issues.append("High confidence without explicit measurement evidence")
                validated_material["confidence"] = min(confidence, 0.6)
        
        # Very low confidence with quantity should be questioned
        if confidence < 0.3 and quantity is not None:
            confidence_issues.append("Very low confidence but quantity provided")
            
        # Inferred materials should have lower confidence
        if any(keyword in notes.lower() for keyword in ["inferred", "assumed", "unclear", "blurry"]):
            if confidence > 0.5:
                confidence_issues.append("Inferred/uncertain material with high confidence")
                validated_material["confidence"] = min(confidence, 0.4)
        
        # Nullify ambiguous or unsupported values
        nullification_reasons = []
        
        # Remove quantities that seem guessed rather than measured
        if quantity is not None:
            guessing_indicators = ["assumed", "estimated", "guessed", "approximately", "roughly"]
            explicit_indicators = ["measured", "labeled", "callout", "schedule", "dimension"]
            
            has_guessing = any(indicator in notes.lower() for indicator in guessing_indicators)
            has_explicit = any(indicator in notes.lower() for indicator in explicit_indicators)
            
            if has_guessing and not has_explicit and confidence < 0.4:
                nullification_reasons.append("Quantity appears guessed rather than measured")
                validated_material["quantity"] = None
                validated_material["unit"] = None
        
        # Remove units when quantity is null
        if validated_material.get("quantity") is None and unit is not None:
            nullification_reasons.append("Unit removed due to null quantity")
            validated_material["unit"] = None
        
        # Update notes with validation information
        validation_notes = []
        if validation_flags:
            validation_notes.extend([f"Validation flag: {flag}" for flag in validation_flags])
        if confidence_issues:
            validation_notes.extend([f"Confidence issue: {issue}" for issue in confidence_issues])
        if nullification_reasons:
            validation_notes.extend([f"Nullified: {reason}" for reason in nullification_reasons])
        
        if validation_notes:
            existing_notes = validated_material.get("notes", "")
            validation_summary = "; ".join(validation_notes)
            validated_material["notes"] = f"{existing_notes}. {validation_summary}" if existing_notes else validation_summary
        
        # Add validation metadata
        validated_material["validation_metadata"] = {
            "unit_validation_passed": unit_validation_passed,
            "validation_flags_count": len(validation_flags),
            "confidence_issues_count": len(confidence_issues),
            "nullification_count": len(nullification_reasons),
            "original_confidence": confidence
        }
        
        validated_materials.append(validated_material)
    
    # Log validation statistics
    if materials:
        flagged_count = sum(1 for m in validated_materials if m.get("validation_metadata", {}).get("validation_flags_count", 0) > 0)
        nullified_count = sum(1 for m in validated_materials if m.get("validation_metadata", {}).get("nullification_count", 0) > 0)
        confidence_adjusted = sum(1 for i, m in enumerate(validated_materials) if m.get("confidence", 0.5) != materials[i].get("confidence", 0.5))
        
        logger.info(f"Post-agent validation complete: {flagged_count}/{len(materials)} flagged, {nullified_count} nullified, {confidence_adjusted} confidence adjusted")
    
    return validated_materials

def post_agent_result_validation(materials: List[Dict], context_results: List[Dict] = None) -> List[Dict]:
    """
    Enhanced post-agent result validation implementing comprehensive checks
    
    Args:
        materials: List of material dictionaries from agent analysis
        context_results: Context results for cross-validation
        
    Returns:
        List of validated and corrected materials
    """
    logger.info("Starting enhanced post-agent result validation...")
    
    # Apply the existing validation first
    validated_materials = validate_material_quantities(materials)
    
    # Additional validation checks specific to the improvements
    enhanced_materials = []
    
    for material in validated_materials:
        enhanced_material = material.copy()
        
        # Get material properties
        item_name = material.get("item_name", "").lower()
        quantity = material.get("quantity")
        unit = material.get("unit")
        confidence = material.get("confidence", 0.5)
        notes = material.get("notes", "")
        
        additional_flags = []
        
        # Recheck confidence vs visibility vs type (enhanced check)
        visibility_confidence = 0.5  # Default
        if "clear" in notes.lower() or "visible" in notes.lower():
            visibility_confidence += 0.2
        if "unclear" in notes.lower() or "blurry" in notes.lower():
            visibility_confidence -= 0.3
        if "counted" in notes.lower() or "measured" in notes.lower():
            visibility_confidence += 0.3
        
        visibility_confidence = max(0.1, min(1.0, visibility_confidence))
        
        # If agent confidence is significantly higher than visibility suggests
        if confidence > visibility_confidence + 0.3:
            additional_flags.append("Agent confidence exceeds visibility assessment")
            enhanced_material["confidence"] = (confidence + visibility_confidence) / 2
        
        # Enhanced unit matching for specialized plumbing items
        specialized_unit_checks = {
            "water": ["LF", "EA", "GAL"],  # Water systems can be linear, discrete, or volume
            "sewer": ["LF", "EA"],        # Sewer systems typically linear or discrete
            "gas": ["LF", "EA"],          # Gas lines typically linear or discrete
            "storm": ["LF", "EA", "CF"],  # Storm systems can include volume for retention
            "fire": ["LF", "EA", "GAL"],  # Fire systems include pipes, equipment, and water volume
        }
        
        for system_type, valid_units in specialized_unit_checks.items():
            if system_type in item_name and unit and unit not in valid_units:
                additional_flags.append(f"Unusual unit for {system_type} system: {unit}")
        
        # Flag quantities that seem inconsistent with item type
        if quantity is not None:
            # Very small quantities for large infrastructure
            if "main" in item_name and quantity < 10 and unit == "LF":
                additional_flags.append("Unusually short main line")
            
            # Very large quantities for small items
            if any(keyword in item_name for keyword in ["fitting", "valve", "coupling"]) and quantity > 100:
                additional_flags.append("Unusually high quantity for individual component")
        
        # Update notes with additional validation
        if additional_flags:
            existing_notes = enhanced_material.get("notes", "")
            additional_summary = "; ".join([f"Enhanced validation: {flag}" for flag in additional_flags])
            enhanced_material["notes"] = f"{existing_notes}. {additional_summary}" if existing_notes else additional_summary
        
        # Update validation metadata
        if "validation_metadata" not in enhanced_material:
            enhanced_material["validation_metadata"] = {}
        enhanced_material["validation_metadata"]["additional_flags"] = additional_flags
        enhanced_material["validation_metadata"]["visibility_confidence"] = visibility_confidence
        
        enhanced_materials.append(enhanced_material)
    
    logger.info(f"Enhanced post-agent validation complete: {len(materials)} materials processed")
    return enhanced_materials

def compute_confidence(clarity: float, annotation_clarity: float, measurement_type: str) -> float:
    """
    Compute confidence score based on visual clarity, annotation quality, and measurement type
    
    Args:
        clarity: Visual clarity of the material/symbol (0.0-1.0)
        annotation_clarity: Clarity of text annotations and labels (0.0-1.0) 
        measurement_type: Type of measurement ("explicit", "visual_estimate", "inferred")
    
    Returns:
        Confidence score (0.0-1.0)
    """
    base = (clarity + annotation_clarity) / 2
    
    if measurement_type == "explicit":
        # Explicit measurements (dimension labels, quantity callouts) get bonus
        return min(1.0, base + 0.2)
    elif measurement_type == "visual_estimate":
        # Visual estimates use base score
        return base
    elif measurement_type == "inferred":
        # Inferred quantities get penalty
        return max(0.0, base - 0.3)
    else:
        # Default case
        return base

def analyze_material_metadata(material: Dict, context_data: Dict = None) -> Dict:
    """
    Analyze material metadata to determine confidence factors
    
    Args:
        material: Material dictionary from agent analysis
        context_data: Context information including legends and symbols
        
    Returns:
        Dictionary with confidence metadata
    """
    metadata = {
        "clarity": 0.5,  # Default medium clarity
        "annotation_clarity": 0.5,  # Default medium annotation clarity
        "measurement_type": "visual_estimate",  # Default type
        "symbol_defined": False,
        "quantity_labeled": False,
        "has_dimensions": False,
        "legend_match": False
    }
    
    # Analyze material notes for confidence indicators
    notes = material.get("notes", "").lower()
    item_name = material.get("item_name", "").lower()
    
    # Check for explicit measurements
    explicit_indicators = ["measured from", "dimension label", "quantity callout", "schedule", "detail", "specified"]
    if any(indicator in notes for indicator in explicit_indicators):
        metadata["measurement_type"] = "explicit"
        metadata["annotation_clarity"] = 0.8
        metadata["quantity_labeled"] = True
    
    # Check for visual estimates  
    visual_indicators = ["counted", "visible", "traced route", "estimated length", "plan view"]
    if any(indicator in notes for indicator in visual_indicators):
        metadata["measurement_type"] = "visual_estimate"
        metadata["clarity"] = 0.7
    
    # Check for inferred/uncertain quantities
    inferred_indicators = ["mentioned", "inferred", "assumed", "unclear", "blurry", "partial"]
    if any(indicator in notes for indicator in inferred_indicators):
        metadata["measurement_type"] = "inferred"
        metadata["clarity"] = 0.3
        metadata["annotation_clarity"] = 0.3
    
    # Check for dimension information
    if any(dim in item_name + notes for dim in ["inch", '"', "mm", "cm", "ft", "size"]):
        metadata["has_dimensions"] = True
        metadata["annotation_clarity"] = min(1.0, metadata["annotation_clarity"] + 0.2)
    
    # Check for legend/symbol matching
    if context_data and "legend" in context_data:
        legends = context_data.get("legend", [])
        for legend_entry in legends:
            symbol = legend_entry.get("symbol", "").lower()
            description = legend_entry.get("description", "").lower()
            
            # Check if material matches a legend symbol
            if symbol and (symbol in item_name or symbol in notes):
                metadata["symbol_defined"] = True
                metadata["legend_match"] = True
                metadata["clarity"] = min(1.0, metadata["clarity"] + 0.3)
                break
            
            # Check if material matches legend description
            if description and any(word in description for word in item_name.split()):
                metadata["legend_match"] = True
                metadata["clarity"] = min(1.0, metadata["clarity"] + 0.2)
    
    # Boost confidence for materials with quantities and units
    if material.get("quantity") is not None and material.get("unit"):
        metadata["quantity_labeled"] = True
        metadata["annotation_clarity"] = min(1.0, metadata["annotation_clarity"] + 0.1)
    
    # Adjust clarity based on precision flags
    if "flagged for review" in notes or "precise measurement flagged" in notes:
        metadata["clarity"] = max(0.0, metadata["clarity"] - 0.2)
    
    return metadata

def enhance_material_confidence(materials: List[Dict], context_results: List[Dict] = None) -> List[Dict]:
    """
    Enhance material confidence scores using logic-based analysis
    
    Args:
        materials: List of material dictionaries from agent analysis
        context_results: List of context analysis results for legend matching
        
    Returns:
        List of materials with enhanced confidence scores
    """
    enhanced_materials = []
    
    # Combine all context data for legend matching
    combined_context = {}
    if context_results:
        all_legends = []
        for context in context_results:
            if isinstance(context, dict) and "legend" in context:
                all_legends.extend(context.get("legend", []))
        combined_context["legend"] = all_legends
    
    for material in materials:
        enhanced_material = material.copy()
        
        # Analyze metadata for confidence factors
        metadata = analyze_material_metadata(material, combined_context)
        
        # Compute new confidence score
        new_confidence = compute_confidence(
            clarity=metadata["clarity"],
            annotation_clarity=metadata["annotation_clarity"], 
            measurement_type=metadata["measurement_type"]
        )
        
        # Store original confidence for comparison
        original_confidence = material.get("confidence", 0.5)
        
        # Use the higher of original or computed confidence (don't downgrade good agent scores)
        final_confidence = max(original_confidence, new_confidence)
        
        # Cap confidence for materials without quantities
        if material.get("quantity") is None:
            final_confidence = min(final_confidence, 0.6)
        
        # Update material with enhanced confidence and metadata
        enhanced_material["confidence"] = round(final_confidence, 3)
        enhanced_material["confidence_metadata"] = metadata
        enhanced_material["original_confidence"] = original_confidence
        
        # Add confidence reasoning to notes
        reasoning_parts = []
        if metadata["symbol_defined"]:
            reasoning_parts.append("symbol defined in legend")
        if metadata["quantity_labeled"]:
            reasoning_parts.append("quantity explicitly labeled")
        if metadata["measurement_type"] == "explicit":
            reasoning_parts.append("explicit measurement")
        elif metadata["measurement_type"] == "inferred":
            reasoning_parts.append("inferred quantity")
        
        if reasoning_parts:
            confidence_note = f"Confidence: {final_confidence:.2f} ({', '.join(reasoning_parts)})"
            original_notes = enhanced_material.get("notes", "")
            enhanced_material["notes"] = f"{original_notes}. {confidence_note}" if original_notes else confidence_note
        
        enhanced_materials.append(enhanced_material)
    
    # Log confidence enhancement statistics
    if materials:
        avg_original = sum(m.get("confidence", 0.5) for m in materials) / len(materials)
        avg_enhanced = sum(m["confidence"] for m in enhanced_materials) / len(enhanced_materials)
        improved_count = sum(1 for m in enhanced_materials if m["confidence"] > m["original_confidence"])
        
        logger.info(f"Confidence enhancement: {avg_original:.3f} → {avg_enhanced:.3f} avg, {improved_count}/{len(materials)} materials improved")
    
    return enhanced_materials

def filter_low_confidence_materials(materials: List[Dict], min_confidence: float = 0.6) -> List[Dict]:
    """Filter out materials with confidence below threshold"""
    filtered = []
    excluded_count = 0
    
    for material in materials:
        confidence = material.get("confidence", 1.0)
        if confidence >= min_confidence:
            filtered.append(material)
        else:
            excluded_count += 1
            logger.info(f"Excluded low-confidence material: {material.get('item_name', 'Unknown')} (confidence: {confidence})")
    
    if excluded_count > 0:
        logger.info(f"Filtered out {excluded_count} low-confidence materials (threshold: {min_confidence})")
    
    return filtered

@app.post("/api/upload")
async def upload_pdf(file: UploadFile):
    # Validate file format
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Read content and validate size (limit to 50MB)
    content = await file.read()
    max_size = 50 * 1024 * 1024  # 50MB limit
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB")
    
    if len(content) < 100:  # Minimum viable PDF size
        raise HTTPException(status_code=400, detail="File appears to be corrupted or empty")
    
    # Basic PDF magic number validation
    if not content.startswith(b'%PDF-'):
        raise HTTPException(status_code=400, detail="Invalid PDF file format")
    
    task_id = str(uuid.uuid4())
    safe_filename = sanitize_filename(file.filename)
    temp_file_path = os.path.join(tempfile.gettempdir(), f"{task_id}_{safe_filename}")
    
    try:
        with open(temp_file_path, "wb") as f:
            f.write(content)
        update_progress(task_id, "uploading", 100, 100, "File upload complete")
        return {"taskId": task_id, "filename": safe_filename}
    except Exception as e:
        # Clean up on failure
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

def validate_task_id(task_id: str) -> str:
    """Validate task ID format to prevent injection or traversal."""
    import re
    if not re.match(r'^[a-f0-9\-]{36}$', task_id):
        raise HTTPException(status_code=400, detail="Invalid task ID format")
    return task_id

@app.post("/api/analyze")
async def analyze_pdf(taskId: str = Form(...), filename: str = Form(...)):
    task_id = validate_task_id(taskId)
    safe_filename = sanitize_filename(filename)
    temp_file_path = os.path.join(tempfile.gettempdir(), f"{task_id}_{safe_filename}")
    
    if not os.path.exists(temp_file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # If result already exists, do not re-run analysis
    if task_id in result_store:
        backend_url = AppConfig.BACKEND_URL
        return {
            "taskId": task_id,
            "pdfUrl": f"{backend_url}/api/images/{task_id}_{safe_filename}",
            "analysis": result_store[task_id]["analysis"]
        }
    
    try:
        with open(temp_file_path, "rb") as f:
            content = f.read()
        
        pdf_converter = PDFConverter()
        
        try:
            # Enhanced PDF processing with text extraction
            logger.info("Starting enhanced PDF processing with text extraction")
            update_progress(task_id, "processing", 0, 100, "Starting enhanced PDF conversion...")
            pages_info = pdf_converter.pdf_to_images(content)
            logger.info(f"Enhanced PDF processing complete. Generated {len(pages_info)} pages with metadata")
            
            # Initialize results storage
            all_context_results = []
            all_plumbing_results = []
            consolidated_materials = []
            sheets_processed = []
            processing_errors = []  # Track structured errors
            
            # Process each page with two-stage workflow
            total_stages = len(pages_info) * 2  # context, plumbing
            current_stage = 0
            
            for i, page_info in enumerate(pages_info):
                # Get sheet ID from metadata, but ensure page numbers are formatted correctly
                original_sheet_id = page_info.get("sheet_metadata", {}).get("sheet_id")
                if original_sheet_id:
                    sheet_id = original_sheet_id
                else:
                    sheet_id = str(i+1)  # Use just the number (1, 2, 3, etc.)
                
                # If the extracted sheet_id has page_ format, convert it to just the number
                if sheet_id and sheet_id.startswith("page_"):
                    sheet_id = sheet_id.replace("page_", "")
                
                sheets_processed.append(sheet_id)
                
                logger.info(f"Processing sheet {sheet_id} ({i+1}/{len(pages_info)})")
                
                # Prepare image using ImageProcessor
                try:
                    autogen_image = image_processor.process_image_for_agent(page_info['image_path'])
                except Exception as e:
                    logger.error(f"Error processing image for {sheet_id}: {str(e)}")
                    add_error_to_status(processing_errors, "image_processor", sheet_id, "processing", f"Image processing failed: {str(e)}")
                    continue
                
                # Stage 1: Context Extraction
                current_stage += 1
                progress_percent = int((current_stage / total_stages) * 100)
                update_progress(task_id, "analyzing", progress_percent, 100, f"Extracting context for {sheet_id}...")
                
                agent_input = create_agent_input(page_info)
                # Sanitize inputs before including in prompt to prevent injection
                safe_sheet_id = sanitize_prompt_text(sheet_id)
                safe_filename_for_prompt = sanitize_prompt_text(safe_filename)
                context_prompt = f"""Extract document context from this construction plan:\n\nSheet: {safe_sheet_id}\nFile: {safe_filename_for_prompt}\n\nStructured Input Data:\n{json.dumps(agent_input.model_dump(), indent=2)}\n\nAnalyze this construction document to extract legends, symbols, and organizational information."""
                
                context_message = MultiModalMessage(
                    content=[context_prompt, autogen_image],
                    source="user"
                )
                
                try:
                    context_response = await context_agent.on_messages([context_message], CancellationToken())
                    context_content = context_response.chat_message.content if hasattr(context_response, 'chat_message') else str(context_response)
                    
                    # Parse context response
                    context_data = parse_json_response(context_content, ContextOutput, "context", sheet_id, processing_errors)
                    save_intermediate_result(task_id, sheet_id, "context", context_data)
                    all_context_results.append(context_data)
                    
                    logger.info(f"Context extraction complete for {sheet_id}")
                    
                except Exception as e:
                    logger.error(f"Context extraction failed for {sheet_id}: {str(e)}")
                    add_error_to_status(processing_errors, "context", sheet_id, "analysis", f"Context extraction failed: {str(e)}")
                    context_data = {
                        "sheet_metadata": page_info.get("sheet_metadata", {}),
                        "legend": [],
                        "drawing_types": [],
                        "annotation_systems": {},
                        "technical_standards": [],
                        "document_organization": {}
                    }
                
                # Stage 2: Plumbing Analysis
                current_stage += 1
                progress_percent = int((current_stage / total_stages) * 100)
                update_progress(task_id, "analyzing", progress_percent, 100, f"Analyzing plumbing for {sheet_id}...")
                
                # Create enhanced agent input with context
                plumbing_input = create_agent_input(page_info, context_data)
                
                # Build comprehensive analysis prompt
                sheet_title = page_info.get("sheet_metadata", {}).get("title", "Unknown")
                
                # Determine analysis guidance based on sheet content
                analysis_guidance = ""
                if ("water" in sheet_title.lower() or "recycled" in sheet_title.lower() or 
                    "civil" in sheet_title.lower() or "utility" in sheet_title.lower()):
                    analysis_guidance = """
FOCUS AREAS FOR WATER/CIVIL INFRASTRUCTURE:
- Water mains and distribution lines (measure route lengths)
- Service connections and laterals (count each connection)
- Fire hydrants and valves (count all instances)
- Backflow prevention devices and meters
- Pumping stations and storage facilities
- Air release valves and pressure reducing valves
- Pipe fittings (elbows, tees, reducers, couplings)
- Thrust blocks and anchoring systems
- Manholes, valve boxes, and access structures"""
                else:
                    analysis_guidance = """
FOCUS AREAS FOR PLUMBING SYSTEMS:
- Water supply pipes and fixtures
- Drainage and waste systems
- Plumbing fixtures (sinks, toilets, water heaters)
- Valves, fittings, and connections
- Pumps and water treatment equipment
- Pipe insulation and supports"""
                
                # Sanitize inputs for plumbing prompt as well
                safe_sheet_id_plumbing = sanitize_prompt_text(sheet_id)
                safe_sheet_title = sanitize_prompt_text(sheet_title)
                safe_filename_plumbing = sanitize_prompt_text(safe_filename)
                
                plumbing_prompt = f"""COMPREHENSIVE PLUMBING/WATER INFRASTRUCTURE ANALYSIS

Sheet: {safe_sheet_id_plumbing} ({safe_sheet_title})
File: {safe_filename_plumbing}

CONTEXT DATA PROVIDED:
{json.dumps(context_data, indent=2)}

STRUCTURED INPUT DATA:
{json.dumps(plumbing_input.model_dump(), indent=2)}

{analysis_guidance}

ANALYSIS REQUIREMENTS:
1. **USE CONTEXT AS REFERENCE**: The context agent has provided legend information - use this to understand symbols, but YOU must do the comprehensive material extraction
2. **ANALYZE ALL VISUAL CONTENT**: Examine every part of the construction drawings for materials, quantities, and specifications  
3. **COUNT ACTUAL INSTANCES**: Don't just list legend definitions - count how many of each item you actually see in the plans
4. **MEASURE WHERE POSSIBLE**: For pipes and linear materials, estimate lengths based on drawn routes and scale
5. **EXTRACT FROM MULTIPLE SOURCES**: Look for materials in symbols, text annotations, schedules, details, and specifications
6. **PROVIDE REALISTIC QUANTITIES**: Base quantities on visual analysis of the actual plan content

CRITICAL: Perform COMPREHENSIVE visual analysis to extract ALL plumbing/water infrastructure materials visible in the drawings."""
                
                plumbing_message = MultiModalMessage(
                    content=[plumbing_prompt, autogen_image],
                    source="user"
                )
                
                try:
                    plumbing_response = await plumbing_agent.on_messages([plumbing_message], CancellationToken())
                    plumbing_content = plumbing_response.chat_message.content if hasattr(plumbing_response, 'chat_message') else str(plumbing_response)
                    
                    # Parse plumbing response
                    plumbing_data = parse_json_response(plumbing_content, PlumbingOutput, "plumbing", sheet_id, processing_errors)
                    save_intermediate_result(task_id, sheet_id, "plumbing", plumbing_data)
                    all_plumbing_results.append(plumbing_data)
                    
                    # Deduplicate materials within this page first
                    page_materials = plumbing_data.get("materials", [])
                    if page_materials:
                        logger.debug(f"Deduplicating materials within {sheet_id}: {len(page_materials)} materials found")
                        deduplicated_page_materials = deduplicate_materials(page_materials)
                        logger.debug(f"Page deduplication complete for {sheet_id}: {len(page_materials)} -> {len(deduplicated_page_materials)} unique materials")
                    else:
                        deduplicated_page_materials = []
                    
                    # Apply immediate confidence filtering to catch obvious issues early
                    reliable_materials = [
                        material for material in deduplicated_page_materials
                        if material.get("confidence", 0) > 0.4  # Very low threshold here, main filtering happens later
                    ]
                    
                    # Add to consolidated list with page reference
                    for material in reliable_materials:
                        material["reference_sheet"] = sheet_id
                        consolidated_materials.append(material)
                    
                    excluded_materials = len(plumbing_data.get("materials", [])) - len(reliable_materials)
                    if excluded_materials > 0:
                        logger.info(f"Excluded {excluded_materials} very low-confidence materials from {sheet_id}")
                    
                    logger.info(f"Plumbing analysis complete for {sheet_id}")
                    
                except Exception as e:
                    logger.error(f"Plumbing analysis failed for {sheet_id}: {str(e)}")
                    add_error_to_status(processing_errors, "plumbing", sheet_id, "analysis", f"Plumbing analysis failed: {str(e)}")
                    plumbing_data = {
                        "materials": [],
                        "special_requirements": [],
                        "potential_issues": [f"Analysis failed: {str(e)}"],
                        "summary": "Analysis incomplete due to error"
                    }
                    all_plumbing_results.append(plumbing_data)
            
            # Sort materials by ascending page number (materials already deduplicated per page)
            logger.info(f"Sorting materials by page number: {len(consolidated_materials)} materials found")
            
            def get_sort_key(material):
                """Create sort key for materials - prioritize numeric page numbers"""
                ref_sheet = material.get("reference_sheet", "")
                if not ref_sheet:
                    return (999, "")  # Unknown sheets go to end
                
                # Try to parse as pure number (page numbers like "1", "2", etc.)
                if ref_sheet.isdigit():
                    return (0, int(ref_sheet))  # Numeric pages first, sorted numerically
                
                # Handle sheet IDs like "P-102", "M-201", etc.
                return (1, ref_sheet)  # Sheet IDs second, sorted alphabetically
            
            sorted_materials = sorted(consolidated_materials, key=get_sort_key)
            logger.info(f"Materials sorted by ascending page number")
            
            # Apply validation and enhanced confidence scoring to improve accuracy
            logger.info("Applying material validation and enhanced confidence scoring...")
            validated_materials = post_agent_result_validation(sorted_materials, all_context_results)
            
            # Apply logic-based confidence enhancement using context and annotation metadata
            enhanced_materials = enhance_material_confidence(validated_materials, all_context_results)
            
            # Filter materials based on enhanced confidence scores
            filtered_materials = filter_low_confidence_materials(enhanced_materials, min_confidence=0.6)
            
            logger.info(f"Material processing complete: {len(sorted_materials)} -> {len(enhanced_materials)} -> {len(filtered_materials)} materials after enhancement and filtering")
            
            # Create final analysis result with structured error tracking
            final_processing_status = create_processing_status_with_errors(
                stage="complete",
                progress=100.0,
                message=f"Two-stage analysis complete! {len(processing_errors)} errors encountered." if processing_errors else "Two-stage analysis complete!",
                current_sheet=None,
                errors=processing_errors
            )
            
            final_result = AnalysisResult(
                task_id=task_id,
                sheets_processed=sheets_processed,
                context_results=all_context_results,
                plumbing_results=all_plumbing_results,
                consolidated_materials=filtered_materials,
                processing_status=final_processing_status,
                metadata={
                    "filename": safe_filename,
                    "total_pages": len(pages_info),
                    "workflow": "context -> plumbing",
                    "total_errors": len(processing_errors),
                    "agents_with_errors": list(final_processing_status.agent_error_summary.keys()) if processing_errors else [],
                    "sheets_with_errors": list(final_processing_status.sheet_error_summary.keys()) if processing_errors else [],
                    "total_materials": len(filtered_materials),
                    "materials_before_filtering": len(sorted_materials),
                    "materials_after_validation": len(validated_materials),
                    "materials_after_enhancement": len(enhanced_materials),
                    "confidence_threshold": 0.6,
                    "confidence_enhancement_enabled": True,
                    "post_agent_validation_enabled": True,
                    "deduplication_method": "per_page",
                    "sorted_by_page": True,
                    "validation_stats": {
                        "materials_flagged": sum(1 for m in validated_materials if m.get("validation_metadata", {}).get("validation_flags_count", 0) > 0),
                        "materials_nullified": sum(1 for m in validated_materials if m.get("validation_metadata", {}).get("nullification_count", 0) > 0),
                        "confidence_adjustments": sum(1 for m in validated_materials if m.get("validation_metadata", {}).get("confidence_issues_count", 0) > 0),
                        "unit_validation_failures": sum(1 for m in validated_materials if not m.get("validation_metadata", {}).get("unit_validation_passed", True)),
                        "large_quantities_flagged": sum(1 for m in validated_materials if "Large quantity" in m.get("notes", "")),
                        "symbol_matching_enabled": True,
                        "enhanced_validation_rules": ["quantities_>=_1000", "unit_compatibility", "confidence_vs_visibility", "ambiguous_value_nullification"]
                    },
                    "confidence_stats": {
                        "avg_original": round(sum(m.get("original_confidence", 0.5) for m in enhanced_materials) / len(enhanced_materials), 3) if enhanced_materials else 0,
                        "avg_enhanced": round(sum(m["confidence"] for m in enhanced_materials) / len(enhanced_materials), 3) if enhanced_materials else 0,
                        "materials_improved": sum(1 for m in enhanced_materials if m["confidence"] > m.get("original_confidence", 0.5)),
                        "explicit_measurements": sum(1 for m in enhanced_materials if m.get("confidence_metadata", {}).get("measurement_type") == "explicit"),
                        "visual_estimates": sum(1 for m in enhanced_materials if m.get("confidence_metadata", {}).get("measurement_type") == "visual_estimate"),
                        "inferred_materials": sum(1 for m in enhanced_materials if m.get("confidence_metadata", {}).get("measurement_type") == "inferred"),
                        "legend_matched": sum(1 for m in enhanced_materials if m.get("confidence_metadata", {}).get("legend_match", False))
                    }
                }
            )
            
            update_progress(task_id, "complete", 100, 100, "Two-stage analysis complete!")
            
            # Store results
            backend_url = AppConfig.BACKEND_URL
            result_store[task_id] = {
                "pdfUrl": f"{backend_url}/api/images/{task_id}_{safe_filename}",
                "analysis": final_result.model_dump(),
                "context_results": all_context_results,
                "plumbing_results": all_plumbing_results
            }
            
            pdf_converter.cleanup()
            
            return {
                "taskId": task_id,
                "pdfUrl": f"{backend_url}/api/images/{task_id}_{safe_filename}",
                "analysis": final_result.model_dump()
            }
            
        except Exception as e:
            logger.error(f"Error during enhanced analysis: {str(e)}")
            
            # Create error result with structured error information
            system_error = create_processing_error("system", None, "critical", f"Analysis workflow failed: {str(e)}")
            error_processing_status = create_processing_status_with_errors(
                stage="failed",
                progress=0.0,
                message=f"Analysis failed due to system error: {str(e)}",
                current_sheet=None,
                errors=[system_error]
            )
            
            # Store partial results if available
            backend_url = AppConfig.BACKEND_URL
            result_store[task_id] = {
                "pdfUrl": f"{backend_url}/api/images/{task_id}_{safe_filename}",
                "analysis": {
                    "task_id": task_id,
                    "processing_status": error_processing_status.model_dump()
                },
                "context_results": [],
                "plumbing_results": []
            }
            
            pdf_converter.cleanup()
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing document")

@app.get("/api/progress/{task_id}")
async def get_progress(task_id: str):
    """Get progress updates for a specific task"""
    if task_id not in progress_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return progress_store[task_id]

@app.get("/api/progress/{task_id}/stream")
async def stream_progress(task_id: str):
    """Stream progress updates for a specific task using Server-Sent Events"""
    
    async def event_generator():
        last_update = None
        
        while True:
            if task_id in progress_store:
                current_update = progress_store[task_id]
                
                # Only send if there's a new update
                if last_update != current_update:
                    last_update = current_update
                    yield f"data: {json.dumps(current_update)}\n\n"
                
                # If analysis is complete, stop streaming
                if current_update["stage"] == "complete":
                    break
            
            await asyncio.sleep(0.5)  # Check every 500ms
    
    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.get("/api/images/{filename}")
async def get_image(filename: str):
    """Serve PDF files with security validation"""
    try:
        # Sanitize filename to prevent path traversal
        safe_filename = os.path.basename(filename)
        
        # Additional validation - ensure filename matches expected pattern
        import re
        if not re.match(r'^[a-f0-9\-]{36}_[a-zA-Z0-9._-]+\.pdf$', safe_filename):
            raise HTTPException(status_code=400, detail="Invalid filename format")
        
        # Construct safe file path
        file_path = os.path.join(tempfile.gettempdir(), safe_filename)
        
        # Verify the file path is within temp directory (additional safety)
        real_temp_dir = os.path.realpath(tempfile.gettempdir())
        real_file_path = os.path.realpath(file_path)
        if not real_file_path.startswith(real_temp_dir):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Verify it's actually a PDF file
        with open(file_path, 'rb') as f:
            header = f.read(8)
            if not header.startswith(b'%PDF-'):
                raise HTTPException(status_code=400, detail="Invalid file type")
        
        return FileResponse(file_path, media_type="application/pdf")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error serving file")

@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    task_id = validate_task_id(task_id)
    if task_id not in result_store:
        raise HTTPException(status_code=404, detail="Result not found")
    return result_store[task_id]

@app.get("/api/context/{task_id}")
async def get_context(task_id: str):
    """Get the context analysis for debugging purposes"""
    task_id = validate_task_id(task_id)
    if task_id not in result_store:
        raise HTTPException(status_code=404, detail="Result not found")
    if "context_results" not in result_store[task_id]:
        raise HTTPException(status_code=404, detail="Context not available")
    return {"context_results": result_store[task_id]["context_results"]}

@app.get("/api/debug/text-extraction/{task_id}")
async def get_text_extraction_debug(task_id: str):
    """Get raw text extraction results for debugging PDF processing issues"""
    task_id = validate_task_id(task_id)
    import tempfile
    import json
    
    results_dir = os.path.join(tempfile.gettempdir(), f"analysis_results_{task_id}")
    if not os.path.exists(results_dir):
        raise HTTPException(status_code=404, detail="Debug data not found")
    
    debug_data = {}
    
    # Look for context files with raw text
    for filename in os.listdir(results_dir):
        if filename.endswith("_context_result.json"):
            sheet_id = filename.replace("_context_result.json", "")
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, "r", encoding='utf-8') as f:
                    context_data = json.load(f)
                    debug_data[sheet_id] = {
                        "sheet_metadata": context_data.get("sheet_metadata", {}),
                        "legend_count": len(context_data.get("legend", [])),
                        "legend_entries": context_data.get("legend", []),
                        "text_blocks_count": len(context_data.get("text_blocks", [])),
                        "text_blocks": context_data.get("text_blocks", [])[:5],  # First 5 blocks only
                        "notes": context_data.get("notes", [])
                    }
            except Exception as e:
                debug_data[sheet_id] = {"error": str(e)}
    
    return {"text_extraction_debug": debug_data}

@app.get("/api/debug/intermediate/{task_id}")
async def get_intermediate_results(task_id: str):
    """Get all intermediate analysis results for debugging"""
    task_id = validate_task_id(task_id)
    import tempfile
    import json
    
    results_dir = os.path.join(tempfile.gettempdir(), f"analysis_results_{task_id}")
    if not os.path.exists(results_dir):
        raise HTTPException(status_code=404, detail="Debug data not found")
    
    intermediate_data = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith("_result.json"):
            # Parse filename to get sheet_id and agent_name
            base_name = filename.replace("_result.json", "")
            parts = base_name.rsplit("_", 1)
            if len(parts) == 2:
                sheet_id, agent_name = parts
                filepath = os.path.join(results_dir, filename)
                try:
                    with open(filepath, "r", encoding='utf-8') as f:
                        result_data = json.load(f)
                        
                        if sheet_id not in intermediate_data:
                            intermediate_data[sheet_id] = {}
                        intermediate_data[sheet_id][agent_name] = result_data
                except Exception as e:
                    if sheet_id not in intermediate_data:
                        intermediate_data[sheet_id] = {}
                    intermediate_data[sheet_id][agent_name] = {"error": str(e)}
    
    return {"intermediate_results": intermediate_data}

@app.get("/api/debug/material-validation/{task_id}")
async def get_material_validation_debug(task_id: str):
    """Get details about material validation and filtering process"""
    if task_id not in result_store:
        raise HTTPException(status_code=404, detail="Result not found")
    
    analysis = result_store[task_id].get("analysis", {})
    metadata = analysis.get("metadata", {})
    
    validation_info = {
        "total_materials_before_filtering": metadata.get("materials_before_filtering", 0),
        "total_materials_after_filtering": metadata.get("total_materials", 0),
        "confidence_threshold": metadata.get("confidence_threshold", 0.6),
        "deduplication_method": metadata.get("deduplication_method", "per_page"),
        "materials_filtered": metadata.get("materials_before_filtering", 0) - metadata.get("total_materials", 0)
    }
    
    # Add confidence distribution
    materials = analysis.get("consolidated_materials", [])
    confidence_ranges = {
        "0.0-0.3": 0,
        "0.3-0.5": 0, 
        "0.5-0.7": 0,
        "0.7-0.9": 0,
        "0.9-1.0": 0,
        "missing": 0
    }
    
    for material in materials:
        confidence = material.get("confidence")
        if confidence is None:
            confidence_ranges["missing"] += 1
        elif confidence < 0.3:
            confidence_ranges["0.0-0.3"] += 1
        elif confidence < 0.5:
            confidence_ranges["0.3-0.5"] += 1
        elif confidence < 0.7:
            confidence_ranges["0.5-0.7"] += 1
        elif confidence < 0.9:
            confidence_ranges["0.7-0.9"] += 1
        else:
            confidence_ranges["0.9-1.0"] += 1
    
    validation_info["confidence_distribution"] = confidence_ranges
    
    return {"validation_debug": validation_info}

@app.get("/api/debug/confidence-enhancement/{task_id}")
async def get_confidence_enhancement_debug(task_id: str):
    """Get details about confidence enhancement process and scoring logic"""
    if task_id not in result_store:
        raise HTTPException(status_code=404, detail="Result not found")
    
    analysis = result_store[task_id].get("analysis", {})
    materials = analysis.get("consolidated_materials", [])
    
    enhancement_debug = {
        "enhancement_enabled": analysis.get("metadata", {}).get("confidence_enhancement_enabled", False),
        "confidence_stats": analysis.get("metadata", {}).get("confidence_stats", {}),
        "material_details": []
    }
    
    # Provide detailed breakdown for each material
    for material in materials:
        confidence_metadata = material.get("confidence_metadata", {})
        detail = {
            "item_name": material.get("item_name", "Unknown"),
            "original_confidence": material.get("original_confidence", 0.5),
            "enhanced_confidence": material.get("confidence", 0.5),
            "confidence_change": round(material.get("confidence", 0.5) - material.get("original_confidence", 0.5), 3),
            "measurement_type": confidence_metadata.get("measurement_type", "unknown"),
            "clarity": confidence_metadata.get("clarity", 0.5),
            "annotation_clarity": confidence_metadata.get("annotation_clarity", 0.5),
            "symbol_defined": confidence_metadata.get("symbol_defined", False),
            "quantity_labeled": confidence_metadata.get("quantity_labeled", False),
            "has_dimensions": confidence_metadata.get("has_dimensions", False),
            "legend_match": confidence_metadata.get("legend_match", False),
            "quantity": material.get("quantity"),
            "unit": material.get("unit"),
            "notes": material.get("notes", "")
        }
        enhancement_debug["material_details"].append(detail)
    
    # Sort by confidence change (most improved first)
    enhancement_debug["material_details"].sort(key=lambda x: x["confidence_change"], reverse=True)
    
    # Add summary statistics
    if enhancement_debug["material_details"]:
        details = enhancement_debug["material_details"]
        enhancement_debug["summary"] = {
            "total_materials": len(details),
            "materials_improved": sum(1 for d in details if d["confidence_change"] > 0),
            "materials_unchanged": sum(1 for d in details if d["confidence_change"] == 0),
            "materials_degraded": sum(1 for d in details if d["confidence_change"] < 0),
            "avg_confidence_change": round(sum(d["confidence_change"] for d in details) / len(details), 3),
            "max_improvement": max(d["confidence_change"] for d in details),
            "measurement_type_distribution": {
                "explicit": sum(1 for d in details if d["measurement_type"] == "explicit"),
                "visual_estimate": sum(1 for d in details if d["measurement_type"] == "visual_estimate"),
                "inferred": sum(1 for d in details if d["measurement_type"] == "inferred")
            }
        }
    
    return {"confidence_enhancement_debug": enhancement_debug}

@app.get("/api/debug/post-agent-validation/{task_id}")
async def get_post_agent_validation_debug(task_id: str):
    """Get details about post-agent validation process and flagged issues"""
    if task_id not in result_store:
        raise HTTPException(status_code=404, detail="Result not found")
    
    analysis = result_store[task_id].get("analysis", {})
    materials = analysis.get("consolidated_materials", [])
    
    validation_debug = {
        "validation_enabled": analysis.get("metadata", {}).get("post_agent_validation_enabled", False),
        "validation_stats": analysis.get("metadata", {}).get("validation_stats", {}),
        "material_details": []
    }
    
    # Provide detailed breakdown for each material
    for material in materials:
        validation_metadata = material.get("validation_metadata", {})
        detail = {
            "item_name": material.get("item_name", "Unknown"),
            "quantity": material.get("quantity"),
            "unit": material.get("unit"),
            "confidence": material.get("confidence", 0.5),
            "original_confidence": validation_metadata.get("original_confidence", 0.5),
            "unit_validation_passed": validation_metadata.get("unit_validation_passed", True),
            "validation_flags_count": validation_metadata.get("validation_flags_count", 0),
            "confidence_issues_count": validation_metadata.get("confidence_issues_count", 0),
            "nullification_count": validation_metadata.get("nullification_count", 0),
            "additional_flags": validation_metadata.get("additional_flags", []),
            "visibility_confidence": validation_metadata.get("visibility_confidence", 0.5),
            "notes": material.get("notes", ""),
            "validation_impact": {
                "was_flagged": validation_metadata.get("validation_flags_count", 0) > 0,
                "was_nullified": validation_metadata.get("nullification_count", 0) > 0,
                "confidence_adjusted": validation_metadata.get("confidence_issues_count", 0) > 0,
                "unit_failed": not validation_metadata.get("unit_validation_passed", True)
            }
        }
        validation_debug["material_details"].append(detail)
    
    # Sort by validation impact (most flagged first)
    validation_debug["material_details"].sort(
        key=lambda x: (x["validation_flags_count"] + x["confidence_issues_count"] + x["nullification_count"]), 
        reverse=True
    )
    
    # Add summary statistics
    if validation_debug["material_details"]:
        details = validation_debug["material_details"]
        validation_debug["summary"] = {
            "total_materials": len(details),
            "materials_with_issues": sum(1 for d in details if d["validation_impact"]["was_flagged"] or d["validation_impact"]["was_nullified"]),
            "materials_flagged": sum(1 for d in details if d["validation_impact"]["was_flagged"]),
            "materials_nullified": sum(1 for d in details if d["validation_impact"]["was_nullified"]),
            "confidence_adjusted": sum(1 for d in details if d["validation_impact"]["confidence_adjusted"]),
            "unit_validation_failures": sum(1 for d in details if d["validation_impact"]["unit_failed"]),
            "common_issues": {
                "large_quantities": sum(1 for d in details if "Large quantity" in d["notes"]),
                "unit_mismatches": sum(1 for d in details if "using non-" in d["notes"]),
                "confidence_without_evidence": sum(1 for d in details if "without explicit" in d["notes"]),
                "guessed_quantities": sum(1 for d in details if "appears guessed" in d["notes"]),
                "unmatched_symbols": sum(1 for d in details if "Unmatched symbol" in d["notes"])
            }
        }
    
    return {"post_agent_validation_debug": validation_debug}

@app.delete("/api/cache/clear")
async def clear_cache():
    """Clear all cached results to force re-analysis"""
    global result_store, progress_store
    result_store.clear()
    progress_store.clear()
    return {"message": "Cache cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 