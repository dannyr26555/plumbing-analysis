from fastapi import FastAPI, UploadFile, HTTPException, Depends, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import logging
import tempfile
import shutil
from datetime import datetime
from pdf_converter import PDFConverter
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import Image as AutoGenImage
from autogen_agentchat.messages import MultiModalMessage
import json
from pathlib import Path
from config.agent_config import SYSTEM_MESSAGES, AGENT_CONFIG, JSON_RESPONSE_FORMAT
import asyncio
import uuid
import base64
from typing import List, Dict, Any, Optional
from pydantic import ValidationError
from models import (
    AgentInput, MaterialItem, ContextOutput, PlumbingOutput, 
    PreprocessorOutput, AnalysisResult, ProcessingStatus, 
    LegendEntry, TextBlock, SheetMetadata, BoundingBox, ProcessingError
)
import re
from typing import Dict, List, Any, Optional, Tuple
import difflib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store for progress updates
progress_store = {}
# Store for analysis results
result_store = {}

# Load environment variables from project root
try:
    env_path = Path(__file__).parent.parent / '.env'
    if not env_path.exists():
        raise RuntimeError("Environment configuration not found")
    load_dotenv(env_path)
    logger.info("Environment configuration loaded successfully")
except Exception as e:
    logger.error("Failed to load environment configuration")
    raise RuntimeError("Environment configuration error")

# Initialize FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize AutoGen client
try:
    # Get Azure configuration
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    if not azure_endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
    
    # Ensure endpoint has proper format
    if not azure_endpoint.startswith('https://'):
        azure_endpoint = f'https://{azure_endpoint}'
    if not azure_endpoint.endswith('/'):
        azure_endpoint += '/'
    
    # Get deployment name
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
    if not deployment_name:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME is not set")
    
    # Construct the full endpoint URL
    full_endpoint = f"{azure_endpoint}openai/deployments/{deployment_name}"
    
    # Log configuration (without sensitive data)
    logger.info("Azure OpenAI Configuration:")
    logger.info(f"API Base: {full_endpoint}")
    logger.info(f"API Version: {os.getenv('AZURE_OPENAI_API_VERSION')}")
    logger.info(f"Deployment Name: {deployment_name}")
    
    # Create the client with explicit Azure configuration
    client = OpenAIChatCompletionClient(
        model=deployment_name,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=full_endpoint,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_type="azure",
        deployment_id=deployment_name,
        base_url=full_endpoint,
        default_headers={
            "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
            "Content-Type": "application/json"
        },
        default_query={
            "api-version": os.getenv("AZURE_OPENAI_API_VERSION")
        }
    )
    logger.info("Successfully initialized Azure OpenAI client")
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
    raise RuntimeError("Azure OpenAI configuration error")

# Initialize preprocessor agent
try:
    preprocessor_agent = AssistantAgent(
        "preprocessor_agent",
        client,
        system_message=SYSTEM_MESSAGES["preprocessor"]
    )
    logger.info("Successfully initialized preprocessor agent")
except Exception as e:
    logger.error(f"Failed to initialize preprocessor agent: {str(e)}")
    raise RuntimeError("Preprocessor agent initialization error")

# Initialize context extraction agent
try:
    context_agent = AssistantAgent(
        "context_agent",
        client,
        system_message=SYSTEM_MESSAGES["context"]
    )
    logger.info("Successfully initialized context extraction agent")
except Exception as e:
    logger.error(f"Failed to initialize context extraction agent: {str(e)}")
    raise RuntimeError("Context agent initialization error")

# Initialize plumbing analysis agent
try:
    plumbing_agent = AssistantAgent(
        "plumbing_agent",
        client,
        system_message=SYSTEM_MESSAGES["plumbing"]
    )
    logger.info("Successfully initialized plumbing analysis agent")
except Exception as e:
    logger.error(f"Failed to initialize plumbing analysis agent: {str(e)}")
    raise RuntimeError("Plumbing agent initialization error")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and remove sensitive information."""
    return os.path.basename(filename)

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
        # Clean up the response text
        cleaned_text = response_text.strip()
        
        # Remove any markdown code blocks if present
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        # Parse JSON
        parsed_data = json.loads(cleaned_text)
        
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
        
        # Validate against model if provided
        if expected_model:
            validated_data = expected_model(**parsed_data)
            return validated_data.model_dump()
        
        return parsed_data
        
    except json.JSONDecodeError as e:
        error_msg = f"JSON parsing error: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Raw response: {response_text}")
        
        # Add to structured error tracking if provided
        if processing_errors is not None:
            add_error_to_status(processing_errors, agent_name, sheet_id, "parsing", error_msg)
            
        raise ValueError(f"Invalid JSON response: {str(e)}")
    except ValidationError as e:
        error_msg = f"Model validation error: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Parsed data: {parsed_data}")
        
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

def validate_material_quantities(materials: List[Dict], max_reasonable_quantity: float = 10000) -> List[Dict]:
    """Validate and flag potentially incorrect material quantities"""
    validated_materials = []
    
    for material in materials:
        qty = material.get("quantity")
        confidence = material.get("confidence", 1.0)
        
        # Create a copy to avoid modifying original
        validated_material = material.copy()
        
        if qty and isinstance(qty, (int, float)):
            # Flag suspiciously large quantities
            if qty > max_reasonable_quantity:
                validated_material["confidence"] = min(confidence, 0.3)
                original_notes = validated_material.get("notes", "")
                flag_note = f"Large quantity flagged for review: {qty}"
                validated_material["notes"] = f"{original_notes}. {flag_note}" if original_notes else flag_note
                logger.warning(f"Flagged large quantity for {material.get('item_name', 'Unknown')}: {qty}")
            
            # Flag quantities that seem too precise for visual estimation
            if isinstance(qty, float) and qty > 100:
                # Check if it's suspiciously precise (many decimal places)
                decimal_places = len(str(qty).split('.')[-1]) if '.' in str(qty) else 0
                if decimal_places > 1:
                    validated_material["confidence"] = min(confidence, 0.5)
                    original_notes = validated_material.get("notes", "")
                    precision_note = "Precise measurement flagged - may be estimated"
                    validated_material["notes"] = f"{original_notes}. {precision_note}" if original_notes else precision_note
        
        validated_materials.append(validated_material)
    
    return validated_materials

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
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file format")
    task_id = str(uuid.uuid4())
    safe_filename = sanitize_filename(file.filename)
    temp_file_path = os.path.join(tempfile.gettempdir(), f"{task_id}_{safe_filename}")
    content = await file.read()
    with open(temp_file_path, "wb") as f:
        f.write(content)
    update_progress(task_id, "uploading", 100, 100, "File upload complete")
    return {"taskId": task_id, "filename": safe_filename}

@app.post("/api/analyze")
async def analyze_pdf(taskId: str = Form(...), filename: str = Form(...)):
    task_id = taskId
    safe_filename = sanitize_filename(filename)
    temp_file_path = os.path.join(tempfile.gettempdir(), f"{task_id}_{safe_filename}")
    
    if not os.path.exists(temp_file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # If result already exists, do not re-run analysis
    if task_id in result_store:
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
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
            
            # Process each page with three-stage workflow
            total_stages = len(pages_info) * 3  # preprocessor, context, plumbing
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
                
                # Prepare image
                try:
                    with Image.open(page_info['image_path']) as pil_image:
                        if pil_image.size[0] > 1536 or pil_image.size[1] > 1536:
                            pil_image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
                        autogen_image = AutoGenImage(pil_image)
                except Exception as e:
                    logger.error(f"Error processing image for {sheet_id}: {str(e)}")
                    add_error_to_status(processing_errors, "image_processor", sheet_id, "processing", f"Image processing failed: {str(e)}")
                    continue
                
                # Stage 1: Preprocessor Analysis
                current_stage += 1
                progress_percent = int((current_stage / total_stages) * 100)
                update_progress(task_id, "analyzing", progress_percent, 100, f"Classifying sheet {sheet_id}...")
                
                preprocessor_prompt = f"""Classify and analyze this construction document:\n\nSheet: {sheet_id}\nFile: {safe_filename}\n\nAnalyze this sheet to determine its type, complexity, and recommended processing approach."""
                
                preprocessor_message = MultiModalMessage(
                    content=[preprocessor_prompt, autogen_image],
                    source="user"
                )
                
                try:
                    from autogen_core import CancellationToken
                    preprocessor_response = await preprocessor_agent.on_messages([preprocessor_message], CancellationToken())
                    preprocessor_content = preprocessor_response.chat_message.content if hasattr(preprocessor_response, 'chat_message') else str(preprocessor_response)
                    
                    # Parse preprocessor response
                    preprocessor_data = parse_json_response(preprocessor_content, PreprocessorOutput, "preprocessor", sheet_id, processing_errors)
                    save_intermediate_result(task_id, sheet_id, "preprocessor", preprocessor_data)
                    
                    logger.info(f"Preprocessor classified {sheet_id} as: {preprocessor_data['sheet_type']}")
                    
                except Exception as e:
                    logger.error(f"Preprocessor analysis failed for {sheet_id}: {str(e)}")
                    add_error_to_status(processing_errors, "preprocessor", sheet_id, "analysis", f"Preprocessor analysis failed: {str(e)}")
                    preprocessor_data = {
                        "sheet_type": "unknown",
                        "complexity_score": 0.5,
                        "recommended_agents": ["context", "plumbing"],
                        "processing_notes": [f"Preprocessor failed: {str(e)}"]
                    }
                
                # Stage 2: Context Extraction
                current_stage += 1
                progress_percent = int((current_stage / total_stages) * 100)
                update_progress(task_id, "analyzing", progress_percent, 100, f"Extracting context for {sheet_id}...")
                
                agent_input = create_agent_input(page_info)
                context_prompt = f"""Extract document context from this construction plan:\n\nSheet: {sheet_id}\nDiscipline: {preprocessor_data['sheet_type']}\nComplexity: {preprocessor_data['complexity_score']}\n\nStructured Input Data:\n{json.dumps(agent_input.model_dump(), indent=2)}\n\nAnalyze this construction document to extract legends, symbols, and organizational information."""
                
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
                
                # Stage 3: Plumbing Analysis (if applicable)
                current_stage += 1
                progress_percent = int((current_stage / total_stages) * 100)
                
                if preprocessor_data['sheet_type'] in ['plumbing', 'mixed', 'civil']:
                    update_progress(task_id, "analyzing", progress_percent, 100, f"Analyzing plumbing for {sheet_id}...")
                    
                    # Create enhanced agent input with context
                    plumbing_input = create_agent_input(page_info, context_data)
                    
                    # Build comprehensive analysis prompt
                    discipline = preprocessor_data['sheet_type']
                    sheet_title = page_info.get("sheet_metadata", {}).get("title", "Unknown")
                    
                    analysis_guidance = ""
                    if discipline == "civil" or "water" in sheet_title.lower() or "recycled" in sheet_title.lower():
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
                    
                    plumbing_prompt = f"""COMPREHENSIVE PLUMBING/WATER INFRASTRUCTURE ANALYSIS

Sheet: {sheet_id} ({sheet_title})
Discipline: {discipline}
Complexity: {preprocessor_data['complexity_score']}/1.0

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
                else:
                    update_progress(task_id, "analyzing", progress_percent, 100, f"Skipping plumbing analysis for {sheet_id} (sheet type: {preprocessor_data['sheet_type']})")
                    logger.info(f"Skipping plumbing analysis for {sheet_id} - sheet type: {preprocessor_data['sheet_type']} (only analyzing plumbing, mixed, and civil sheets)")
            
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
            
            # Apply validation and filtering to improve accuracy
            logger.info("Applying material validation and confidence filtering...")
            validated_materials = validate_material_quantities(sorted_materials)
            filtered_materials = filter_low_confidence_materials(validated_materials, min_confidence=0.6)
            logger.info(f"Material validation complete: {len(sorted_materials)} -> {len(filtered_materials)} materials after filtering")
            
            # Create final analysis result with structured error tracking
            final_processing_status = create_processing_status_with_errors(
                stage="complete",
                progress=100.0,
                message=f"Three-stage analysis complete! {len(processing_errors)} errors encountered." if processing_errors else "Three-stage analysis complete!",
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
                    "workflow": "preprocessor -> context -> plumbing",
                    "total_errors": len(processing_errors),
                    "agents_with_errors": list(final_processing_status.agent_error_summary.keys()) if processing_errors else [],
                    "sheets_with_errors": list(final_processing_status.sheet_error_summary.keys()) if processing_errors else [],
                    "total_materials": len(filtered_materials),
                    "materials_before_filtering": len(sorted_materials),
                    "confidence_threshold": 0.6,
                    "deduplication_method": "per_page",
                    "sorted_by_page": True
                }
            )
            
            update_progress(task_id, "complete", 100, 100, "Three-stage analysis complete!")
            
            # Store results
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
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
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
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
    """Serve PDF files"""
    try:
        file_path = os.path.join(tempfile.gettempdir(), filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path, media_type="application/pdf")
    except Exception as e:
        logger.error(f"Error serving file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error serving file")

@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in result_store:
        raise HTTPException(status_code=404, detail="Result not found")
    return result_store[task_id]

@app.get("/api/context/{task_id}")
async def get_context(task_id: str):
    """Get the context analysis for debugging purposes"""
    if task_id not in result_store:
        raise HTTPException(status_code=404, detail="Result not found")
    if "context_results" not in result_store[task_id]:
        raise HTTPException(status_code=404, detail="Context not available")
    return {"context_results": result_store[task_id]["context_results"]}

@app.get("/api/debug/text-extraction/{task_id}")
async def get_text_extraction_debug(task_id: str):
    """Get raw text extraction results for debugging PDF processing issues"""
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