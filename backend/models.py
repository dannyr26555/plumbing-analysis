"""
Data Models for Structured Agent Communication
Defines Pydantic models for agent input/output formats
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime

class BoundingBox(BaseModel):
    """Bounding box coordinates for text elements"""
    x0: float
    y0: float
    x1: float
    y1: float

class LegendEntry(BaseModel):
    """Legend/symbol definition entry"""
    symbol: str = Field(description="Symbol or abbreviation")
    description: str = Field(description="Description of the symbol")
    bbox: Optional[BoundingBox] = Field(default=None, description="Bounding box coordinates")

class TextBlock(BaseModel):
    """Text block with positioning information"""
    text: str = Field(description="Text content")
    bbox: Optional[BoundingBox] = Field(default=None, description="Bounding box coordinates")
    confidence: Optional[float] = Field(default=None, description="Confidence score 0-1 for text extraction quality")
    source: Optional[str] = Field(default=None, description="Source of text extraction (e.g., 'ocr', 'pdf_text')")

class SheetMetadata(BaseModel):
    """Sheet metadata extracted from PDF"""
    sheet_id: Optional[str] = Field(default=None, description="Sheet identifier (e.g., P-102)")
    discipline: Optional[str] = Field(default=None, description="Engineering discipline")
    floor: Optional[str] = Field(default=None, description="Floor or level designation")
    title: Optional[str] = Field(default=None, description="Drawing title")
    scale: Optional[str] = Field(default=None, description="Drawing scale")
    date: Optional[str] = Field(default=None, description="Drawing date")

class AgentInput(BaseModel):
    """Structured input format for agents"""
    sheet_id: Optional[str] = Field(default=None, description="Sheet identifier")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image")
    image_path: Optional[str] = Field(default=None, description="Path to image file")
    legend: List[LegendEntry] = Field(default_factory=list, description="Legend entries")
    text_blocks: List[TextBlock] = Field(default_factory=list, description="Text blocks from PDF")
    sheet_metadata: SheetMetadata = Field(default_factory=SheetMetadata, description="Sheet metadata")
    notes: List[str] = Field(default_factory=list, description="General notes and instructions")
    context_from_previous: Optional[Dict[str, Any]] = Field(default=None, description="Context from previous agent")

class MaterialUnit(str, Enum):
    """Standard units for materials"""
    EA = "EA"  # Each
    LF = "LF"  # Linear Feet
    SF = "SF"  # Square Feet
    CF = "CF"  # Cubic Feet
    LS = "LS"  # Lump Sum
    GAL = "GAL"  # Gallons
    LB = "LB"  # Pounds
    TON = "TON"  # Tons

class MaterialItem(BaseModel):
    """Individual material item in structured output"""
    item_name: str = Field(description="Name/description of the material item")
    quantity: Optional[Union[int, float]] = Field(default=None, description="Quantity needed (None if not determinable from document)")
    unit: Optional[MaterialUnit] = Field(default=None, description="Unit of measurement (None if not determinable from document)")
    reference_sheet: Optional[str] = Field(default=None, description="Sheet where item was found")
    zone: Optional[str] = Field(default=None, description="Zone or area designation")
    size: Optional[str] = Field(default=None, description="Size specification (e.g., '6 inch')")
    specification: Optional[str] = Field(default=None, description="Material specification")
    confidence: Optional[float] = Field(default=None, description="Confidence score 0-1")
    notes: Optional[str] = Field(default=None, description="Additional notes")

class ContextOutput(BaseModel):
    """Context agent output format"""
    sheet_metadata: SheetMetadata = Field(description="Extracted sheet metadata")
    legend: List[LegendEntry] = Field(description="Complete legend entries")
    drawing_types: List[str] = Field(default_factory=list, description="Types of drawings identified")
    annotation_systems: Dict[str, Optional[str]] = Field(default_factory=dict, description="Annotation and numbering systems")
    technical_standards: List[str] = Field(default_factory=list, description="Referenced standards")
    document_organization: Dict[str, Any] = Field(default_factory=dict, description="Document organization info")

class PlumbingOutput(BaseModel):
    """Plumbing agent output format"""
    materials: List[MaterialItem] = Field(description="List of plumbing materials")
    special_requirements: List[str] = Field(default_factory=list, description="Special installation requirements")
    potential_issues: List[str] = Field(default_factory=list, description="Potential concerns or issues")
    summary: Optional[str] = Field(default=None, description="Analysis summary")

class AgentMessage(BaseModel):
    """Message format for agent-to-agent communication"""
    from_agent: str = Field(description="Source agent name")
    to_agent: str = Field(description="Target agent name")
    message_type: str = Field(description="Type of message (question, clarification, etc.)")
    content: str = Field(description="Message content")
    reference_data: Optional[Dict[str, Any]] = Field(default=None, description="Referenced data")

class ProcessingError(BaseModel):
    """Structured error information for granular tracking"""
    agent: str = Field(description="Agent name that encountered the error")
    sheet_id: Optional[str] = Field(default=None, description="Sheet ID where error occurred")
    error_type: str = Field(description="Type of error (processing, validation, communication, etc.)")
    error_message: str = Field(description="Detailed error message")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="When the error occurred")

class ProcessingStatus(BaseModel):
    """Processing status for tracking"""
    stage: str = Field(description="Current processing stage")
    progress: float = Field(description="Progress percentage 0-100")
    message: str = Field(description="Status message")
    current_sheet: Optional[str] = Field(default=None, description="Currently processing sheet")
    errors: List[ProcessingError] = Field(default_factory=list, description="Structured errors encountered")
    total_errors: int = Field(default=0, description="Total number of errors encountered")
    agent_error_summary: Dict[str, int] = Field(default_factory=dict, description="Error count per agent")
    sheet_error_summary: Dict[str, int] = Field(default_factory=dict, description="Error count per sheet")

class AnalysisResult(BaseModel):
    """Complete analysis result"""
    task_id: str = Field(description="Unique task identifier")
    sheets_processed: List[str] = Field(description="List of processed sheet IDs")
    context_results: List[ContextOutput] = Field(description="Context extraction results per sheet")
    plumbing_results: List[PlumbingOutput] = Field(description="Plumbing analysis results per sheet")
    consolidated_materials: List[MaterialItem] = Field(description="Consolidated material list")
    processing_status: ProcessingStatus = Field(description="Final processing status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class PreprocessorOutput(BaseModel):
    """Preprocessor agent output for sheet classification"""
    sheet_type: str = Field(description="Type of sheet (plumbing, mechanical, etc.)")
    complexity_score: float = Field(description="Complexity score 0-1")
    recommended_agents: List[str] = Field(description="Recommended analysis agents")
    processing_notes: List[str] = Field(default_factory=list, description="Processing recommendations") 