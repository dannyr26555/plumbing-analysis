"""
Agent Response Parser Module
Provides centralized JSON response parsing with standardized error handling and validation
"""

import json
import logging
import re
from typing import Type, Dict, Any, List, Optional, Union
from pydantic import BaseModel, ValidationError
from models import ContextOutput, PlumbingOutput, ProcessingError

logger = logging.getLogger(__name__)

class AgentResponseParser:
    """Centralized parser for agent responses with validation and error handling"""
    
    @staticmethod
    def clean_json_response(content: str) -> str:
        """
        Clean agent response content to extract valid JSON
        
        Args:
            content: Raw response content from agent
            
        Returns:
            Cleaned JSON string
        """
        if not content or not isinstance(content, str):
            raise ValueError("Invalid content: must be non-empty string")
        
        content = content.strip()
        
        # Remove markdown code blocks if present
        if "```json" in content:
            # Extract content between ```json and ```
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
        elif "```" in content:
            # Extract content between generic code blocks
            json_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
        
        # Remove any leading/trailing text that's not JSON
        # Look for the first { and last }
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            content = content[start_idx:end_idx + 1]
        
        # Additional cleaning to remove common formatting issues
        # Remove any null bytes that might interfere with parsing
        content = content.replace('\x00', '')
        
        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove any BOM (Byte Order Mark) characters
        content = content.lstrip('\ufeff')
        
        return content
    
    @staticmethod
    def parse_json_safely(content: str) -> Dict[str, Any]:
        """
        Parse JSON content with error handling
        
        Args:
            content: JSON string to parse
            
        Returns:
            Parsed dictionary
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            fixed_content = AgentResponseParser._attempt_json_fix(content)
            if fixed_content != content:
                try:
                    return json.loads(fixed_content)
                except json.JSONDecodeError:
                    pass  # Fall through to original error
            
            logger.error("JSON parsing failed - content could not be parsed")
            raise e
    
    @staticmethod
    def _attempt_json_fix(content: str) -> str:
        """
        Attempt to fix common JSON formatting issues
        
        Args:
            content: Potentially malformed JSON string
            
        Returns:
            Fixed JSON string
        """
        # Fix trailing commas before closing brackets/braces
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        # Fix missing quotes around unquoted property names
        # Look for word characters at start of line or after { or , that are followed by :
        content = re.sub(r'(^|[{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', content, flags=re.MULTILINE)
        
        # Fix single quotes to double quotes for strings
        # But preserve single quotes inside double-quoted strings
        def replace_single_quotes(match):
            # If we're inside double quotes, don't replace
            return match.group(0).replace("'", '"')
        
        # Simple single quote to double quote replacement
        content = re.sub(r"'([^']*)'", r'"\1"', content)
        
        # Fix missing commas between array/object elements
        # Add comma between } and { or between } and "
        content = re.sub(r'}\s*(?=["{])', r'},', content)
        content = re.sub(r'"\s*(?=")', r'",', content)
        
        # Fix common escaped character issues
        content = content.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
        
        return content
    
    @staticmethod
    def validate_with_model(data: Dict[str, Any], model_class: Type[BaseModel]) -> BaseModel:
        """
        Validate parsed data against Pydantic model
        
        Args:
            data: Parsed dictionary data
            model_class: Pydantic model class for validation
            
        Returns:
            Validated model instance
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            return model_class(**data)
        except ValidationError as e:
            logger.error(f"Model validation failed for {model_class.__name__}: {str(e)}")
            logger.error("Data failed model validation - check response structure")
            raise e
    
    @staticmethod
    def _clean_none_values_for_context(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean up None values in ContextOutput fields that should be lists or dicts
        
        Args:
            data: Response data dictionary
            
        Returns:
            Cleaned data dictionary
        """
        # Fix None values for list fields that should have empty lists as default
        list_fields = ['technical_standards', 'drawing_types', 'legend']
        for field in list_fields:
            if field in data and data[field] is None:
                data[field] = []
                logger.info(f"Converted None to empty list for field: {field}")
        
        # Fix None values for dict fields that should have empty dicts as default
        dict_fields = ['annotation_systems', 'document_organization']
        for field in dict_fields:
            if field in data and data[field] is None:
                data[field] = {}
                logger.info(f"Converted None to empty dict for field: {field}")
        
        # Special handling for annotation_systems if it's a dict with None values
        if 'annotation_systems' in data and isinstance(data['annotation_systems'], dict):
            cleaned_annotation_systems = {}
            for key, value in data['annotation_systems'].items():
                cleaned_annotation_systems[key] = value  # Allow None values as model now supports them
            data['annotation_systems'] = cleaned_annotation_systems
        
        return data

    @staticmethod
    def normalize_field_names(data: Dict[str, Any], model_class: Type[BaseModel]) -> Dict[str, Any]:
        """
        Normalize field names to match model expectations and clean problematic values
        
        Args:
            data: Dictionary with potentially misnamed fields
            model_class: Pydantic model class to match against
            
        Returns:
            Dictionary with normalized field names and cleaned values
        """
        # Apply model-specific cleaning for ContextOutput
        if model_class.__name__ == 'ContextOutput':
            data = AgentResponseParser._clean_none_values_for_context(data)
        
        # Get expected field names from the model
        expected_fields = set(model_class.model_fields.keys())
        
        # Create a mapping of common field name variations
        field_mappings = {
            # Context model field mappings
            'legends': 'legend',  # Common AI mistake: legends vs legend
            'symbols': 'legend',  # Alternative naming
            # Add more mappings as needed
        }
        
        # Apply mappings
        normalized_data = {}
        for key, value in data.items():
            # Check if we need to map this field name
            if key in field_mappings and field_mappings[key] in expected_fields:
                normalized_key = field_mappings[key]
                logger.info(f"Normalizing field name: '{key}' → '{normalized_key}'")
                normalized_data[normalized_key] = value
            else:
                normalized_data[key] = value
        
        return normalized_data

    @staticmethod
    def parse_agent_response(
        content: str, 
        model_class: Type[BaseModel], 
        agent_name: str, 
        sheet_id: Optional[str] = None,
        error_list: Optional[List[ProcessingError]] = None
    ) -> Union[Dict[str, Any], None]:
        """
        Parse and validate agent response with comprehensive error handling
        
        Args:
            content: Raw response content from agent
            model_class: Pydantic model class for validation
            agent_name: Name of the agent that generated the response
            sheet_id: Optional sheet ID for error tracking
            error_list: Optional list to append errors to
            
        Returns:
            Validated response data as dictionary, or None if parsing failed
        """
        try:
            # Clean the response content
            cleaned_content = AgentResponseParser.clean_json_response(content)
            
            # Parse JSON
            parsed_data = AgentResponseParser.parse_json_safely(cleaned_content)
            
            # Normalize field names to match model expectations
            normalized_data = AgentResponseParser.normalize_field_names(parsed_data, model_class)
            
            # Validate against model
            validated_model = AgentResponseParser.validate_with_model(normalized_data, model_class)
            
            logger.info(f"Successfully parsed {agent_name} response for sheet {sheet_id}")
            return validated_model.model_dump()
            
        except (ValueError, json.JSONDecodeError) as e:
            error_msg = f"JSON parsing failed: {str(e)}"
            logger.error(f"{agent_name} response parsing failed for sheet {sheet_id}: {error_msg}")
            
            if error_list is not None:
                AgentResponseParser._add_parsing_error(
                    error_list, agent_name, sheet_id, "json_parsing", error_msg
                )
            
            return None
            
        except ValidationError as e:
            error_msg = f"Model validation failed: {str(e)}"
            logger.error(f"{agent_name} response validation failed for sheet {sheet_id}: {error_msg}")
            
            if error_list is not None:
                AgentResponseParser._add_parsing_error(
                    error_list, agent_name, sheet_id, "validation", error_msg
                )
            
            return None
            
        except Exception as e:
            error_msg = f"Unexpected parsing error: {str(e)}"
            logger.error(f"{agent_name} response parsing failed for sheet {sheet_id}: {error_msg}")
            
            if error_list is not None:
                AgentResponseParser._add_parsing_error(
                    error_list, agent_name, sheet_id, "unexpected", error_msg
                )
            
            return None
    
    @staticmethod
    def _add_parsing_error(
        error_list: List[ProcessingError], 
        agent_name: str, 
        sheet_id: Optional[str], 
        error_type: str, 
        error_message: str
    ) -> None:
        """Add parsing error to error list"""
        error = ProcessingError(
            agent=agent_name,
            sheet_id=sheet_id,
            error_type=error_type,
            error_message=error_message
        )
        error_list.append(error)
    
    @staticmethod
    def get_fallback_response(model_class: Type[BaseModel]) -> Dict[str, Any]:
        """
        Get fallback response for failed parsing
        
        Args:
            model_class: Pydantic model class to create fallback for
            
        Returns:
            Fallback response dictionary
        """
        if model_class == ContextOutput:
            return {
                "sheet_metadata": {
                    "sheet_id": None,
                    "discipline": None,
                    "floor": None,
                    "title": None,
                    "scale": None,
                    "date": None
                },
                "legend": [],
                "drawing_types": [],
                "annotation_systems": {},
                "technical_standards": [],
                "document_organization": {}
            }
        elif model_class == PlumbingOutput:
            return {
                "materials": [],
                "special_requirements": [],
                "potential_issues": [],
                "summary": "Analysis failed - no materials extracted"
            }
        else:
            # Generic fallback - create instance with minimal data
            try:
                return model_class().model_dump()
            except Exception:
                logger.warning(f"Could not create fallback for {model_class.__name__}")
                return {}
    
    @staticmethod
    def parse_context_response(
        content: str, 
        sheet_id: Optional[str] = None,
        error_list: Optional[List[ProcessingError]] = None
    ) -> Dict[str, Any]:
        """
        Parse context agent response with fallback
        
        Args:
            content: Raw response content
            sheet_id: Optional sheet ID for error tracking
            error_list: Optional error list for tracking
            
        Returns:
            Parsed context data or fallback
        """
        result = AgentResponseParser.parse_agent_response(
            content, ContextOutput, "context", sheet_id, error_list
        )
        
        if result is None:
            logger.warning(f"Using fallback context response for sheet {sheet_id}")
            result = AgentResponseParser.get_fallback_response(ContextOutput)
        
        return result
    
    @staticmethod
    def parse_plumbing_response(
        content: str, 
        sheet_id: Optional[str] = None,
        error_list: Optional[List[ProcessingError]] = None
    ) -> Dict[str, Any]:
        """
        Parse plumbing agent response with fallback
        
        Args:
            content: Raw response content
            sheet_id: Optional sheet ID for error tracking
            error_list: Optional error list for tracking
            
        Returns:
            Parsed plumbing data or fallback
        """
        result = AgentResponseParser.parse_agent_response(
            content, PlumbingOutput, "plumbing", sheet_id, error_list
        )
        
        if result is None:
            logger.warning(f"Using fallback plumbing response for sheet {sheet_id}")
            result = AgentResponseParser.get_fallback_response(PlumbingOutput)
        
        return result 