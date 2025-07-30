"""
Prompt Manager Module
Handles loading and assembly of modular prompt templates from external files
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages loading and assembly of modular prompt templates"""
    
    CONFIDENCE_GUIDANCE = """NOTES DOCUMENTATION FOR CONFIDENCE ENHANCEMENT:
In the notes field, clearly document your evidence and measurement approach using specific keywords:
- EXPLICIT EVIDENCE: "measured from dimension label", "quantity callout shows X", "detail schedule specifies", "dimension label indicates"
- VISUAL ESTIMATES: "counted X visible symbols", "traced route length", "estimated from plan view", "visible pipe segments"
- INFERRED/UNCERTAIN: "mentioned", "inferred", "assumed", "unclear", "blurry", "partial"
- SYMBOL MATCHING: "matches legend symbol [X]", "identified from legend as [description]", "symbol defined in legend"
- ANNOTATION QUALITY: "clear dimension label", "explicit quantity callout", "well-marked", "unclear marking", "partial annotation"

QUALITY ASSURANCE:
- If context agent found symbols but you see more instances, count what YOU see
- If you find materials not in the context agent's legend, include them with notes
- Only provide quantities/units when there's clear evidence in the document
- Use confidence scores to reflect uncertainty (0.0 = very uncertain, 1.0 = very certain)
- When in doubt about quantity or unit, leave as null and note the limitation

BEFORE RETURNING YOUR RESULTS:
- Review how each quantity was determined: explicitly labeled, visually counted, or estimated
- Adjust confidence score based on measurement method:
  • 0.9–1.0 → clearly labeled quantity with dimension callouts or schedules
  • 0.6–0.8 → visual symbol count with clear annotations
  • 0.3–0.5 → layout-based estimates or unclear markings
  • <0.3   → unclear, speculative, or inferred quantities
- Add detailed notes field per item explaining uncertainty, method, and evidence source
- Ensure compliance with MATERIAL EXTRACTION RULES (see above)"""

    SYMBOL_MATCHING_RULES = """SYMBOL MATCHING RULES:
- PRIMARY RULE: When counting material instances, use ONLY symbols defined in the legend from the context agent
- UNMATCHED SYMBOLS: If a symbol is visible but not defined in the context agent's legend, include it with confidence < 0.5 and add note: "Unmatched symbol - not found in legend"
- LEGEND VALIDATION: Cross-reference every counted item against the context agent's legend entries
- UNKNOWN SYMBOLS: For symbols not in the legend, provide best interpretation but clearly flag uncertainty"""
    
    def __init__(self, prompts_dir: Optional[Path] = None, preload_templates: bool = True):
        """
        Initialize prompt manager
        
        Args:
            prompts_dir: Directory containing prompt template files
            preload_templates: Whether to preload known templates for performance
        """
        if prompts_dir is None:
            # Default to prompts directory relative to this file
            self.prompts_dir = Path(__file__).parent / "prompts"
        else:
            self.prompts_dir = prompts_dir
            
        self._template_cache: Dict[str, str] = {}
        self._system_message_cache: Dict[str, str] = {}
        
        logger.debug(f"PromptManager initialized with prompts directory: {self.prompts_dir}")
        
        # Preload known templates for better performance
        if preload_templates:
            self._preload_known_templates()
    
    def _preload_known_templates(self) -> None:
        """
        Preload known static templates for better performance
        """
        known_templates = [
            "json_response_format.txt",
            "material_extraction_rules.txt", 
            "context_json_format.txt",
            "context_agent_base.txt",
            "plumbing_agent_base.txt"
        ]
        
        preloaded_count = 0
        for template_name in known_templates:
            try:
                self._load_template(template_name, use_cache=True)
                preloaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to preload template {template_name}: {e}")
        
        logger.debug(f"Preloaded {preloaded_count}/{len(known_templates)} templates")
    
    def _load_template(self, template_name: str, use_cache: bool = True) -> str:
        """
        Load a template file from disk with whitespace normalization
        
        Args:
            template_name: Name of the template file (with or without .txt extension)
            use_cache: Whether to use cached templates
            
        Returns:
            Template content as normalized string
        """
        # Add .txt extension if not present
        if not template_name.endswith('.txt'):
            template_name += '.txt'
            
        # Check cache first
        if use_cache and template_name in self._template_cache:
            return self._template_cache[template_name]
        
        template_path = self.prompts_dir / template_name
        
        try:
            if not template_path.exists():
                logger.warning(f"Template file not found: {template_name}")
                return f"[Missing: {template_name}]"
            
            with open(template_path, 'r', encoding='utf-8') as f:
                # Trim and normalize whitespace for token efficiency
                content = f.read().strip().replace("  ", " ").replace("\r\n", "\n")
                
            # Cache the template
            if use_cache:
                self._template_cache[template_name] = content
                
            logger.debug(f"Loaded template: {template_name}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to load template {template_name}: {str(e)}")
            return f"[Error: {template_name}]"
    
    def get_material_extraction_rules(self) -> str:
        """Get the material extraction rules template"""
        return self._load_template("material_extraction_rules.txt")
    
    def get_symbol_matching_rules(self) -> str:
        """Get the symbol matching rules (static constant)"""
        return self.SYMBOL_MATCHING_RULES
    
    def get_confidence_guidance(self) -> str:
        """Get the confidence guidance (static constant)"""
        return self.CONFIDENCE_GUIDANCE
    
    def get_json_response_format(self) -> str:
        """Get the JSON response format template"""
        return self._load_template("json_response_format.txt")
    
    def get_context_json_format(self) -> str:
        """Get the context-specific JSON response format template"""
        return self._load_template("context_json_format.txt")
    
    def build_context_system_message(self) -> str:
        """
        Build the complete context agent system message from modular templates
        
        Returns:
            Complete system message for context agent
        """
        cache_key = "context_agent_system_message"
        
        if cache_key in self._system_message_cache:
            return self._system_message_cache[cache_key]
        
        try:
            # Load the base template
            base_template = self._load_template("context_agent_base.txt")
            
            # Load supporting templates
            json_format = self.get_context_json_format()
            
            # Assemble the complete message
            complete_message = base_template.format(
                json_format=json_format
            )
            
            # Cache the result
            self._system_message_cache[cache_key] = complete_message
            
            logger.debug("Built context agent system message from templates")
            return complete_message
            
        except Exception as e:
            logger.error(f"Failed to build context system message: {str(e)}")
            return "[Error: context system message]"
    
    def build_plumbing_system_message(self) -> str:
        """
        Build the complete plumbing agent system message from modular templates
        
        Returns:
            Complete system message for plumbing agent
        """
        cache_key = "plumbing_agent_system_message"
        
        if cache_key in self._system_message_cache:
            return self._system_message_cache[cache_key]
        
        try:
            # Load the base template
            base_template = self._load_template("plumbing_agent_base.txt")
            
            # Load supporting templates
            json_format = self.get_json_response_format()
            material_extraction_rules = self.get_material_extraction_rules()
            symbol_matching_rules = self.get_symbol_matching_rules()
            confidence_guidance = self.get_confidence_guidance()
            
            # Assemble the complete message
            complete_message = base_template.format(
                json_format=json_format,
                material_extraction_rules=material_extraction_rules,
                symbol_matching_rules=symbol_matching_rules,
                confidence_guidance=confidence_guidance
            )
            
            # Cache the result
            self._system_message_cache[cache_key] = complete_message
            
            logger.debug("Built plumbing agent system message from templates")
            return complete_message
            
        except Exception as e:
            logger.error(f"Failed to build plumbing system message: {str(e)}")
            return "[Error: plumbing system message]"
    
    def get_system_messages(self) -> Dict[str, str]:
        """
        Get all system messages for the standard agents
        
        Returns:
            Dictionary mapping agent types to their system messages
        """
        return {
            "context": self.build_context_system_message(),
            "plumbing": self.build_plumbing_system_message()
        }
    
    def clear_cache(self) -> None:
        """Clear all cached templates and system messages"""
        self._template_cache.clear()
        self._system_message_cache.clear()
        logger.info("Cleared all prompt template caches")
    
    def reload_templates(self) -> None:
        """Reload all templates from disk (clears cache first)"""
        self.clear_cache()
        logger.info("Reloaded all prompt templates from disk")
    
    def get_template_info(self) -> Dict[str, any]:
        """
        Get information about loaded templates
        
        Returns:
            Dictionary with template information
        """
        template_files = []
        if self.prompts_dir.exists():
            template_files = [f.name for f in self.prompts_dir.glob("*.txt")]
        
        return {
            "prompts_directory": str(self.prompts_dir),
            "directory_exists": self.prompts_dir.exists(),
            "available_templates": template_files,
            "cached_templates": list(self._template_cache.keys()),
            "cached_system_messages": list(self._system_message_cache.keys())
        }
    
    def validate_templates(self) -> Dict[str, bool]:
        """
        Validate that all required template files exist
        
        Returns:
            Dictionary mapping template names to existence status
        """
        required_templates = [
            "material_extraction_rules.txt",
            "json_response_format.txt",
            "context_agent_base.txt",
            "context_json_format.txt",
            "plumbing_agent_base.txt"
        ]
        
        validation_results = {}
        for template in required_templates:
            template_path = self.prompts_dir / template
            validation_results[template] = template_path.exists()
        
        missing_templates = [name for name, exists in validation_results.items() if not exists]
        if missing_templates:
            logger.warning(f"Missing template files: {missing_templates}")
        else:
            logger.info("All required template files are present")
        
        return validation_results 