"""
Application Configuration Module
Centralizes environment variable access with validation and sensible defaults
"""

import os
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AppConfig:
    """Centralized application configuration with environment variable management"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_DEPLOYMENT_NAME: str = ""
    AZURE_OPENAI_API_VERSION: str = ""
    
    # PDF Processing Configuration
    PDF_RENDER_DPI: int = 400  # Conservative default for quality/performance balance
    PDF_MAX_DPI: int = 900     # Conservative limit to prevent memory issues
    PDF_ENABLE_SHARPENING: bool = True
    
    # Application URLs
    FRONTEND_URL: str = "http://localhost:3000"
    BACKEND_URL: str = "http://localhost:8000"
    
    # Processing Configuration
    CONFIDENCE_THRESHOLD: float = 0.6
    MAX_IMAGE_SIZE: int = 1536  # Maximum image dimension for agent processing
    AI_OPTIMIZED_IMAGE_SIZE: int = 768  # Smaller size for AI analysis to reduce token usage
    
    _initialized: bool = False
    
    @classmethod
    def initialize(cls, env_file: Optional[Path] = None) -> None:
        """
        Initialize configuration from environment variables
        
        Args:
            env_file: Optional path to .env file (defaults to project root)
        """
        if cls._initialized:
            return
            
        try:
            # Load environment variables
            if env_file is None:
                # Default to project root .env file
                current_file = Path(__file__).resolve()
                env_file = current_file.parent.parent.parent / '.env'
            
            if env_file.exists():
                load_dotenv(env_file)
                logger.info("Loaded environment configuration from .env file")
            else:
                logger.warning("Environment file not found - using system environment variables")
            
            # Load Azure OpenAI configuration
            cls.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
            cls.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
            cls.AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
            cls.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01").strip()
            
            # Load PDF processing configuration
            cls.PDF_RENDER_DPI = int(os.getenv('PDF_RENDER_DPI', 400))
            cls.PDF_MAX_DPI = int(os.getenv('PDF_MAX_DPI', 900))
            cls.PDF_ENABLE_SHARPENING = os.getenv('PDF_ENABLE_SHARPENING', 'true').lower() in ('true', '1', 'yes', 'on')
            
            # Load application URLs
            cls.FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
            cls.BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
            
            # Load processing configuration
            cls.CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))
            cls.MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 1536))
            cls.AI_OPTIMIZED_IMAGE_SIZE = int(os.getenv("AI_OPTIMIZED_IMAGE_SIZE", 768))
            
            cls._initialized = True
            logger.info("Application configuration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize application configuration: {str(e)}")
            raise RuntimeError("Application configuration initialization failed")
    
    @classmethod
    def validate_required(cls) -> None:
        """Validate that all required configuration is present"""
        if not cls._initialized:
            raise RuntimeError("Configuration not initialized. Call AppConfig.initialize() first.")
        
        required_fields = {
            'AZURE_OPENAI_ENDPOINT': cls.AZURE_OPENAI_ENDPOINT,
            'AZURE_OPENAI_API_KEY': cls.AZURE_OPENAI_API_KEY,
            'AZURE_OPENAI_DEPLOYMENT_NAME': cls.AZURE_OPENAI_DEPLOYMENT_NAME,
        }
        
        missing = [key for key, value in required_fields.items() if not value]
        if missing:
            raise ValueError(f"Missing required configuration: {missing}")
        
        logger.info("Configuration validation passed")
    
    @classmethod
    def get_azure_endpoint_url(cls) -> str:
        """Get properly formatted Azure OpenAI endpoint URL"""
        if not cls._initialized:
            raise RuntimeError("Configuration not initialized")
            
        endpoint = cls.AZURE_OPENAI_ENDPOINT
        
        # Ensure proper format
        if not endpoint.startswith('https://'):
            endpoint = f'https://{endpoint}'
        if not endpoint.endswith('/'):
            endpoint += '/'
            
        return f"{endpoint}openai/deployments/{cls.AZURE_OPENAI_DEPLOYMENT_NAME}"
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """Get configuration summary for logging (excluding sensitive data)"""
        if not cls._initialized:
            return {"status": "not_initialized"}
            
        return {
            "status": "initialized",
            "azure_endpoint": cls.AZURE_OPENAI_ENDPOINT[:20] + "..." if cls.AZURE_OPENAI_ENDPOINT else "not_set",
            "deployment_name": cls.AZURE_OPENAI_DEPLOYMENT_NAME,
            "api_version": cls.AZURE_OPENAI_API_VERSION,
            "pdf_render_dpi": cls.PDF_RENDER_DPI,
            "pdf_max_dpi": cls.PDF_MAX_DPI,
            "pdf_sharpening": cls.PDF_ENABLE_SHARPENING,
            "frontend_url": cls.FRONTEND_URL,
            "backend_url": cls.BACKEND_URL,
            "confidence_threshold": cls.CONFIDENCE_THRESHOLD,
            "max_image_size": cls.MAX_IMAGE_SIZE
        } 