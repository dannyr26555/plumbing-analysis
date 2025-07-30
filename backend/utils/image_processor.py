"""
Image Processing Utility Module
Provides streamlined image processing operations for agent consumption and PDF conversion
"""

import logging
from PIL import Image, ImageFilter
from pathlib import Path
from typing import Optional, Tuple
from autogen_core import Image as AutoGenImage
from config.app_config import AppConfig

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Utility class for standardized image processing operations"""
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize image processor with configuration
        
        Args:
            config: Application configuration (will use AppConfig if not provided)
        """
        self.config = config or AppConfig
        
        # Get configuration values
        if hasattr(self.config, '_initialized') and self.config._initialized:
            self.max_image_size = self.config.MAX_IMAGE_SIZE
            self.ai_optimized_size = self.config.AI_OPTIMIZED_IMAGE_SIZE
            self.enable_sharpening = self.config.PDF_ENABLE_SHARPENING
        else:
            # Fallback defaults if config not initialized
            self.max_image_size = 1536
            self.ai_optimized_size = 768
            self.enable_sharpening = True
            
        logger.debug(f"ImageProcessor initialized: max_size={self.max_image_size}, ai_optimized_size={self.ai_optimized_size}, sharpening={self.enable_sharpening}")
    
    def process_image_for_agent(self, image_path: str, max_size: Optional[int] = None, optimize_for_tokens: bool = True) -> AutoGenImage:
        """
        Process image for agent consumption with standardized resizing and optimization
        
        Args:
            image_path: Path to the image file
            max_size: Maximum dimension (overrides config default)
            optimize_for_tokens: Whether to aggressively optimize for token usage
            
        Returns:
            AutoGenImage instance ready for agent use
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: If image processing fails
        """
        # Use much smaller dimensions for token optimization
        if optimize_for_tokens:
            max_dimension = self.ai_optimized_size  # Configurable size to save ~75% tokens
        else:
            max_dimension = max_size or self.max_image_size
        
        try:
            # Validate file exists
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            with Image.open(image_path) as pil_image:
                # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Resize if image is too large
                if pil_image.size[0] > max_dimension or pil_image.size[1] > max_dimension:
                    logger.debug(f"Resizing image from {pil_image.size} to fit within {max_dimension}x{max_dimension} (token optimized: {optimize_for_tokens})")
                    pil_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                
                # Apply additional token optimization
                if optimize_for_tokens:
                    pil_image = self._optimize_for_ai_analysis(pil_image)
                
                # Estimate token usage for monitoring
                token_estimate = self.estimate_token_usage(pil_image)
                
                # Create AutoGenImage instance
                autogen_image = AutoGenImage(pil_image)
                
                logger.info(f"Successfully processed image -> {pil_image.size} (~{token_estimate:,} tokens, optimized: {optimize_for_tokens})")
                return autogen_image
                
        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}")
            raise Exception(f"Image processing failed: {str(e)}")
    
    def _optimize_for_ai_analysis(self, image: Image.Image) -> Image.Image:
        """
        Apply aggressive optimization specifically for AI token reduction while preserving analysis quality
        
        Args:
            image: PIL Image to optimize
            
        Returns:
            Optimized PIL Image with reduced token footprint
        """
        try:
            # Apply sharpening first to enhance details before compression
            if self.enable_sharpening:
                image = self.apply_sharpening(image, radius=0.8, percent=120, threshold=1)
            
            # Enhance contrast slightly to improve text/symbol readability at lower resolution
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)  # 10% contrast boost
            
            logger.debug("Applied AI analysis optimization: sharpening + contrast enhancement")
            return image
            
        except Exception as e:
            logger.warning(f"Failed to apply AI optimization: {e}, returning original image")
            return image
    
    def estimate_token_usage(self, image: Image.Image) -> int:
        """
        Estimate the approximate token usage for an image when base64 encoded
        
        Args:
            image: PIL Image to estimate tokens for
            
        Returns:
            Estimated token count (rough approximation)
        """
        try:
            # Rough estimation: base64 encoding increases size by ~33%
            # Typical compression ratio for PNG is 2-4x
            # Each token is roughly 4 characters
            width, height = image.size
            pixel_count = width * height * 3  # RGB
            estimated_base64_size = pixel_count * 0.4  # Compressed + base64
            estimated_tokens = int(estimated_base64_size / 4)  # 4 chars per token
            
            logger.debug(f"Image {width}x{height} estimated ~{estimated_tokens:,} tokens")
            return estimated_tokens
            
        except Exception as e:
            logger.warning(f"Failed to estimate token usage: {e}")
            return 0
    
    def apply_sharpening(self, image: Image.Image, radius: float = 0.5, percent: int = 110, threshold: int = 2) -> Image.Image:
        """
        Apply unsharp mask sharpening to enhance symbol clarity
        
        Args:
            image: PIL Image to sharpen
            radius: Sharpening radius (default: 0.5 for construction plans)
            percent: Sharpening strength (default: 110%)
            threshold: Sharpening threshold (default: 2)
            
        Returns:
            Sharpened PIL Image
        """
        try:
            if not self.enable_sharpening:
                logger.debug("Sharpening disabled in configuration")
                return image
            
            # Apply UnsharpMask with conservative settings for construction drawings
            sharpened = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
            logger.debug(f"Applied sharpening filter: radius={radius}, percent={percent}, threshold={threshold}")
            return sharpened
            
        except Exception as e:
            logger.warning(f"Failed to apply sharpening filter: {e}, returning original image")
            return image
    
    def resize_with_aspect_ratio(self, image: Image.Image, max_width: int, max_height: int) -> Image.Image:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: PIL Image to resize
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            Resized PIL Image
        """
        original_width, original_height = image.size
        
        # Calculate aspect ratio
        aspect_ratio = original_width / original_height
        
        # Determine new dimensions
        if original_width > max_width or original_height > max_height:
            if aspect_ratio > 1:  # Landscape
                new_width = min(max_width, original_width)
                new_height = int(new_width / aspect_ratio)
            else:  # Portrait
                new_height = min(max_height, original_height)
                new_width = int(new_height * aspect_ratio)
            
            # Ensure we don't exceed maximums
            if new_width > max_width:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
            if new_height > max_height:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
            
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.debug(f"Resized image: {original_width}x{original_height} -> {new_width}x{new_height}")
            return resized_image
        
        return image
    
    def optimize_for_analysis(self, image_path: str, enhance_symbols: bool = True) -> Image.Image:
        """
        Optimize image specifically for construction plan analysis
        
        Args:
            image_path: Path to the image file
            enhance_symbols: Whether to apply symbol enhancement filters
            
        Returns:
            Optimized PIL Image
        """
        try:
            with Image.open(image_path) as pil_image:
                # Convert to RGB for consistent processing
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Apply sharpening if enabled and requested
                if enhance_symbols and self.enable_sharpening:
                    pil_image = self.apply_sharpening(pil_image)
                
                # Resize if necessary
                pil_image = self.resize_with_aspect_ratio(pil_image, self.max_image_size, self.max_image_size)
                
                logger.debug("Optimized image for analysis")
                return pil_image.copy()  # Return a copy to avoid issues with context manager
                
        except Exception as e:
            logger.error(f"Failed to optimize image: {str(e)}")
            raise Exception(f"Image optimization failed: {str(e)}")
    
    def get_image_info(self, image_path: str) -> dict:
        """
        Get basic information about an image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image information
        """
        try:
            with Image.open(image_path) as pil_image:
                return {
                    "path": image_path,
                    "size": pil_image.size,
                    "mode": pil_image.mode,
                    "format": pil_image.format,
                    "width": pil_image.size[0],
                    "height": pil_image.size[1],
                    "file_size": Path(image_path).stat().st_size if Path(image_path).exists() else 0
                }
        except Exception as e:
            logger.error(f"Failed to get image info: {str(e)}")
            return {
                "path": image_path,
                "error": str(e)
            }
    
    def batch_process_for_agents(self, image_paths: list, max_size: Optional[int] = None) -> list:
        """
        Process multiple images for agent consumption
        
        Args:
            image_paths: List of image file paths
            max_size: Maximum dimension for all images
            
        Returns:
            List of AutoGenImage instances
        """
        processed_images = []
        max_dimension = max_size or self.max_image_size
        
        for image_path in image_paths:
            try:
                autogen_image = self.process_image_for_agent(image_path, max_dimension, optimize_for_tokens=True)
                processed_images.append(autogen_image)
            except Exception as e:
                logger.error(f"Failed to process image in batch: {str(e)}")
                # Continue processing other images
                continue
        
        logger.info(f"Batch processed {len(processed_images)}/{len(image_paths)} images successfully")
        return processed_images
    
    def validate_image_file(self, image_path: str) -> bool:
        """
        Validate that a file is a valid image that can be processed
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if valid image, False otherwise
        """
        try:
            if not Path(image_path).exists():
                return False
                
            with Image.open(image_path) as pil_image:
                # Try to load the image
                pil_image.verify()
                return True
                
        except Exception as e:
            logger.debug(f"Image validation failed: {str(e)}")
            return False 