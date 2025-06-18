from fastapi import FastAPI, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
from config.agent_config import SYSTEM_MESSAGES, AGENT_CONFIG, RESPONSE_FORMAT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    raise RuntimeError("Agent initialization error")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and remove sensitive information."""
    return os.path.basename(filename)

@app.post("/api/analyze")
async def analyze_pdf(file: UploadFile):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file format")

    try:
        # Read file content
        content = await file.read()
        safe_filename = sanitize_filename(file.filename)
        logger.info(f"Processing PDF file: {safe_filename}, size: {len(content)} bytes")
        
        # Convert PDF to images
        pdf_converter = PDFConverter()
        try:
            logger.info("Starting PDF to image conversion")
            pages_info = pdf_converter.pdf_to_images(content)
            logger.info(f"PDF conversion complete. Generated {len(pages_info)} pages")
            
            # Prepare images for analysis
            images = []
            for page in pages_info:
                try:
                    with Image.open(page['image_path']) as pil_image:
                        # Optimize image size
                        if pil_image.size[0] > 768 or pil_image.size[1] > 768:
                            pil_image.thumbnail((768, 768), Image.Resampling.LANCZOS)
                        
                        # Convert to AutoGen Image
                        autogen_image = AutoGenImage(pil_image)
                        images.append(autogen_image)
                except Exception as e:
                    logger.error("Error processing image page")
                    continue

            if not images:
                raise HTTPException(status_code=500, detail="Failed to process PDF images")

            # Prepare the prompt
            prompt = f"""Analyze the following construction plan images for plumbing systems:

Document Information:
- File Name: {safe_filename}
- Total Pages: {len(pages_info)}

Please analyze this construction plan focusing specifically on plumbing systems."""

            # Create multimodal message
            multimodal_message = MultiModalMessage(
                content=[prompt] + images,
                source="user"
            )

            # Get analysis from AutoGen agent
            from autogen_core import CancellationToken
            try:
                logger.info("Sending request to Azure OpenAI")
                response = await plumbing_agent.on_messages([multimodal_message], CancellationToken())
                analysis_content = response.chat_message.content if hasattr(response, 'chat_message') else str(response)
                logger.info("Successfully received response from Azure OpenAI")
            except Exception as e:
                logger.error(f"Error during AI analysis: {str(e)}")
                raise HTTPException(status_code=500, detail="AI analysis failed")

            # Clean up temporary files
            pdf_converter.cleanup()

            return {
                "status": "success",
                "data": {
                    "document_info": {
                        "filename": safe_filename,
                        "pages": len(pages_info)
                    },
                    "analysis": analysis_content
                }
            }

        except Exception as e:
            logger.error(f"Error during PDF conversion: {str(e)}")
            pdf_converter.cleanup()
            raise HTTPException(status_code=500, detail="Error processing document")

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing document")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 