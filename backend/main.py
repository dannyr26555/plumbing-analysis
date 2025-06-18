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
from config.agent_config import SYSTEM_MESSAGES, AGENT_CONFIG, RESPONSE_FORMAT
import asyncio
import uuid

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

def update_progress(task_id: str, stage: str, current: int, total: int, message: str):
    """Update progress for a specific task"""
    progress_store[task_id] = {
        "stage": stage,
        "current": current,
        "total": total,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }

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
            logger.info("Starting PDF to image conversion")
            update_progress(task_id, "processing", 0, 100, "Starting PDF conversion...")
            pages_info = pdf_converter.pdf_to_images(content)
            logger.info(f"PDF conversion complete. Generated {len(pages_info)} pages")
            for i, page in enumerate(pages_info):
                progress_percent = int((i + 1) / len(pages_info) * 100)
                update_progress(task_id, "processing", progress_percent, 100, f"Processed page {i + 1} of {len(pages_info)}")
                await asyncio.sleep(0.1)
            update_progress(task_id, "analyzing", 0, 100, "Preparing images for analysis...")
            images = []
            for i, page in enumerate(pages_info):
                try:
                    with Image.open(page['image_path']) as pil_image:
                        if pil_image.size[0] > 768 or pil_image.size[1] > 768:
                            pil_image.thumbnail((768, 768), Image.Resampling.LANCZOS)
                        autogen_image = AutoGenImage(pil_image)
                        images.append(autogen_image)
                        progress_percent = int((i + 1) / len(pages_info) * 50)
                        update_progress(task_id, "analyzing", progress_percent, 100, f"Prepared image {i + 1} of {len(pages_info)}")
                except Exception as e:
                    logger.error("Error processing image page")
                    continue
            if not images:
                raise HTTPException(status_code=500, detail="Failed to process PDF images")
            prompt = f"""Analyze the following construction plan images for plumbing systems:\n\nDocument Information:\n- File Name: {safe_filename}\n- Total Pages: {len(pages_info)}\n\nPlease analyze this construction plan focusing specifically on plumbing systems."""
            multimodal_message = MultiModalMessage(
                content=[prompt] + images,
                source="user"
            )
            update_progress(task_id, "analyzing", 50, 100, "Sending to AI for analysis...")
            from autogen_core import CancellationToken
            try:
                logger.info("Sending request to Azure OpenAI")
                response = await plumbing_agent.on_messages([multimodal_message], CancellationToken())
                analysis_content = response.chat_message.content if hasattr(response, 'chat_message') else str(response)
                logger.info("Successfully received response from Azure OpenAI")
                update_progress(task_id, "complete", 100, 100, "Analysis complete!")
                backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                # Store result
                result_store[task_id] = {
                    "pdfUrl": f"{backend_url}/api/images/{task_id}_{safe_filename}",
                    "analysis": analysis_content
                }
            except Exception as e:
                logger.error(f"Error during AI analysis: {str(e)}")
                raise HTTPException(status_code=500, detail="AI analysis failed")
            pdf_converter.cleanup()
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            return {
                "taskId": task_id,
                "pdfUrl": f"{backend_url}/api/images/{task_id}_{safe_filename}",
                "analysis": result_store[task_id]["analysis"]
            }
        except Exception as e:
            logger.error(f"Error during PDF conversion: {str(e)}")
            pdf_converter.cleanup()
            raise HTTPException(status_code=500, detail="Error processing document")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 