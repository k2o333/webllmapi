import os
import asyncio
import time
from typing import Dict, Set, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
from pydantic import BaseModel

from config_manager import get_config, get_llm_site_config, get_enabled_llm_sites
from browser_handler import LLMWebsiteAutomator
from utils import setup_logging
from models import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse

# Load environment variables from .env file
load_dotenv()

# Setup logging
logger = setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file="logs/wrapper_api.log"
)

app = FastAPI(
    title="LLM API Wrapper",
    description="Wrapper API for LLM websites with OpenAI API compatibility",
    version="1.0"
)

# Global state
active_automators: Dict[str, LLMWebsiteAutomator] = {}
automator_locks: Dict[str, asyncio.Lock] = {}

class HealthResponse(BaseModel):
    status: str
    details: Dict[str, str]

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting LLM API Wrapper application")
    
    # Load configuration
    config = get_config()
    logger.info("Configuration loaded successfully")
    
    # Initialize automators for enabled sites
    enabled_sites = get_enabled_llm_sites()
    for site in enabled_sites:
        logger.info(f"Initializing automator for site: {site.id}")
        automator = LLMWebsiteAutomator(site)
        active_automators[site.id] = automator
        automator_locks[site.id] = asyncio.Lock()
    
    logger.info(f"Initialized {len(active_automators)} automators")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down LLM API Wrapper application")
    
    # Close all automators
    for automator in active_automators.values():
        await automator.close()
    
    logger.info("All automators closed successfully")

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI compatible chat completions endpoint"""
    # Get model ID from request
    model_id = request.model
    if model_id not in active_automators:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' not found or disabled"
        )
    
    # Get automator and lock for this model
    automator = active_automators[model_id]
    lock = automator_locks[model_id]
    
    # Process messages to construct prompt
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    
    # Acquire lock to ensure single access to this automator
    async with lock:
        try:
            if request.stream:
                # Streaming response
                async def generate_stream() -> AsyncGenerator[str, None]:
                    start_time = time.time()
                    completion_text = ""
                    
                    try:
                        async for chunk in automator.send_prompt_and_get_response(prompt, stream=True):
                            completion_text += chunk
                            
                            # Format as OpenAI streaming response
                            response_chunk = ChatCompletionStreamResponse(
                                choices=[{
                                    "delta": {
                                        "content": chunk,
                                        "role": "assistant"
                                    },
                                    "finish_reason": None
                                }],
                                model=model_id
                            )
                            yield f"data: {response_chunk.json()}\n\n"
                            
                        # Final completion message
                        final_response = ChatCompletionStreamResponse(
                            choices=[{
                                "delta": {},
                                "finish_reason": "stop"
                            }],
                            model=model_id
                        )
                        yield f"data: {final_response.json()}\n\n"
                        
                    finally:
                        # Log performance metrics
                        duration = time.time() - start_time
                        logger.info(
                            f"Streaming request completed - Model: {model_id}, "
                            f"Duration: {duration:.2f}s, "
                            f"Completion length: {len(completion_text)} chars"
                        )
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming response
                start_time = time.time()
                response_text = await automator.send_prompt_and_get_response(prompt)
                
                # Log performance metrics
                duration = time.time() - start_time
                logger.info(
                    f"Request completed - Model: {model_id}, "
                    f"Duration: {duration:.2f}s, "
                    f"Completion length: {len(response_text)} chars"
                )
                
                # Construct OpenAI compatible response
                return ChatCompletionResponse(
                    choices=[{
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }],
                    model=model_id,
                    usage={
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(response_text.split()),
                        "total_tokens": len(prompt.split()) + len(response_text.split())
                    }
                )
            
        except Exception as e:
            logger.error(f"Error processing request for model {model_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"
            )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    status = "healthy"
    details = {}
    
    # Check health of each automator
    for model_id, automator in active_automators.items():
        try:
            is_healthy = await automator.is_healthy()
            details[model_id] = "healthy" if is_healthy else "unhealthy"
            if not is_healthy:
                status = "degraded"
        except Exception as e:
            details[model_id] = f"error: {str(e)}"
            status = "unhealthy"
    
    return HealthResponse(status=status, details=details)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)