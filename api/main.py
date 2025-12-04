"""
FastAPI application for AgentOps API
Provides REST endpoints and WebSocket for real-time agent monitoring
"""

import asyncio
import json
import logging
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .events import event_bus, EventType
from .pipeline_manager import pipeline_manager
from .models import (
    PipelineRunCreate,
    PipelineRunFromMetadata,
    PipelineRunResponse,
    PipelineListResponse,
    PipelineApproval,
    PipelineStatus,
    AgentHierarchy,
    AgentListResponse,
    OverallStats,
    RateLimitStats,
    SystemStats,
    WSMessage,
)

logger = logging.getLogger(__name__)

# Data directory for uploaded files
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Track if Ray is available
RAY_AVAILABLE = False
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    logger.warning("Ray not installed - engine features will be limited")


def initialize_ray():
    """Initialize Ray for distributed processing."""
    global RAY_AVAILABLE
    if not RAY_AVAILABLE:
        logger.warning("Ray not available - skipping initialization")
        return False
    
    try:
        import ray
        import warnings
        
        # Suppress Ray's noisy warnings about metrics exporter
        warnings.filterwarnings("ignore", category=FutureWarning, module="ray")
        os.environ["RAY_DEDUP_LOGS"] = "0"
        
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                num_cpus=4,
                include_dashboard=False,  # Disable dashboard to reduce noise
                _metrics_export_port=None,  # Disable metrics export
                logging_level=logging.WARNING,  # Reduce Ray log verbosity
            )
            logger.info("✅ Ray initialized successfully")
        else:
            logger.info("Ray already initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Ray: {e}")
        RAY_AVAILABLE = False
        return False


def shutdown_ray():
    """Shutdown Ray gracefully."""
    if not RAY_AVAILABLE:
        return
    
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")
    except Exception as e:
        logger.warning(f"Error shutting down Ray: {e}")


# ==================== Lifespan ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown"""
    # Startup
    logger.info("Starting AgentOps API server...")
    
    # Initialize Ray engine
    initialize_ray()
    
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    logger.info(f"Data directory: {DATA_DIR}")
    
    # Set event loop for EventBus
    loop = asyncio.get_running_loop()
    event_bus.set_loop(loop)
    
    logger.info("EventBus initialized with async loop")
    logger.info("🚀 AgentOps engine ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AgentOps API server...")
    shutdown_ray()


# ==================== App Setup ====================

app = FastAPI(
    title="AgentOps API",
    description="API for controlling and monitoring the AgentOps document processing engine",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health Check ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/")
async def root():
    """Root endpoint with API info"""
    ray_status = "not installed"
    if RAY_AVAILABLE:
        try:
            import ray
            ray_status = "running" if ray.is_initialized() else "not initialized"
        except:
            ray_status = "error"
    
    return {
        "name": "AgentOps API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "engine_status": ray_status,
        "data_directory": str(DATA_DIR),
    }


# ==================== File Upload Endpoints ====================

@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
):
    """
    Upload a PDF file for processing.
    
    The file will be saved to the data/ directory and can be used
    to start a pipeline via /api/pipeline/start.
    
    Returns the file path to use in pipeline requests.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate unique filename to avoid conflicts
    file_id = uuid.uuid4().hex[:8]
    safe_filename = f"{file_id}_{file.filename.replace(' ', '_')}"
    file_path = DATA_DIR / safe_filename
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        
        logger.info(f"✅ File uploaded: {safe_filename} ({file_size:.2f} MB)")
        
        return {
            "status": "success",
            "filename": safe_filename,
            "original_filename": file.filename,
            "file_path": str(file_path),
            "size_mb": round(file_size, 2),
            "message": f"File uploaded successfully. Use file_path in /api/pipeline/start"
        }
        
    except Exception as e:
        logger.error(f"❌ File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@app.get("/api/files")
async def list_files():
    """
    List all PDF files available in the data directory.
    
    These files can be used to start pipelines.
    """
    files = []
    
    for file_path in DATA_DIR.glob("*.pdf"):
        stat = file_path.stat()
        files.append({
            "filename": file_path.name,
            "file_path": str(file_path),
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x["modified_at"], reverse=True)
    
    return {
        "files": files,
        "total": len(files),
        "data_directory": str(DATA_DIR),
    }


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """Delete a file from the data directory."""
    file_path = DATA_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    if not file_path.suffix.lower() == '.pdf':
        raise HTTPException(status_code=400, detail="Can only delete PDF files")
    
    try:
        file_path.unlink()
        logger.info(f"🗑️ File deleted: {filename}")
        return {"status": "deleted", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@app.post("/api/upload-and-process")
async def upload_and_process(
    file: UploadFile = File(..., description="PDF file to process"),
    auto_approve: bool = Form(True, description="Auto-approve SubMaster plan"),
    user_notes: Optional[str] = Form(None, description="Optional notes for processing"),
):
    """
    Upload a PDF and immediately start processing.
    
    This is a convenience endpoint that combines:
    1. POST /api/upload - Upload the file
    2. POST /api/pipeline/start - Start processing
    
    Returns pipeline status with the pipeline_id for tracking.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate unique filename
    file_id = uuid.uuid4().hex[:8]
    safe_filename = f"{file_id}_{file.filename.replace(' ', '_')}"
    file_path = DATA_DIR / safe_filename
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ File uploaded: {safe_filename} ({file_size:.2f} MB)")
        
        # Start pipeline
        request = PipelineRunCreate(
            file_path=str(file_path),
            auto_approve=auto_approve,
            user_notes=user_notes,
        )
        
        response = await pipeline_manager.start_pipeline(request)
        
        return {
            "status": "processing",
            "file": {
                "filename": safe_filename,
                "original_filename": file.filename,
                "file_path": str(file_path),
                "size_mb": round(file_size, 2),
            },
            "pipeline": response.model_dump(),
        }
        
    except Exception as e:
        logger.error(f"❌ Upload and process failed: {e}")
        # Clean up file if pipeline start failed
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ==================== Pipeline Endpoints ====================

@app.post("/api/pipeline/start", response_model=PipelineRunResponse)
async def start_pipeline(request: PipelineRunCreate):
    """
    Start a new pipeline run from a PDF file.
    
    The pipeline will:
    1. Extract metadata and sections from the PDF (Mapper)
    2. Generate SubMaster execution plan (MasterAgent)
    3. Wait for approval (if auto_approve=False)
    4. Process document with SubMasters and Workers
    5. Generate final report
    """
    try:
        response = await pipeline_manager.start_pipeline(request)
        return response
    except Exception as e:
        logger.exception("Failed to start pipeline")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pipeline/start-from-metadata", response_model=PipelineRunResponse)
async def start_pipeline_from_metadata(request: PipelineRunFromMetadata):
    """
    Start a pipeline from an existing metadata JSON file.
    
    This skips the Mapper step and uses pre-generated metadata.
    """
    try:
        response = await pipeline_manager.start_pipeline_from_metadata(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to start pipeline from metadata")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pipeline", response_model=PipelineListResponse)
async def list_pipelines(
    status: Optional[PipelineStatus] = Query(None, description="Filter by status")
):
    """List all pipeline runs, optionally filtered by status"""
    pipelines = pipeline_manager.list_pipelines(status)
    return PipelineListResponse(pipelines=pipelines, total=len(pipelines))


@app.get("/api/pipeline/{pipeline_id}", response_model=PipelineRunResponse)
async def get_pipeline(pipeline_id: str):
    """Get status of a specific pipeline"""
    pipeline = pipeline_manager.get_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    return pipeline


@app.post("/api/pipeline/{pipeline_id}/approve")
async def approve_pipeline(pipeline_id: str, approval: PipelineApproval):
    """
    Approve or reject a pipeline's SubMaster execution plan.
    
    Only applicable when pipeline status is 'awaiting_approval'.
    """
    try:
        pipeline_manager.approve_pipeline(pipeline_id, approval.approved, approval.feedback)
        return {"status": "approved" if approval.approved else "rejected", "pipeline_id": pipeline_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pipeline/{pipeline_id}/cancel")
async def cancel_pipeline(pipeline_id: str):
    """Cancel a running pipeline"""
    success = pipeline_manager.cancel_pipeline(pipeline_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline {pipeline_id} cannot be cancelled (not found or already finished)"
        )
    return {"status": "cancelled", "pipeline_id": pipeline_id}


@app.get("/api/pipeline/{pipeline_id}/events")
async def get_pipeline_events(
    pipeline_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Max events to return")
):
    """Get event history for a pipeline"""
    events = event_bus.get_history(pipeline_id, limit)
    return {"pipeline_id": pipeline_id, "events": events, "count": len(events)}


# ==================== Agent Endpoints ====================

@app.get("/api/agents", response_model=List[Dict[str, Any]])
async def list_all_agents():
    """List all active agents across all pipelines"""
    # Get agent info from active pipelines
    pipelines = event_bus.get_active_pipelines()
    agents = []
    
    for pipeline in pipelines:
        pipeline_id = pipeline.get("id")
        # Get recent events to determine active agents
        events = event_bus.get_history(pipeline_id, 100)
        
        # Extract agent info from events
        agent_map = {}
        for event in events:
            agent_id = event.get("agent_id")
            if agent_id:
                agent_map[agent_id] = {
                    "agent_id": agent_id,
                    "agent_type": event.get("agent_type"),
                    "pipeline_id": pipeline_id,
                    "last_event": event.get("event_type"),
                    "last_timestamp": event.get("timestamp"),
                }
        
        agents.extend(agent_map.values())
    
    return agents


@app.get("/api/pipeline/{pipeline_id}/agents", response_model=AgentListResponse)
async def get_pipeline_agents(pipeline_id: str):
    """Get agent hierarchy for a specific pipeline"""
    pipeline = pipeline_manager.get_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    
    # Build agent hierarchy from events
    events = event_bus.get_history(pipeline_id, 500)
    
    hierarchy = AgentHierarchy(
        pipeline_id=pipeline_id,
        master=None,
        submasters=[],
        total_workers=0,
    )
    
    # TODO: Parse events to build actual hierarchy
    # This is a placeholder - will be populated when agents emit events
    
    return AgentListResponse(pipeline_id=pipeline_id, agents=hierarchy)


# ==================== Stats Endpoints ====================

@app.get("/api/stats")
async def get_overall_stats():
    """Get overall system statistics"""
    try:
        from utils.llm_helper import global_rate_limiter
        rate_stats = global_rate_limiter.get_stats() if global_rate_limiter else None
    except ImportError:
        rate_stats = None
    
    pipelines = pipeline_manager.list_pipelines()
    active_count = sum(1 for p in pipelines if p.status == PipelineStatus.RUNNING)
    
    return {
        "active_pipelines": active_count,
        "total_pipelines": len(pipelines),
        "total_submasters": sum(p.num_submasters or 0 for p in pipelines),
        "total_workers": sum(p.num_workers or 0 for p in pipelines),
        "rate_limit": rate_stats,
    }


@app.get("/api/stats/ratelimit")
async def get_rate_limit_stats():
    """Get rate limiter statistics"""
    try:
        from utils.llm_helper import global_rate_limiter
        if global_rate_limiter:
            stats = global_rate_limiter.get_stats()
            return RateLimitStats(**stats)
        return {"error": "Rate limiter not available"}
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rate limit stats: {e}")


@app.get("/api/stats/system")
async def get_system_stats():
    """Get system resource statistics"""
    try:
        from utils.monitor import SystemMonitor
        monitor = SystemMonitor()
        stats = monitor.get_stats()
        return stats
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system stats: {e}")


# ==================== WebSocket Endpoint ====================

@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming.
    
    Connect and optionally subscribe to specific pipelines.
    Messages:
    - {"type": "subscribe", "payload": {"pipeline_id": "xxx"}}  - Subscribe to pipeline
    - {"type": "unsubscribe", "payload": {"pipeline_id": "xxx"}} - Unsubscribe
    - {"type": "ping"} -> {"type": "pong"}  - Keep-alive
    
    Server sends:
    - {"type": "event", "payload": {...event data...}}
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    # Subscribe to all events by default
    event_queue = event_bus.register_ws_client("*")
    subscribed_pipelines = {"*"}
    
    try:
        # Task to receive messages from client
        async def receive_messages():
            while True:
                try:
                    data = await websocket.receive_json()
                    msg_type = data.get("type")
                    payload = data.get("payload", {})
                    
                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                    
                    elif msg_type == "subscribe":
                        pipeline_id = payload.get("pipeline_id", "*")
                        if pipeline_id not in subscribed_pipelines:
                            # Register for specific pipeline
                            new_queue = event_bus.register_ws_client(pipeline_id)
                            subscribed_pipelines.add(pipeline_id)
                            await websocket.send_json({
                                "type": "subscribed",
                                "payload": {"pipeline_id": pipeline_id}
                            })
                    
                    elif msg_type == "unsubscribe":
                        pipeline_id = payload.get("pipeline_id")
                        if pipeline_id and pipeline_id in subscribed_pipelines:
                            subscribed_pipelines.discard(pipeline_id)
                            await websocket.send_json({
                                "type": "unsubscribed",
                                "payload": {"pipeline_id": pipeline_id}
                            })
                    
                    elif msg_type == "get_history":
                        pipeline_id = payload.get("pipeline_id")
                        limit = payload.get("limit", 100)
                        if pipeline_id:
                            events = event_bus.get_history(pipeline_id, limit)
                            await websocket.send_json({
                                "type": "history",
                                "payload": {"pipeline_id": pipeline_id, "events": events}
                            })
                            
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
        
        # Task to send events to client
        async def send_events():
            while True:
                try:
                    event = await event_queue.get()
                    await websocket.send_json({
                        "type": "event",
                        "payload": event.to_dict()
                    })
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    logger.error(f"Error sending WebSocket event: {e}")
        
        # Run both tasks concurrently
        await asyncio.gather(receive_messages(), send_events())
        
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup
        event_bus.unregister_ws_client(event_queue, "*")
        for pipeline_id in subscribed_pipelines:
            event_bus.unregister_ws_client(event_queue, pipeline_id)


@app.websocket("/api/ws/{pipeline_id}")
async def websocket_pipeline_endpoint(websocket: WebSocket, pipeline_id: str):
    """
    WebSocket endpoint for streaming events from a specific pipeline.
    
    Automatically subscribes to the specified pipeline.
    """
    await websocket.accept()
    logger.info(f"WebSocket client connected for pipeline: {pipeline_id}")
    
    # Subscribe to specific pipeline
    event_queue = event_bus.register_ws_client(pipeline_id)
    
    # Send event history first
    history = event_bus.get_history(pipeline_id, 100)
    if history:
        await websocket.send_json({
            "type": "history",
            "payload": {"pipeline_id": pipeline_id, "events": history}
        })
    
    try:
        async def receive_messages():
            while True:
                try:
                    data = await websocket.receive_json()
                    if data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except WebSocketDisconnect:
                    raise
                except Exception:
                    pass
        
        async def send_events():
            while True:
                try:
                    event = await event_queue.get()
                    await websocket.send_json({
                        "type": "event",
                        "payload": event.to_dict()
                    })
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    logger.error(f"Error sending event: {e}")
        
        await asyncio.gather(receive_messages(), send_events())
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from pipeline: {pipeline_id}")
    finally:
        event_bus.unregister_ws_client(event_queue, pipeline_id)


# ==================== Error Handlers ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )
