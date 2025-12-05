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
from fastapi.responses import JSONResponse, FileResponse

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
    SessionCreate,
    IntentSubmit,
    MetadataResponse,
    MetadataApproval,
    SessionStatus,
    ChatRequest,
    ChatResponse,
    ChatMessage,
    ChatHistory,
)

logger = logging.getLogger(__name__)

# Data directory for uploaded files
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Track if Ray is available
RAY_AVAILABLE = False
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    logger.warning("Ray not installed - engine features will be limited")


# ==================== Session Manager ====================

class SessionManager:
    """Manages processing sessions for the step-by-step workflow."""
    
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._chat_histories: Dict[str, List[Dict[str, str]]] = {}
    
    def create_session(self, file_path: str, file_name: str) -> str:
        """Create a new processing session."""
        session_id = f"session-{uuid.uuid4().hex[:8]}"
        self._sessions[session_id] = {
            "session_id": session_id,
            "file_path": file_path,
            "file_name": file_name,
            "status": "awaiting_intent",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "high_level_intent": None,
            "document_context": None,
            "metadata_path": None,
            "metadata": None,
            "pipeline_id": None,
            "error": None,
        }
        self._chat_histories[session_id] = []
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        return self._sessions.get(session_id)
    
    def update_session(self, session_id: str, **kwargs) -> bool:
        """Update session fields."""
        if session_id not in self._sessions:
            return False
        self._sessions[session_id].update(kwargs)
        self._sessions[session_id]["updated_at"] = datetime.utcnow()
        return True
    
    def list_sessions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all sessions, optionally filtered by status."""
        sessions = list(self._sessions.values())
        if status:
            sessions = [s for s in sessions if s["status"] == status]
        return sorted(sessions, key=lambda x: x["created_at"], reverse=True)
    
    def add_chat_message(self, session_id: str, role: str, content: str):
        """Add a message to chat history."""
        if session_id not in self._chat_histories:
            self._chat_histories[session_id] = []
        self._chat_histories[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get chat history for a session."""
        return self._chat_histories.get(session_id, [])
    
    def clear_chat_history(self, session_id: str):
        """Clear chat history for a session."""
        if session_id in self._chat_histories:
            self._chat_histories[session_id] = []


# Global session manager instance
session_manager = SessionManager()


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
    high_level_intent: Optional[str] = Form(None, description="High-level intent (e.g., 'Summarize for presentation')"),
    document_context: Optional[str] = Form(None, description="Additional context about the document"),
):
    """
    Upload a PDF and immediately start processing.
    
    This is the main endpoint for the API workflow:
    1. Upload the PDF file
    2. Start the pipeline with provided intent and context
    3. Returns pipeline_id for tracking progress via WebSocket
    
    Parameters:
    - file: PDF file to process
    - auto_approve: Auto-approve the SubMaster plan (default: True)
    - user_notes: Optional notes for processing
    - high_level_intent: What you want to do with the document (e.g., "Summarize for presentation")
    - document_context: Additional context about the document content
    
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
        
        # Start pipeline with intent and context
        request = PipelineRunCreate(
            file_path=str(file_path),
            auto_approve=auto_approve,
            user_notes=user_notes,
            high_level_intent=high_level_intent or "Analyze and summarize the document",
            document_context=document_context,
        )
        
        response = await pipeline_manager.start_pipeline(request)
        
        return {
            "status": "processing",
            "message": "Pipeline started successfully. Use the pipeline_id to track progress.",
            "file": {
                "filename": safe_filename,
                "original_filename": file.filename,
                "file_path": str(file_path),
                "size_mb": round(file_size, 2),
            },
            "pipeline": response.model_dump(),
            "tracking": {
                "status_url": f"/api/pipeline/{response.pipeline_id}",
                "websocket_url": f"/ws/pipeline/{response.pipeline_id}",
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Upload and process failed: {e}")
        # Clean up file if pipeline start failed
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ==================== Session Workflow Endpoints ====================
# Step-by-step workflow: Upload -> Intent -> Metadata Approval -> Process -> Chat

@app.post("/api/session/upload", response_model=SessionCreate)
async def session_upload(
    file: UploadFile = File(..., description="PDF file to upload"),
):
    """
    STEP 1: Upload a PDF and create a processing session.
    
    This starts the workflow. After upload, the client should:
    1. Call POST /api/session/{session_id}/intent to submit user intent
    2. Receive metadata and approve it
    3. Start processing
    4. Chat with the processed document
    
    Returns a session_id for tracking the workflow.
    """
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
        logger.info(f"✅ Session file uploaded: {safe_filename} ({file_size:.2f} MB)")
        
        # Create session
        session_id = session_manager.create_session(str(file_path), safe_filename)
        
        return SessionCreate(
            session_id=session_id,
            file_path=str(file_path),
            file_name=safe_filename,
            status="awaiting_intent"
        )
        
    except Exception as e:
        logger.error(f"❌ Session upload failed: {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/session/{session_id}/intent", response_model=MetadataResponse)
async def session_submit_intent(session_id: str, intent: IntentSubmit):
    """
    STEP 2: Submit user intent and generate metadata.
    
    After uploading, the user provides:
    - high_level_intent: What they want to do (e.g., "Summarize for presentation")
    - document_context: Optional additional context about the document
    
    This triggers the Mapper to extract metadata from the PDF.
    Returns the metadata for user approval.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    if session["status"] != "awaiting_intent":
        raise HTTPException(
            status_code=400, 
            detail=f"Session is in '{session['status']}' state, expected 'awaiting_intent'"
        )
    
    try:
        # Update session with intent
        session_manager.update_session(
            session_id,
            high_level_intent=intent.high_level_intent,
            document_context=intent.document_context,
            status="generating_metadata"
        )
        
        # Run Mapper to extract metadata
        from workflows.mapper import Mapper
        
        user_config = {
            "document_type": "research_paper",
            "processing_requirements": [
                "summary_generation",
                "entity_extraction", 
                "keyword_indexing"
            ],
            "user_notes": intent.high_level_intent,
            "high_level_intent": intent.high_level_intent,
            "document_context": intent.document_context or "",
            "complexity_level": "high",
            "preferred_model": "mistral-small-latest",
            "max_parallel_submasters": 3,
            "num_workers_per_submaster": 4,
            "feedback_required": True
        }
        
        mapper = Mapper(output_dir=str(OUTPUT_DIR))
        mapper_result = mapper.execute(session["file_path"], user_config)
        
        if mapper_result.get("status") != "success":
            raise RuntimeError(f"Mapper failed: {mapper_result.get('error', 'Unknown error')}")
        
        # Load generated metadata
        metadata_path = mapper_result["metadata_path"]
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Enrich metadata with user intent
        metadata["high_level_intent"] = intent.high_level_intent
        metadata["user_document_context"] = intent.document_context
        
        # Save enriched metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Update session
        session_manager.update_session(
            session_id,
            metadata_path=metadata_path,
            metadata=metadata,
            status="awaiting_approval"
        )
        
        logger.info(f"✅ Metadata generated for session {session_id}")
        
        return MetadataResponse(
            session_id=session_id,
            metadata_path=metadata_path,
            metadata=metadata,
            status="awaiting_approval"
        )
        
    except Exception as e:
        logger.exception(f"Failed to generate metadata for session {session_id}")
        session_manager.update_session(session_id, status="failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metadata generation failed: {str(e)}")


@app.get("/api/session/{session_id}/metadata")
async def session_get_metadata(session_id: str):
    """
    Get the current metadata for a session.
    
    Use this to view or edit the metadata before approval.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    if not session.get("metadata"):
        raise HTTPException(status_code=400, detail="Metadata not yet generated. Submit intent first.")
    
    return {
        "session_id": session_id,
        "metadata_path": session["metadata_path"],
        "metadata": session["metadata"],
        "status": session["status"]
    }


@app.put("/api/session/{session_id}/metadata")
async def session_update_metadata(session_id: str, metadata: Dict[str, Any]):
    """
    Update/edit metadata before approval.
    
    Allows the user to modify the generated metadata before processing.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    if session["status"] not in ["awaiting_approval", "awaiting_intent"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot edit metadata in '{session['status']}' state"
        )
    
    try:
        # Save updated metadata
        metadata_path = session.get("metadata_path")
        if metadata_path:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        session_manager.update_session(session_id, metadata=metadata)
        
        return {
            "session_id": session_id,
            "status": "metadata_updated",
            "metadata": metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update metadata: {str(e)}")


@app.post("/api/session/{session_id}/approve")
async def session_approve_and_process(session_id: str, approval: MetadataApproval):
    """
    STEP 3: Approve metadata and start processing.
    
    If approved=True, starts the processing pipeline with current metadata.
    If approved=False with modified_metadata, updates metadata first then processes.
    
    Returns the pipeline_id for tracking processing progress.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    if session["status"] != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Session is in '{session['status']}' state, expected 'awaiting_approval'"
        )
    
    try:
        metadata_path = session["metadata_path"]
        
        # Update metadata if modified
        if not approval.approved and approval.modified_metadata:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(approval.modified_metadata, f, indent=2, ensure_ascii=False)
            session_manager.update_session(session_id, metadata=approval.modified_metadata)
        elif not approval.approved:
            raise HTTPException(
                status_code=400,
                detail="Rejection requires modified_metadata to be provided"
            )
        
        # Start pipeline from metadata
        session_manager.update_session(session_id, status="processing")
        
        request = PipelineRunFromMetadata(
            metadata_path=metadata_path,
            high_level_intent=session.get("high_level_intent"),
            document_context=session.get("document_context"),
            auto_approve=True
        )
        
        response = await pipeline_manager.start_pipeline_from_metadata(request)
        
        session_manager.update_session(
            session_id,
            pipeline_id=response.pipeline_id,
            status="processing"
        )
        
        logger.info(f"✅ Processing started for session {session_id}, pipeline: {response.pipeline_id}")
        
        return {
            "session_id": session_id,
            "status": "processing",
            "pipeline_id": response.pipeline_id,
            "pipeline": response.model_dump(),
            "tracking": {
                "status_url": f"/api/pipeline/{response.pipeline_id}",
                "websocket_url": f"/ws/pipeline/{response.pipeline_id}"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to start processing for session {session_id}")
        session_manager.update_session(session_id, status="failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/api/session/{session_id}", response_model=SessionStatus)
async def session_get_status(session_id: str):
    """
    Get the current status of a session.
    
    Statuses:
    - awaiting_intent: Upload complete, waiting for user intent
    - generating_metadata: Mapper is extracting metadata
    - awaiting_approval: Metadata ready for user approval
    - processing: Pipeline is running
    - completed: Processing complete, ready for chat
    - failed: An error occurred
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # If processing, check pipeline status
    if session["status"] == "processing" and session.get("pipeline_id"):
        pipeline = pipeline_manager.get_pipeline(session["pipeline_id"])
        if pipeline:
            if pipeline.status == PipelineStatus.COMPLETED:
                session_manager.update_session(session_id, status="completed")
                session["status"] = "completed"
            elif pipeline.status == PipelineStatus.FAILED:
                session_manager.update_session(session_id, status="failed", error="Pipeline failed")
                session["status"] = "failed"
    
    return SessionStatus(
        session_id=session["session_id"],
        status=session["status"],
        file_name=session["file_name"],
        file_path=session["file_path"],
        pipeline_id=session.get("pipeline_id"),
        metadata_path=session.get("metadata_path"),
        created_at=session["created_at"],
        updated_at=session.get("updated_at"),
        error=session.get("error")
    )


@app.get("/api/sessions")
async def list_sessions(status: Optional[str] = Query(None)):
    """List all sessions, optionally filtered by status."""
    sessions = session_manager.list_sessions(status)
    return {
        "sessions": sessions,
        "total": len(sessions)
    }


# ==================== Chat/RAG Endpoints ====================

@app.post("/api/session/{session_id}/chat", response_model=ChatResponse)
async def session_chat(session_id: str, request: ChatRequest):
    """
    STEP 4: Chat with the processed document.
    
    After processing is complete, use this endpoint to ask questions
    about the document content. Uses RAG (Retrieval Augmented Generation)
    to find relevant context and generate answers.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    if session["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Chat not available. Session status: '{session['status']}'. Processing must be completed first."
        )
    
    try:
        # Import and use RAG system
        from RAG.query_system import RAGSystem
        
        rag = RAGSystem()
        result = rag.query(request.question, top_k=request.top_k, verbose=False)
        
        # Store in chat history
        session_manager.add_chat_message(session_id, "user", request.question)
        session_manager.add_chat_message(session_id, "assistant", result["answer"])
        
        return ChatResponse(
            question=result["query"],
            answer=result["answer"],
            sources=[{
                "text": s["text"][:200] + "..." if len(s["text"]) > 200 else s["text"],
                "score": s["score"],
                "doc_id": s["metadata"].get("doc_id", "N/A")
            } for s in result.get("sources", [])],
            context_used=len(result.get("context", ""))
        )
        
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="RAG system not available. Ensure vector store is built."
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store not found. Run the reducer pipeline first to build it. Error: {e}"
        )
    except Exception as e:
        logger.exception(f"Chat failed for session {session_id}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/api/session/{session_id}/chat/history", response_model=ChatHistory)
async def session_chat_history(session_id: str):
    """Get the chat history for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    history = session_manager.get_chat_history(session_id)
    
    return ChatHistory(
        session_id=session_id,
        messages=[ChatMessage(role=m["role"], content=m["content"]) for m in history]
    )


@app.delete("/api/session/{session_id}/chat/history")
async def session_clear_chat_history(session_id: str):
    """Clear the chat history for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session_manager.clear_chat_history(session_id)
    return {"status": "cleared", "session_id": session_id}


# ==================== Standalone Chat Endpoint ====================

@app.post("/api/chat", response_model=ChatResponse)
async def standalone_chat(request: ChatRequest):
    """
    Chat with processed documents without a session.
    
    Uses the global vector store to answer questions.
    This works if documents have been processed via the unified pipeline.
    """
    try:
        from RAG.query_system import RAGSystem
        
        rag = RAGSystem()
        result = rag.query(request.question, top_k=request.top_k, verbose=False)
        
        return ChatResponse(
            question=result["query"],
            answer=result["answer"],
            sources=[{
                "text": s["text"][:200] + "..." if len(s["text"]) > 200 else s["text"],
                "score": s["score"],
                "doc_id": s["metadata"].get("doc_id", "N/A")
            } for s in result.get("sources", [])],
            context_used=len(result.get("context", ""))
        )
        
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="RAG system not available. Ensure dependencies are installed."
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store not found. Process documents first. Error: {e}"
        )
    except Exception as e:
        logger.exception("Standalone chat failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# ==================== Pipeline Endpoints ====================

@app.post("/api/pipeline/start", response_model=PipelineRunResponse)
async def start_pipeline(request: PipelineRunCreate):
    """
    Start a new pipeline run from a PDF file.
    
    Request body:
    - file_path: Path to the PDF file to process
    - high_level_intent: What to do with the document (e.g., "Summarize for presentation")
    - document_context: Additional context about the document
    - auto_approve: Auto-approve SubMaster plan (default: True)
    - user_notes: Optional notes for processing
    - max_parallel_submasters: Max parallel SubMasters (1-10)
    - num_workers_per_submaster: Workers per SubMaster (1-10)
    
    The pipeline will:
    1. Extract metadata and sections from the PDF (Mapper)
    2. Generate SubMaster execution plan (MasterAgent) - uses high_level_intent
    3. Wait for approval (if auto_approve=False)
    4. Process document with SubMasters and Workers
    5. Generate final report
    
    Returns pipeline_id for tracking progress.
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
    
    Request body:
    - metadata_path: Path to metadata JSON file
    - high_level_intent: What to do with the document
    - document_context: Additional document context
    - auto_approve: Auto-approve SubMaster plan (default: True)
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


# ==================== Internal Event Endpoint ====================

@app.post("/api/internal/emit")
async def emit_event_internal(event_data: Dict[str, Any]):
    """
    Internal endpoint for agents running in Ray actors to emit events.
    This is needed because Ray actors run in separate processes and can't
    share the EventBus singleton with the main FastAPI process.
    """
    try:
        event_type_str = event_data.get("event_type")
        pipeline_id = event_data.get("pipeline_id")
        data = event_data.get("data", {})
        agent_id = event_data.get("agent_id")
        agent_type = event_data.get("agent_type")
        
        if not event_type_str or not pipeline_id:
            raise HTTPException(status_code=400, detail="event_type and pipeline_id required")
        
        # Get event type enum
        event_type = getattr(EventType, event_type_str, None)
        if not event_type:
            # Try with the full string value
            for et in EventType:
                if et.value == event_type_str:
                    event_type = et
                    break
        
        if not event_type:
            logger.warning(f"Unknown event type: {event_type_str}")
            return {"status": "skipped", "reason": f"Unknown event type: {event_type_str}"}
        
        event_bus.emit_simple(
            event_type,
            pipeline_id,
            data,
            agent_id=agent_id,
            agent_type=agent_type,
        )
        
        return {"status": "ok", "event_type": event_type_str}
    except Exception as e:
        logger.error(f"Failed to emit event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


# ==================== Download Endpoints ====================

@app.get("/api/pipeline/{pipeline_id}/download/report")
async def download_pipeline_report(pipeline_id: str):
    """Download the PDF report for a completed pipeline"""
    pipeline = pipeline_manager.get_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    
    # First, check if the pipeline has a pdf_path in its result
    if pipeline.result and isinstance(pipeline.result, dict):
        pdf_path = pipeline.result.get("pdf_path")
        if pdf_path and Path(pdf_path).exists():
            return FileResponse(
                path=str(pdf_path),
                media_type="application/pdf",
                filename=f"final_summary_{pipeline_id}.pdf"
            )
    
    # Look for report in output directory
    output_dir = Path(__file__).parent.parent / "output"
    
    # Try different naming patterns - prioritize final_summary PDFs from unified pipeline
    possible_patterns = [
        f"final_summary_{pipeline_id}_*.pdf",  # Unified pipeline output
        f"final_summary_*{pipeline_id}*.pdf",
        f"*{pipeline_id}*report*.pdf",
        f"analysis_report_*{pipeline_id}*.pdf",
        f"*report*{pipeline_id}*.pdf",
    ]
    
    report_path = None
    for pattern in possible_patterns:
        matches = list(output_dir.glob(pattern))
        if matches:
            report_path = matches[0]
            break
    
    # Fallback: find most recent final_summary PDF, then any report PDF
    if not report_path:
        summary_pdfs = list(output_dir.glob("final_summary_*.pdf"))
        if summary_pdfs:
            report_path = max(summary_pdfs, key=lambda p: p.stat().st_mtime)
        else:
            pdf_files = list(output_dir.glob("*report*.pdf"))
            if pdf_files:
                report_path = max(pdf_files, key=lambda p: p.stat().st_mtime)
    
    if not report_path or not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found. Pipeline may still be processing.")
    
    return FileResponse(
        path=str(report_path),
        media_type="application/pdf",
        filename=f"final_summary_{pipeline_id}.pdf"
    )


@app.get("/api/pipeline/{pipeline_id}/final-summary")
async def get_pipeline_final_summary(pipeline_id: str):
    """
    Get the final comprehensive summary from the Master Merger agent.
    
    This returns the complete synthesis including:
    - Executive summary
    - Detailed synthesis
    - Metadata (entities, keywords, themes)
    - Insights and conclusions
    - PDF path if available
    """
    pipeline = pipeline_manager.get_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    
    if pipeline.status != PipelineStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Pipeline is '{pipeline.status.value}'. Final summary only available for completed pipelines."
        )
    
    result = pipeline.result
    if not result:
        raise HTTPException(status_code=404, detail="No results found for this pipeline")
    
    # Extract final summary from unified pipeline result
    if isinstance(result, dict):
        final_summary = result.get("final_summary", {})
        reducer_results = result.get("reducer_results", {})
        
        return {
            "pipeline_id": pipeline_id,
            "status": "completed",
            "final_summary": final_summary,
            "executive_summary": final_summary.get("executive_summary", ""),
            "insights_and_conclusions": final_summary.get("insights_and_conclusions", {}),
            "source_statistics": final_summary.get("source_statistics", {}),
            "pdf_path": result.get("pdf_path"),
            "reducer_phases": {
                "reducer": reducer_results.get("phases", {}).get("reducer", {}).get("status"),
                "residual": reducer_results.get("phases", {}).get("residual", {}).get("status"),
                "merger": reducer_results.get("phases", {}).get("merger", {}).get("status"),
            }
        }
    
    return {"pipeline_id": pipeline_id, "result": result}


@app.get("/api/pipeline/{pipeline_id}/download/json")
async def download_pipeline_json(pipeline_id: str):
    """Download the JSON results for a completed pipeline"""
    pipeline = pipeline_manager.get_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    
    # Look for results in output directory
    output_dir = Path(__file__).parent.parent / "output"
    
    # Try different naming patterns
    possible_patterns = [
        f"{pipeline_id}_*_results_*.json",
        f"*{pipeline_id}*results*.json",
        f"*results*{pipeline_id}*.json",
    ]
    
    json_path = None
    for pattern in possible_patterns:
        matches = list(output_dir.glob(pattern))
        if matches:
            json_path = matches[0]
            break
    
    # Fallback: find most recent results JSON
    if not json_path:
        json_files = list(output_dir.glob("*_results_*.json"))
        if json_files:
            # Sort by modification time, get most recent
            json_path = max(json_files, key=lambda p: p.stat().st_mtime)
    
    if not json_path or not json_path.exists():
        raise HTTPException(status_code=404, detail="Results not found. Pipeline may still be processing.")
    
    return FileResponse(
        path=str(json_path),
        media_type="application/json",
        filename=f"analysis_results_{pipeline_id}.json"
    )


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
