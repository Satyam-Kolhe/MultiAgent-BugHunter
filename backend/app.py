"""
FastAPI Backend for Infineon Bug Hunter Multi-Agent System
Provides REST API endpoints for code analysis pipeline
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import logging
import os
import tempfile
from pathlib import Path

# Import agents
from agents.librarian import LibrarianAgent
from agents.inspector import InspectorAgent
from agents.diagnostician import DiagnosticianAgent
from agents.fixer import FixerAgent

# Import document processors
from utils.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Infineon Bug Hunter API",
    description="Multi-Agent System for Bug Detection and Fixing in Embedded Code",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents (singleton pattern)
librarian_agent = None
inspector_agent = None
diagnostician_agent = None
fixer_agent = None
document_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup"""
    global librarian_agent, inspector_agent, diagnostician_agent, fixer_agent, document_processor
    logger.info("Initializing agents...")
    librarian_agent = LibrarianAgent()
    inspector_agent = InspectorAgent()
    diagnostician_agent = DiagnosticianAgent()
    fixer_agent = FixerAgent()
    document_processor = DocumentProcessor()
    logger.info("All agents initialized successfully")

# ==================== Request/Response Models ====================

class CodeAnalysisRequest(BaseModel):
    """Request model for code analysis"""
    code: str
    context_type: Optional[str] = "embedded"
    include_fixes: Optional[bool] = True

class CodeAnalysisResponse(BaseModel):
    """Response model for code analysis"""
    status: str
    librarian_result: Dict[str, Any]
    inspector_result: Dict[str, Any]
    diagnostician_result: Optional[Dict[str, Any]] = None
    fixer_result: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    agents_ready: bool
    message: str

# ==================== API Endpoints ====================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Infineon Bug Hunter API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    agents_ready = all([
        librarian_agent is not None,
        inspector_agent is not None,
        diagnostician_agent is not None,
        fixer_agent is not None
    ])
    
    return HealthResponse(
        status="healthy" if agents_ready else "degraded",
        agents_ready=agents_ready,
        message="All agents ready" if agents_ready else "Some agents not initialized"
    )

@app.post("/api/analyze", response_model=CodeAnalysisResponse)
async def analyze_code(request: CodeAnalysisRequest):
    """
    Main endpoint: Analyze code through the complete pipeline
    Librarian → Inspector → Diagnostician → Fixer
    """
    import time
    start_time = time.time()
    
    try:
        # Step 1: Librarian - Context Retrieval
        logger.info("Step 1: Librarian - Context Retrieval")
        librarian_result = await librarian_agent.analyze_context(
            request.code, 
            request.context_type
        )
        
        # Step 2: Inspector - Bug Detection
        logger.info("Step 2: Inspector - Bug Detection")
        inspector_result = await inspector_agent.analyze_code(
            request.code,
            librarian_result
        )
        
        diagnostician_result = None
        fixer_result = None
        
        # Step 3: Diagnostician - Root Cause Analysis (if bugs found)
        if inspector_result.get("total_findings", 0) > 0:
            logger.info("Step 3: Diagnostician - Root Cause Analysis")
            diagnostician_result = await diagnostician_agent.diagnose_bugs(
                inspector_result,
                librarian_result,
                request.code
            )
            
            # Step 4: Fixer - Generate Fixes (if requested)
            if request.include_fixes and diagnostician_result:
                logger.info("Step 4: Fixer - Generate Fixes")
                fixer_result = await fixer_agent.generate_fixes(
                    diagnostician_result,
                    request.code,
                    librarian_result
                )
        
        execution_time = time.time() - start_time
        
        return CodeAnalysisResponse(
            status="success",
            librarian_result=librarian_result,
            inspector_result=inspector_result,
            diagnostician_result=diagnostician_result,
            fixer_result=fixer_result,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        execution_time = time.time() - start_time
        return CodeAnalysisResponse(
            status="error",
            librarian_result={},
            inspector_result={},
            error=str(e),
            execution_time=execution_time
        )

@app.post("/api/analyze/file")
async def analyze_file(
    file: UploadFile = File(...),
    context_type: Optional[str] = "embedded",
    include_fixes: Optional[bool] = True
):
    """
    Analyze code from uploaded file (supports .c, .cpp, .h, .txt, .pdf, .docx)
    """
    import time
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        file_extension = Path(file.filename).suffix.lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process document and extract code
            logger.info(f"Processing file: {file.filename}")
            extracted_data = await document_processor.process_file(tmp_file_path)
            
            # Extract code from processed document
            code = extracted_data.get("code", "")
            if not code:
                raise HTTPException(
                    status_code=400,
                    detail="No code found in the uploaded file. Please ensure the file contains code."
                )
            
            # Run analysis pipeline
            request = CodeAnalysisRequest(
                code=code,
                context_type=context_type,
                include_fixes=include_fixes
            )
            
            result = await analyze_code(request)
            result.execution_time = time.time() - start_time
            
            # Add file metadata to response
            result.librarian_result["source_file"] = file.filename
            result.librarian_result["file_type"] = file_extension
            
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.post("/api/librarian/analyze")
async def librarian_only(request: CodeAnalysisRequest):
    """Run only the Librarian agent"""
    try:
        result = await librarian_agent.analyze_context(
            request.code,
            request.context_type
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Librarian analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/inspector/analyze")
async def inspector_only(request: CodeAnalysisRequest):
    """Run only the Inspector agent"""
    try:
        librarian_result = await librarian_agent.analyze_context(
            request.code,
            request.context_type
        )
        result = await inspector_agent.analyze_code(
            request.code,
            librarian_result
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Inspector analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "agents": {
            "librarian": librarian_agent is not None,
            "inspector": inspector_agent is not None,
            "diagnostician": diagnostician_agent is not None,
            "fixer": fixer_agent is not None
        },
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
