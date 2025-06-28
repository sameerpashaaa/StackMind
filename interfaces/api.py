import logging
import os
import json
import base64
import tempfile
from typing import Dict, List, Any, Optional, Union
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime
import uuid

# Import core components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.agent import ProblemSolverAgent
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load settings
settings = Settings()

# Create FastAPI app
app = FastAPI(
    title="AI Problem Solver API",
    description="API for the AI Problem Solver - A multi-domain problem-solving system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a global agent instance
agent = ProblemSolverAgent(settings=settings)

# Helper function to get the agent instance
def get_agent():
    """Get the agent instance from app.state if available, otherwise use the global agent"""
    return getattr(app.state, 'agent', agent)

# Define request and response models
class TextInput(BaseModel):
    text: str = Field(..., description="Text input for the problem solver")
    session_id: Optional[str] = Field(None, description="Session ID for continuing a conversation")
    domain: Optional[str] = Field(None, description="Optional domain hint (math, code, science, general)")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional options for processing")

class ImageInput(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    session_id: Optional[str] = Field(None, description="Session ID for continuing a conversation")
    domain: Optional[str] = Field(None, description="Optional domain hint (math, code, science, general)")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional options for processing")

class VoiceInput(BaseModel):
    audio_data: str = Field(..., description="Base64 encoded audio data")
    session_id: Optional[str] = Field(None, description="Session ID for continuing a conversation")
    domain: Optional[str] = Field(None, description="Optional domain hint (math, code, science, general)")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional options for processing")

class CodeInput(BaseModel):
    code: str = Field(..., description="Code input for the problem solver")
    language: Optional[str] = Field(None, description="Programming language of the code")
    session_id: Optional[str] = Field(None, description="Session ID for continuing a conversation")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional options for processing")

class FeedbackInput(BaseModel):
    session_id: str = Field(..., description="Session ID for the solution being rated")
    solution_id: str = Field(..., description="Solution ID being rated")
    rating: int = Field(..., description="Rating from 1-5")
    feedback_text: Optional[str] = Field(None, description="Optional feedback text")

class SolverResponse(BaseModel):
    session_id: str = Field(..., description="Session ID for this conversation")
    solution_id: str = Field(..., description="Unique ID for this solution")
    problem_domain: str = Field(..., description="Detected problem domain")
    solution: str = Field(..., description="Solution to the problem")
    explanation: Optional[str] = Field(None, description="Explanation of the solution")
    reasoning_steps: Optional[List[str]] = Field(None, description="Step-by-step reasoning")
    confidence: float = Field(..., description="Confidence score for the solution")
    alternative_solutions: Optional[List[Dict[str, Any]]] = Field(None, description="Alternative solutions if available")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the solution")

# Active sessions storage
active_sessions = {}

# Helper functions
def get_or_create_session_id(session_id: Optional[str] = None) -> str:
    """Get an existing session ID or create a new one"""
    if session_id and session_id in active_sessions:
        return session_id
    
    new_session_id = str(uuid.uuid4())
    active_sessions[new_session_id] = {
        "created_at": datetime.now(),
        "last_active": datetime.now(),
        "solutions": []
    }
    return new_session_id

def update_session_activity(session_id: str) -> None:
    """Update the last active timestamp for a session"""
    if session_id in active_sessions:
        active_sessions[session_id]["last_active"] = datetime.now()

def save_solution_to_session(session_id: str, solution_data: Dict[str, Any]) -> None:
    """Save a solution to the session history"""
    if session_id in active_sessions:
        active_sessions[session_id]["solutions"].append(solution_data)

# API endpoints
@app.get("/")
async def root():
    return {"message": "AI Problem Solver API is running"}

@app.post("/solve/text", response_model=SolverResponse)
async def solve_text_problem(input_data: TextInput):
    """Solve a problem from text input"""
    try:
        session_id = get_or_create_session_id(input_data.session_id)
        update_session_activity(session_id)
        
        # Process the text input
        result = get_agent().process_input(
            input_text=input_data.text,
            input_type="text",
            domain_hint=input_data.domain,
            options=input_data.options or {},
            session_id=session_id
        )
        
        # Save the solution to the session
        solution_id = str(uuid.uuid4())
        solution_data = {
            "id": solution_id,
            "timestamp": datetime.now(),
            "input": {"type": "text", "content": input_data.text},
            "result": result
        }
        save_solution_to_session(session_id, solution_data)
        
        # Prepare the response
        response = {
            "session_id": session_id,
            "solution_id": solution_id,
            "problem_domain": result["domain"],
            "solution": result["solution"],
            "explanation": result.get("explanation"),
            "reasoning_steps": result.get("reasoning_steps"),
            "confidence": result.get("confidence", 0.0),
            "alternative_solutions": result.get("alternative_solutions"),
            "metadata": result.get("metadata")
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing text input: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing text input: {str(e)}")

@app.post("/solve/image", response_model=SolverResponse)
async def solve_image_problem(input_data: ImageInput):
    """Solve a problem from image input"""
    try:
        session_id = get_or_create_session_id(input_data.session_id)
        update_session_activity(session_id)
        
        # Decode the base64 image
        try:
            image_bytes = base64.b64decode(input_data.image_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Process the image input
            result = get_agent().process_input(
                input_image=temp_file_path,
                input_type="image",
                domain_hint=input_data.domain,
                options=input_data.options or {},
                session_id=session_id
            )
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        # Save the solution to the session
        solution_id = str(uuid.uuid4())
        solution_data = {
            "id": solution_id,
            "timestamp": datetime.now(),
            "input": {"type": "image", "content": "[Image data]"}, # Don't store the full image data
            "result": result
        }
        save_solution_to_session(session_id, solution_data)
        
        # Prepare the response
        response = {
            "session_id": session_id,
            "solution_id": solution_id,
            "problem_domain": result["domain"],
            "solution": result["solution"],
            "explanation": result.get("explanation"),
            "reasoning_steps": result.get("reasoning_steps"),
            "confidence": result.get("confidence", 0.0),
            "alternative_solutions": result.get("alternative_solutions"),
            "metadata": result.get("metadata")
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing image input: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image input: {str(e)}")

@app.post("/solve/voice", response_model=SolverResponse)
async def solve_voice_problem(input_data: VoiceInput):
    """Solve a problem from voice input"""
    try:
        session_id = get_or_create_session_id(input_data.session_id)
        update_session_activity(session_id)
        
        # Decode the base64 audio
        try:
            audio_bytes = base64.b64decode(input_data.audio_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio data: {str(e)}")
        
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Process the voice input
            result = get_agent().process_input(
                input_audio=temp_file_path,
                input_type="voice",
                domain_hint=input_data.domain,
                options=input_data.options or {},
                session_id=session_id
            )
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        # Save the solution to the session
        solution_id = str(uuid.uuid4())
        solution_data = {
            "id": solution_id,
            "timestamp": datetime.now(),
            "input": {"type": "voice", "content": "[Audio data]"}, # Don't store the full audio data
            "result": result
        }
        save_solution_to_session(session_id, solution_data)
        
        # Prepare the response
        response = {
            "session_id": session_id,
            "solution_id": solution_id,
            "problem_domain": result["domain"],
            "solution": result["solution"],
            "explanation": result.get("explanation"),
            "reasoning_steps": result.get("reasoning_steps"),
            "confidence": result.get("confidence", 0.0),
            "alternative_solutions": result.get("alternative_solutions"),
            "metadata": result.get("metadata")
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing voice input: {str(e)}")

@app.post("/solve/code", response_model=SolverResponse)
async def solve_code_problem(input_data: CodeInput):
    """Solve a problem from code input"""
    try:
        session_id = get_or_create_session_id(input_data.session_id)
        update_session_activity(session_id)
        
        # Process the code input
        options = input_data.options or {}
        if input_data.language:
            options["language"] = input_data.language
            
        result = get_agent().process_input(
            input_text=input_data.code,
            input_type="code",
            domain_hint="code",
            options=options,
            session_id=session_id
        )
        
        # Save the solution to the session
        solution_id = str(uuid.uuid4())
        solution_data = {
            "id": solution_id,
            "timestamp": datetime.now(),
            "input": {"type": "code", "content": input_data.code, "language": input_data.language},
            "result": result
        }
        save_solution_to_session(session_id, solution_data)
        
        # Prepare the response
        response = {
            "session_id": session_id,
            "solution_id": solution_id,
            "problem_domain": result["domain"],
            "solution": result["solution"],
            "explanation": result.get("explanation"),
            "reasoning_steps": result.get("reasoning_steps"),
            "confidence": result.get("confidence", 0.0),
            "alternative_solutions": result.get("alternative_solutions"),
            "metadata": result.get("metadata")
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing code input: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing code input: {str(e)}")

@app.post("/feedback", response_model=Dict[str, Any])
async def provide_feedback(input_data: FeedbackInput):
    """Provide feedback on a solution"""
    try:
        if input_data.session_id not in active_sessions:
            raise HTTPException(status_code=404, detail=f"Session {input_data.session_id} not found")
        
        update_session_activity(input_data.session_id)
        
        # Find the solution in the session
        session = active_sessions[input_data.session_id]
        solution_found = False
        
        for solution in session["solutions"]:
            if solution["id"] == input_data.solution_id:
                # Add feedback to the solution
                solution["feedback"] = {
                    "rating": input_data.rating,
                    "feedback_text": input_data.feedback_text,
                    "timestamp": datetime.now()
                }
                solution_found = True
                break
        
        if not solution_found:
            raise HTTPException(status_code=404, detail=f"Solution {input_data.solution_id} not found in session {input_data.session_id}")
        
        # Process the feedback with the agent
        get_agent().process_feedback(
            session_id=input_data.session_id,
            solution_id=input_data.solution_id,
            rating=input_data.rating,
            feedback_text=input_data.feedback_text
        )
        
        return {"status": "success", "message": "Feedback received"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

@app.get("/sessions/{session_id}", response_model=Dict[str, Any])
async def get_session(session_id: str):
    """Get information about a session"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        update_session_activity(session_id)
        
        session = active_sessions[session_id]
        
        # Prepare the response (exclude large data like images/audio)
        response = {
            "session_id": session_id,
            "created_at": session["created_at"].isoformat(),
            "last_active": session["last_active"].isoformat(),
            "solution_count": len(session["solutions"]),
            "solutions": [
                {
                    "id": sol["id"],
                    "timestamp": sol["timestamp"].isoformat(),
                    "input_type": sol["input"]["type"],
                    "domain": sol["result"]["domain"],
                    "has_feedback": "feedback" in sol
                }
                for sol in session["solutions"]
            ]
        }
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")

@app.get("/sessions/{session_id}/solutions/{solution_id}", response_model=Dict[str, Any])
async def get_solution(session_id: str, solution_id: str):
    """Get a specific solution from a session"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        update_session_activity(session_id)
        
        session = active_sessions[session_id]
        solution = None
        
        for sol in session["solutions"]:
            if sol["id"] == solution_id:
                solution = sol
                break
        
        if not solution:
            raise HTTPException(status_code=404, detail=f"Solution {solution_id} not found in session {session_id}")
        
        # Prepare the response
        response = {
            "id": solution["id"],
            "timestamp": solution["timestamp"].isoformat(),
            "input": {
                "type": solution["input"]["type"],
                # Include content only for text and code inputs
                "content": solution["input"]["content"] if solution["input"]["type"] in ["text", "code"] else "[Binary data]"
            },
            "result": solution["result"],
            "feedback": solution.get("feedback")
        }
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving solution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving solution: {str(e)}")

@app.post("/sessions/{session_id}/refine/{solution_id}", response_model=SolverResponse)
async def refine_solution(session_id: str, solution_id: str, refinement: TextInput):
    """Refine an existing solution based on feedback"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        update_session_activity(session_id)
        
        # Find the original solution
        session = active_sessions[session_id]
        original_solution = None
        
        for sol in session["solutions"]:
            if sol["id"] == solution_id:
                original_solution = sol
                break
        
        if not original_solution:
            raise HTTPException(status_code=404, detail=f"Solution {solution_id} not found in session {session_id}")
        
        # Process the refinement
        result = get_agent().refine_solution(
            session_id=session_id,
            solution_id=solution_id,
            feedback_text=refinement.text,
            options=refinement.options or {}
        )
        
        # Save the refined solution to the session
        new_solution_id = str(uuid.uuid4())
        solution_data = {
            "id": new_solution_id,
            "timestamp": datetime.now(),
            "input": {
                "type": "refinement",
                "content": refinement.text,
                "original_solution_id": solution_id
            },
            "result": result
        }
        save_solution_to_session(session_id, solution_data)
        
        # Prepare the response
        response = {
            "session_id": session_id,
            "solution_id": new_solution_id,
            "problem_domain": result["domain"],
            "solution": result["solution"],
            "explanation": result.get("explanation"),
            "reasoning_steps": result.get("reasoning_steps"),
            "confidence": result.get("confidence", 0.0),
            "alternative_solutions": result.get("alternative_solutions"),
            "metadata": result.get("metadata")
        }
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refining solution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error refining solution: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Run the API server
def start_api_server(agent=None, settings=None, host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server
    
    Args:
        agent: ProblemSolverAgent instance (optional)
        settings: Settings instance (optional)
        host: Host to bind the server to
        port: Port to bind the server to
    """
    # If agent is provided, use it instead of the global one
    if agent is not None:
        global app
        app.state.agent = agent
    
    # If settings are provided, update the global settings
    if settings is not None:
        # Update the global settings
        globals()['settings'] = settings
    
    uvicorn.run("interfaces.api:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    # This is used when running the file directly
    port = int(os.environ.get("PORT", 8000))
    start_api_server(port=port)