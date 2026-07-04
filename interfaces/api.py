import os

from dotenv import load_dotenv

# NOTE: KMP_DUPLICATE_LIB_OK is set conditionally in start_api_server() for dev only

load_dotenv()  # Load .env variables (MISTRAL_API_KEY, etc.)

import base64
import logging
import sys
import tempfile
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Settings
from core.agent import ProblemSolverAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Lazy-loaded globals (deferred so load_dotenv() runs first)
_settings = None
_agent = None


def _get_settings():
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def _get_agent():
    """Get the agent instance, initializing lazily on first call"""
    global _agent
    if hasattr(app.state, "agent") and app.state.agent is not None:
        return app.state.agent
    if _agent is None:
        _agent = ProblemSolverAgent(settings=_get_settings())
    return _agent


# ── Structured error response ────────────────────────────────────
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type/code")
    detail: str = Field(..., description="Human-readable error message")
    request_id: Optional[str] = Field(None, description="Request tracking ID")


# ── API Key Security ─────────────────────────────────────────────
# Set STACKMIND_API_KEY in your .env file. Leave unset to disable auth (dev mode).
_API_KEY = os.getenv("STACKMIND_API_KEY")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(_api_key_header)) -> None:
    """Dependency that validates the X-API-Key header when STACKMIND_API_KEY is configured."""
    if _API_KEY and api_key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


# Create FastAPI app
app = FastAPI(
    title="StackMind API",
    description="StackMind — Multi-step reasoning and problem-solving AI agent API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS — configurable via environment
_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handler ─────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return all HTTP errors as structured ErrorResponse JSON."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            detail=exc.detail,
            request_id=request.headers.get("X-Request-Id"),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unexpected server errors."""
    logger.error("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            detail="An unexpected error occurred. Check server logs.",
            request_id=request.headers.get("X-Request-Id"),
        ).model_dump(),
    )


# ── Startup: Eager Agent Warmup ──────────────────────────────────
@app.on_event("startup")
async def _startup_warmup() -> None:
    """Pre-initialize the agent on server startup to eliminate first-request latency."""
    logger.info("Warming up ProblemSolverAgent on startup...")
    _get_agent()
    logger.info("ProblemSolverAgent ready.")


# Alias for all route handlers — enforces API key on every call
def get_agent(auth: None = Depends(verify_api_key)):
    return _get_agent()


# Define request and response models
class TextInput(BaseModel):
    text: str = Field(..., description="Text input for the problem solver")
    session_id: Optional[str] = Field(
        None, description="Session ID for continuing a conversation"
    )
    domain: Optional[str] = Field(
        None, description="Optional domain hint (math, code, science, general)"
    )
    options: Optional[Dict[str, Any]] = Field(
        None, description="Additional options for processing"
    )


class ImageInput(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    session_id: Optional[str] = Field(
        None, description="Session ID for continuing a conversation"
    )
    domain: Optional[str] = Field(
        None, description="Optional domain hint (math, code, science, general)"
    )
    options: Optional[Dict[str, Any]] = Field(
        None, description="Additional options for processing"
    )


class VoiceInput(BaseModel):
    audio_data: str = Field(..., description="Base64 encoded audio data")
    session_id: Optional[str] = Field(
        None, description="Session ID for continuing a conversation"
    )
    domain: Optional[str] = Field(
        None, description="Optional domain hint (math, code, science, general)"
    )
    options: Optional[Dict[str, Any]] = Field(
        None, description="Additional options for processing"
    )


class CodeInput(BaseModel):
    code: str = Field(..., description="Code input for the problem solver")
    language: Optional[str] = Field(
        None, description="Programming language of the code"
    )
    session_id: Optional[str] = Field(
        None, description="Session ID for continuing a conversation"
    )
    options: Optional[Dict[str, Any]] = Field(
        None, description="Additional options for processing"
    )


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
    reasoning_steps: Optional[List[str]] = Field(
        None, description="Step-by-step reasoning"
    )
    confidence: float = Field(..., description="Confidence score for the solution")
    alternative_solutions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Alternative solutions if available"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata about the solution"
    )


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
        "solutions": [],
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


def normalize_reasoning_steps(steps) -> Optional[List[str]]:
    """Convert reasoning steps from List[Dict] to List[str] for the API response."""
    if not steps:
        return None
    result = []
    for step in steps:
        if isinstance(step, dict):
            result.append(step.get("content", str(step)))
        else:
            result.append(str(step))
    return result


# API endpoints
@app.get("/")
async def root():
    return {
        "message": "StackMind API is running",
        "version": "1.0.0",
        "docs": "/api/docs",
    }


# ── API v1 Router ────────────────────────────────────────────────
from fastapi import APIRouter

v1 = APIRouter(prefix="/api/v1")


# Legacy route (backwards compat) + versioned route
@app.post("/solve/text", response_model=SolverResponse, include_in_schema=False)
@v1.post("/solve/text", response_model=SolverResponse, tags=["Solve"])
async def solve_text_problem(input_data: TextInput):
    """Solve a problem from text input"""
    try:
        session_id = get_or_create_session_id(input_data.session_id)
        update_session_activity(session_id)

        # Process the text input
        result = get_agent().solve_problem(
            input_data=input_data.text, input_type="text"
        )

        # Save the solution to the session
        solution_id = str(uuid.uuid4())
        solution_data = {
            "id": solution_id,
            "timestamp": datetime.now(),
            "input": {"type": "text", "content": input_data.text},
            "result": result,
        }
        save_solution_to_session(session_id, solution_data)

        # Prepare the response
        response = {
            "session_id": session_id,
            "solution_id": solution_id,
            "problem_domain": result["domain"],
            "solution": result["solution"],
            "explanation": result.get("explanation"),
            "reasoning_steps": normalize_reasoning_steps(result.get("reasoning_steps")),
            "confidence": result.get("confidence", 0.0),
            "alternative_solutions": result.get("alternative_solutions"),
            "metadata": result.get("metadata"),
        }

        return response

    except Exception as e:
        logger.error(f"Error processing text input: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing text input: {str(e)}"
        )


@app.post("/solve/image", response_model=SolverResponse, include_in_schema=False)
@v1.post("/solve/image", response_model=SolverResponse, tags=["Solve"])
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
            result = get_agent().solve_problem(
                input_data=temp_file_path, input_type="image"
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
            "input": {
                "type": "image",
                "content": "[Image data]",
            },  # Don't store the full image data
            "result": result,
        }
        save_solution_to_session(session_id, solution_data)

        # Prepare the response
        response = {
            "session_id": session_id,
            "solution_id": solution_id,
            "problem_domain": result["domain"],
            "solution": result["solution"],
            "explanation": result.get("explanation"),
            "reasoning_steps": normalize_reasoning_steps(result.get("reasoning_steps")),
            "confidence": result.get("confidence", 0.0),
            "alternative_solutions": result.get("alternative_solutions"),
            "metadata": result.get("metadata"),
        }

        return response

    except Exception as e:
        logger.error(f"Error processing image input: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing image input: {str(e)}"
        )


@app.post("/solve/voice", response_model=SolverResponse, include_in_schema=False)
@v1.post("/solve/voice", response_model=SolverResponse, tags=["Solve"])
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
            result = get_agent().solve_problem(
                input_data=temp_file_path, input_type="voice"
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
            "input": {
                "type": "voice",
                "content": "[Audio data]",
            },  # Don't store the full audio data
            "result": result,
        }
        save_solution_to_session(session_id, solution_data)

        # Prepare the response
        response = {
            "session_id": session_id,
            "solution_id": solution_id,
            "problem_domain": result["domain"],
            "solution": result["solution"],
            "explanation": result.get("explanation"),
            "reasoning_steps": normalize_reasoning_steps(result.get("reasoning_steps")),
            "confidence": result.get("confidence", 0.0),
            "alternative_solutions": result.get("alternative_solutions"),
            "metadata": result.get("metadata"),
        }

        return response

    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing voice input: {str(e)}"
        )


@app.post("/solve/code", response_model=SolverResponse, include_in_schema=False)
@v1.post("/solve/code", response_model=SolverResponse, tags=["Solve"])
async def solve_code_problem(input_data: CodeInput):
    """Solve a problem from code input"""
    try:
        session_id = get_or_create_session_id(input_data.session_id)
        update_session_activity(session_id)

        # Process the code input
        options = input_data.options or {}
        if input_data.language:
            options["language"] = input_data.language

        result = get_agent().solve_problem(
            input_data=input_data.code, input_type="code"
        )

        # Save the solution to the session
        solution_id = str(uuid.uuid4())
        solution_data = {
            "id": solution_id,
            "timestamp": datetime.now(),
            "input": {
                "type": "code",
                "content": input_data.code,
                "language": input_data.language,
            },
            "result": result,
        }
        save_solution_to_session(session_id, solution_data)

        # Prepare the response
        response = {
            "session_id": session_id,
            "solution_id": solution_id,
            "problem_domain": result["domain"],
            "solution": result["solution"],
            "explanation": result.get("explanation"),
            "reasoning_steps": normalize_reasoning_steps(result.get("reasoning_steps")),
            "confidence": result.get("confidence", 0.0),
            "alternative_solutions": result.get("alternative_solutions"),
            "metadata": result.get("metadata"),
        }

        return response

    except Exception as e:
        logger.error(f"Error processing code input: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing code input: {str(e)}"
        )


@app.post("/feedback", response_model=Dict[str, Any], include_in_schema=False)
@v1.post("/feedback", response_model=Dict[str, Any], tags=["Feedback"])
async def provide_feedback(input_data: FeedbackInput):
    """Provide feedback on a solution"""
    try:
        if input_data.session_id not in active_sessions:
            raise HTTPException(
                status_code=404, detail=f"Session {input_data.session_id} not found"
            )

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
                    "timestamp": datetime.now(),
                }
                solution_found = True
                break

        if not solution_found:
            raise HTTPException(
                status_code=404,
                detail=f"Solution {input_data.solution_id} not found in session {input_data.session_id}",
            )

        # Process the feedback with the agent
        get_agent().process_feedback(
            session_id=input_data.session_id,
            solution_id=input_data.solution_id,
            rating=input_data.rating,
            feedback_text=input_data.feedback_text,
        )

        return {"status": "success", "message": "Feedback received"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing feedback: {str(e)}"
        )


@app.get(
    "/sessions/{session_id}", response_model=Dict[str, Any], include_in_schema=False
)
@v1.get("/sessions/{session_id}", response_model=Dict[str, Any], tags=["Sessions"])
async def get_session(session_id: str):
    """Get information about a session"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(
                status_code=404, detail=f"Session {session_id} not found"
            )

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
                    "has_feedback": "feedback" in sol,
                }
                for sol in session["solutions"]
            ],
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving session: {str(e)}"
        )


@app.get(
    "/sessions/{session_id}/solutions/{solution_id}",
    response_model=Dict[str, Any],
    include_in_schema=False,
)
@v1.get(
    "/sessions/{session_id}/solutions/{solution_id}",
    response_model=Dict[str, Any],
    tags=["Sessions"],
)
async def get_solution(session_id: str, solution_id: str):
    """Get a specific solution from a session"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(
                status_code=404, detail=f"Session {session_id} not found"
            )

        update_session_activity(session_id)

        session = active_sessions[session_id]
        solution = None

        for sol in session["solutions"]:
            if sol["id"] == solution_id:
                solution = sol
                break

        if not solution:
            raise HTTPException(
                status_code=404,
                detail=f"Solution {solution_id} not found in session {session_id}",
            )

        # Prepare the response
        response = {
            "id": solution["id"],
            "timestamp": solution["timestamp"].isoformat(),
            "input": {
                "type": solution["input"]["type"],
                # Include content only for text and code inputs
                "content": (
                    solution["input"]["content"]
                    if solution["input"]["type"] in ["text", "code"]
                    else "[Binary data]"
                ),
            },
            "result": solution["result"],
            "feedback": solution.get("feedback"),
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving solution: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving solution: {str(e)}"
        )


@app.post(
    "/sessions/{session_id}/refine/{solution_id}",
    response_model=SolverResponse,
    include_in_schema=False,
)
@v1.post(
    "/sessions/{session_id}/refine/{solution_id}",
    response_model=SolverResponse,
    tags=["Solve"],
)
async def refine_solution(session_id: str, solution_id: str, refinement: TextInput):
    """Refine an existing solution based on feedback"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(
                status_code=404, detail=f"Session {session_id} not found"
            )

        update_session_activity(session_id)

        # Find the original solution
        session = active_sessions[session_id]
        original_solution = None

        for sol in session["solutions"]:
            if sol["id"] == solution_id:
                original_solution = sol
                break

        if not original_solution:
            raise HTTPException(
                status_code=404,
                detail=f"Solution {solution_id} not found in session {session_id}",
            )

        # Process the refinement — agent.refine_solution() accepts a feedback string
        result = get_agent().refine_solution(feedback=refinement.text)

        # Save the refined solution to the session
        new_solution_id = str(uuid.uuid4())
        solution_data = {
            "id": new_solution_id,
            "timestamp": datetime.now(),
            "input": {
                "type": "refinement",
                "content": refinement.text,
                "original_solution_id": solution_id,
            },
            "result": result,
        }
        save_solution_to_session(session_id, solution_data)

        # Prepare the response
        response = {
            "session_id": session_id,
            "solution_id": new_solution_id,
            "problem_domain": result["domain"],
            "solution": result["solution"],
            "explanation": result.get("explanation"),
            "reasoning_steps": normalize_reasoning_steps(result.get("reasoning_steps")),
            "confidence": result.get("confidence", 0.0),
            "alternative_solutions": result.get("alternative_solutions"),
            "metadata": result.get("metadata"),
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refining solution: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error refining solution: {str(e)}"
        )


# ── Multi-Agent Pipeline Endpoints ─────────────────────────────────


class PipelineDecomposeInput(BaseModel):
    query: str = Field(..., description="User query to decompose into sub-tasks")
    context: Optional[str] = Field(
        None, description="Optional prior conversation context"
    )


class PipelineAgentInput(BaseModel):
    task: str = Field(..., description="Sub-task for the agent to execute")
    agent_id: int = Field(..., description="Agent ID (1, 2, or 3)")
    query: str = Field(..., description="Original user query for context")


class PipelineSynthesizeInput(BaseModel):
    query: str = Field(..., description="Original user query")
    agents: List[Dict[str, Any]] = Field(
        ..., description="List of {id, task, output} from all agents"
    )


def _call_mistral_direct(
    prompt: str, max_tokens: int = 2000, temperature: float = 0.7
) -> str:
    """Make a direct Mistral API call using the existing integration with retry logic for 429 Rate Limits."""
    import time

    from integrations.mistral_integration import generate_completion

    max_retries = 5
    backoff = 2.0  # Start with 2 seconds

    for attempt in range(max_retries):
        try:
            result = generate_completion(
                prompt, temperature=temperature, max_tokens=max_tokens
            )
            if result and result.strip():
                return result.strip()

            # If result is empty, log and trigger retry
            logger.warning(
                f"Mistral returned empty result, retrying in {backoff}s... (Attempt {attempt+1}/{max_retries})"
            )
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "rate_limited" in err_str:
                logger.warning(
                    f"Mistral rate limit hit, backing off for {backoff}s... (Attempt {attempt+1}/{max_retries})"
                )
            else:
                logger.warning(
                    f"Mistral call failed: {e}, retrying in {backoff}s... (Attempt {attempt+1}/{max_retries})"
                )

        time.sleep(backoff)
        backoff *= 2.0  # Exponential backoff

    raise HTTPException(
        status_code=502,
        detail="Mistral API rate limit exceeded. Please wait a moment and try again.",
    )


@app.post("/pipeline/decompose")
async def pipeline_decompose(input_data: PipelineDecomposeInput):
    """
    Stage 2: Decompose a user query into a summary + 3 agent sub-tasks.
    Returns structured JSON: { summary, agents: [{ id, task }] }
    """
    context_block = ""
    if input_data.context:
        context_block = f"\n\nPrior conversation context:\n{input_data.context}\n"

    prompt = f"""You are a multi-agent AI orchestrator. A user has submitted the following query:

"{input_data.query}"{context_block}

Your job is to:
1. Write a clear, concise problem statement that rephrases the query (2-3 sentences max).
2. Decompose the problem into exactly 3 distinct, non-overlapping sub-tasks that together cover the full scope of the query.
3. Assign each sub-task to a named agent.

Respond ONLY with a valid JSON object in this exact format, no markdown, no extra text:
{{
  "summary": "<concise rephrased problem statement>",
  "agents": [
    {{"id": 1, "task": "<sub-task for Agent 1>"}},
    {{"id": 2, "task": "<sub-task for Agent 2>"}},
    {{"id": 3, "task": "<sub-task for Agent 3>"}}
  ]
}}"""

    raw = _call_mistral_direct(prompt, max_tokens=800, temperature=0.4)

    # Strip markdown code fences if present
    cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        data = json.loads(cleaned)
        # Validate shape
        assert "summary" in data and "agents" in data
        assert len(data["agents"]) == 3
        return data
    except Exception:
        logger.error(f"Decompose JSON parse failed. Raw output: {raw}")
        raise HTTPException(
            status_code=500, detail="LLM returned invalid decomposition JSON"
        )


@app.post("/pipeline/agent")
async def pipeline_agent(input_data: PipelineAgentInput):
    """
    Stage 3: Execute a single agent's sub-task.
    Returns: { agent_id, output }
    """
    prompt = f"""You are Agent {input_data.agent_id}, a specialized AI research assistant.

Original user query: "{input_data.query}"

Your specific assigned sub-task: "{input_data.task}"

Research and answer this sub-task thoroughly. Provide a well-structured response of 3-6 sentences. Be specific, factual, and informative. Do not refer to other agents or the broader pipeline — just answer your sub-task directly."""

    output = _call_mistral_direct(prompt, max_tokens=600, temperature=0.6)
    return {"agent_id": input_data.agent_id, "output": output}


@app.post("/pipeline/synthesize")
async def pipeline_synthesize(input_data: PipelineSynthesizeInput):
    """
    Stage 4+5: Merge 3 agent outputs into a final, synthesized answer.
    Returns: { final_answer }
    """
    agents_block = "\n\n".join(
        f"Agent {a['id']} — {a['task']}:\n{a['output']}" for a in input_data.agents
    )

    prompt = f"""You are a senior AI analyst tasked with synthesizing research from three specialist agents.

Original user query: "{input_data.query}"

Agent outputs:
{agents_block}

Your task:
- Synthesize all three agent outputs into a single, cohesive, well-structured final answer.
- The answer should flow naturally and not feel like three separate sections stitched together.
- Use paragraphs or short bullets where appropriate.
- Be comprehensive yet concise. Aim for 150-300 words.
- Do not mention the agents, pipeline, or decomposition process in your answer."""

    final_answer = _call_mistral_direct(prompt, max_tokens=1200, temperature=0.5)
    return {"final_answer": final_answer}


# ── Health ───────────────────────────────────────────────────────────
@app.get("/health")
@v1.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint with system info"""
    import platform

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "python": platform.python_version(),
        "active_sessions": len(active_sessions),
    }


# Register the versioned router
app.include_router(v1)


# Run the API server
def start_api_server(
    agent=None, settings=None, host: str = "0.0.0.0", port: int = 8010
):
    """Start the FastAPI server

    Args:
        agent: ProblemSolverAgent instance (optional)
        settings: Settings instance (optional)
        host: Host to bind the server to
        port: Port to bind the server to
    """
    # Prevent OpenMP crash with faster-whisper in dev environments
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # If agent is provided, use it instead of the global one
    if agent is not None:
        app.state.agent = agent

    # If settings are provided, update the lazy-loaded settings
    if settings is not None:
        global _settings
        _settings = settings

    uvicorn.run("interfaces.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    # This is used when running the file directly
    port = int(os.environ.get("PORT", 8010))
    start_api_server(port=port)
