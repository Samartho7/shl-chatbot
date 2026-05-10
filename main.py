"""
SHL Assessment Recommendation API
===================================
Endpoints:
  GET  /health  →  {"status": "ok"}
  POST /chat    →  {"reply": "...", "recommendations": [...], "end_of_conversation": false}

Chat request body:
  {"messages": [{"role": "user"|"assistant", "content": "..."}]}

Start server:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import uvicorn

from chatbot import process_conversation

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommendation Chatbot",
    description=(
        "Conversational AI that helps recruiters choose the right SHL assessments. "
        "Stateless — full conversation history is supplied in every /chat request."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Startup: pre-load FAISS index + embedding model so first /chat isn't slow
# ─────────────────────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm resources on boot (avoids 30s timeout on first /chat)."""
    import sys
    print("[startup] Pre-loading FAISS index and embedding model...", file=sys.stderr)
    from chatbot import _load_resources
    _load_resources()
    print("[startup] Ready.", file=sys.stderr)
    yield  # server runs here

app.router.lifespan_context = lifespan


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role:    Literal["user", "assistant"] = Field(
        ..., description="Who sent this message: 'user' or 'assistant'"
    )
    content: str = Field(..., description="The message text")


class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        ...,
        description="Full conversation history (oldest first). Every request must include all prior turns.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": "user", "content": "I want to hire a Java developer."},
                    {"role": "assistant", "content": "Sure! What experience level — junior, mid, or senior?"},
                    {"role": "user", "content": "Mid-level. I need both a coding test and a personality assessment."},
                ]
            }
        }
    }


class Recommendation(BaseModel):
    name:      str = Field(..., description="Assessment name as shown in SHL catalog")
    url:       str = Field(..., description="Direct link to the assessment product page")
    test_type: str = Field(..., description="Comma-joined SHL type code(s), e.g. 'K' or 'A, P'")


class ChatResponse(BaseModel):
    reply:               str                  = Field(..., description="Assistant's conversational reply")
    recommendations:     list[Recommendation] = Field(..., description="0–10 recommended assessments")
    end_of_conversation: bool                 = Field(..., description="True when the session is complete")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get(
    "/health",
    summary="Health check",
    response_description="Service liveness indicator",
)
def health() -> dict:
    """Simple liveness probe. Returns 200 + {status: ok} when the service is up."""
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a chat message and receive recommendations",
    response_description="Assistant reply, optional recommendations, and conversation-end flag",
)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless conversational endpoint.

    - Supply the **full** conversation history in `messages` on every call.
    - The assistant will ask clarifying questions until it has enough context,
      then recommend relevant SHL assessments grounded in the catalog.
    - `end_of_conversation` is `true` when the recruiter has finished.
    """
    try:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        result   = process_conversation(messages)

        # Normalise recommendations to Recommendation objects
        recs = [
            Recommendation(
                name      = r["name"],
                url       = r["url"],
                test_type = r["test_type"],
            )
            for r in result.get("recommendations", [])
        ]

        return ChatResponse(
            reply               = result["reply"],
            recommendations     = recs,
            end_of_conversation = result["end_of_conversation"],
        )

    except RuntimeError as exc:
        # Missing API key or resource file
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
