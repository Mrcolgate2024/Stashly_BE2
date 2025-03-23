from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

from graph.orchestrator_graph import orchestrator_graph
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()

# Allow frontend (adjust origin for deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
session_store = {}

# --- Data Models ---

class ChatInput(BaseModel):
    session_id: str | None = None
    message: str
    persona: str = "friendly"  # or "rogue"

# --- Endpoint ---

@app.post("/chat")
async def chat_endpoint(input: ChatInput):
    # Use existing or generate new session
    session_id = input.session_id or str(uuid.uuid4())

    # Load chat history or start fresh
    messages = session_store.get(session_id, [])
    messages.append(HumanMessage(content=input.message))

    # Build LangGraph state
    state = {
        "session_id": session_id,
        "messages": messages,
        "persona_mode": input.persona
    }

    # Invoke orchestrator LangGraph
    result = orchestrator_graph.invoke(state)

    # Append AI message to history
    messages += result["messages"]
    session_store[session_id] = messages

    # Extract and return AI reply
    ai_reply = next((m for m in result["messages"] if isinstance(m, AIMessage)), None)
    return {
        "session_id": session_id,
        "response": ai_reply.content if ai_reply else "Sorry, I had trouble generating a response."
    }
