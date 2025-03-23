from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uuid
import asyncio
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from graph.orchestrator_graph import orchestrator_graph
from utils.token_utils import trim_messages_to_fit_token_limit

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
session_store = {}

class ChatInput(BaseModel):
    session_id: Optional[str] = None
    message: str
    persona: str = "friendly"

@app.post("/chat")
async def chat_endpoint(input: ChatInput):
    session_id = input.session_id or str(uuid.uuid4())
    messages = session_store.get(session_id, [])
    messages.append(HumanMessage(content=input.message))

    # Trim messages if token budget is exceeded (e.g., 4096 or 8192)
    trimmed_messages = trim_messages_to_fit_token_limit(messages, max_tokens=4000)
    state = {
        "session_id": session_id,
        "messages": trimmed_messages,
        "persona_mode": input.persona
    }

    result = orchestrator_graph.invoke(state)
    messages += result["messages"]
    session_store[session_id] = messages

    ai_msg = next((m for m in result["messages"] if isinstance(m, AIMessage)), None)
    return {
        "session_id": session_id,
        "response": ai_msg.content if ai_msg else "Sorry, I couldn't generate a response."
    }

@app.post("/chat/stream")
async def stream_chat(input: ChatInput):
    session_id = input.session_id or str(uuid.uuid4())
    messages = session_store.get(session_id, [])
    messages.append(HumanMessage(content=input.message))
    trimmed_messages = trim_messages_to_fit_token_limit(messages, max_tokens=4000)

    state = {
        "session_id": session_id,
        "messages": trimmed_messages,
        "persona_mode": input.persona
    }

    async def stream():
        async for event in orchestrator_graph.stream(state):
            for _, value in event.items():
                if isinstance(value, list):
                    for msg in value:
                        if isinstance(msg, AIMessage):
                            yield f"data: {msg.content}\n\n"
                elif isinstance(value, AIMessage):
                    yield f"data: {value.content}\n\n"
            await asyncio.sleep(0.1)
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/sessions")
async def list_sessions():
    return {"sessions": list(session_store.keys())}
