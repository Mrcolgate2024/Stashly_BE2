from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from config.settings import persona_prompts, Settings
from agents.return_agent import run_return_agent
from agents.market_summary_agent import run_market_summary_agent
from agents.fund_transparency_agent import run_fund_transparency_agent
from agents.research_agent.run_research_agent import run_research_agent

settings = Settings()

# Define the state for the graph
class OrchestratorState(dict):
    session_id: str
    messages: list
    persona_mode: str  # 'friendly' or 'rogue'
    output: str

def orchestrator_router(state: OrchestratorState):
    """Route user message to the correct agent based on intent."""
    last_message = state['messages'][-1].content.lower()

    if any(keyword in last_message for keyword in ["return", "performance", "volatility", "sharpe"]):
        return "return_agent"
    elif any(keyword in last_message for keyword in ["market summary", "what happened", "macro", "news"]):
        return "market_summary_agent"
    elif any(keyword in last_message for keyword in ["exposure", "fund", "holding", "xml"]):
        return "fund_transparency_agent"
    elif any(keyword in last_message for keyword in ["research", "report", "analyst", "study", "analyze"]):
        return "research_agent"
    else:
        return "return_agent"  # default fallback

def orchestrator_handler(state: OrchestratorState):
    """Pre-process input, inject persona prompt, post-process output."""
    messages = state['messages']
    persona_prompt = persona_prompts.get(state['persona_mode'], persona_prompts['friendly'])

    # Prepend system message with persona style
    full_prompt = [
        {"role": "system", "content": persona_prompt},
        *[{"role": "user", "content": msg.content} for msg in messages if isinstance(msg, HumanMessage)]
    ]

    return {
        "session_id": state["session_id"],
        "messages": messages,
        "persona_mode": state["persona_mode"]
    }

# Build the LangGraph
builder = StateGraph(OrchestratorState)
builder.add_node("return_agent", run_return_agent)
builder.add_node("market_summary_agent", run_market_summary_agent)
builder.add_node("fund_transparency_agent", run_fund_transparency_agent)
builder.add_node("research_agent", run_research_agent)

builder.set_entry_point("orchestrator_handler")
builder.add_node("orchestrator_handler", orchestrator_handler)
builder.add_conditional_edges(
    "orchestrator_handler",
    orchestrator_router,
    {
        "return_agent": END,
        "market_summary_agent": END,
        "fund_transparency_agent": END,
        "research_agent": END,
    }
)

orchestrator_graph = builder.compile(checkpointer=MemorySaver())
