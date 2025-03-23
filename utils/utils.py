from langchain_core.messages import BaseMessage
from typing import List

# Simple token estimator (can be replaced with tiktoken if needed)
def estimate_tokens(messages: List[BaseMessage]) -> int:
    return sum(len(m.content.split()) for m in messages)  # ~1 token per word

def trim_messages_to_fit_token_limit(messages: List[BaseMessage], max_tokens: int = 4000) -> List[BaseMessage]:
    if estimate_tokens(messages) <= max_tokens:
        return messages

    # Trim from start until it fits
    trimmed = messages[:]
    while trimmed and estimate_tokens(trimmed) > max_tokens:
        trimmed.pop(0)
    return trimmed
