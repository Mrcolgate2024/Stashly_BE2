from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

from config.settings import Settings

def get_model(settings: Settings) -> BaseChatModel:
    """
    Returns a configured LLM based on the settings.
    Priority: OpenAI > Claude > Groq
    Includes timeout, streaming, and graceful fallback.
    """
    try:
        if "gpt" in settings.model:
            return ChatOpenAI(
                model=settings.model,
                temperature=settings.temperature,
                max_retries=2,
                timeout=30,
                streaming=True
            )
        elif "claude" in settings.model:
            return ChatAnthropic(
                model=settings.model,
                temperature=settings.temperature,
                max_retries=2,
                timeout=30,
                streaming=True
            )
        elif "mixtral" in settings.model or "groq" in settings.model:
            return ChatGroq(
                model_name=settings.model,
                temperature=settings.temperature,
                max_retries=2,
                timeout=30,
                streaming=True
            )
        else:
            raise ValueError(f"Unsupported model: {settings.model}")
    except Exception as e:
        # Fallback to GPT-4o-mini if primary fails
        return ChatOpenAI(model="gpt-4o", temperature=0.3, streaming=True)
