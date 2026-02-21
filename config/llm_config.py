
from pydantic import BaseModel
from typing import Optional
import os
from pathlib import Path

# Load .env so os.getenv() picks up API keys even when
# this module is imported outside of the FastAPI pydantic-settings flow.
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on real env vars


class GeminiConfig(BaseModel):
    """Configuration for Google Gemini (primary provider)."""
    api_key: str = ""
    model: str = "gemini-3-flash-preview"


class GroqConfig(BaseModel):
    """Configuration for Groq (fallback provider)."""
    api_key: str = ""
    model: str = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com/openai/v1"


class VLLMConfig(BaseModel):
    """Configuration for vLLM (OpenAI-compatible API on Lightning AI H200 via ngrok)."""
    base_url: str = "https://overglad-queen-unstapled.ngrok-free.dev/v1"
    model: str = "parzivalprime/TrialPulse-8B-Zenith-V2.1"
    api_key: str = "EMPTY"


class LLMConfig(BaseModel):
    primary_provider: str = "gemini"
    secondary_provider: str = "groq"
    gemini: GeminiConfig = GeminiConfig()
    groq: GroqConfig = GroqConfig()
    vllm: VLLMConfig = VLLMConfig()


def get_config() -> LLMConfig:
    """Load config, preferring environment variables."""
    return LLMConfig(
        primary_provider=os.getenv("AI_PRIMARY_PROVIDER", "gemini"),
        secondary_provider=os.getenv("AI_SECONDARY_PROVIDER", "groq"),
        gemini=GeminiConfig(
            api_key=os.getenv("GOOGLE_API_KEY", ""),
            model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        ),
        groq=GroqConfig(
            api_key=os.getenv("GROQ_API_KEY", ""),
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            base_url="https://api.groq.com/openai/v1",
        ),
        vllm=VLLMConfig(
            base_url=os.getenv("VLLM_BASE_URL", VLLMConfig().base_url),
            model=os.getenv("VLLM_MODEL", VLLMConfig().model),
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        ),
    )
