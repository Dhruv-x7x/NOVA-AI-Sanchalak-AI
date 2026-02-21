import os
import time
import logging
from typing import Dict, Any, Optional, List
from config.llm_config import get_config

logger = logging.getLogger(__name__)


class LLMResponse:
    def __init__(self, content: str, model: str, total_tokens: int = 0, latency_ms: float = 0):
        self.content = content
        self.model = model
        self.total_tokens = total_tokens
        self.latency_ms = latency_ms


class LLMWrapper:
    """
    Multi-provider LLM interface.
    Provider chain: Gemini (primary) → Groq (fallback) → vLLM (legacy).
    """

    def __init__(self):
        self.config = get_config()
        self._gemini_client = None
        self._groq_client = None
        self._vllm_client = None
        self._init_clients()

    # ── Initialisation ──────────────────────────────────────────

    def _init_clients(self):
        """Initialise available provider clients."""
        # Gemini
        if self.config.gemini.api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.config.gemini.api_key)
                logger.info(f"Gemini client initialised → model={self.config.gemini.model}")
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")

        # Groq (OpenAI-compatible)
        if self.config.groq.api_key:
            try:
                from openai import OpenAI
                self._groq_client = OpenAI(
                    base_url=self.config.groq.base_url,
                    api_key=self.config.groq.api_key,
                )
                logger.info("Groq client initialised")
            except Exception as e:
                logger.warning(f"Groq init failed: {e}")

        # vLLM (legacy)
        try:
            from openai import OpenAI
            self._vllm_client = OpenAI(
                base_url=self.config.vllm.base_url,
                api_key=self.config.vllm.api_key,
                default_headers={"bypass-tunnel-reminder": "true"},
            )
            logger.info(f"vLLM client initialised → {self.config.vllm.base_url}")
        except Exception as e:
            logger.warning(f"vLLM init failed: {e}")

    # ── Provider helpers ────────────────────────────────────────

    def _generate_gemini(self, prompt: str, system_prompt: str) -> LLMResponse:
        """Generate using Google Gemini."""
        start = time.time()
        contents = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = self._gemini_client.models.generate_content(
            model=self.config.gemini.model,
            contents=contents,
        )
        latency = (time.time() - start) * 1000
        content = response.text or ""
        tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens = getattr(response.usage_metadata, "total_token_count", 0)
        return LLMResponse(
            content=content,
            model=f"gemini/{self.config.gemini.model}",
            total_tokens=tokens,
            latency_ms=latency,
        )

    def _generate_groq(self, prompt: str, system_prompt: str) -> LLMResponse:
        """Generate using Groq (OpenAI-compatible)."""
        start = time.time()
        chat = self._groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=self.config.groq.model,
            temperature=0.7,
            max_tokens=4096,
        )
        latency = (time.time() - start) * 1000
        content = chat.choices[0].message.content or ""
        tokens = chat.usage.total_tokens if chat.usage else 0
        return LLMResponse(
            content=content,
            model=f"groq/{self.config.groq.model}",
            total_tokens=tokens,
            latency_ms=latency,
        )

    def _generate_vllm(self, prompt: str, system_prompt: str, model: Optional[str] = None) -> LLMResponse:
        """Generate using vLLM (OpenAI-compatible)."""
        start = time.time()
        target_model = model or self.config.vllm.model
        chat = self._vllm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=target_model,
            temperature=0.7,
            max_tokens=4096,
            extra_body={
                "repetition_penalty": 1.05,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.1,
                "top_p": 0.9,
            },
        )
        latency = (time.time() - start) * 1000
        content = chat.choices[0].message.content or ""
        tokens = chat.usage.total_tokens if chat.usage else 0
        return LLMResponse(
            content=content,
            model=f"vllm/{target_model}",
            total_tokens=tokens,
            latency_ms=latency,
        )

    # ── Public API ──────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        model: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a response using the configured provider chain."""
        providers = self._get_provider_chain()

        for name, fn in providers:
            try:
                if name == "vllm":
                    return fn(prompt, system_prompt, model)
                return fn(prompt, system_prompt)
            except Exception as e:
                logger.warning(f"{name} generation failed: {e}")

        # All providers exhausted
        return LLMResponse(
            content=(
                "[AI SERVICE OFFLINE] All configured AI providers (Gemini, Groq, vLLM) are "
                "currently unreachable. Please check your API keys and network connectivity."
            ),
            model="offline-fallback",
        )

    def _get_provider_chain(self) -> List:
        """Return ordered list of (name, callable) based on config."""
        chain = []
        order = [self.config.primary_provider, self.config.secondary_provider, "vllm"]
        seen = set()
        for p in order:
            if p in seen:
                continue
            seen.add(p)
            if p == "gemini" and self._gemini_client:
                chain.append(("gemini", self._generate_gemini))
            elif p == "groq" and self._groq_client:
                chain.append(("groq", self._generate_groq))
            elif p == "vllm" and self._vllm_client:
                chain.append(("vllm", self._generate_vllm))
        return chain

    def health_check(self) -> Dict[str, Any]:
        """Check availability of all configured providers."""
        result: Dict[str, Any] = {}

        # Gemini
        gemini_ok = False
        gemini_err = None
        if self._gemini_client:
            try:
                self._gemini_client.models.generate_content(
                    model=self.config.gemini.model,
                    contents="ping",
                )
                gemini_ok = True
            except Exception as e:
                gemini_err = str(e)
        result["gemini"] = {
            "available": gemini_ok,
            "error": gemini_err,
            "model": self.config.gemini.model,
            "is_primary": self.config.primary_provider == "gemini",
        }

        # Groq
        groq_ok = False
        groq_err = None
        if self._groq_client:
            try:
                self._groq_client.models.list()
                groq_ok = True
            except Exception as e:
                groq_err = str(e)
        result["groq"] = {
            "available": groq_ok,
            "error": groq_err,
            "model": self.config.groq.model,
            "is_primary": self.config.primary_provider == "groq",
        }

        # vLLM
        vllm_ok = False
        vllm_err = None
        models: List[str] = []
        if self._vllm_client:
            try:
                model_list = self._vllm_client.models.list()
                models = [m.id for m in model_list.data]
                vllm_ok = True
            except Exception as e:
                vllm_err = str(e)
        result["vllm"] = {
            "available": vllm_ok,
            "error": vllm_err,
            "base_url": self.config.vllm.base_url,
            "model": self.config.vllm.model,
            "available_models": models,
        }

        return result


def get_llm():
    return LLMWrapper()
