from __future__ import annotations

import os
from typing import Optional

from chell.models.base import ModelBackend

_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {
        "model": "claude-sonnet-4-6",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
    },
    "gemini": {
        "model": "gemini-1.5-pro",
        "env_key": "GOOGLE_API_KEY",
    },
}


class APIBackend(ModelBackend):
    """Hosted-API backend supporting Anthropic, OpenAI, and Gemini providers.

    Parameters
    ----------
    provider:
        One of "anthropic", "openai", or "gemini".
    model:
        Model identifier. Defaults to the provider's recommended model when
        None is passed.
    api_key:
        Explicit API key. Falls back to the relevant environment variable
        (ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY) when None.
    max_tokens:
        Maximum tokens to generate per call (default 2048).
    temperature:
        Sampling temperature (default 0.2).
    """

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> None:
        provider = provider.lower()
        if provider not in _PROVIDER_DEFAULTS:
            raise ValueError(
                f"Unknown provider {provider!r}. "
                f"Supported providers: {list(_PROVIDER_DEFAULTS)}"
            )

        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature

        defaults = _PROVIDER_DEFAULTS[provider]
        self.model = model or defaults["model"]

        resolved_key = api_key or os.environ.get(defaults["env_key"])
        self._api_key = resolved_key

        # Initialise the provider client eagerly so missing credentials surface
        # at construction time rather than at first generate() call.
        self._client = self._build_client(provider, resolved_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self, provider: str, api_key: Optional[str]):
        if provider == "anthropic":
            try:
                import anthropic  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "The 'anthropic' package is required for provider='anthropic'. "
                    "Install it with: pip install anthropic"
                ) from exc
            return anthropic.Anthropic(api_key=api_key)

        if provider == "openai":
            try:
                import openai  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "The 'openai' package is required for provider='openai'. "
                    "Install it with: pip install openai"
                ) from exc
            return openai.OpenAI(api_key=api_key)

        if provider == "gemini":
            try:
                import google.generativeai as genai  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "The 'google-generativeai' package is required for provider='gemini'. "
                    "Install it with: pip install google-generativeai"
                ) from exc
            genai.configure(api_key=api_key)
            return genai

        # Unreachable given the guard in __init__, but keeps type checkers happy.
        raise ValueError(f"Unknown provider: {provider!r}")

    # ------------------------------------------------------------------
    # ModelBackend interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Generate text via the configured provider API."""
        effective_max_tokens = max_tokens or self.max_tokens
        effective_temperature = temperature if temperature != 0.2 else self.temperature

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        if self.provider == "gemini":
            import google.generativeai as genai  # type: ignore[import]

            gen_model = genai.GenerativeModel(self.model)
            response = gen_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                ),
            )
            return response.text

        raise ValueError(f"Unknown provider: {self.provider!r}")

    def embed(self, text: str) -> list[float]:  # noqa: ARG002
        raise NotImplementedError(
            "Use a local embedding model for .embed() — APIBackend does not support embeddings"
        )

    @property
    def torch_model(self):
        """API backends have no local nn.Module."""
        return None
