"""Minimal LLM clients (plain HTTP, no vendor SDK required)."""

from __future__ import annotations

from typing import Protocol

import httpx

from ..config import Settings


class LLM(Protocol):
    async def complete(self, system: str, user: str, max_tokens: int = 400) -> str: ...


class AnthropicLLM:
    def __init__(self, http: httpx.AsyncClient, api_key: str, model: str) -> None:
        self._http, self._key, self._model = http, api_key, model

    async def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        resp = await self._http.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self._key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": self._model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        resp.raise_for_status()
        blocks = resp.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


class OpenAICompatLLM:
    def __init__(self, http: httpx.AsyncClient, api_key: str, model: str, base_url: str) -> None:
        self._http, self._key, self._model, self._base = http, api_key, model, base_url.rstrip("/")

    async def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        resp = await self._http.post(
            f"{self._base}/chat/completions",
            headers={"authorization": f"Bearer {self._key}"},
            json={
                "model": self._model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""


def build_llm(settings: Settings, http: httpx.AsyncClient) -> LLM | None:
    provider = settings.llm_provider.lower()
    if provider == "none" or not settings.llm_api_key:
        return None
    if provider == "anthropic":
        return AnthropicLLM(http, settings.llm_api_key, settings.llm_model)
    if provider == "openai":
        return OpenAICompatLLM(http, settings.llm_api_key, settings.llm_model, settings.llm_base_url)
    raise ValueError(f"unknown LLM_PROVIDER '{settings.llm_provider}'")
