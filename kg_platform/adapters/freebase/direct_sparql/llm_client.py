from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """OpenAI-compatible client settings for relation / flatten LLM filters."""

    model: str
    base_url: str | None = None
    base_url_env: str | None = "OPENAI_BASE_URL"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.6
    max_tokens: int = 4096
    stop: list[str] | None = None
    timeout: float = 10.0


class BaseModelClient:
    """Thin OpenAI-compatible chat client used by Freebase relation filters."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        api_key = cfg.api_key or os.getenv(cfg.api_key_env, "")
        base_url = cfg.base_url
        if not base_url and cfg.base_url_env:
            base_url = os.getenv(cfg.base_url_env) or None

        from openai import AsyncOpenAI, OpenAI

        client_kwargs: dict[str, Any] = {"timeout": cfg.timeout}
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url, **client_kwargs)
            self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, **client_kwargs)
            logger.info("Using API server at %s with model %s", base_url, cfg.model)
        else:
            self.client = OpenAI(api_key=api_key, **client_kwargs)
            self.async_client = AsyncOpenAI(api_key=api_key, **client_kwargs)
            logger.info("Using default OpenAI API with model %s", cfg.model)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **gen_kwargs: Any,
    ) -> str:
        if messages is not None:
            return self.generate_from_messages(messages, **gen_kwargs)
        msg_list: list[dict[str, str]] = []
        if system:
            msg_list.append({"role": "system", "content": system})
        msg_list.append({"role": "user", "content": prompt})
        return self.generate_from_messages(msg_list, **gen_kwargs)

    def generate_from_messages(self, messages: list[dict[str, str]], **gen_kwargs: Any) -> str:
        temperature = gen_kwargs.get("temperature", self.cfg.temperature)
        max_tokens = gen_kwargs.get("max_tokens", self.cfg.max_tokens)
        stop = gen_kwargs.get("stop", self.cfg.stop)
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        if not resp.choices:
            return "[No response choices returned]"
        return resp.choices[0].message.content or ""

    async def agenerate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **gen_kwargs: Any,
    ) -> str:
        if messages is not None:
            return await self.agenerate_from_messages(messages, **gen_kwargs)
        msg_list: list[dict[str, str]] = []
        if system:
            msg_list.append({"role": "system", "content": system})
        msg_list.append({"role": "user", "content": prompt})
        return await self.agenerate_from_messages(msg_list, **gen_kwargs)

    async def agenerate_from_messages(self, messages: list[dict[str, str]], **gen_kwargs: Any) -> str:
        temperature = gen_kwargs.get("temperature", self.cfg.temperature)
        max_tokens = gen_kwargs.get("max_tokens", self.cfg.max_tokens)
        stop = gen_kwargs.get("stop", self.cfg.stop)
        resp = await self.async_client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        if not resp.choices:
            return "[No response choices returned]"
        return resp.choices[0].message.content or ""
