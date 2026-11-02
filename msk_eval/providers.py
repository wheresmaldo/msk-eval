"""Talk to AI models.

Every provider has one method, complete(model, prompt) -> (text, Usage).
Databricks serving endpoints, OpenAI and Anthropic all speak the OpenAI chat
format, so one client covers them; only the address and key change.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Protocol

from .schema import Usage


class Provider(Protocol):
    def complete(self, model: str, prompt: str) -> tuple[str, Usage]: ...


class OpenAICompatible:
    def __init__(self, base_url: str | None, api_key: str, temperature: float = 0.0, max_tokens: int = 800):
        from openai import OpenAI  # imported here so tests don't need network libraries

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, model: str, prompt: str) -> tuple[str, Usage]:
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = resp.choices[0].message.content or ""
        u = resp.usage
        return text, Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
            seconds=round(time.perf_counter() - start, 2),
        )


def databricks(host: str | None = None, token: str | None = None) -> OpenAICompatible:
    """Databricks Model Serving. Model names are serving endpoint names."""
    host = (host or os.environ["DATABRICKS_HOST"]).rstrip("/")
    if not host.startswith("http"):
        host = "https://" + host
    return OpenAICompatible(base_url=f"{host}/serving-endpoints", api_key=token or os.environ["DATABRICKS_TOKEN"])


class DatabricksNotebook:
    """Inside a Databricks notebook: uses the notebook's own login, so no token to manage."""

    def __init__(self, temperature: float = 0.0, max_tokens: int = 800):
        from mlflow.deployments import get_deploy_client

        self.client = get_deploy_client("databricks")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, model: str, prompt: str) -> tuple[str, Usage]:
        start = time.perf_counter()
        resp = self.client.predict(endpoint=model, inputs={
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature, "max_tokens": self.max_tokens,
        })
        content = resp["choices"][0]["message"]["content"]
        if isinstance(content, list):  # some models return a list of content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        u = resp.get("usage") or {}
        return content or "", Usage(u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0,
                                    round(time.perf_counter() - start, 2))


def openai_direct() -> OpenAICompatible:
    return OpenAICompatible(base_url=None, api_key=os.environ["OPENAI_API_KEY"])


def anthropic_direct() -> OpenAICompatible:
    return OpenAICompatible(base_url="https://api.anthropic.com/v1/", api_key=os.environ["ANTHROPIC_API_KEY"])


class FakeProvider:
    """Returns canned replies. Lets tests and dry runs work with no keys and no cost."""

    def __init__(self, reply: Callable[[str, str], str]):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, model: str, prompt: str) -> tuple[str, Usage]:
        self.calls.append((model, prompt))
        return self.reply(model, prompt), Usage(input_tokens=len(prompt) // 4, output_tokens=50)
