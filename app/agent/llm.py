from __future__ import annotations

from typing import Callable, List, Optional

from app.config import get_provider_config, get_settings

# (system_prompt, user_prompt) -> raw model text
CompleteFn = Callable[[str, str], str]


def make_complete_fn(model: Optional[str] = None, temperature: float = 0.1) -> CompleteFn:
    """Build a (system, user) -> text function backed by the OpenAI-compatible chat API.

    Used by the extractor / merger / router so they can call a real LLM, while
    staying injectable (and therefore testable with fakes) in unit tests.
    """
    from openai import OpenAI  # lazy: only needed at runtime

    settings = get_settings()
    cfg = get_provider_config(settings, settings.llm_provider)
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    mdl = model or settings.llm_model

    def complete(system_prompt: str, user_prompt: str) -> str:
        resp = client.chat.completions.create(
            model=mdl,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "") if resp.choices else ""

    return complete


def make_embed_fn() -> Callable[[str], List[float]]:
    """Build a text -> embedding vector function from the configured embedder."""
    from app.embedder.embedder import build_text_embedder

    embedder = build_text_embedder()

    def embed(text: str) -> List[float]:
        vec = embedder.embed_text(text or " ", output_type="list")
        return [float(x) for x in vec]

    return embed
