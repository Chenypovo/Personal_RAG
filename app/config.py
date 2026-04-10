import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass
class ProviderConfig:
    provider: str
    api_key: str
    base_url: str


@dataclass
class Settings:
    llm_provider: str
    embed_provider: str
    llm_model: str
    embed_model: str
    openai_compatible_api_key: str
    openai_compatible_base_url: str


def _getenv(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def get_settings() -> Settings:
    return Settings(
        llm_provider=_getenv("LLM_PROVIDER", "openai_compatible"),
        embed_provider=_getenv("EMBED_PROVIDER", "openai_compatible"),
        llm_model=_getenv("LLM_MODEL", "glm-4-flash"),
        embed_model=_getenv("EMBED_MODEL", "embedding-3"),
        openai_compatible_api_key=_getenv("OPENAI_COMPAT_API_KEY"),
        openai_compatible_base_url=_getenv("OPENAI_COMPAT_BASE_URL"),
    )



def get_provider_config(settings: Settings, provider: str) -> ProviderConfig:
    p = provider.lower()
    if p != "openai_compatible":
        raise ValueError(f"Unsupported provider: {provider}")
    if not settings.openai_compatible_api_key:
        raise ValueError("Missing OPENAI_COMPAT_API_KEY")
    if not settings.openai_compatible_base_url:
        raise ValueError("Missing OPENAI_COMPAT_BASE_URL")

    return ProviderConfig(
        provider=p,
        api_key=settings.openai_compatible_api_key,
        base_url=settings.openai_compatible_base_url,
    )