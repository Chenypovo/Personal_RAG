from typing import List
from openai import OpenAI
from app.config import get_provider_config, get_settings


class OpenAICompatibleEmbedder:
    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        cfg = get_provider_config(settings, settings.embed_provider)
        self.client = OpenAI(api_key = cfg.api_key, base_url = cfg.base_url)
        self.model = model or settings.embed_model

    
    def embed_text(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            model = self.model,
            input = text,
        )

        return resp.data[0].embedding
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        resp = self.client.embeddings.create(
            model = self.model,
            input = texts,
        )
        return [item.embedding for item in resp.data]