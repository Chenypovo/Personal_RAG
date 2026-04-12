import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.config import get_provider_config, get_settings


class OpenAICompatibleGenerator:
    def __init__(self, model: Optional[str] = None) -> None:
        settings = get_settings()
        cfg = get_provider_config(settings, settings.llm_provider)
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.model = model or settings.llm_model

    @staticmethod
    def _citation_label(meta: Dict[str, Any]) -> str:
        source = str(meta.get("source", "unknown"))
        source_name = os.path.basename(source) if source not in {"", "unknown"} else "unknown"
        chunk_id = meta.get("chunk_id", "NA")
        modality = str(meta.get("modality", "text"))

        if modality == "image":
            if "time_sec" in meta:
                try:
                    t = float(meta["time_sec"])
                    return f"Frame {chunk_id}, {source_name}, t={t:.2f}s"
                except Exception:
                    return f"Frame {chunk_id}, {source_name}"
            return f"Image {chunk_id}, {source_name}"

        return f"Chunk {chunk_id}, {source_name}"

    def _build_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        lines: List[str] = []

        for item in retrieved_chunks:
            meta_raw = item.get("metadata")
            meta = meta_raw if isinstance(meta_raw, dict) else {}

            label = self._citation_label(meta)
            text_raw = meta.get("text", "")
            text = text_raw if isinstance(text_raw, str) else ("" if text_raw is None else str(text_raw))
            text = text.strip()

            modality = str(meta.get("modality", "text"))
            image_path = meta.get("image_path")
            extra = f"\n(modality={modality}" + (f", image_path={image_path}" if image_path else "") + ")"

            lines.append(f"[{label}]\n{text}{extra}")

        return "\n\n".join(lines).strip()

    def generate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        context = self._build_context(retrieved_chunks)

        system_prompt = (
            "You are a RAG assistant. "
            "Prioritize answering from the given context. "
            "If context is insufficient, you may add common knowledge but clearly separate it. "
            "Use only evidence labels from provided chunks, such as [Chunk x, filename] or [Frame x, filename, t=xx.xx s]. "
            "Do not treat URLs inside chunks as retrieved citations."
        )

        user_prompt = (
            f"User question:\n{query}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Provide a concise and accurate answer. Append evidence labels to key claims."
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        answer = resp.choices[0].message.content if resp.choices else ""
        sources: List[Dict[str, Any]] = []

        for item in retrieved_chunks:
            meta_raw = item.get("metadata")
            meta = meta_raw if isinstance(meta_raw, dict) else {}
            sources.append(
                {
                    "citation": self._citation_label(meta),
                    "source": meta.get("source"),
                    "chunk_id": meta.get("chunk_id"),
                    "modality": meta.get("modality", "text"),
                    "start": meta.get("start"),
                    "end": meta.get("end"),
                    "time_sec": meta.get("time_sec"),
                    "image_path": meta.get("image_path"),
                }
            )

        return {"answer": answer, "sources": sources}
