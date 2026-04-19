from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Union
import warnings

import numpy as np
from openai import OpenAI
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

from app.config import get_provider_config, get_settings


OutputType = Literal["list", "numpy", "tensor"]
BatchOutput = Union[List[List[float]], np.ndarray, torch.Tensor]
VectorOutput = Union[List[float], np.ndarray, torch.Tensor]


def _validate_texts(texts: Sequence[str]) -> List[str]:
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of str, not a single str")
    if not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence of str")

    clean: List[str] = []
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise TypeError(f"texts[{i}] must be str, got {type(t).__name__}")
        t2 = t.strip()
        if t2:
            clean.append(t2)
    return clean


def _to_output(vectors: np.ndarray, output_type: OutputType) -> BatchOutput:
    if output_type == "list":
        return vectors.tolist()
    if output_type == "numpy":
        return vectors
    if output_type == "tensor":
        return torch.from_numpy(vectors)
    raise ValueError(f"Unsupported output_type: {output_type}")


def _empty_output(output_type: OutputType) -> BatchOutput:
    if output_type == "list":
        return []
    if output_type == "numpy":
        return np.zeros((0, 0), dtype=np.float32)
    return torch.zeros((0, 0), dtype=torch.float32)


def _l2_normalize_rows(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vectors, got shape={vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (vectors / norms).astype(np.float32, copy=False)


class OpenAICompatibleEmbedder:
    """Text embedding via OpenAI-compatible API."""

    def __init__(self, model: Optional[str] = None, batch_size: int = 64) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        settings = get_settings()
        cfg = get_provider_config(settings, settings.embed_provider)

        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.model = model or settings.embed_model
        self.batch_size = batch_size

    def embed_text(self, text: str, output_type: OutputType = "list") -> VectorOutput:
        out = self.embed_texts([text], output_type=output_type)
        return out[0]  # type: ignore[index]

    def embed_texts(
        self,
        texts: Sequence[str],
        output_type: OutputType = "list",
    ) -> BatchOutput:
        clean = _validate_texts(texts)
        if not clean:
            return _empty_output(output_type)

        all_vecs: List[List[float]] = []
        for i in range(0, len(clean), self.batch_size):
            batch = clean[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            all_vecs.extend([item.embedding for item in resp.data])

        arr = np.asarray(all_vecs, dtype=np.float32)
        arr = _l2_normalize_rows(arr)
        return _to_output(arr, output_type)


class CLIPMultimodalEmbedder:
    """
    CLIP multimodal encoder (text + image).
    Note: it does not process video directly; extract frames first, then call embed_images.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        batch_size: int = 32,
        warn_on_truncate: bool = True,
    ) -> None:
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.warn_on_truncate = warn_on_truncate

        # transformers stubs are incomplete in Pylance; route through Any to avoid false positives.
        processor_cls: Any = CLIPProcessor
        model_cls: Any = CLIPModel
        tokenizer_cls: Any = AutoTokenizer

        self.processor: Any = processor_cls.from_pretrained(model_name)
        model_obj: Any = model_cls.from_pretrained(model_name)
        self.model: Any = model_obj.to(self.device)
        self.model.eval()
        self.tokenizer: Any = tokenizer_cls.from_pretrained(model_name, use_fast=True)

        max_pos = 77
        text_cfg = getattr(getattr(self.model, "config", None), "text_config", None)
        if text_cfg is not None:
            max_pos = int(getattr(text_cfg, "max_position_embeddings", 77))
        self.max_text_tokens = max_pos

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if not device:
            return "cpu"

        d = device.strip().lower()
        if d == "cpu":
            return "cpu"
        if d.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ValueError("CUDA requested but not available")
            return d
        if d == "mps":
            mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if not mps_ok:
                raise ValueError("MPS requested but not available")
            return d

        raise ValueError(f"Unsupported device: {device}")

    def _warn_truncation(self, texts: Sequence[str]) -> None:
        if not self.warn_on_truncate:
            return

        truncated = 0
        for t in texts:
            ids = self.tokenizer(
                t,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            if len(ids) > self.max_text_tokens:
                truncated += 1

        if truncated > 0:
            warnings.warn(
                f"{truncated}/{len(texts)} texts exceed CLIP token limit "
                f"({self.max_text_tokens}) and will be truncated.",
                stacklevel=2,
            )

    @staticmethod
    def _coerce_to_tensor(raw_feats: Any) -> torch.Tensor:
        if isinstance(raw_feats, torch.Tensor):
            return raw_feats

        if isinstance(raw_feats, (tuple, list)) and raw_feats:
            head = raw_feats[0]
            if isinstance(head, torch.Tensor):
                return head

        pooler = getattr(raw_feats, "pooler_output", None)
        if isinstance(pooler, torch.Tensor):
            return pooler

        last_hidden = getattr(raw_feats, "last_hidden_state", None)
        if isinstance(last_hidden, torch.Tensor):
            if last_hidden.ndim >= 2:
                return last_hidden[:, 0, :]
            return last_hidden

        if isinstance(raw_feats, np.ndarray):
            return torch.from_numpy(raw_feats)

        raise TypeError(f"Unsupported features type: {type(raw_feats).__name__}")

    def _maybe_fallback_to_cpu(self, err: Exception) -> bool:
        if self.device.startswith("cuda"):
            msg = str(err).lower()
            if "sm80" in msg or "fmha_cutlass" in msg or "no kernel image is available" in msg:
                self.device = "cpu"
                self.model = self.model.to("cpu")
                return True
        return False

    def embed_text(self, text: str, output_type: OutputType = "list") -> VectorOutput:
        out = self.embed_texts([text], output_type=output_type)
        return out[0]  # type: ignore[index]

    def embed_texts(
        self,
        texts: Sequence[str],
        output_type: OutputType = "list",
    ) -> BatchOutput:
        clean = _validate_texts(texts)
        if not clean:
            return _empty_output(output_type)

        self._warn_truncation(clean)

        all_batches: List[np.ndarray] = []
        for i in range(0, len(clean), self.batch_size):
            batch = clean[i : i + self.batch_size]

            def _run_once() -> torch.Tensor:
                inputs: Any = self.processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_text_tokens,
                )
                if hasattr(inputs, "to"):
                    inputs = inputs.to(self.device)
                with torch.inference_mode():
                    raw_feats: Any = self.model.get_text_features(**inputs)
                    feats_tensor = self._coerce_to_tensor(raw_feats)
                    return F.normalize(feats_tensor, p=2, dim=-1)

            try:
                feats = _run_once()
            except RuntimeError as e:
                if self._maybe_fallback_to_cpu(e):
                    feats = _run_once()
                else:
                    raise

            all_batches.append(feats.detach().cpu().numpy().astype(np.float32))

        arr = np.concatenate(all_batches, axis=0)
        return _to_output(arr, output_type)

    def embed_image(self, image_path: str, output_type: OutputType = "list") -> VectorOutput:
        out = self.embed_images([image_path], output_type=output_type)
        return out[0]  # type: ignore[index]

    def embed_images(
        self,
        image_paths: Sequence[str],
        output_type: OutputType = "list",
    ) -> BatchOutput:
        if isinstance(image_paths, str):
            raise TypeError("image_paths must be a sequence of str, not a single str")
        if not isinstance(image_paths, Sequence):
            raise TypeError("image_paths must be a sequence of str")

        clean_paths: List[str] = []
        for i, p in enumerate(image_paths):
            if not isinstance(p, str):
                raise TypeError(f"image_paths[{i}] must be str, got {type(p).__name__}")
            p2 = p.strip()
            if not p2:
                continue
            pp = Path(p2)
            if not pp.exists() or not pp.is_file():
                raise FileNotFoundError(f"Image file not found: {p2}")
            clean_paths.append(str(pp))

        if not clean_paths:
            return _empty_output(output_type)

        all_batches: List[np.ndarray] = []
        for i in range(0, len(clean_paths), self.batch_size):
            batch_paths = clean_paths[i : i + self.batch_size]

            with ExitStack() as stack:
                images: List[Image.Image] = []
                for p in batch_paths:
                    try:
                        pil = stack.enter_context(Image.open(p))
                    except UnidentifiedImageError as e:
                        raise ValueError(f"Invalid/corrupted image file: {p}") from e

                    rgb = pil.convert("RGB")
                    images.append(rgb)
                    stack.callback(rgb.close)

                def _run_once() -> torch.Tensor:
                    inputs: Any = self.processor(images=images, return_tensors="pt")
                    if hasattr(inputs, "to"):
                        inputs = inputs.to(self.device)
                    with torch.inference_mode():
                        raw_feats: Any = self.model.get_image_features(**inputs)
                        feats_tensor = self._coerce_to_tensor(raw_feats)
                        return F.normalize(feats_tensor, p=2, dim=-1)

                try:
                    feats = _run_once()
                except RuntimeError as e:
                    if self._maybe_fallback_to_cpu(e):
                        feats = _run_once()
                    else:
                        raise

                all_batches.append(feats.detach().cpu().numpy().astype(np.float32))

        arr = np.concatenate(all_batches, axis=0)
        return _to_output(arr, output_type)
