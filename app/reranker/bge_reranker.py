from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

# pairs of (query, document_text) -> one relevance score per pair
ScoreFn = Callable[[List[Tuple[str, str]]], List[float]]


class BGEReranker:
    """Cross-encoder reranker.

    By default it loads a BAAI/bge-reranker model (torch + transformers, imported
    lazily so importing this module is cheap). A ``score_fn`` can be injected to
    supply scores directly, which keeps the ranking logic testable without the
    heavy model.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        batch_size: int = 16,
        score_fn: Optional[ScoreFn] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._score_fn = score_fn

        if score_fn is None:
            self._load_model(device)

    def _load_model(self, device: Optional[str]) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _sdp_context(self):
        import torch

        # Some Windows + new GPU combinations can crash on fused FMHA kernels.
        # Force math SDP on CUDA to avoid incompatible kernels.
        if self.device.type != "cuda":
            return nullcontext()
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
        except Exception:
            return nullcontext()

    def _score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        import torch

        scores: List[float] = []

        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i: i + self.batch_size]
                queries = [x[0] for x in batch]
                docs = [x[1] for x in batch]

                inputs = self.tokenizer(
                    queries,
                    docs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt',
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                try:
                    with self._sdp_context():
                        logits = self.model(**inputs).logits.view(-1).float()
                except RuntimeError as e:
                    msg = str(e)
                    if self.device.type == "cuda" and ("fmha_cutlass" in msg or "sm80-sm100" in msg):
                        print("[reranker] CUDA FMHA kernel mismatch detected, fallback to CPU for rerank.")
                        self.device = torch.device("cpu")
                        self.model.to(self.device)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        logits = self.model(**inputs).logits.view(-1).float()
                    else:
                        raise
                scores.extend(logits.cpu().tolist())

        return scores

    def _score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if self._score_fn is not None:
            return self._score_fn(pairs)
        return self._score_pairs(pairs)

    def rerank(
        self,
        query: str,
        retrieved: List[Dict[str, Any]],
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:

        q = (query or "").strip()
        if not q or not retrieved:
            return retrieved[:top_k]

        pairs: List[Tuple[str, str]] = []
        valid_indices: List[int] = []

        for idx, item in enumerate(retrieved):
            meta_raw = item.get("metadata")
            meta = meta_raw if isinstance(meta_raw, dict) else {}
            text_raw = meta.get("text", "")
            text = text_raw if isinstance(text_raw, str) else ("" if text_raw is None else str(text_raw))
            text = text.strip()
            if not text:
                continue
            pairs.append((q, text))
            valid_indices.append(idx)

        if not pairs:
            return retrieved[:top_k]

        pair_scores = self._score(pairs)
        score_map = {idx: float(score) for idx, score in zip(valid_indices, pair_scores)}

        rescored: List[Dict[str, Any]] = []
        for idx, item in enumerate(retrieved):
            new_item = dict(item)
            new_item["rerank_score"] = score_map.get(idx, float("-inf"))
            rescored.append(new_item)

        rescored.sort(key=lambda x: x.get("rerank_score", float('-inf')), reverse=True)
        return rescored[:top_k]
