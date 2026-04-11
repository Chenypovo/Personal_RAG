from typing import Any, List, Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BGEReranker:
    def __init__(
            self,
            model_name: str = "BAAI/bge-reranker-base",
            device: Optional[str] = None,
            batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _score_pairs(self, pairs: List[tuple[str, str]]) -> List[float]:
        scores: List[float] = []

        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i: i + self.batch_size]
                queries = [x[0] for x in batch]
                docs = [x[1] for x in batch]

                inputs = self.tokenizer(
                    queries,
                    docs,
                    padding = True,
                    truncation = True,
                    max_length = 512,
                    return_tensors = 'pt',
                )
                inputs = {k: v.to(self.device) for k ,v in inputs.items()}
                logits = self.model(**inputs).logits.view(-1).float()
                scores.extend(logits.cpu().tolist())

        return scores
    
    def rerank(
            self,
            query: str,
            retrieved: List[Dict[str, Any]],
            top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        
        q = (query or "").strip()
        if not q or not retrieved:
            return retrieved[:top_k]
        
        pairs: List[tuple[str, str]] = []
        valid_indices: List[int] = []

        for idx, item in enumerate(retrieved):
            meta = item.get("metadata", {})
            text = (meta.get("text", "") or "").strip()
            if not text:
                continue
            pairs.append((q, text))
            valid_indices.append(idx)

        if not pairs:
            return retrieved[:top_k]
        
        pair_scores = self._score_pairs(pairs)
        score_map = {idx: float(score) for idx, score in zip(valid_indices, pair_scores)}

        rescored: List[Dict[str, Any]] = []
        for idx, item in enumerate(retrieved):
            new_item = dict(item)
            new_item["rerank_score"] = score_map.get(idx, float("-inf"))
            rescored.append(new_item)

        rescored.sort(key = lambda x: x.get("rerank_score", float('-inf')), reverse = True)
        return rescored[:top_k]
