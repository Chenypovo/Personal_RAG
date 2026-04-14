from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

try:
    import jieba  # type: ignore
except Exception:  # pragma: no cover
    jieba = None


Tokenizer = Callable[[str], List[str]]

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
# Protect mixed/code-like tokens first (path, model, version, acronym, english word, number)
PROTECTED_TOKEN_RE = re.compile(
    r"""
    (?<![A-Za-z0-9_])
    (
        [A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)+
        |
        [A-Za-z]+[0-9]+(?:[._/-][A-Za-z0-9]+)*
        |
        [vV]?\d+(?:\.\d+)+
        |
        [A-Z]{2,}(?:-[A-Z0-9]+)*
        |
        [A-Za-z]{2,}(?:'[A-Za-z]+)?
        |
        \d{2,}
    )
    (?![A-Za-z0-9_])
    """,
    re.VERBOSE,
)


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "").strip())


def _extract_protected_tokens(text: str) -> Tuple[List[str], str]:
    tokens: List[str] = []
    masked = list(text)

    for m in PROTECTED_TOKEN_RE.finditer(text):
        token = m.group(1).strip()
        if token:
            tokens.append(token.lower())
        for i in range(m.start(), m.end()):
            masked[i] = " "

    remaining = "".join(masked)
    return tokens, remaining


def _build_cjk_bigrams(text: str) -> List[str]:
    # Keep unique fallback bigrams to avoid overweighting noise.
    grams: List[str] = []
    seen = set()

    for seg in CJK_RE.findall(text):
        if len(seg) == 1:
            if seg not in seen:
                seen.add(seg)
                grams.append(seg)
            continue

        for i in range(len(seg) - 1):
            g = seg[i : i + 2]
            if g not in seen:
                seen.add(g)
                grams.append(g)

    return grams


class BM25Store:
    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        user_dict_paths: Optional[Sequence[str]] = None,
        enable_cjk_bigram_fallback: bool = True,
    ) -> None:
        self.enable_cjk_bigram_fallback = enable_cjk_bigram_fallback
        self._jieba_tokenizer: Any = None

        if tokenizer is not None:
            self.tokenizer: Tokenizer = tokenizer
        else:
            self._jieba_tokenizer = self._build_jieba_tokenizer(user_dict_paths or [])
            self.tokenizer = self._default_tokenize

        self.metadatas: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    @staticmethod
    def _build_jieba_tokenizer(user_dict_paths: Sequence[str]) -> Any:
        if jieba is None:
            return None

        tok = jieba.Tokenizer()
        for p in user_dict_paths:
            pp = Path(p)
            if not pp.exists():
                raise FileNotFoundError(f"jieba user dict not found: {p}")
            tok.load_userdict(str(pp))
        return tok

    def _default_tokenize(self, text: str) -> List[str]:
        # 1) normalize
        norm = _normalize_text(text)
        if not norm:
            return []

        # 2) regex protected token extraction
        protected_tokens, remaining = _extract_protected_tokens(norm)

        tokens: List[str] = []
        tokens.extend(protected_tokens)  # 4) keep original protected tokens

        # 3) jieba on remaining chinese chunks
        cjk_segments = CJK_RE.findall(remaining)
        if cjk_segments:
            if self._jieba_tokenizer is not None:
                for seg in cjk_segments:
                    for t in self._jieba_tokenizer.lcut(seg, HMM=True):
                        tt = t.strip()
                        if tt:
                            tokens.append(tt)
            else:
                # fallback when jieba is unavailable
                for seg in cjk_segments:
                    tokens.extend(list(seg))

        # 5) optional cjk 2-gram fallback
        if self.enable_cjk_bigram_fallback:
            tokens.extend(_build_cjk_bigrams(remaining))

        return [t for t in tokens if t]

    @staticmethod
    def _make_doc_id(meta: Dict[str, Any], idx: int) -> str:
        if "doc_id" in meta and isinstance(meta["doc_id"], str) and meta["doc_id"].strip():
            return meta["doc_id"].strip()
        source = str(meta.get("source", "unknown"))
        chunk_id = meta.get("chunk_id", idx)
        return f"{source}#{chunk_id}"

    def build(self, texts: Sequence[str], metadatas: Sequence[Dict[str, Any]]) -> None:
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas length mismatch")

        self.metadatas = []
        self.corpus_tokens = []

        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            raw = (text or "").strip()
            if not raw:
                continue

            tokens = self.tokenizer(raw)
            if not tokens:
                continue

            m = dict(meta)
            m["doc_id"] = self._make_doc_id(m, i)

            self.metadatas.append(m)
            self.corpus_tokens.append(tokens)

        self._bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def build_from_records(self, records: Sequence[Dict[str, Any]]) -> None:
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []

        for i, r in enumerate(records):
            modality = str(r.get("modality", "text"))
            if modality != "text":
                continue

            text = str(r.get("text", "")).strip()
            if not text:
                continue

            m = dict(r)
            m["doc_id"] = self._make_doc_id(m, i)

            texts.append(text)
            metas.append(m)

        self.build(texts=texts, metadatas=metas)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self._bm25 is None:
            return []

        q_tokens = self.tokenizer(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[Dict[str, Any]] = []
        for idx in ranked_idx:
            score = float(scores[idx])
            results.append(
                {
                    "score": score,
                    "distance": -score,  # keep compatibility with current downstream
                    "metadata": self.metadatas[idx],
                }
            )
        return results

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadatas": self.metadatas,
            "corpus_tokens": self.corpus_tokens,
            "enable_cjk_bigram_fallback": self.enable_cjk_bigram_fallback,
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str,
        tokenizer: Optional[Tokenizer] = None,
        user_dict_paths: Optional[Sequence[str]] = None,
    ) -> "BM25Store":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"BM25 file not found: {path}")

        data = json.loads(p.read_text(encoding="utf-8"))

        obj = cls(
            tokenizer=tokenizer,
            user_dict_paths=user_dict_paths,
            enable_cjk_bigram_fallback=bool(data.get("enable_cjk_bigram_fallback", True)),
        )
        obj.metadatas = list(data.get("metadatas", []))
        obj.corpus_tokens = list(data.get("corpus_tokens", []))
        obj._bm25 = BM25Okapi(obj.corpus_tokens) if obj.corpus_tokens else None
        return obj
