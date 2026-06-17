"""Real-model smoke test: OCR + BGE reranker actually run (not mocked).

Run AFTER installing the full deps (torch, transformers, rapidocr-onnxruntime):

    python scripts/verify_real.py

The first run downloads the BGE reranker model (~1.1GB) and the rapidocr
models. Subsequent runs use the local cache and are fast.
"""

import os
import sys
import tempfile
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def check_ocr() -> bool:
    print("\n[1/2] OCR (rapidocr via image_loader) ...")
    try:
        from PIL import Image, ImageDraw, ImageFont

        from app.loader.image_loader import load_image
    except Exception as e:  # pragma: no cover - depends on optional deps
        print(f"  SKIP: deps missing: {e}")
        return False

    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "ocr_probe.png"
        img = Image.new("RGB", (520, 160), "white")
        draw = ImageDraw.Draw(img)
        text = "MEMORY RAG 2026"

        font = None
        for fp in (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ):
            try:
                font = ImageFont.truetype(fp, 56)
                break
            except Exception:
                continue
        draw.text((20, 50), text, fill="black", font=font)
        img.save(img_path)

        doc = load_image(str(img_path))
        recognized = doc["entries"][0].get("ocr_text", "")
        print(f"  rendered  : {text!r}")
        print(f"  recognized: {recognized!r}")
        ok = bool(recognized.strip())
        print("  PASS" if ok else "  FAIL: OCR returned empty text")
        return ok


def check_reranker() -> bool:
    print("\n[2/2] BGE reranker (torch + transformers) ...")
    try:
        from app.reranker.bge_reranker import BGEReranker
    except Exception as e:  # pragma: no cover - depends on optional deps
        print(f"  SKIP: deps missing: {e}")
        return False

    print("  loading model (first run downloads ~1.1GB) ...")
    reranker = BGEReranker()  # real cross-encoder model

    query = "How do I play the guitar?"
    docs = [
        {"metadata": {"text": "Guitar practice: start with basic chords and strumming.",
                      "chunk_id": 0, "source": "music.md"}},
        {"metadata": {"text": "Quarterly revenue grew across all product lines this year.",
                      "chunk_id": 1, "source": "finance.md"}},
    ]
    ranked = reranker.rerank(query=query, retrieved=docs, top_k=2)
    print(f"  query: {query!r}")
    for r in ranked:
        print(f"    score={r.get('rerank_score'):.4f}  {r['metadata']['text'][:48]!r}")
    top_text = ranked[0]["metadata"]["text"].lower()
    ok = "guitar" in top_text
    print("  PASS" if ok else "  FAIL: relevant doc not ranked first")
    return ok


def main() -> None:
    print("=== Real-model verification (OCR + reranker) ===")
    results = {"OCR": check_ocr(), "reranker": check_reranker()}
    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {name:10s}: {'PASS' if ok else 'FAIL/SKIP'}")
    if not all(results.values()):
        sys.exit(1)
    print("\nAll real-model checks passed.")


if __name__ == "__main__":
    main()
