import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.chunker.text_chunker import chunk_text  # noqa: E402
from app.embedder.embedder import OpenAICompatibleEmbedder  # noqa: E402
from app.generator.generator import OpenAICompatibleGenerator  # noqa: E402
from app.loader import SUPPORTED_EXTENSIONS, load_document  # noqa: E402
from app.retriever.faiss_retriever import FaissRetriever  # noqa: E402
from app.vectordb.faiss_store import FaissStore  # noqa: E402


def _collect_files(input_dir: Path) -> List[str]:
    files: List[str] = []
    for fp in input_dir.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(str(fp))
    return files


def _build_chunks(file_paths: List[str], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for file_path in file_paths:
        doc = load_document(file_path)
        chunks = chunk_text(
            text=doc["text"],
            source=doc["source"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(chunks)
    return all_chunks


def _save_uploaded_files(uploaded_files: List[Any], upload_dir: Path) -> List[Path]:
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for f in uploaded_files:
        target = upload_dir / f.name
        target.write_bytes(f.getbuffer())
        saved_paths.append(target)
    return saved_paths


def _build_index(input_dir: Path, index_path: Path, meta_path: Path, chunk_size: int, overlap: int) -> Dict[str, Any]:
    file_paths = _collect_files(input_dir)
    if not file_paths:
        raise ValueError(f"No supported files found in: {input_dir}")

    chunks = _build_chunks(file_paths, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("No chunks generated from input files")

    embedder = OpenAICompatibleEmbedder()
    vectors = embedder.embed_texts([c["text"] for c in chunks])
    if not vectors:
        raise ValueError("No vectors generated")

    dim = len(vectors[0])
    store = FaissStore(dim=dim)
    store.add(vectors=vectors, metadatas=chunks)
    store.save(index_path=str(index_path), meta_path=str(meta_path))

    return {
        "files": len(file_paths),
        "chunks": len(chunks),
        "dim": dim,
        "index_path": str(index_path),
        "meta_path": str(meta_path),
    }


def _run_query(
    query: str,
    top_k: int,
    index_path: Path,
    meta_path: Path,
    use_rerank: bool,
    rerank_model: str,
    rerank_top_n: int,
    rerank_batch_size: int,
    rerank_device: str,
) -> Dict[str, Any]:
    store = FaissStore.load(index_path=str(index_path), meta_path=str(meta_path))
    embedder = OpenAICompatibleEmbedder()
    retriever = FaissRetriever(store=store, embedder=embedder)

    retrieve_k = max(top_k, rerank_top_n) if use_rerank else top_k
    retrieved = retriever.retrieve(query=query, top_k=retrieve_k)

    if use_rerank and retrieved:
        from app.reranker.bge_reranker import BGEReranker  # lazy import

        reranker = BGEReranker(
            model_name=rerank_model,
            device=rerank_device or None,
            batch_size=rerank_batch_size,
        )
        retrieved = reranker.rerank(query=query, retrieved=retrieved, top_k=top_k)

    generator = OpenAICompatibleGenerator()
    result = generator.generate(query=query, retrieved_chunks=retrieved)
    return {"result": result, "retrieved": retrieved}


def main() -> None:
    st.set_page_config(page_title="Personal RAG", layout="wide")
    st.title("Personal RAG")
    st.caption("上传文档 -> 建立索引 -> 提问检索问答")

    data_dir = Path(PROJECT_ROOT) / "data"
    upload_dir = data_dir / "uploads"
    index_path = data_dir / "index" / "faiss.index"
    meta_path = data_dir / "index" / "metadatas.json"

    with st.sidebar:
        st.header("参数设置")
        chunk_size = st.number_input("chunk_size", min_value=100, max_value=4000, value=700, step=50)
        overlap = st.number_input("overlap", min_value=0, max_value=1000, value=120, step=10)
        top_k = st.number_input("top_k", min_value=1, max_value=20, value=4, step=1)

        st.subheader("Rerank")
        use_rerank = st.checkbox("启用 Cross-Encoder Rerank", value=False)
        rerank_model = st.text_input("rerank_model", value="BAAI/bge-reranker-base")
        rerank_top_n = st.number_input("rerank_top_n", min_value=1, max_value=200, value=20, step=1)
        rerank_batch_size = st.number_input("rerank_batch_size", min_value=1, max_value=128, value=16, step=1)
        rerank_device = st.selectbox("rerank_device", options=["", "cpu", "cuda"], index=0)

    st.subheader("1) 上传文档")
    uploaded_files = st.file_uploader(
        "支持 txt / md / pdf",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    cols = st.columns([1, 1, 3])
    with cols[0]:
        if st.button("保存上传文件", use_container_width=True):
            if not uploaded_files:
                st.warning("请先选择文件。")
            else:
                saved = _save_uploaded_files(uploaded_files, upload_dir)
                st.success(f"已保存 {len(saved)} 个文件到 {upload_dir}")

    with cols[1]:
        if st.button("构建索引", use_container_width=True):
            try:
                with st.spinner("正在构建索引..."):
                    stats = _build_index(
                        input_dir=upload_dir,
                        index_path=index_path,
                        meta_path=meta_path,
                        chunk_size=int(chunk_size),
                        overlap=int(overlap),
                    )
                st.success(
                    f"构建完成：files={stats['files']} chunks={stats['chunks']} dim={stats['dim']}"
                )
            except Exception as e:
                st.error(f"构建失败：{e}")

    st.divider()
    st.subheader("2) 提问")
    query = st.text_input("请输入问题", placeholder="例如：什么是 FAISS？")
    ask = st.button("开始问答", type="primary")

    if ask:
        if not query.strip():
            st.warning("请输入问题。")
        elif not index_path.exists() or not meta_path.exists():
            st.error("索引文件不存在，请先构建索引。")
        else:
            try:
                with st.spinner("正在检索并生成回答..."):
                    data = _run_query(
                        query=query,
                        top_k=int(top_k),
                        index_path=index_path,
                        meta_path=meta_path,
                        use_rerank=use_rerank,
                        rerank_model=rerank_model.strip(),
                        rerank_top_n=int(rerank_top_n),
                        rerank_batch_size=int(rerank_batch_size),
                        rerank_device=rerank_device,
                    )

                result = data["result"]
                retrieved = data["retrieved"]

                st.markdown("### Answer")
                st.write(result.get("answer", ""))

                st.markdown("### Sources")
                for i, s in enumerate(result.get("sources", []), start=1):
                    st.write(
                        f"[{i}] {s.get('citation')} | source={s.get('source')} "
                        f"| span=({s.get('start')}, {s.get('end')})"
                    )

                with st.expander("查看检索到的 Chunks"):
                    for i, item in enumerate(retrieved, start=1):
                        meta_raw = item.get("metadata")
                        meta = meta_raw if isinstance(meta_raw, dict) else {}
                        text_raw = meta.get("text", "")
                        text = text_raw if isinstance(text_raw, str) else ("" if text_raw is None else str(text_raw))
                        text = text.strip()
                        score = item.get("rerank_score")
                        score_text = f", rerank_score={score:.4f}" if isinstance(score, (int, float)) else ""
                        st.markdown(
                            f"**[{i}] source={meta.get('source')} chunk_id={meta.get('chunk_id')}{score_text}**"
                        )
                        st.write(text)
                        st.divider()
            except Exception as e:
                st.error(f"问答失败：{e}")


if __name__ == "__main__":
    main()
