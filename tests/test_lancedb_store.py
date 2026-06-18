import pytest

pytest.importorskip("lancedb")

from app.vectordb.lancedb_store import LanceDBStore  # noqa: E402


def test_lancedb_roundtrip_search_and_all_metadatas(tmp_path):
    store = LanceDBStore(uri=str(tmp_path / "ldb"), table_name="t")
    store.add(
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[
            {"source": "a.md", "chunk_id": 0, "text": "guitar chords", "parent_id": "a.md::Sec A"},
            {"source": "b.md", "chunk_id": 0, "text": "quarterly revenue"},
        ],
    )

    # nearest neighbour to [1,0] is the first row, and metadata is reconstructed
    res = store.search([1.0, 0.0], top_k=2)
    assert res[0]["metadata"]["text"] == "guitar chords"
    assert isinstance(res[0]["distance"], float)

    # the unified row preserves arbitrary metadata (parent_id) for parent-child
    metas = store.all_metadatas()
    assert len(metas) == 2
    assert {m["source"] for m in metas} == {"a.md", "b.md"}
    assert any(m.get("parent_id") == "a.md::Sec A" for m in metas)


def test_lancedb_reload_from_disk(tmp_path):
    uri = str(tmp_path / "ldb")
    LanceDBStore(uri=uri, table_name="t").add(
        vectors=[[1.0, 0.0]], metadatas=[{"source": "a.md", "chunk_id": 0, "text": "x"}]
    )
    reopened = LanceDBStore.load(uri=uri, table_name="t")
    assert reopened.search([1.0, 0.0], top_k=1)[0]["metadata"]["text"] == "x"
