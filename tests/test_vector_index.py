from app.memory.vector_index import NumpyVectorIndex


def test_search_returns_nearest_by_cosine():
    index = NumpyVectorIndex(dim=2)
    index.add("a", [1.0, 0.0])
    index.add("b", [0.0, 1.0])

    results = index.search([0.9, 0.1], top_k=2)

    assert [r[0] for r in results] == ["a", "b"]
    assert results[0][1] > results[1][1]


def test_add_same_id_upserts():
    index = NumpyVectorIndex(dim=2)
    index.add("a", [1.0, 0.0])
    index.add("a", [0.0, 1.0])  # overwrite

    results = index.search([0.0, 1.0], top_k=5)

    ids = [r[0] for r in results]
    assert ids.count("a") == 1
    assert ids == ["a"]


def test_remove_drops_id():
    index = NumpyVectorIndex(dim=2)
    index.add("a", [1.0, 0.0])
    index.add("b", [0.0, 1.0])

    index.remove("a")
    results = index.search([1.0, 0.0], top_k=5)

    assert [r[0] for r in results] == ["b"]


def test_search_empty_index_returns_empty():
    index = NumpyVectorIndex(dim=3)
    assert index.search([1.0, 0.0, 0.0], top_k=5) == []


def test_dimension_mismatch_raises():
    index = NumpyVectorIndex(dim=2)
    try:
        index.add("a", [1.0, 0.0, 0.0])
        assert False, "expected ValueError"
    except ValueError:
        pass
