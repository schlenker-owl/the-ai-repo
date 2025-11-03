from airoad.rag.eval import exact_match, retrieve_topk, evaluate_qa

def test_exact_match_case_insensitive():
    assert exact_match("Paris", "paris") == 1.0
    assert exact_match("Berlin", "Paris") == 0.0

def test_retrieve_tfidf_fallback_ranks_right_doc():
    corpus = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "The Moon orbits Earth.",
    ]
    idxs = retrieve_topk("capital of France", corpus, k=1)
    assert idxs[0] == 0  # first doc contains the right answer

def test_evaluate_qa_shapes():
    items = [{"question": "q", "predicted": "a", "reference": "a"}]
    m = evaluate_qa(items)
    assert "exact_match" in m and "cosine_sim" in m
