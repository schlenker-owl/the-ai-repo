from airoad.sft.eval_sft import exact_match, rouge_l_f, evaluate_pairs

def test_exact_match_and_rouge():
    assert exact_match("Paris", "paris") == 1.0
    assert exact_match("Paris", "Berlin") == 0.0
    # ROUGE-L: identical
    assert rouge_l_f("a b c", "a b c") == 1.0
    # partial overlap
    r = rouge_l_f("a b c", "a c d")
    assert 0.0 < r < 1.0

def test_evaluate_pairs_averages():
    pairs = [("a b", "a b"), ("a b", "a c")]
    m = evaluate_pairs(pairs)
    assert "exact_match" in m and "rougeL_f" in m
    assert 0.0 <= m["exact_match"] <= 1.0
    assert 0.0 <= m["rougeL_f"] <= 1.0
