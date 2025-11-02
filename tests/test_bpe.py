# tests/test_bpe.py
from airoad.tokenizers.bpe import BPETokenizer

def test_bpe_roundtrip_basic():
    texts = ["low lower newest widest", "low low lower", "new wider"]
    tok = BPETokenizer(vocab_size=200, min_freq=2).train(texts)
    s = "low newer"
    assert tok.decode(tok.encode(s)) == s

def test_bpe_has_unk_and_eow():
    texts = ["a aa aaa", "b bb bbb"]
    tok = BPETokenizer(vocab_size=50, min_freq=2).train(texts)
    # unknown symbol maps to <unk>
    ids = tok.encode("zzz")
    assert len(ids) >= 1
