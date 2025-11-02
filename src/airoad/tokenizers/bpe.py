# src/airoad/tokenizers/bpe.py
from __future__ import annotations
import json
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import List, Tuple

_EOW = "</w>"    # end-of-word marker for training
_UNK = "<unk>"

def _word_to_symbols(word: str) -> Tuple[str, ...]:
    # split into unicode chars and append EOW
    return tuple(list(word)) + (_EOW,)

def _get_stats(corpus: List[Tuple[str, ...]]) -> Counter:
    stats = Counter()
    for symbols in corpus:
        for i in range(len(symbols) - 1):
            stats[(symbols[i], symbols[i+1])] += 1
    return stats

def _merge_corpus(corpus: List[Tuple[str, ...]], pair: Tuple[str, str]) -> List[Tuple[str, ...]]:
    a, b = pair
    merged = []
    bigram = a + b
    for symbols in corpus:
        out: List[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                out.append(bigram)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        merged.append(tuple(out))
    return merged

@dataclass
class BPETokenizer:
    vocab_size: int = 2000
    min_freq: int = 2
    merges_: List[Tuple[str, str]] | None = None
    stoi_: dict | None = None
    itos_: List[str] | None = None

    def train(self, texts: List[str]):
        # initial corpus: words -> symbols with EOW
        words = []
        for line in texts:
            for w in line.strip().split():
                words.append(_word_to_symbols(w))
        if not words:
            words = [tuple("hello") + (_EOW,)]

        merges: List[Tuple[str, str]] = []
        corpus = words

        while True:
            stats = _get_stats(corpus)
            # drop infrequent pairs
            for p in list(stats.keys()):
                if stats[p] < self.min_freq:
                    del stats[p]
            if not stats:
                break
            pair, freq = stats.most_common(1)[0]
            merges.append(pair)
            corpus = _merge_corpus(corpus, pair)
            # stop if vocab limit reached approximately
            if len(merges) >= max(1, self.vocab_size - 256):
                break

        # build vocabulary from final corpus
        vocab = Counter()
        for symbols in corpus:
            vocab.update(symbols)
        # ensure special tokens
        if _UNK not in vocab:
            vocab[_UNK] = 1

        itos = list(vocab.keys())
        stoi = {t: i for i, t in enumerate(itos)}
        self.merges_, self.stoi_, self.itos_ = merges, stoi, itos
        return self

    def _apply_merges(self, symbols: List[str]) -> List[str]:
        if not self.merges_:
            return symbols
        # greedy left-to-right merge using learned order
        for a, b in self.merges_:
            i = 0
            out: List[str] = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(symbols[i]); i += 1
            symbols = out
        return symbols

    def encode(self, text: str) -> List[int]:
        tokens: List[int] = []
        for w in text.strip().split():
            symbols = list(w) + [_EOW]
            symbols = self._apply_merges(symbols)
            for s in symbols:
                tokens.append(self.stoi_.get(s, self.stoi_[_UNK]))
        return tokens

    def decode(self, ids: List[int]) -> str:
        words: List[str] = []
        cur: List[str] = []
        for i in ids:
            tok = self.itos_[i]
            if tok == _EOW:
                words.append("".join(cur))
                cur = []
            elif tok == _UNK:
                cur.append("?")
            else:
                cur.append(tok)
        if cur:
            words.append("".join(cur))
        return " ".join(words)

    def save(self, path_json: str):
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump({"merges": self.merges_, "itos": self.itos_}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path_json: str) -> "BPETokenizer":
        with open(path_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tok = cls()
        tok.merges_ = [tuple(m) for m in obj["merges"]]
        tok.itos_ = list(obj["itos"])
        tok.stoi_ = {t: i for i, t in enumerate(tok.itos_)}
        return tok
