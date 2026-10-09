import math
import re
from collections import Counter
from dataclasses import dataclass

from .models import Chunk


STOP_WORDS = set("a an the is are was were of to in on for and or with what which how does did this that from by as it paper study".split())


def tokens(text):
    text = text.casefold()
    words = [w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text) if w not in STOP_WORDS]
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        words.extend(run[i:i+2] for i in range(len(run) - 1))
        if len(run) == 1:
            words.append(run)
    return words


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    score_kind: str = "bm25"


class BM25:
    """Transparent lexical baseline. The score is not a confidence probability."""
    def __init__(self, chunks):
        self.chunks = tuple(chunks)
        self.counts = [Counter(tokens(c.text)) for c in self.chunks]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average = sum(self.lengths) / len(self.lengths) if self.lengths else 1
        self.document_frequency = Counter(t for count in self.counts for t in count)

    def search(self, question, k=5):
        if not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20")
        query, hits = set(tokens(question)), []
        n = len(self.chunks)
        for chunk, counts, length in zip(self.chunks, self.counts, self.lengths):
            score = 0.0
            for term in query:
                frequency = counts[term]
                if not frequency:
                    continue
                df = self.document_frequency[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * frequency * 2.5 / (
                    frequency + 1.5 * (0.25 + 0.75 * length / (self.average or 1))
                )
            if score > 0:
                hits.append(Hit(chunk, score))
        return tuple(sorted(hits, key=lambda h: (-h.score, h.chunk.id))[:k])
