import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from paper_evidence.models import Chunk
from paper_evidence.semantic import DenseRetriever, HybridRetriever


def chunk(cid,text):
    return Chunk(cid,"paper",1,"Results","text",text,(cid+"-block",),((0,0,10,10),))


class FixtureEmbedder:
    metadata = {"model_id":"deterministic-test-vectors","revision":"v1"}
    calls = 0

    def passages(self,texts):
        self.calls += 1
        # First chunk has a low-similarity window followed by a relevant tail.
        return np.array([[0,1],[1,0],[0.2,0.98]],dtype=float),np.array([0,0,1])

    def query(self,text):
        return [[1,0]]


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.chunks = (chunk("target","the relevant tail"),chunk("other","unrelated"))

    def test_late_window_retrieval_preserves_original_source(self):
        index = DenseRetriever(self.chunks,FixtureEmbedder())
        hits = index.search("中文问题",1)
        self.assertEqual(hits[0].chunk,self.chunks[0])
        self.assertEqual(hits[0].score_kind,"cosine")
        self.assertAlmostEqual(hits[0].score,1)

    def test_cache_reused_but_changed_text_reembedded(self):
        with tempfile.TemporaryDirectory() as path:
            embedder = FixtureEmbedder()
            a = DenseRetriever(self.chunks,embedder,path)
            b = DenseRetriever(self.chunks,embedder,path)
            self.assertEqual(embedder.calls,1)
            self.assertEqual(a.search("x"),b.search("x"))
            altered = (replace(self.chunks[0],text="different content"),self.chunks[1])
            c = DenseRetriever(altered,embedder,path)
            self.assertNotEqual(a.fingerprint,c.fingerprint)
            self.assertEqual(embedder.calls,2)

    def test_model_revision_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as path:
            first = FixtureEmbedder()
            a = DenseRetriever(self.chunks,first,path)
            second = FixtureEmbedder()
            second.metadata = {**first.metadata,"revision":"v2"}
            b = DenseRetriever(self.chunks,second,path)
            self.assertNotEqual(a.fingerprint,b.fingerprint)
            self.assertEqual(second.calls,1)

    def test_corrupt_cache_is_not_used(self):
        with tempfile.TemporaryDirectory() as path:
            index = DenseRetriever(self.chunks,FixtureEmbedder(),path)
            np.savez(Path(path)/(index.fingerprint+".npz"),vectors=np.array([[float('nan'),1]]),owners=[0])
            with self.assertRaises(ValueError):
                DenseRetriever(self.chunks,FixtureEmbedder(),path)

    def test_missing_window_owner_rejected(self):
        embedder = FixtureEmbedder()
        embedder.passages = lambda texts:(np.array([[1,0]]),np.array([0]))
        with self.assertRaises(ValueError):
            DenseRetriever(self.chunks,embedder)

    def test_fusion_deduplicates_and_keeps_provenance(self):
        dense = DenseRetriever(self.chunks,FixtureEmbedder())
        hits = HybridRetriever(self.chunks,dense).search("relevant",2)
        self.assertEqual(len({h.chunk.id for h in hits}),2)
        self.assertEqual(hits[0].chunk,self.chunks[0])
        self.assertTrue(all(h.score_kind=="rrf" for h in hits))

    def test_empty_corpus_never_calls_embedder(self):
        embedder = FixtureEmbedder()
        index = DenseRetriever([],embedder)
        self.assertEqual(index.search("question"),())
        self.assertEqual(embedder.calls,0)
