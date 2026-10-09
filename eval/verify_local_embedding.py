"""Optional integration check with prepared real weights, using synthetic text."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_evidence.models import Chunk
from paper_evidence.semantic import DenseRetriever, E5Embedder


def main():
    embedder = E5Embedder()
    text = ("论文测试段落只用于验证长文本窗口。 " * 180) + "\nThe proposed method achieves an accuracy of 91.2%."
    windows = embedder.windows(text)
    lengths = [len(embedder.model.tokenizer.encode("passage: " + w, verbose=False)) for w in windows]
    assert len(windows) > 1 and max(lengths) <= 512
    assert any("91.2%" in w for w in windows)
    target = Chunk("long-fixture", "synthetic", 2, "Fixture", "text", text, ("block",), ((0,0,10,10),))
    other = Chunk("other-fixture", "synthetic", 1, "Fixture", "text", "The sky is blue.", ("other",), ((0,0,10,10),))
    hit = DenseRetriever([target,other],embedder).search("提出的方法准确率是多少？",1)[0]
    assert hit.chunk == target and hit.chunk.text.endswith("91.2%.")
    try:
        embedder.query(text)
    except ValueError:
        pass
    else:
        raise AssertionError("Overlong question was silently truncated")
    report = {"dataset":"synthetic text integration check", "model":embedder.metadata,
              "windows":len(windows), "window_token_counts":lengths,
              "tail_retained":True, "original_chunk_retained":True,
              "overlong_question_rejected":True, "quality_benchmark":False}
    (ROOT / "eval/local-model-check.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
