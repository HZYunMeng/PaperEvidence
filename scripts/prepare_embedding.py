"""Explicit model download; the application itself only loads local weights."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main():
    parser = argparse.ArgumentParser(description="Prepare the multilingual E5 retrieval model")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "multilingual-e5-small")
    parser.add_argument("--revision", default="614241f622f53c4eeff9890bdc4f31cfecc418b3",
                        help="pinned model commit; override explicitly to evaluate another revision")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    model_id = "intfloat/multilingual-e5-small"
    info = HfApi().model_info(model_id, revision=args.revision)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} at {info.sha}", flush=True)
    snapshot_download(repo_id=model_id, revision=info.sha, local_dir=args.out,
        allow_patterns=["config.json", "config_sentence_transformers.json", "modules.json",
                        "sentence_bert_config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "sentencepiece.bpe.model", "model.safetensors",
                        "1_Pooling/config.json", "2_Normalize/config.json"])
    if not (args.out / "model.safetensors").exists():
        raise RuntimeError("Expected safetensors model weights were not downloaded")
    (args.out / "provenance.json").write_text(json.dumps({
        "model_id":model_id, "revision":info.sha, "license":"MIT (model card)",
        "source":"https://huggingface.co/intfloat/multilingual-e5-small",
    }, indent=2)+"\n")
    print(f"Ready: {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
