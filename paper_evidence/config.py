import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    base_url: str = ""
    model: str = ""
    api_key: str = field(default="", repr=False)
    max_tokens: int = 1200
    max_evidence_chars: int = 16000
    timeout: float = 90
    json_mode: bool = True
    enable_thinking: bool | None = None
    embedding_path: str = ""


def load_settings(path=None, environ=None):
    env = os.environ if environ is None else environ
    config_path = Path(path or ROOT / "config.toml")
    data = tomllib.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    llm, retrieval = data.get("llm", {}), data.get("retrieval", {})
    key_env = llm.get("api_key_env", "PAPER_EVIDENCE_API_KEY")
    thinking = llm.get("enable_thinking")
    if thinking is not None and type(thinking) is not bool:
        raise ValueError("enable_thinking must be boolean")
    return Settings(
        base_url=env.get("PAPER_EVIDENCE_BASE_URL", llm.get("base_url", "")),
        model=env.get("PAPER_EVIDENCE_MODEL", llm.get("model", "")),
        api_key=env.get(key_env, ""),
        max_tokens=int(llm.get("max_tokens", 1200)),
        max_evidence_chars=int(llm.get("max_evidence_chars", 16000)),
        timeout=float(llm.get("timeout", 90)), json_mode=llm.get("json_mode", True),
        enable_thinking=thinking,
        embedding_path=env.get("PAPER_EVIDENCE_EMBEDDING_PATH", retrieval.get("model_path", "")),
    )
