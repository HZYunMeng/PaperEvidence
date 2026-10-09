"""Build a source-only archive, excluding local data, weights and credentials."""
import argparse
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".venv", ".cache", "models", "data", "__pycache__", ".git", ".streamlit"}
EXCLUDED_FILES = {"config.toml", ".DS_Store"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT.parent / "academic-rag-starter.zip")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(args.out,"w",ZIP_DEFLATED) as archive:
        for directory, dirs, files in os.walk(ROOT):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
            for name in sorted(files):
                path = Path(directory) / name
                if (name in EXCLUDED_FILES or name.startswith(".env")
                        or path.relative_to(ROOT) == Path("eval/results.json")
                        or path.suffix in {".pyc", ".log", ".zip"} or path.is_symlink()):
                    continue
                archive.write(path,Path("academic-rag") / path.relative_to(ROOT))
    print(f"Source archive: {args.out.resolve()} ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
