"""python examples/make_synthetic_clip.py out.mp4 — a 6 s generated court clip (used by the smoke test)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_e2e_synthetic import synthetic_clip  # noqa: E402

if __name__ == "__main__":
    print(synthetic_clip(Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic.mp4")))
