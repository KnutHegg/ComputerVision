import subprocess
import sys
from pathlib import Path

# Deprecated compatibility entry point.
# Preferred command: `python src/infer_live.py`

def main() -> int:
    src_script = Path(__file__).resolve().parent / "src" / "infer_live.py"
    cmd = [sys.executable, str(src_script), *sys.argv[1:]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
