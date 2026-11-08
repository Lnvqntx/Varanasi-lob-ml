"""Isolated v33 entry point; unchanged train_multi data, loss and optimization.
Output directory must be new; refuses to overwrite prior outputs.
"""
import os
import sys
from pathlib import Path
from v33_model import register


def main():
    if '--output-dir' not in sys.argv:
        raise SystemExit('Required: --output-dir NEW_DIRECTORY')
    i = sys.argv.index('--output-dir')
    out = Path(sys.argv[i + 1]).resolve()
    del sys.argv[i:i + 2]
    out.mkdir(parents=True, exist_ok=False)
    tm = register()
    tm.CKPT_DIR = str(out)
    # Existing trainer logs use /home/jovyan/work/{name}; route only FileHandler
    # paths into the new output directory without changing training logic.
    handler = tm.logging.FileHandler
    tm.logging.FileHandler = lambda filename, *a, **kw: handler(out / Path(filename).name, *a, **kw)
    tm.GPU_MEM_CAP_GB = 2.5
    tm.main()


if __name__ == '__main__':
    main()
