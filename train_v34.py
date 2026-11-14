"""Fresh-output v34 shared-stream entry point. Original trainer remains unchanged."""
import sys
from pathlib import Path
from v34_model import register


def main():
    if '--output-dir' not in sys.argv:
        raise SystemExit('Required: --output-dir NEW_DIRECTORY')
    index = sys.argv.index('--output-dir')
    out = Path(sys.argv[index + 1]).resolve()
    del sys.argv[index:index + 2]
    tm = register()
    out.mkdir(parents=True, exist_ok=False)
    tm.CKPT_DIR = str(out)
    handler = tm.logging.FileHandler
    tm.logging.FileHandler = lambda filename, *a, **kw: handler(out / Path(filename).name, *a, **kw)
    tm.GPU_MEM_CAP_GB = 2.5
    tm.main()


if __name__ == '__main__':
    main()
