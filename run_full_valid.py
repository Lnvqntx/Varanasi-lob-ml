"""Score a submission zip on the full validation set with the official ScorerStepByStep.

  taskset -c 31 python run_full_valid.py ZIP EXTRACT_DIR

Extracts ZIP into EXTRACT_DIR (must not exist), imports solution.PredictionModel from there and runs
utils.ScorerStepByStep(valid.parquet).score(). A thin wrapper only counts rows and logs progress.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "1"

import importlib.util
import json
import sys
import time
import zipfile

STARTER = "/home/jovyan/work/wnn/wnn_connectome_starterpack"
sys.path.insert(0, STARTER)
from utils import SEQUENCE_LENGTH, ScorerStepByStep  # noqa: E402


class Progress:
    def __init__(self, model, n_seq):
        self.model, self.n_seq = model, n_seq
        self.rows, self.t0, self.t_predict = 0, time.time(), 0.0

    def predict(self, data_point):
        t = time.perf_counter()
        out = self.model.predict(data_point)
        self.t_predict += time.perf_counter() - t
        self.rows += 1
        if self.rows % (100 * SEQUENCE_LENGTH) == 0:
            el = time.time() - self.t0
            seqs = self.rows // SEQUENCE_LENGTH
            print(f"{seqs}/{self.n_seq} seqs, {el / 60:.1f} min elapsed, {el / self.rows * 1e6:.1f} us/row overall, "
                  f"{self.t_predict / self.rows * 1e6:.1f} us/row in predict(), eta {el / seqs * (self.n_seq - seqs) / 60:.1f} min",
                  flush=True)
        return out


def main():
    zip_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir)
    with zipfile.ZipFile(zip_path) as z:
        print(f"{zip_path}: {[i.filename for i in z.infolist()]}")
        z.extractall(out_dir)
    spec = importlib.util.spec_from_file_location("solution", os.path.join(out_dir, "solution.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    scorer = ScorerStepByStep(f"{STARTER}/datasets/valid.parquet")
    n_seq = scorer.parquet.num_row_groups
    print(f"affinity {sorted(os.sched_getaffinity(0))}, {n_seq} sequences, {n_seq * SEQUENCE_LENGTH:,} rows", flush=True)
    model = Progress(mod.PredictionModel(), n_seq)
    t = time.time()
    result = scorer.score(model)
    el = time.time() - t
    print(json.dumps(result, indent=1))
    print(f"total {el / 60:.1f} min for {model.rows:,} rows: {el / model.rows * 1e6:.1f} us/row overall "
          f"(incl. parquet reads + scorer loop), {model.t_predict / model.rows * 1e6:.1f} us/row in predict(); "
          f"projected for 39.4M rows: {el / model.rows * 39.4e6 / 60:.1f} min overall")


if __name__ == "__main__":
    main()
