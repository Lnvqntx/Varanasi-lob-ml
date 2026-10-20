"""solution_v2.py for graphs with any set of recurrent states (export_v4.py: LSTM h0, h1, c0, c1).

The recurrent pairs are read from the graph: every input other than state and prev_prices is a hidden
state X fed back from output X_out, and all of them start at zero for a new sequence.

solution_v2.py's description:

Stateful single-step ONNX inference: one ORT run per row.

model.onnx contains the full train_v1 preprocessing for one row (prices, price diffs, float16-rounded
other features, 5.199 fill-value flags; standardization folded into the first GRU layer), two GRU
layers and the linear head. stats.npz holds the stats for reference; only price_idx is read here.

Inputs and outputs are bound once to preallocated buffers (ORT IOBinding). Two bindings alternate
so each row's recurrent outputs (h0, h1, prices) become the next row's inputs without copies.
"""
from pathlib import Path
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent


class PredictionModel:
    def __init__(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.use_per_session_threads = True
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(HERE / "model.onnx"), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        shapes = {i.name: tuple(i.shape) for i in self.session.get_inputs()}
        self.hidden = [n for n in shapes if n not in ("state", "prev_prices")]
        RECURRENT = [(n, n + "_out") for n in self.hidden] + [("prev_prices", "prices")]
        self.price_idx = np.load(HERE / "stats.npz")["price_idx"]

        self.state = np.zeros(shapes["state"], dtype=np.float32)
        self.pred = np.zeros((1, 1, 2), dtype=np.float32)
        # bufs[k][name]: recurrent buffer set k; binding k reads set k and writes set 1 - k.
        self.bufs = [{name: np.zeros(shapes[name], dtype=np.float32) for name, _ in RECURRENT} for _ in range(2)]
        self.bindings = []
        for k in range(2):
            b = self.session.io_binding()
            self._bind(b.bind_input, "state", self.state)
            self._bind(b.bind_output, "pred", self.pred)
            for name, out in RECURRENT:
                self._bind(b.bind_input, name, self.bufs[k][name])
                self._bind(b.bind_output, out, self.bufs[1 - k][name])
            self.bindings.append(b)
        self.parity = 0
        self.current_seq_ix = None
        self.previous_step = None

    @staticmethod
    def _bind(fn, name, arr):
        fn(name, "cpu", 0, np.float32, list(arr.shape), arr.ctypes.data)

    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        step = int(data_point.step_in_seq)
        if self.current_seq_ix != seq_ix:
            if step != 0:
                raise ValueError("a new sequence must start at step zero")
            self.current_seq_ix = seq_ix
            self.previous_step = None
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")

        state = np.asarray(data_point.state, dtype=np.float32)
        if state.shape != (112,) or not np.isfinite(state).all():
            raise ValueError("expected 112 finite float32 features")
        self.state[0, 0] = state
        if self.previous_step is None:
            # New sequence: zero hidden state; prev_prices = this row's prices, so the diff is exactly 0.
            self.parity = 0
            cur = self.bufs[0]
            for n in self.hidden:
                cur[n].fill(0.0)
            cur["prev_prices"][0, 0] = state[self.price_idx]
        self.previous_step = step

        self.session.run_with_iobinding(self.bindings[self.parity])
        self.parity ^= 1
        if not data_point.need_prediction:
            return None
        return self.pred[0, 0].copy()
