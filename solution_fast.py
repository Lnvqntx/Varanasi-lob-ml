"""Stateful single-step ONNX inference: one session.run per row.

model.onnx contains the full train_v1 preprocessing for one row (prices, price diffs, float16-rounded
other features, standardization with embedded stats, 12 fill-value flags), the GRU cells and the head.
stats.npz holds the same stats for reference; only price_idx is read here (for the step-0 diff).
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
        self.h_shape = tuple(self.session.get_inputs()[2].shape)
        self.price_idx = np.load(HERE / "stats.npz")["price_idx"]
        self.h = np.zeros(self.h_shape, dtype=np.float32)
        self.prev_prices = None
        self.current_seq_ix = None
        self.previous_step = None

    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        step = int(data_point.step_in_seq)
        if self.current_seq_ix != seq_ix:
            if step != 0:
                raise ValueError("a new sequence must start at step zero")
            self.current_seq_ix = seq_ix
            self.previous_step = None
            self.h = np.zeros(self.h_shape, dtype=np.float32)
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")

        state = np.asarray(data_point.state, dtype=np.float32)
        if state.shape != (112,) or not np.isfinite(state).all():
            raise ValueError("expected 112 finite float32 features")
        state = state.reshape(1, 112)
        if self.previous_step is None:
            self.prev_prices = state[:, self.price_idx]  # diff is exactly 0 at step 0
        self.previous_step = step

        pred, self.h, self.prev_prices = self.session.run(
            None, {"state": state, "prev_prices": self.prev_prices, "h": self.h})
        if not data_point.need_prediction:
            return None
        return np.asarray(pred[0], dtype=np.float32)
