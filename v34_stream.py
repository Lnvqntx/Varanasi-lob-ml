"""Disposable streaming graph wrapper and one-thread IOBinding adapter for v34."""
import numpy as np
import torch
from torch import nn


class StreamingGraph(nn.Module):
    def __init__(self, model, ck):
        super().__init__()
        from export_v1 import FEATURE_COLUMNS
        self.model=model
        for name,value in [('price_idx',[FEATURE_COLUMNS.index(c) for c in ck['price_cols']]),
                           ('other_idx',[FEATURE_COLUMNS.index(c) for c in ck['other_cols']]),
                           ('flag_idx',[ck['other_cols'].index(c) for c in ck['flag_cols']])]:
            self.register_buffer(name,torch.tensor(value))
        self.register_buffer('mean',ck['stats']['mean'].cpu())
        self.register_buffer('std',ck['stats']['std'].cpu())

    def forward(self, raw, previous_prices, first, *history):
        prices=raw[:,:,self.price_idx]
        other=raw[:,:,self.other_idx].half().float()
        diff=torch.where(first,torch.zeros_like(prices),prices-previous_prices)
        flags=(other[:,:,self.flag_idx].abs()==5.19921875).float()
        x=torch.cat([(torch.cat([prices,diff,other],2)-self.mean)/self.std,flags],2)
        state=tuple(torch.where(first,torch.zeros_like(h),h) for h in history)
        pred,state=self.model.step(x[:,0],state)
        return (pred[:,None].clamp(-2,2),prices,*state)


class PredictionModel:
    def __init__(self, path):
        import onnxruntime as ort
        options=ort.SessionOptions()
        options.intra_op_num_threads=options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        options.add_session_config_entry('session.inter_op.allow_spinning','0')
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        shapes={item.name:tuple(item.shape) for item in self.session.get_inputs()}
        self.raw=np.zeros(shapes['raw'],np.float32)
        self.first=np.ones(shapes['first'],bool)
        self.pred=np.zeros((1,1,2),np.float32)
        self.names=['previous_prices']+[f'h{i}' for i in range(8)]
        self.buffers=[{name:np.zeros(shapes[name],np.float32) for name in self.names} for _ in range(2)]
        self.bindings=[]
        for k in range(2):
            binding=self.session.io_binding()
            self.bind(binding.bind_input,'raw',self.raw)
            self.bind(binding.bind_input,'first',self.first)
            self.bind(binding.bind_output,'pred',self.pred)
            for name in self.names:
                self.bind(binding.bind_input,name,self.buffers[k][name])
                self.bind(binding.bind_output,'prices' if name=='previous_prices' else name+'_out',self.buffers[1-k][name])
            self.bindings.append(binding)
        self.parity=0;self.seq=None;self.previous_step=None

    @staticmethod
    def bind(fn,name,array):
        fn(name,'cpu',0,array.dtype.type,list(array.shape),array.ctypes.data)

    def predict(self, point):
        seq=int(point.seq_ix);step=int(point.step_in_seq)
        if self.seq!=seq:
            if step!=0:raise ValueError('a new sequence must start at step zero')
            self.seq=seq;self.previous_step=None;self.parity=0
        if self.previous_step is not None and step!=self.previous_step+1:
            raise ValueError('rows must arrive in sequence order')
        state=np.asarray(point.state,dtype=np.float32)
        if state.shape!=(112,) or not np.isfinite(state).all():raise ValueError('expected 112 finite features')
        self.raw[0,0]=state
        self.first.fill(self.previous_step is None)
        self.previous_step=step
        self.session.run_with_iobinding(self.bindings[self.parity]);self.parity^=1
        if not point.need_prediction:return None
        return self.pred[0,0].copy()
