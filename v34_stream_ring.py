"""Same v34 network, compact ONNX and Python-managed circular block-input caches.
256 prebound phases avoid per-row pointer rebinding, tap copies and history shifts.
One ORT call/row writes eight 64-channel block inputs into preallocated scratch;
only then are they copied to the circular buffers (no input/output aliasing).
"""
from pathlib import Path
import numpy as np


def build(ck):
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    from export_v1 import FEATURE_COLUMNS
    sd={k:v.cpu().numpy() for k,v in ck['model'].items()}
    assert sd['projection.weight'].shape==(64,168) and len([k for k in sd if k.endswith('depthwise.weight')])==8
    nodes=[];inits=[]
    def init(name,a):inits.append(nh.from_array(np.asarray(a),name));return name
    def node(op,inputs,out,**kw):nodes.append(h.make_node(op,inputs,[out],name=out,**kw));return out
    price_idx=np.array([FEATURE_COLUMNS.index(c) for c in ck['price_cols']],np.int64)
    other_idx=np.array([FEATURE_COLUMNS.index(c) for c in ck['other_cols']],np.int64)
    flag_idx=np.array([ck['other_cols'].index(c) for c in ck['flag_cols']],np.int64)
    node('Gather',['raw',init('price_idx',price_idx)],'prices',axis=1)
    node('Gather',['raw',init('other_idx',other_idx)],'other_raw',axis=1)
    node('Cast',['other_raw'],'other16',to=T.FLOAT16);node('Cast',['other16'],'other',to=T.FLOAT)
    node('Sub',['prices','previous_prices'],'difference')
    node('Gather',['other',init('flag_idx',flag_idx)],'flag_values',axis=1)
    node('Abs',['flag_values'],'absolute_flags')
    node('Equal',['absolute_flags',init('fill',np.float32(5.19921875))],'is_fill')
    node('Cast',['is_fill'],'flags',to=T.FLOAT)
    node('Concat',['prices','difference','other'],'continuous',axis=1)
    node('Sub',['continuous',init('mean',ck['stats']['mean'].cpu().numpy())],'centered')
    node('Div',['centered',init('std',ck['stats']['std'].cpu().numpy())],'standardized')
    node('Concat',['standardized','flags'],'x',axis=1)
    init('projection_w',sd['projection.weight'].T.copy());init('projection_b',sd['projection.bias'])
    node('Gemm',['x','projection_w','projection_b'],'projection');node('Relu',['projection'],'z0')
    for i in range(8):
        prefix=f'blocks.{i}.';current=f'z{i}'
        w=sd[prefix+'depthwise.weight'][:,0]
        for tap,inp in enumerate([f'past{i}a',f'past{i}b',current]):
            node('Mul',[inp,init(f'dw{i}w{tap}',w[:,tap].copy())],f'term{i}_{tap}')
        node('Sum',[f'term{i}_{j}' for j in range(3)]+[init(f'dw{i}bias',sd[prefix+'depthwise.bias'])],f'dw{i}')
        node('Relu',[f'dw{i}'],f'temporal{i}')
        # Reassociate the residual bias to allow Gemm+ReLU fusion. Same real-
        # arithmetic function, measured fp32 roundoff checked against batched TCN.
        node('Add',[current,init(f'mix{i}bias',sd[prefix+'pointwise.bias'])],f'residual{i}')
        node('Gemm',[f'temporal{i}',init(f'mix{i}w',sd[prefix+'pointwise.weight'][:,:,0].T.copy()),f'residual{i}'],f'mix{i}')
        node('Relu',[f'mix{i}'],f'z{i+1}')
    node('Gemm',['z8',init('head_w',sd['head.weight'].T.copy()),init('head_b',sd['head.bias'])],'unclipped')
    node('Clip',['unclipped',init('lo',np.float32(-2)),init('hi',np.float32(2))],'pred')
    info=lambda name,shape:h.make_tensor_value_info(name,T.FLOAT,shape)
    graph=h.make_graph(nodes,'v34_same_architecture_circular',
        [info('raw',[1,112]),info('previous_prices',[1,44])]+[info(f'past{i}{k}',[1,64]) for i in range(8) for k in ('a','b')],
        [info('pred',[1,2]),info('prices',[1,44])]+[info(f'z{i}',[1,64]) for i in range(8)],inits)
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)],producer_name='v34_ring_same_architecture');model.ir_version=8
    onnx.checker.check_model(model,full_check=True)
    return model,price_idx


class PredictionModel:
    def __init__(self,path,price_idx,profile_prefix=None):
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        options.add_session_config_entry('session.inter_op.allow_spinning','0')
        if profile_prefix:
            options.enable_profiling=True;options.profile_file_prefix=str(profile_prefix)
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        self.price_idx=price_idx
        self.raw=np.zeros((1,112),np.float32);self.prev=np.zeros((1,44),np.float32)
        self.prices=np.zeros((1,44),np.float32);self.pred=np.zeros((1,2),np.float32)
        self.finite=np.zeros(112,bool)
        self.z=np.zeros((8,1,64),np.float32)
        self.rings=[np.zeros((2*d,64),np.float32) for d in (1,2,4,8,16,32,64,128)]
        self.masks=[len(r)-1 for r in self.rings]
        self.bindings=[]
        for phase in range(256):
            binding=self.session.io_binding()
            for name,array in [('raw',self.raw),('previous_prices',self.prev)]:self.bind(binding.bind_input,name,array)
            for name,array in [('pred',self.pred),('prices',self.prices)]:self.bind(binding.bind_output,name,array)
            for i,ring in enumerate(self.rings):
                pos=phase&self.masks[i];near=(pos+len(ring)//2)&self.masks[i]
                self.bind(binding.bind_input,f'past{i}a',ring[pos:pos+1])
                self.bind(binding.bind_input,f'past{i}b',ring[near:near+1])
                self.bind(binding.bind_output,f'z{i}',self.z[i])
            self.bindings.append(binding)
        self.phase=0;self.seq=None;self.previous_step=None

    @staticmethod
    def bind(fn,name,array):fn(name,'cpu',0,np.float32,list(array.shape),array.ctypes.data)

    def update_history(self):
        phase=self.phase
        for i,ring in enumerate(self.rings):ring[phase&self.masks[i]]=self.z[i,0]
        np.copyto(self.prev,self.prices)
        self.phase=(phase+1)&255

    def predict(self,point):
        seq=int(point.seq_ix);step=int(point.step_in_seq)
        reset=self.seq!=seq
        if reset:
            if step!=0:raise ValueError('a new sequence must start at step zero')
            self.seq=seq;self.previous_step=None;self.phase=0
        if self.previous_step is not None and step!=self.previous_step+1:raise ValueError('rows must arrive in sequence order')
        state=np.asarray(point.state,dtype=np.float32)
        if state.shape!=(112,) or not np.isfinite(state,out=self.finite).all():raise ValueError('expected 112 finite features')
        self.raw[0]=state
        if reset:
            for ring in self.rings:ring.fill(0)
            np.take(state,self.price_idx,out=self.prev[0])  # first price difference exactly zero
        self.previous_step=step
        self.session.run_with_iobinding(self.bindings[self.phase])
        self.update_history()
        if not point.need_prediction:return None
        return self.pred[0].copy()
