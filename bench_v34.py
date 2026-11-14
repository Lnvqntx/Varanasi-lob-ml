"""Streaming prototype export + correctness + pinned one-thread predict timings.
Use a bounded-smoke checkpoint; never creates a submission or trains.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from v34_model import DILATIONS,STATE_FLOATS,register
from v34_stream import StreamingGraph,PredictionModel


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ckpt',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,{31});torch.set_num_threads(1)
    tm=register();model,ck=tm.load_model(a.ckpt,'cpu')
    wrapper=StreamingGraph(model,ck).eval()
    inputs=['raw','previous_prices','first']+[f'h{i}' for i in range(8)]
    outputs=['pred','prices']+[f'h{i}_out' for i in range(8)]
    args=(torch.zeros(1,1,112),torch.zeros(1,1,44),torch.ones(1,1,1,dtype=torch.bool),*[torch.zeros(1,64,2*d) for d in DILATIONS])
    fp32=out/'v34_fp32.onnx';int8=out/'v34_int8.onnx'
    torch.onnx.export(wrapper,args,str(fp32),input_names=inputs,output_names=outputs,opset_version=17,dynamo=False)
    from onnxruntime.quantization import quantize_dynamic,QuantType
    quantize_dynamic(str(fp32),str(int8),weight_type=QuantType.QInt8,op_types_to_quantize=['MatMul','Gemm'])
    # Actual graph coverage check: cached temporal taps are elementwise fp32;
    # projection, eight pointwise mixers and head are dynamically quantized.
    import onnx
    graph=onnx.load(int8)
    quant_nodes=sum(n.op_type=='MatMulInteger' for n in graph.graph.node)
    assert quant_nodes>=10,quant_nodes
    from export_v1 import FEATURE_COLUMNS
    data=tm.load_split('valid');steps=1600;sequences=3
    raw=np.empty((sequences,steps,112),np.float32)
    raw[:,:,[FEATURE_COLUMNS.index(c) for c in ck['price_cols']]]=data['prices'][:sequences,:steps]
    raw[:,:,[FEATURE_COLUMNS.index(c) for c in ck['other_cols']]]=data['features'][:sequences,:steps]
    with torch.no_grad():
        x=tm.build_inputs(torch.from_numpy(np.array(data['prices'][:sequences,:steps])),torch.from_numpy(np.array(data['features'][:sequences,:steps])),ck['stats'])
        reference=model(x)[0].clamp(-2,2).numpy()
    fp=PredictionModel(fp32);quant=PredictionModel(int8)
    spec=importlib.util.spec_from_file_location('deployed_v18cv19a','/home/jovyan/work/submission_ens_v18cv19a_int8/solution.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    deployed=module.PredictionModel()
    models={'v34_fp32':fp,'v34_int8':quant,'deployed_pair_int8':deployed}
    times={k:[] for k in models};preds={k:np.empty((sequences,steps-99,2),np.float32) for k in models}
    # Alternate model order by row to reduce sequential timing/order bias.
    for seq in range(sequences):
        for step in range(steps):
            point=SimpleNamespace(seq_ix=seq,step_in_seq=step,state=raw[seq,step],need_prediction=step>=99)
            names=list(models)
            if step%2:names.reverse()
            for name in names:
                start=time.perf_counter();pred=models[name].predict(point);elapsed=time.perf_counter()-start
                if step>=200:times[name].append(elapsed)
                if step>=99:preds[name][seq,step-99]=pred
                else:assert pred is None
    expected=reference[:,99:]
    fp_error=np.abs(preds['v34_fp32']-expected);int_error=np.abs(preds['v34_int8']-preds['v34_fp32'])
    assert fp_error.max()<1e-5,fp_error.max()
    # Sequence reset with dirty caches must reproduce a fresh adapter exactly.
    point=SimpleNamespace(seq_ix=99,step_in_seq=0,state=raw[0,0],need_prediction=True)
    fresh=PredictionModel(fp32)
    assert np.array_equal(fp.predict(point),fresh.predict(point))
    # Repeating the same data in a distinct sequence is bitwise deterministic.
    sequence_again=[]
    for step in range(steps):
        p=fp.predict(SimpleNamespace(seq_ix=100,step_in_seq=step,state=raw[0,step],need_prediction=step>=99))
        if p is not None:sequence_again.append(p)
    assert np.array_equal(np.array(sequence_again),preds['v34_fp32'][0])
    result=dict(checkpoint=str(Path(a.ckpt).resolve()),checkpoint_step=ck['step'],parameters=sum(p.numel() for p in model.parameters()),
                state_floats=STATE_FLOATS,state_bytes_fp32=STATE_FLOATS*4,previous_prices_bytes=44*4,
                iobinding_double_buffer_history_bytes=STATE_FLOATS*4*2,affinity=[31],threads=1,sequences=sequences,rows_per_sequence=steps,
                timed_rows_per_model=len(times['v34_fp32']),timing='matching predict() calls, raw preprocessing + validation + IOBinding + output copy; alternating model order; first 200 rows per sequence excluded',
                predict_us={k:dict(mean=float(np.mean(v)*1e6),median=float(np.median(v)*1e6),p95=float(np.percentile(v,95)*1e6)) for k,v in times.items()},
                fp32_vs_pytorch=dict(max=float(fp_error.max()),mean=float(fp_error.mean())),
                int8_vs_fp32=dict(max=float(int_error.max()),mean=float(int_error.mean()),rmse=float(np.sqrt(np.mean(int_error**2)))),
                errors_on='clipped predictions at all required rows, including after step 1000, across three independent sequences',
                warmup_and_reset_verified=True,deterministic_streaming_exact=True,
                dynamic_int8_matmul_nodes=quant_nodes,quantization='projection + eight pointwise mixers + head, with depthwise temporal taps/state kept fp32',
                graph_bytes=dict(fp32=fp32.stat().st_size,int8=int8.stat().st_size),
                limitation='100-update smoke weights; errors are not validation metric/accuracy guarantees; no full training or submission built')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
