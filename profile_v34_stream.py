"""Profile the existing graph and callback; CPU-only, fresh outputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,collections
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import onnx
import onnxruntime as ort
from v34_stream import PredictionModel


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);os.sched_setaffinity(0,{31})
    path=Path('v34_results/onnx_20261109_0551/v34_fp32.onnx').resolve()
    g=onnx.load(path)
    model=PredictionModel(path)
    rng=np.random.default_rng(0);rows=rng.normal(size=(1000,112)).astype(np.float32)
    # Warm ORT and memory; isolate components without ORT profiler overhead.
    for i in range(100):model.predict(SimpleNamespace(seq_ix=0,step_in_seq=i,state=rows[i],need_prediction=True))
    parts=collections.defaultdict(list)
    for i in range(100,1000):
        t=time.perf_counter_ns();state=np.asarray(rows[i],dtype=np.float32)
        assert state.shape==(112,) and np.isfinite(state).all()
        t1=time.perf_counter_ns();model.raw[0,0]=state;model.first.fill(False);model.previous_step=i
        t2=time.perf_counter_ns();model.session.run_with_iobinding(model.bindings[model.parity]);model.parity^=1
        t3=time.perf_counter_ns();model.pred[0,0].copy();t4=time.perf_counter_ns()
        for k,v in [('python_validation',t1-t),('python_input_state_setup',t2-t1),('onnx_run_including_history',t3-t2),('python_output_copy',t4-t3)]:parts[k].append(v/1000)
    opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1;opts.enable_profiling=True
    opts.profile_file_prefix=str(out/'ort_profile');opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    prof=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    feed={i.name:np.zeros(i.shape,dtype=bool if i.type=='tensor(bool)' else np.float32) for i in prof.get_inputs()}
    for i in range(200):
        feed['raw'][0,0]=rows[i];feed['first'].fill(i==0)
        results=prof.run(None,feed)
        for name,arr in zip([o.name for o in prof.get_outputs()],results):
            if name=='prices':feed['previous_prices']=arr
            elif name.startswith('h'):feed[name[:-4]]=arr
    events=json.loads(Path(prof.end_profiling()).read_text())
    byop=collections.defaultdict(list);bynode=collections.defaultdict(list)
    for e in events:
        if e.get('cat')=='Node' and e['name'].endswith('_kernel_time'):
            byop[e['args']['op_name']].append(e['dur']);bynode[e['name']].append(e['dur'])
    operators={k:dict(total_us=sum(v),calls=len(v),us_per_row=sum(v)/200) for k,v in byop.items()}
    # Distinguish history movement nodes (zero-reset Where, shift Slice+Concat),
    # versus temporal taps (Gather/Slice/Mul). Trace timings are profiler-instrumented.
    result=dict(callback_components_us={k:float(np.mean(v)) for k,v in parts.items()},operators=operators,
                top_nodes=sorted([dict(name=k,total_us=sum(v),us_per_row=sum(v)/200) for k,v in bynode.items()],key=lambda v:-v['total_us'])[:30],
                original_graph_op_counts=dict(collections.Counter(n.op_type for n in g.graph.node)),
                window_convolution_nodes=sum(n.op_type=='Conv' for n in g.graph.node),
                history_reset_shift_ops_instrumented_us_per_row=sum(v['us_per_row'] for k,v in operators.items() if k in ('Where','Concat','Slice')),
                caveat='ORT event durations include profiler overhead; use unprofiled component and callback benchmarks for latency. Where/Concat/Slice aggregate also includes some preprocessing/small tensor movement.')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
