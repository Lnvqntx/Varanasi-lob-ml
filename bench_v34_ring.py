"""Repeated warmed matched callbacks: old fp32/int8, same-architecture ring, deployed pair."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,collections,importlib.util,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import onnx
from v34_model import register
from v34_stream import PredictionModel as Old
from v34_stream_ring import build,PredictionModel as Ring


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);os.sched_setaffinity(0,{31});torch.set_num_threads(1)
    ckpt=Path('checkpoints/v34_preflight_20261109_0549/v34b_tcn_final.pt')
    tm=register();model,ck=tm.load_model(str(ckpt),'cpu')
    graph,price_idx=build(ck);path=out/'v34_ring_fp32.onnx';onnx.save(graph,path)
    data=tm.load_split('valid');nseq=2;length=2048
    from export_v1 import FEATURE_COLUMNS
    raw=np.empty((nseq,length,112),np.float32)
    raw[:,:,[FEATURE_COLUMNS.index(c) for c in ck['price_cols']]]=data['prices'][:nseq,:length]
    raw[:,:,[FEATURE_COLUMNS.index(c) for c in ck['other_cols']]]=data['features'][:nseq,:length]
    with torch.no_grad():
        x=tm.build_inputs(torch.from_numpy(np.array(data['prices'][:nseq,:length])),torch.from_numpy(np.array(data['features'][:nseq,:length])),ck['stats'])
        ref=model(x)[0].clamp(-2,2).numpy()
        # Explicit training-chunk interface to verify the same reference across 1000 boundaries.
        state=None;pieces=[]
        for s in range(0,length,1000):
            p,state=model(x[:,s:s+1000],state);pieces.append(p)
        chunk_error=float((torch.cat(pieces,1).clamp(-2,2)-torch.from_numpy(ref)).abs().max())
    spec=importlib.util.spec_from_file_location('deployed','/home/jovyan/work/submission_ens_v18cv19a_int8/solution.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    models={'old_fp32':Old('v34_results/onnx_20261109_0551/v34_fp32.onnx'),
            'old_int8':Old('v34_results/onnx_20261109_0551/v34_int8.onnx'),
            'ring_fp32':Ring(path,price_idx),'deployed_pair':module.PredictionModel()}
    names=list(models);repeat_means=[];all_times={k:[] for k in names};errors=[]
    # Five full repeats, independent seq IDs; first 256 rows/sequence excluded.
    for repeat in range(5):
        times={k:[] for k in names};preds={k:np.empty((nseq,length-99,2),np.float32) for k in names}
        for seq in range(nseq):
            for step in range(length):
                point=SimpleNamespace(seq_ix=repeat*nseq+seq,step_in_seq=step,state=raw[seq,step],need_prediction=step>=99)
                shift=(step+repeat)%len(names);order=names[shift:]+names[:shift]
                if step%2:order=order[::-1]
                for name in order:
                    start=time.perf_counter_ns();p=models[name].predict(point);elapsed=(time.perf_counter_ns()-start)/1000
                    if step>=256:times[name].append(elapsed)
                    if step>=99:preds[name][seq,step-99]=p
                    else:assert p is None
        err=float(np.max(np.abs(preds['ring_fp32']-ref[:,99:])))
        assert err<1e-5,err
        err_old=float(np.max(np.abs(preds['ring_fp32']-preds['old_fp32'])))
        assert err_old<1e-5
        errors.append(err)
        means={k:float(np.mean(v)) for k,v in times.items()};repeat_means.append(means)
        for k in names:all_times[k]+=times[k]
        print('repeat',repeat+1,json.dumps(means),'max error',err,flush=True)
    ring=models['ring_fp32']
    fresh=Ring(path,price_idx)
    point=SimpleNamespace(seq_ix=100,step_in_seq=0,state=raw[0,0],need_prediction=True)
    assert np.array_equal(ring.predict(point),fresh.predict(point))
    # Future changed rows cannot affect already-returned predictions.
    before=[];after=[]
    for target,seq in [(before,101),(after,102)]:
        for step in range(700):
            r=raw[0,step].copy()
            if seq==102 and step>=600:r+=10
            target.append(ring.predict(SimpleNamespace(seq_ix=seq,step_in_seq=step,state=r,need_prediction=True)))
    assert np.array_equal(np.array(before[:600]),np.array(after[:600]))
    # Fine-grained unprofiled accounting, after full repeated-callback measurements.
    comp=collections.defaultdict(list)
    for step in range(700,1700):
        t=time.perf_counter_ns();r=np.asarray(raw[0,step],np.float32);np.isfinite(r,out=ring.finite).all()
        t1=time.perf_counter_ns();ring.raw[0]=r
        t2=time.perf_counter_ns();ring.session.run_with_iobinding(ring.bindings[ring.phase])
        t3=time.perf_counter_ns();ring.update_history()
        t4=time.perf_counter_ns();ring.pred[0].copy();t5=time.perf_counter_ns()
        for k,v in [('python_validate',t1-t),('python_input_copy',t2-t1),('onnx_execution',t3-t2),('circular_state_update',t4-t3),('output_copy',t5-t4)]:comp[k].append(v/1000)
    profiler=Ring(path,price_idx,profile_prefix=out/'ort_profile')
    for step in range(250):profiler.predict(SimpleNamespace(seq_ix=0,step_in_seq=step,state=raw[0,step],need_prediction=True))
    events=json.loads(Path(profiler.session.end_profiling()).read_text());op=collections.defaultdict(list)
    for e in events:
        if e.get('cat')=='Node' and e['name'].endswith('_kernel_time'):op[e['args']['op_name']].append(e['dur'])
    result=dict(checkpoint=str(ckpt.resolve()),same_architecture=True,parameters=46274,channels=64,blocks=8,kernel=3,dilations=[1,2,4,8,16,32,64,128],
                repeats=5,sequences_per_repeat=nseq,rows_per_sequence=length,warmup_rows_excluded=256,timed_rows_per_model=len(all_times[names[0]]),
                core=31,threads=1,measurement='complete predict callbacks including validation, raw preprocessing, one ORT call, circular state updates and prediction copy; alternating/rotating model order',
                repeats_mean_us=repeat_means,latency_us={k:dict(mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95))) for k,v in all_times.items()},
                ratio_ring_to_deployed=float(np.mean(all_times['ring_fp32'])/np.mean(all_times['deployed_pair'])),
                screening_pass=bool(np.mean(all_times['ring_fp32'])<=1.2*np.mean(all_times['deployed_pair'])),
                ring_to_pytorch_max=max(errors),ring_to_old_fp32_max=err_old,chunk_to_whole_max=chunk_error,reset_exact=True,future_past_exact=True,
                callback_components_us={k:float(np.mean(v)) for k,v in comp.items()},
                instrumented_operators={k:dict(us_per_row=sum(v)/250,calls=len(v)) for k,v in op.items()},
                graph_op_counts=dict(collections.Counter(n.op_type for n in graph.graph.node)),
                recurrent_history_bytes=130560,history_bytes_written_per_row=2048,price_bytes_copied_per_row=176,
                state_bindings=256,per_row_ort_calls=1,graph_bytes=path.stat().st_size,
                limitations=['100-update smoke weights; screening latency is not platform runtime proof','FP32 retained; dynamic int8 previously slower','ORT operator profile includes instrumentation overhead; callback timings are unprofiled'])
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
