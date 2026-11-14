"""Two-way warmed confirmation (no old-model cache pollution), unchanged architecture."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,collections,importlib.util,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from v34_model import register
from v34_stream_ring import PredictionModel


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);os.sched_setaffinity(0,{31});torch.set_num_threads(1)
    tm=register();model,ck=tm.load_model('checkpoints/v34_preflight_20261109_0549/v34b_tcn_final.pt','cpu')
    from export_v1 import FEATURE_COLUMNS
    price_idx=np.array([FEATURE_COLUMNS.index(c) for c in ck['price_cols']])
    valid=tm.load_split('valid');length=2048
    raw=np.empty((2,length,112),np.float32)
    raw[:,:,[FEATURE_COLUMNS.index(c) for c in ck['price_cols']]]=valid['prices'][:2,:length]
    raw[:,:,[FEATURE_COLUMNS.index(c) for c in ck['other_cols']]]=valid['features'][:2,:length]
    ring=PredictionModel('v34_results/ring_20261109_0600/v34_ring_fp32.onnx',price_idx)
    spec=importlib.util.spec_from_file_location('deployed','/home/jovyan/work/submission_ens_v18cv19a_int8/solution.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);deployed=module.PredictionModel()
    with torch.no_grad():
        x=tm.build_inputs(torch.from_numpy(np.array(valid['prices'][:2,:length])),torch.from_numpy(np.array(valid['features'][:2,:length])),ck['stats'])
        expected=model(x)[0].clamp(-2,2).numpy()
    repeats=[];max_error=0.;repeat_outputs=[]
    for repeat in range(5):
        times=[[],[]];outputs=np.empty((2,length-99,2),np.float32)
        for seq in range(2):
            for step in range(length):
                point=SimpleNamespace(seq_ix=repeat*2+seq,step_in_seq=step,state=raw[seq,step],need_prediction=step>=99)
                for i in ([0,1] if (step+repeat)%2 else [1,0]):
                    start=time.perf_counter_ns();p=(ring if i==0 else deployed).predict(point);elapsed=(time.perf_counter_ns()-start)/1000
                    if step>=256:times[i].append(elapsed)
                    if i==0 and step>=99:outputs[seq,step-99]=p
        err=float(np.max(np.abs(outputs-expected[:,99:])));assert err<1e-5;max_error=max(err,max_error)
        if repeat_outputs:assert np.array_equal(outputs,repeat_outputs[0])
        else:repeat_outputs.append(outputs)
        record=dict(ring_mean_us=float(np.mean(times[0])),deployed_mean_us=float(np.mean(times[1])),
                    ring_median_us=float(np.median(times[0])),deployed_median_us=float(np.median(times[1])))
        record['ratio']=record['ring_mean_us']/record['deployed_mean_us'];repeats.append(record);print('repeat',repeat+1,record,flush=True)
    # Components under matching alternating ring/deployed cache conditions.
    parts=collections.defaultdict(list)
    for step in range(length):
        point=SimpleNamespace(seq_ix=100,step_in_seq=step,state=raw[0,step],need_prediction=step>=99)
        deployed.predict(point)
        if step<256:ring.predict(point);continue
        start=time.perf_counter_ns()
        assert ring.seq==100 and ring.previous_step==step-1
        r=np.asarray(point.state,np.float32)
        if r.shape!=(112,) or not np.isfinite(r,out=ring.finite).all():raise ValueError()
        ring.raw[0]=r;ring.previous_step=step
        t1=time.perf_counter_ns();ring.session.run_with_iobinding(ring.bindings[ring.phase])
        t2=time.perf_counter_ns();ring.update_history();t3=time.perf_counter_ns();ring.pred[0].copy();t4=time.perf_counter_ns()
        for k,v in [('python_validation_and_input',t1-start),('onnx_execution',t2-t1),('history_and_price_update',t3-t2),('output_copy',t4-t3)]:parts[k].append(v/1000)
    # Ring physical layout vs batched per-block input histories after wraparound.
    with torch.no_grad():_,state=model(x[:1])
    errors=[]
    for i,buffer in enumerate(ring.rings):
        pos=ring.phase&ring.masks[i]
        chronological=np.concatenate([buffer[pos:],buffer[:pos]],axis=0).T
        errors.append(float(np.max(np.abs(chronological-state[i][0].numpy()))))
    assert max(errors)<1e-5,errors
    mean_ring=float(np.mean([r['ring_mean_us'] for r in repeats]));mean_ref=float(np.mean([r['deployed_mean_us'] for r in repeats]))
    result=dict(repeats=repeats,ring_mean_us=mean_ring,deployed_mean_us=mean_ref,ratio=mean_ring/mean_ref,
                screening_pass=all(r['ratio']<=1.2 for r in repeats),core=31,threads=1,timed_rows_per_model=5*2*(length-256),
                max_output_error=max_error,deterministic_repeats_exact=True,state_error_after_wrap_by_block=errors,
                alternating_callback_components_us={k:float(np.mean(v)) for k,v in parts.items()},
                caveat='Component timers add instrumentation overhead; complete uninstrumented callbacks above are authoritative. Smoke weights, screening not platform proof.')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
