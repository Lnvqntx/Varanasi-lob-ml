"""Actual mixed exported callbacks: pinned one-core timing, fidelity and isolated memory.
No training/submission building. Use --out NEW_DIR after fixed-blend validation.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,importlib.util,json,resource,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def load(directory):
    spec=importlib.util.spec_from_file_location('audit_'+str(abs(hash(str(directory)))),str(Path(directory)/'solution.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.PredictionModel()


def rss():
    return int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))*1024


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--memory-dir');a=ap.parse_args()
    os.sched_setaffinity(0,{31})
    if a.memory_dir:
        baseline=rss();start=time.perf_counter();model=load(a.memory_dir);startup=time.perf_counter()-start
        for step in range(2000):model.predict(SimpleNamespace(seq_ix=0,step_in_seq=step,need_prediction=step>=99,state=np.zeros(112,np.float32)))
        print(json.dumps(dict(baseline_RSS_bytes=baseline,loaded_warmed_RSS_bytes=rss(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                              startup_seconds=startup,buffer_bytes=sum(v.nbytes for block in model.bufs for v in block.values())+model.state.nbytes+model.pred.nbytes)))
        return
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    audit=json.loads(Path('audit_results/final_20261114_1940/results.json').read_text())
    strongest=max(audit['pairs'][1:],key=lambda p:p['scored_wp']);names=strongest['pair'].split('+')
    paths=[audit['ckpts'][name]['path'] for name in names]
    from audit_mixed_stream import export
    cks=export(paths,out/'export')
    import torch
    import train_v1 as base
    import train_v4m
    from export_v1 import FEATURE_COLUMNS
    torch.set_num_threads(1)
    data=base.load_split('valid');nseq=2;length=20000
    raw=np.empty((nseq,length,112),np.float32)
    raw[:,:,[FEATURE_COLUMNS.index(c) for c in cks[0]['price_cols']]]=data['prices'][:nseq]
    raw[:,:,[FEATURE_COLUMNS.index(c) for c in cks[0]['other_cols']]]=data['features'][:nseq]
    references=[]
    with torch.no_grad():
        for name,ck in zip(names,cks):
            model=(train_v4m.LSTMModel if name=='v13' else base.GRUModel)(hidden=ck['args']['hidden']).eval();model.load_state_dict(ck['model'])
            x=base.build_inputs(torch.from_numpy(np.array(data['prices'][:nseq])),torch.from_numpy(np.array(data['features'][:nseq])),ck['stats'])
            state=None;parts=[]
            for s in range(0,length,1000):
                p,state=model(x[:,s:s+1000],state);parts.append(p)
            references.append(torch.cat(parts,1).numpy())
    ref=np.clip(np.mean(references,axis=0,dtype=np.float32),-2,2)
    deployed=Path('/home/jovyan/work/submission_ens_v18cv19a_int8')
    directories=dict(candidate_fp32=out/'export/fp32',candidate_int8=out/'export/int8',incumbent_int8=deployed)
    models={k:load(v) for k,v in directories.items()};repeats=[];sample_outputs=None
    for repeat in range(3):
        samples={k:[] for k in models};outputs={k:np.empty((nseq,length-99,2),np.float32) for k in models}
        for seq in range(nseq):
            for step in range(length):
                point=SimpleNamespace(seq_ix=repeat*nseq+seq,step_in_seq=step,need_prediction=step>=99,state=raw[seq,step])
                order=list(models)
                if (step+repeat)%2:order.reverse()
                for name in order:
                    start=time.perf_counter_ns();pred=models[name].predict(point);elapsed=(time.perf_counter_ns()-start)/1000
                    # Include every callback, including cold sequence reset/warm-up; warm sessions across repeats.
                    samples[name].append(elapsed)
                    if step>=99:outputs[name][seq,step-99]=pred
                    else:assert pred is None
        record={name:dict(mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95))) for name,v in samples.items()}
        repeats.append(record);print('repeat',repeat+1,json.dumps(record),flush=True)
        if sample_outputs is None:sample_outputs=outputs
        else:
            for name in outputs:assert np.array_equal(sample_outputs[name],outputs[name]),name
    fperr=np.abs(sample_outputs['candidate_fp32']-ref[:,99:]);qerr=np.abs(sample_outputs['candidate_int8']-sample_outputs['candidate_fp32'])
    assert fperr.max()<2e-5,float(fperr.max())
    # Explicit reset against fresh session after long recurrent trajectories.
    fresh=load(out/'export/fp32');point=SimpleNamespace(seq_ix=100,step_in_seq=0,need_prediction=True,state=raw[0,0])
    assert np.array_equal(models['candidate_fp32'].predict(point),fresh.predict(point))
    memory={}
    for name,directory in directories.items():
        completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--memory-dir',str(directory)],capture_output=True,text=True,check=True)
        memory[name]=json.loads(completed.stdout)
    means={k:float(np.mean([r[k]['mean'] for r in repeats])) for k in models}
    runtime={k:dict(callback_minutes=us*39.4e6/60e6,margin_25pct_plus_5min=us*39.4e6/60e6*1.25+5,
                    historical_1p89_speedup_minutes=us*39.4e6/60e6/1.89,
                    historical_speedup_with_margin=us*39.4e6/60e6/1.89*1.25+5) for k,us in means.items()}
    result=dict(strongest=strongest['pair'],checkpoints=paths,rows_per_repeat=nseq*length,repeats=repeats,mean_callback_us=means,
                measurement='complete predict(), raw input checks/preprocessing + all GRU h and LSTM h/c state + one ORT call + output copy; core31/thread1; alternating model order; 3 repeats of 2 full 20000-row sequences; no warm-up callbacks excluded',
                fp32_vs_pytorch=dict(max=float(fperr.max()),mean=float(fperr.mean())),int8_vs_fp32=dict(max=float(qerr.max()),mean=float(qerr.mean())),
                reset_exact=True,deterministic_repeats_exact=True,memory=memory,runtime=runtime,
                graph_bytes={k:(v/'model.onnx').stat().st_size for k,v in directories.items()},
                export_support='Original export_ens.py only two GRUs; composed prefixed explicit cells from existing export_v4.build(cell=matmul,fold=False), preserving each own stats; generic solution_v4.py holds all eight hidden/cell states; raw heads averaged then clipped; audit-only graphs, no zip.',
                limitation='FP32 table is checkpoint validation, not full-validation int8 score. Fidelity/timing cover two whole sequences. Historical 1.89 speedup is an assumption, not a current platform guarantee; margin includes 25% plus five minutes for I/O/framework overhead.')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
