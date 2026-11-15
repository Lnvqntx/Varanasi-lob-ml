"""Isolated streaming memory via /proc VmRSS/VmHWM (not inherited rusage peaks)."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,importlib.util,json,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace


def memory():
    lines=Path('/proc/self/status').read_text().splitlines()
    return {name:int(next(line.split()[1] for line in lines if line.startswith(name+':')))*1024 for name in ['VmRSS','VmHWM']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--worker-dir');a=ap.parse_args();os.sched_setaffinity(0,{31})
    if a.worker_dir:
        import numpy as np
        before=memory();start=time.perf_counter()
        spec=importlib.util.spec_from_file_location('isolated_solution',str(Path(a.worker_dir)/'solution.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);model=module.PredictionModel()
        startup=time.perf_counter()-start;state=np.zeros(112,np.float32)
        for step in range(2000):model.predict(SimpleNamespace(seq_ix=0,step_in_seq=step,need_prediction=step>=99,state=state))
        print(json.dumps(dict(before=before,after_warmup=memory(),startup_seconds=startup,rows=2000,
                             buffer_bytes=sum(v.nbytes for b in model.bufs for v in b.values())+model.state.nbytes+model.pred.nbytes)))
        return
    output=Path(a.out);assert not output.exists()
    root=Path('audit_results/final_stream_20261114_1945').resolve()
    directories=dict(candidate_fp32=root/'export/fp32',candidate_int8=root/'export/int8',incumbent_int8=Path('/home/jovyan/work/submission_ens_v18cv19a_int8'))
    results={name:json.loads(subprocess.check_output([sys.executable,str(Path(__file__).resolve()),'--worker-dir',str(path)],text=True)) for name,path in directories.items()}
    result=dict(method='Fresh no-PyTorch workers; /proc/self/status VmHWM measures current address-space peak. Earlier resource.ru_maxrss peaks in benchmark workers were inherited from the large parent and are not used.',models=results)
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
