"""Post-training export gate for the one bounded residual attempt; no training.
B folds the linear correction into v18c's head; C appends its small pointwise head.
Existing explicit two-GRU graph/preprocessing/state layout retained. Never packages
unless all configured gates pass; this checker itself does not create a zip.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,copy,importlib.util,json,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import onnx
from onnx import helper as h,numpy_helper as nh
import export_ens


def load(directory):
    spec=importlib.util.spec_from_file_location('bounded_export_'+str(abs(hash(str(directory)))),str(Path(directory)/'solution.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.PredictionModel()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    run=Path(a.run).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,{31});torch.set_num_threads(1)
    scores=json.loads((run/'results.json').read_text());base_results={r['model']:r for r in scores['results']}
    originals=[torch.load(path,map_location='cpu',weights_only=False) for path in ['/home/jovyan/work/checkpoints/train_v18c_wd05_nodrop_final.pt','/home/jovyan/work/checkpoints/train_v19a_wd10_ds1_final.pt']]
    kind=max(['B','C'],key=lambda n:base_results[n+'+v19a']['scored_wp'])
    residual=torch.load(run/f'{kind}_final.pt',map_location='cpu',weights_only=False)
    cks=copy.deepcopy(originals)
    if kind=='B':
        cks[0]['model']['head.weight']+=residual['head']['weight'];cks[0]['model']['head.bias']+=residual['head']['bias']
    model,stats=export_ens.build(cks)
    if kind=='C':
        sd=residual['head'];inits=model.graph.initializer
        for name,v in [('res_w0',sd['0.weight'].T),('res_b0',sd['0.bias']),('res_w2',sd['2.weight'].T),('res_b2',sd['2.bias'])]:inits.append(nh.from_array(v.numpy().astype(np.float32),name))
        extra=[h.make_node('MatMul',['hA1_out','res_w0'],['res_mm0'],name='res_mm0'),
               h.make_node('Add',['res_mm0','res_b0'],['res_pre']),h.make_node('Relu',['res_pre'],['res_act']),
               h.make_node('MatMul',['res_act','res_w2'],['res_mm2'],name='res_mm2'),
               h.make_node('Add',['res_mm2','res_b2'],['correction']),h.make_node('Add',['predA','correction'],['predA_corrected'])]
        nodes=[]
        for node in model.graph.node:
            if node.op_type=='Add' and list(node.input)==['predA','predB']:
                nodes.extend(extra);node.input[0]='predA_corrected'
            nodes.append(node)
        del model.graph.node[:];model.graph.node.extend(nodes)
    onnx.checker.check_model(model,full_check=True)
    from onnxruntime.quantization import quantize_dynamic,QuantType
    import shutil
    for mode in ('fp32','int8'):
        directory=out/mode;directory.mkdir(exist_ok=False)
        if mode=='fp32':onnx.save(model,directory/'model.onnx')
        else:quantize_dynamic(str(out/'fp32/model.onnx'),str(directory/'model.onnx'),weight_type=QuantType.QInt8,op_types_to_quantize=['MatMul'],nodes_to_exclude=['res_mm0','res_mm2'])
        np.savez(directory/'stats.npz',**stats);shutil.copyfile('solution_ens.py',directory/'solution.py')
    import train_v2m as base
    from export_v1 import FEATURE_COLUMNS
    data=base.load_split('valid');length=3000;n=3
    raw=np.empty((n,length,112),np.float32)
    raw[:,:,[FEATURE_COLUMNS.index(c) for c in originals[0]['price_cols']]]=data['prices'][:n,:length]
    raw[:,:,[FEATURE_COLUMNS.index(c) for c in originals[0]['other_cols']]]=data['features'][:n,:length]
    peer=base.GRUModel(hidden=128).eval();peer.load_state_dict(originals[1]['model'])
    trunk=base.GRUModel(hidden=128).eval();trunk.load_state_dict(originals[0]['model'])
    if kind=='B':head=torch.nn.Linear(128,2)
    else:head=torch.nn.Sequential(torch.nn.Linear(128,32),torch.nn.ReLU(),torch.nn.Linear(32,2))
    head.load_state_dict(residual['head']);head.eval()
    with torch.no_grad():
        x=base.build_inputs(torch.from_numpy(np.array(data['prices'][:n,:length])),torch.from_numpy(np.array(data['features'][:n,:length])),originals[0]['stats'])
        xp=base.build_inputs(torch.from_numpy(np.array(data['prices'][:n,:length])),torch.from_numpy(np.array(data['features'][:n,:length])),originals[1]['stats'])
        z,_=trunk.gru(x);pa=trunk.head(z)+head(z);pp=peer(xp)[0];expected=((pa+pp)/2).clamp(-2,2).numpy()
    models={'candidate_fp32':load(out/'fp32'),'candidate_int8':load(out/'int8'),'incumbent_int8':load('/home/jovyan/work/submission_ens_v18cv19a_int8')}
    names=list(models);means=[];stored=None
    for repeat in range(3):
        times={k:[] for k in names};predictions={k:np.empty((n,length-99,2),np.float32) for k in names}
        for seq in range(n):
            for step in range(length):
                point=SimpleNamespace(seq_ix=repeat*n+seq,step_in_seq=step,need_prediction=step>=99,state=raw[seq,step])
                for name in (names if (step+repeat)%2 else names[::-1]):
                    start=time.perf_counter_ns();p=models[name].predict(point);elapsed=(time.perf_counter_ns()-start)/1000
                    if step>=200:times[name].append(elapsed)
                    if step>=99:predictions[name][seq,step-99]=p
                    else:assert p is None
        if stored is not None:
            for name in names:assert np.array_equal(stored[name],predictions[name])
        else:stored=predictions
        record={k:float(np.mean(v)) for k,v in times.items()};means.append(record);print('REPEAT',repeat+1,record,flush=True)
    fp_error=float(np.max(np.abs(stored['candidate_fp32']-expected[:,99:])));assert fp_error<2e-5
    q_error=float(np.max(np.abs(stored['candidate_int8']-stored['candidate_fp32'])))
    point=SimpleNamespace(seq_ix=100,step_in_seq=0,need_prediction=True,state=raw[0,0])
    fresh=load(out/'fp32');assert np.array_equal(models['candidate_fp32'].predict(point),fresh.predict(point))
    mu={k:float(np.mean([r[k] for r in means])) for k in names}
    result=dict(kind=kind,scored_gain_fp32=base_results[kind+'+v19a']['delta_vs_incumbent'],mean_callback_us=mu,repeats=means,
                fp32_vs_pytorch_max=fp_error,int8_vs_fp32_max=q_error,determinism_exact=True,reset_exact=True,
                local_test_minutes={k:v*39.4e6/60e6 for k,v in mu.items()},
                local_test_with_25pct_plus5_minutes={k:v*39.4e6/60e6*1.25+5 for k,v in mu.items()},
                full_validation_int8_verified=False,package_ready=False,zip_path=None,
                decision='incumbent',reason='No candidate package: short export checks do not establish full-validation quantized improvement or credible sub-60-minute platform margin. Local complete-callback extrapolation is reported, with no unverified platform speedup guarantee.')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
