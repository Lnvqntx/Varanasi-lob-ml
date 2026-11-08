"""Bounded correctness, GPU compute benchmark, disposable ONNX CPU prototype.
Run in project venv; --out must be a new directory. No full training.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from v33_model import averages, register

tm = register()
torch.set_num_threads(1)


def correct():
    tm.set_seed(0)
    x = torch.randn(2, 211, 168)
    result = {}
    for arch in ('gru', 'v33proj', 'v33context'):
        torch.manual_seed(0)
        model = tm.ARCHS[arch](hidden=128).eval()
        with torch.no_grad():
            whole, _ = model(x)
            errors = []
            for chunk in (1, 17, 64, 100):
                state, parts = None, []
                for s in range(0, x.shape[1], chunk):
                    p, state = model(x[:, s:s+chunk], state)
                    parts.append(p)
                errors.append((torch.cat(parts, 1) - whole).abs().max().item())
            changed = x.clone()
            changed[:, 120:] += 100
            past = (model(changed)[0][:, :120] - whole[:, :120]).abs().max().item()
            assert past == 0 and max(errors) < 2e-6
            # Warm-up processed as part of the sequence; loss excludes it only.
            _, state = model(x[:, :99])
            after, _ = model(x[:, 99:], state)
            warm = (after - whole[:, 99:]).abs().max().item()
            reset = (model(x)[0] - whole).abs().max().item()
            assert warm < 2e-6 and reset == 0
        result[arch] = dict(chunk_max=max(errors), future_past=past, warmup_max=warm, reset=reset,
                            parameters=sum(p.numel() for p in model.parameters()))
    z = torch.randn(2, 1000, 32, requires_grad=True)
    av, _ = averages(z)
    ref = z[:, 0, None].expand(-1, 3, -1)
    alpha = z.new_tensor([1-2**(-1/h) for h in (8,64,512)])[None,:,None]
    err = 0
    for t in range(z.shape[1]):
        ref = ref + alpha * (z[:, t, None] - ref)
        err = max(err, (av[:, t]-ref).abs().max().item())
    assert err < 2e-5
    # Loss depends ONLY on averages, eliminating the direct projection path.
    model = tm.ARCHS['v33context'](hidden=128)
    av, _ = averages(torch.tanh(model.projection(x)))
    av[:, -1].square().sum().backward()
    grad = model.projection.weight.grad
    assert torch.isfinite(grad).all() and grad.norm() > 0
    result['ema'] = dict(serial_max=err, projection_average_only_grad=grad.norm().item())
    # Independent construction of companions leaves control init unchanged.
    torch.manual_seed(0)
    a = tm.ARCHS['gru'](hidden=128)
    for arch in ('v33proj','v33context','gru'):
        torch.manual_seed(0)
        b = tm.ARCHS[arch](hidden=128)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    result['rng_control_equal'] = True
    torch.manual_seed(0); proj=tm.ARCHS['v33proj'](hidden=128)
    torch.manual_seed(0); context=tm.ARCHS['v33context'](hidden=128)
    assert torch.equal(proj.projection.weight,context.projection.weight)
    # All half-lives propagate gradients to earlier rows across scan blocks.
    z=torch.randn(1,130,32,requires_grad=True)
    av,state=averages(z)
    assert torch.equal(av[:,0],z[:,0,None].expand(-1,3,-1))
    for scale in range(3):
        g=torch.autograd.grad(av[:,-1,scale].sum(),z,retain_graph=True)[0]
        expected=(1-2**(-1/(8,64,512)[scale]))*2**(-29/(8,64,512)[scale])
        assert torch.isfinite(g).all() and abs(g[0,100,0].item()-expected)<1e-7
    detached=tm.detach_state((torch.zeros(2,1,128,requires_grad=True),state))
    assert all(t.grad_fn is None and not t.requires_grad for t in detached)
    result['first_row_exact_and_all_scale_gradients_and_detach']=True
    with torch.no_grad():
        long_z=torch.randn(1,20000,32)
        long_av,_=averages(long_z)
        assert torch.isfinite(long_av).all()
        serial=long_z[:,0,None].double().expand(-1,3,-1).clone()
        alpha=serial.new_tensor([1-2**(-1/h) for h in (8,64,512)])[None,:,None]
        err=0.
        for t in range(20000):
            serial=serial+alpha*(long_z[:,t,None].double()-serial)
            err=max(err,(long_av[:,t].double()-serial).abs().max().item())
        assert err<2e-5
        result['ema_20000_rows_serial_float64_max']=err

    return result


def gpu():
    tm.set_seed(0)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(2.5*2**30/torch.cuda.get_device_properties(0).total_memory)
    result = {}
    x = torch.randn(16,1000,168,device='cuda')
    y = torch.randn(16,1000,2,device='cuda')
    vm = (torch.arange(1000,device='cuda') >= 99)[None,:,None].float()
    m = torch.rand(16,1000,1,device='cuda') + .5
    for arch in ('gru','v33proj','v33context'):
        torch.manual_seed(0)
        model = tm.ARCHS[arch](hidden=128).cuda()
        opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
        state = None
        torch.cuda.reset_peak_memory_stats()
        times=[]
        for i in range(13):
            torch.cuda.synchronize(); start=time.perf_counter()
            pred,state=model(x,state)
            loss=tm.loss_fn(pred,y,vm,m)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step(); state=tm.detach_state(state)
            torch.cuda.synchronize()
            if i>=3: times.append(time.perf_counter()-start)
        result[arch]=dict(steps_s=1/np.mean(times),rows_s=16000/np.mean(times),peak_MiB=torch.cuda.max_memory_allocated()/2**20)
        del model,opt,state,pred,loss
        torch.cuda.empty_cache()
    models=[tm.ARCHS[a](hidden=128).cuda() for a in ('gru','v33proj','v33context')]
    opts=[torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05) for model in models]
    states=[None]*3
    torch.cuda.reset_peak_memory_stats(); times=[]
    for i in range(8):
        torch.cuda.synchronize(); start=time.perf_counter()
        for k,(model,opt) in enumerate(zip(models,opts)):
            pred,states[k]=model(x,states[k]); loss=tm.loss_fn(pred,y,vm,m)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
            states[k]=tm.detach_state(states[k])
        torch.cuda.synchronize()
        if i>=2: times.append(time.perf_counter()-start)
    result['shared_ABC']=dict(steps_s=1/np.mean(times),rows_s=16000/np.mean(times),peak_MiB=torch.cuda.max_memory_allocated()/2**20)
    del models,opts,states,model,opt,pred,loss
    torch.cuda.empty_cache()
    # Control vs old trainer: identical synthetic inputs, masks, ten updates.
    from train_v2m import GRUModel, loss_fn
    tm.set_seed(0); a=tm.ARCHS['gru'](hidden=128).cuda()
    tm.set_seed(0); b=GRUModel(hidden=128).cuda()
    oa=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.05)
    ob=torch.optim.AdamW(b.parameters(),lr=.001,weight_decay=.05)
    ha=hb=None
    for i in range(10):
        pa,ha=a(x,ha); pb,hb=b(x,hb)
        la=tm.loss_fn(pa,y,vm,m); lb=loss_fn(pb,y,vm,m)
        assert torch.equal(la,lb)
        for model,opt,l in ((a,oa,la),(b,ob,lb)):
            opt.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        ha=ha.detach(); hb=hb.detach()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    result['old_control_10_updates_bitwise']=True
    return result


class Step(torch.nn.Module):
    def __init__(self,model,ck):
        super().__init__(); self.model=model
        from export_v1 import FEATURE_COLUMNS
        self.register_buffer('price_idx',torch.tensor([FEATURE_COLUMNS.index(c) for c in ck['price_cols']]))
        self.register_buffer('other_idx',torch.tensor([FEATURE_COLUMNS.index(c) for c in ck['other_cols']]))
        self.register_buffer('flag_idx',torch.tensor([ck['other_cols'].index(c) for c in ck['flag_cols']]))
        self.register_buffer('mean',ck['stats']['mean'])
        self.register_buffer('std',ck['stats']['std'])
    def forward(self,x,h,ema,first,prev_prices):
        prices=x[:,:,self.price_idx]
        other=x[:,:,self.other_idx].half().float()
        diff=torch.where(first,torch.zeros_like(prices),prices-prev_prices)
        flags=(other[:,:,self.flag_idx].abs()==tm.FILL).float()
        x=torch.cat([(torch.cat([prices,diff,other],2)-self.mean)/self.std,flags],2)
        model=self.model
        if hasattr(model,'projection'):
            z=torch.tanh(model.projection(x))
            if model.context:
                alpha=z.new_tensor([1-2**(-1/k) for k in (8,64,512)])[None,:,None]
                ema=torch.where(first, z[:,0,None].expand(-1,3,-1), ema+alpha*(z[:,0,None]-ema))
                x=torch.cat([z,(z[:,0,None]-ema).flatten(1)[:,None]],2)
            else: x=z
        states=[]
        for layer in range(2):
            gi=torch.nn.functional.linear(x,getattr(model.gru,f'weight_ih_l{layer}'),getattr(model.gru,f'bias_ih_l{layer}'))
            gh=torch.nn.functional.linear(h[layer:layer+1],getattr(model.gru,f'weight_hh_l{layer}'),getattr(model.gru,f'bias_hh_l{layer}'))
            ir,iz,inn=gi.chunk(3,-1); hr,hz,hn=gh.chunk(3,-1)
            r=torch.sigmoid(ir+hr); update=torch.sigmoid(iz+hz)
            candidate=torch.tanh(inn+r*hn)
            x=(1-update)*candidate+update*h[layer:layer+1]
            states.append(x)
        out=x; h=torch.cat(states,0)
        # Keep every explicit state input in every graph for identical call overhead.
        return model.head(out),h,ema+first.to(x.dtype)*0,prices


def onnx_bench(out):
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    os.sched_setaffinity(0,{31})
    result={}
    ck=torch.load('/home/jovyan/work/checkpoints/train_v18c_wd05_nodrop_final.pt',map_location='cpu',weights_only=False)
    data=tm.load_split('valid')
    from export_v1 import FEATURE_COLUMNS
    rows=np.empty((1200,1,1,112),np.float32)
    rows[:,0,0,[FEATURE_COLUMNS.index(c) for c in ck['price_cols']]]=data['prices'][0,:1200]
    rows[:,0,0,[FEATURE_COLUMNS.index(c) for c in ck['other_cols']]]=data['features'][0,:1200]
    reference_x=tm.build_inputs(torch.from_numpy(np.array(data['prices'][0:1,:1200])),torch.from_numpy(np.array(data['features'][0:1,:1200])),ck['stats'])
    for arch in ('gru','v33proj','v33context'):
        torch.manual_seed(0); model=tm.ARCHS[arch](hidden=128).eval()
        if arch=='gru': model.load_state_dict(ck['model'])
        wrapper=Step(model,ck).eval()
        args=(torch.from_numpy(rows[0]),torch.zeros(2,1,128),torch.zeros(1,3,32),torch.ones(1,1,1,dtype=torch.bool),torch.zeros(1,1,44))
        path=out/f'{arch}.onnx'
        torch.onnx.export(wrapper,args,str(path),input_names=['x','h','ema','first','prev_prices'],output_names=['pred','h_out','ema_out','prices'],opset_version=17,dynamo=False)
        quant=out/f'{arch}_int8.onnx'
        quantize_dynamic(str(path),str(quant),weight_type=QuantType.QInt8,op_types_to_quantize=['MatMul','Gemm'])
        outputs={}
        for label,p in (('fp32',path),('int8',quant)):
            so=ort.SessionOptions(); so.intra_op_num_threads=so.inter_op_num_threads=1
            so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            session=ort.InferenceSession(str(p),so,providers=['CPUExecutionProvider'])
            feed={'h':np.zeros((2,1,128),np.float32),'ema':np.zeros((1,3,32),np.float32),'prev_prices':np.zeros((1,1,44),np.float32)}
            preds=[]; times=[]
            for i,row in enumerate(rows):
                feed.update(x=row,first=np.full((1,1,1),i==0,dtype=bool))
                inputs={v.name:feed[v.name] for v in session.get_inputs()}
                start=time.perf_counter(); pred,h,ema,prices=session.run(None,inputs)
                if i>=200: times.append(time.perf_counter()-start)
                feed.update(h=h,ema=ema,prev_prices=prices); preds.append(pred)
            outputs[label]=np.concatenate(preds,1)
            result[f'{arch}_{label}']=dict(us_row=np.mean(times)*1e6,bytes=p.stat().st_size)
        with torch.no_grad(): reference=model(reference_x)[0].numpy()
        fp_err=float(np.max(np.abs(reference-outputs['fp32'])))
        assert fp_err<2e-5
        result[arch+'_agreement']=dict(torch_fp32_max=fp_err,int8_max=float(np.max(np.abs(outputs['int8']-outputs['fp32']))),int8_mean=float(np.mean(np.abs(outputs['int8']-outputs['fp32']))))
    import importlib.util
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('current_solution','/home/jovyan/work/submission_ens_v18cv19a_int8/solution.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    current=module.PredictionModel(); times=[]
    for i,row in enumerate(rows):
        dp=SimpleNamespace(seq_ix=0,step_in_seq=i,state=row[0,0],need_prediction=i>=99)
        start=time.perf_counter(); current.predict(dp)
        if i>=200: times.append(time.perf_counter()-start)
    result['current_deployed_pair_int8_predict_us']=np.mean(times)*1e6
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--mode',choices=['correct','gpu','onnx'],required=True)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    result=correct() if a.mode=='correct' else gpu() if a.mode=='gpu' else onnx_bench(out)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
