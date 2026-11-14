"""Bounded causal/state/gradient checks and GPU control equality, fresh results only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import torch
from v34_model import CausalTCN, DILATIONS, STATE_FLOATS, RECEPTIVE_FIELD, register


def correctness():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = CausalTCN().eval()
    x = torch.randn(2, 2103, 168)
    with torch.no_grad():
        whole, final_state = model(x)
        z = x.clone(); z[:, 1201:] += 10
        assert torch.equal(whole[:, :1201], model(z)[0][:, :1201])
        errors = {}
        for chunk in [1, 17, 511, 1000]:
            state, outputs = None, []
            for s in range(0, len(x[0]), chunk):
                p, state = model(x[:, s:s+chunk], state)
                outputs.append(p)
            err = (torch.cat(outputs,1)-whole).abs().max().item()
            state_err = max((a-b).abs().max().item() for a,b in zip(state,final_state))
            assert err < 1e-5 and state_err < 1e-5
            errors[str(chunk)] = dict(output_max=err,state_max=state_err)
        outputs, state = [], None
        for row in x.unbind(1):
            p,state = model.step(row,state); outputs.append(p)
        row_error = (torch.stack(outputs,1)-whole).abs().max().item()
        assert row_error < 1e-5
        # Mixed interfaces, including streaming directly after a 1000-row training chunk.
        _,state = model(x[:,:1000])
        p,_ = model.step(x[:,1000],state)
        boundary_error = (p-whole[:,1000]).abs().max().item()
        assert boundary_error < 1e-5
        _,warm = model(x[:,:99]); after,_ = model(x[:,99:],warm)
        warm_error = (after-whole[:,99:]).abs().max().item()
        assert warm_error < 1e-5
        assert torch.equal(model(x)[0],whole)
        other = torch.randn(2,99,168); model(other)
        assert torch.equal(model(x,None)[0],whole)
        z = x.clone(); z[:,:1000] += 20
        rf_error = (model(z)[0][:,1510:] - whole[:,1510:]).abs().max().item()
        assert rf_error == 0, rf_error
    # Gradient crosses chunk boundary when NOT detached; no look-ahead.
    a = torch.randn(1,1000,168,requires_grad=True)
    b = torch.randn(1,7,168,requires_grad=True)
    _,history = model(a)
    output,_ = model(b,history)
    loss = output.square().mean(); loss.backward()
    gradients = {name:p.grad.norm().item() for name,p in model.named_parameters()}
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert gradients['projection.weight']>0 and a.grad.abs().sum()>0
    tm=register(); detached=tm.detach_state(history)
    assert all(s.grad_fn is None and not s.requires_grad for s in detached)
    with torch.no_grad():
        assert torch.equal(model(b,history)[0],model(b,detached)[0])
    torch.manual_seed(0); before=torch.random.get_rng_state(); CausalTCN()
    assert torch.equal(before,torch.random.get_rng_state())
    return dict(parameters=sum(p.numel() for p in model.parameters()),state_floats=STATE_FLOATS,
                state_bytes_fp32=STATE_FLOATS*4,receptive_field=RECEPTIVE_FIELD,
                future_past_max=0.,chunk_errors=errors,single_row_max=row_error,chunk_to_row_max=boundary_error,
                warmup_max=warm_error,reset_exact=True,deterministic_exact=True,receptive_field_cutoff_max=rf_error,
                gradients_finite=True,projection_grad_norm=gradients['projection.weight'],past_chunk_input_grad_norm=a.grad.norm().item(),
                all_block_gradient_norms={k:v for k,v in gradients.items() if 'depthwise.weight' in k},state_detach_verified=True,initialization_rng_isolated=True)


def control():
    torch.set_num_threads(1)
    tm=register(); tm.set_seed(0)
    torch.cuda.set_per_process_memory_fraction(2.5*2**30/torch.cuda.get_device_properties(0).total_memory)
    from train_v2m import GRUModel,loss_fn
    x=torch.randn(16,1000,168,device='cuda');y=torch.randn(16,1000,2,device='cuda')
    vm=(torch.arange(1000,device='cuda')>=99)[None,:,None].float(); m=torch.rand(16,1000,1,device='cuda')+.5
    tm.set_seed(0);a=tm.ARCHS['gru'](hidden=128).cuda()
    tm.set_seed(0);CausalTCN();b=GRUModel(hidden=128).cuda()
    opts=[torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05) for model in (a,b)]
    sched=[torch.optim.lr_scheduler.LambdaLR(o,lambda s: .5*(1+__import__('math').cos(__import__('math').pi*s/64500))) for o in opts]
    states=[None,None]
    for i in range(10):
        losses=[]
        for k,model in enumerate((a,b)):
            p,states[k]=model(x,states[k]);l=(tm.loss_fn if k==0 else loss_fn)(p,y,vm,m)
            opts[k].zero_grad(set_to_none=True);l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opts[k].step();sched[k].step();states[k]=states[k].detach();losses.append(l.detach())
        assert torch.equal(*losses)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    return dict(original_trainer_10_updates_bitwise_equal=True,adding_tcn_does_not_change_control=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--gpu-control',action='store_true');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    result=control() if a.gpu_control else correctness()
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
