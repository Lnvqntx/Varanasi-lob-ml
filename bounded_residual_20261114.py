"""One deadline-bounded frozen-v18c residual attempt. No validation-target training.
A unchanged; B linear correction; C 128->32 ReLU->2 correction. Shared frozen GRU.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
import argparse,datetime,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import train_v2m as base
from spread_v2 import STAT_NAMES,seq_stats,wp_from_stats

START=datetime.datetime(2026,11,14,19,52,18,tzinfo=datetime.timezone.utc).timestamp()
TRAIN_CUTOFF=START+45*60
DEADLINE=START+60*60
BACKBONE=Path('/home/jovyan/work/checkpoints/train_v18c_wd05_nodrop_final.pt')
TEACHER=Path('/home/jovyan/work/checkpoints/train_v19a_wd10_ds1_final.pt')


def log(text):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),text,flush=True)

def heads(device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(35000000)
        b=nn.Linear(128,2)
        c=nn.Sequential(nn.Linear(128,32),nn.ReLU(),nn.Linear(32,2))
        nn.init.zeros_(b.weight);nn.init.zeros_(b.bias)
        nn.init.zeros_(c[2].weight);nn.init.zeros_(c[2].bias)
    return [b.to(device),c.to(device)]


def load_backbone(device):
    ck=torch.load(BACKBONE,map_location='cpu',weights_only=False)
    model=base.GRUModel(hidden=128).to(device);model.load_state_dict(ck['model']);model.eval();model.requires_grad_(False)
    return model,ck,{k:v.to(device) for k,v in ck['stats'].items()}


def checks(model):
    # Check the SAME frozen network after pointwise corrections, with fresh heads.
    device=next(model.parameters()).device
    b,c=heads(device);x=torch.randn(2,211,168,device=device)
    with torch.no_grad():
        hidden,_=model.gru(x);a=model.head(hidden)
        assert torch.equal(a,a+b(hidden)) and torch.equal(a,a+c(hidden))
    y=torch.randn_like(a);valid=torch.ones_like(a[:,:,:1]);opt=torch.optim.AdamW(c.parameters(),lr=.001,weight_decay=.05)
    norms={}
    for iteration in range(2):
        loss=base.loss_fn(a+c(hidden),y,valid)
        opt.zero_grad(set_to_none=True);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in c.parameters())
        norms[str(iteration)]=[float(p.grad.norm()) for p in c.parameters()]
        if iteration==0:assert c[0].weight.grad.abs().sum()==0 and c[2].weight.grad.abs().sum()>0
        else:assert c[0].weight.grad.abs().sum()>0
        opt.step()
    lb=base.loss_fn(a+b(hidden),y,valid);lb.backward()
    assert torch.isfinite(b.weight.grad).all() and b.weight.grad.norm()>0
    with torch.no_grad():
        def predict(xs,state=None):
            z,state=model.gru(xs,state);return model.head(z)+c(z),state
        whole,_=predict(x)
        z=x.clone();z[:,120:]+=10
        assert torch.equal(predict(z)[0][:,:120],whole[:,:120])
        errors=[]
        for chunk in (1,17,99):
            state=None;parts=[]
            for s in range(0,211,chunk):p,state=predict(x[:,s:s+chunk],state);parts.append(p)
            errors.append(float((torch.cat(parts,1)-whole).abs().max()))
        assert max(errors)<1e-5
        assert torch.equal(predict(x,None)[0],whole)
        assert torch.equal(predict(x,None)[0],predict(x,None)[0])
    assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
    return dict(initial_equivalence_bitwise=True,C_gradient_norms_first_two_updates=norms,B_gradient_norm=float(b.weight.grad.norm()),
                note='C first-layer initial gradient is exactly zero because its output layer is zero; becomes nonzero after first update.',
                future_past_exact=True,chunk_single_row_max_error=max(errors),resets_exact=True,deterministic_exact=True,backbone_gradients_absent=True)


def update(head_modules,optimizers,schedulers,hidden,a,y,valid,mask):
    losses=[]
    for i,(head,opt) in enumerate(zip(head_modules,optimizers)):
        pred=a+head(hidden)
        loss=base.loss_fn(pred,y,valid,mask);opt.zero_grad(set_to_none=True);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        nn.utils.clip_grad_norm_(head.parameters(),1.0);opt.step()
        if schedulers is not None:schedulers[i].step()
        losses.append(float(loss))
    return losses


def train_stream(model,head_modules,data,train_idx,rng,stats,mean_p,budget,stop_at,pilot=False):
    device=next(model.parameters()).device
    opts=[torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.05) for h in head_modules]
    sched=[torch.optim.lr_scheduler.LambdaLR(opt,lambda step:.5*(1+math.cos(math.pi*min(step,budget)/budget))) for opt in opts]
    step=0;started=time.time();compute=0.;wait=0.;last=time.time();loss_sum=np.zeros(2);curve=[]
    while step<budget and time.time()<stop_at:
        order=rng.permutation(train_idx)
        batches=[order[i:i+16] for i in range(0,len(order),16)]
        loader=iter(base.Prefetcher(data,batches,depth=1))
        while step<budget and time.time()<stop_at:
            t=time.time();batch=next(loader,None);wait+=time.time()-t
            if batch is None:break
            x=base.build_inputs(torch.from_numpy(batch['prices']).to(device),torch.from_numpy(batch['features']).to(device),stats)
            y=torch.from_numpy(batch['targets']).to(device)
            p=torch.from_numpy(batch['maskp']).to(device).float()
            mask=(.5+.5*(p/mean_p).clamp(max=20))[:,:,None]
            state=None
            for s in range(0,20000,1000):
                if step>=budget or time.time()>=stop_at:break
                t=time.time()
                with torch.no_grad():hidden,state=model.gru(x[:,s:s+1000],state);a=model.head(hidden)
                valid=(torch.arange(s,s+1000,device=device)>=99)[None,:,None].float()
                losses=update(head_modules,opts,sched,hidden,a,y[:,s:s+1000],valid,mask[:,s:s+1000])
                state=state.detach();step+=1;loss_sum+=losses;compute+=time.time()-t
                if step%100==0 or (pilot and step%20==0):
                    dt=time.time()-last;denom=20 if pilot else 100
                    record=dict(step=step,loss_B=losses[0],loss_C=losses[1],interval_updates_s=denom/dt,elapsed_s=time.time()-started)
                    curve.append(record);log(('PILOT ' if pilot else 'TRAIN ')+json.dumps(record));last=time.time()
    return dict(updates=step,seconds=time.time()-started,compute_seconds=compute,loader_wait_seconds=wait,
                wall_updates_s=step/(time.time()-started),curve=curve,deadline_stop=step<budget,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    assert time.time()<TRAIN_CUTOFF-300,'Insufficient safe training time; retain incumbent'
    base.set_seed(0);torch.set_num_threads(1);device=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(2.5*2**30/torch.cuda.get_device_properties(0).total_memory)
    model,ck,stats=load_backbone(device)
    checked=checks(model);(out/'checks.json').write_text(json.dumps(checked,indent=2)+'\n');log('CHECKS '+json.dumps(checked))
    train=base.load_split('train')  # Validation targets are never loaded during training.
    rng=np.random.default_rng(0);perm=rng.permutation(train['prices'].shape[0]);holdout=np.sort(perm[:300]);train_idx=perm[300:]
    rng.choice(1873,1873,replace=False)  # Exactly the established data-stream RNG advancement.
    maskp=np.load('/home/jovyan/work/cache/train_maskp.npy',mmap_mode='r');srt=np.sort(train_idx)
    total=sum(np.asarray(maskp[srt[i:i+500],99:]).sum(dtype=np.float64) for i in range(0,len(srt),500))
    mean_p=total/(len(srt)*19901);data=dict(train,maskp=maskp)
    # Pilot is disposable. Main heads, optimizer, schedule and data RNG restart fresh.
    pilot_heads=heads(device)
    pilot=train_stream(model,pilot_heads,data,train_idx,np.random.Generator(np.random.PCG64(0)),stats,mean_p,100,min(TRAIN_CUTOFF,time.time()+120),pilot=True)
    (out/'pilot.json').write_text(json.dumps(pilot,indent=2)+'\n');log('PILOT DONE '+json.dumps(pilot))
    del pilot_heads
    # No tuning: one fixed training budget based only on measured cost/time.
    remaining=TRAIN_CUTOFF-time.time()-120
    conservative_rate=min(pilot['wall_updates_s']*.65,15.)
    epoch_steps=math.ceil(len(train_idx)/16)*20
    epochs=min(2,int(remaining*conservative_rate/epoch_steps))
    if epochs<1:
        budget=max(100,int(remaining*conservative_rate)//20*20)
    else:budget=epochs*epoch_steps
    assert budget>0 and remaining>180
    heads_main=heads(device)
    plan=dict(original_start_utc=datetime.datetime.fromtimestamp(START,datetime.timezone.utc).isoformat(),
              hard_deadline_utc=datetime.datetime.fromtimestamp(DEADLINE,datetime.timezone.utc).isoformat(),
              training_cutoff_utc=datetime.datetime.fromtimestamp(TRAIN_CUTOFF,datetime.timezone.utc).isoformat(),
              planned_updates=budget,chosen_epochs=epochs,conservative_rate=conservative_rate,mean_mask_probability=mean_p,
              training_sequences=len(train_idx),holdout_sequences=len(holdout),lr=.001,wd=.05,alpha=.5,batch=16,chunk=1000,
              backbone=str(BACKBONE),backbone_sha256=hashlib.sha256(BACKBONE.read_bytes()).hexdigest(),
              train_index_sha256=hashlib.sha256(train_idx.tobytes()).hexdigest(),holdout_index_sha256=hashlib.sha256(holdout.tobytes()).hexdigest(),
              training_selection='Budget determined from pilot/time only. No validation read, early stopping or hyperparameter search.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');log('FIXED PLAN '+json.dumps(plan))
    results=train_stream(model,heads_main,data,train_idx,rng,stats,mean_p,budget,TRAIN_CUTOFF)
    for name,head in zip(('B','C'),heads_main):
        torch.save(dict(head=head.state_dict(),kind=name,backbone=str(BACKBONE),stats=ck['stats'],updates=results['updates'],plan=plan),out/f'{name}_final.pt')
    assert all(torch.equal(v.cpu(),ck['model'][k]) for k,v in model.state_dict().items())
    assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
    results['backbone_bitwise_unchanged']=True
    (out/'training.json').write_text(json.dumps(results,indent=2)+'\n');log('TRAINING DONE '+json.dumps({k:v for k,v in results.items() if k!='curve'}))
    # Full validation begins only after heads are frozen; no model selection loop.
    for head in heads_main:head.eval();head.requires_grad_(False)
    peer_ck=torch.load(TEACHER,map_location='cpu',weights_only=False)
    peer=base.GRUModel(hidden=128).to(device).eval();peer.load_state_dict(peer_ck['model']);peer.requires_grad_(False)
    peer_stats={k:v.to(device) for k,v in peer_ck['stats'].items()}
    valid_data=base.load_split('valid');n=1873;need=np.arange(20000)>=99
    names=['A','B','C','A+v19a','B+v19a','C+v19a']
    S=np.zeros((6,n,2,2,6),np.float64);acc=[base.GlobalAccumulator() for _ in names]
    batches=[np.arange(i,min(i+16,n)) for i in range(0,n,16)];started=time.time()
    with torch.no_grad():
        for bi,batch in enumerate(base.Prefetcher(valid_data,batches,depth=1)):
            if time.time()>DEADLINE-90:raise RuntimeError('Evaluation deadline; retain incumbent, no package')
            prices=torch.from_numpy(batch['prices']).to(device);features=torch.from_numpy(batch['features']).to(device)
            x=base.build_inputs(prices,features,stats);xp=base.build_inputs(prices,features,peer_stats)
            state=None;peer_state=None;parts=[[] for _ in range(4)]
            for s in range(0,20000,1000):
                hidden,state=model.gru(x[:,s:s+1000],state);a=model.head(hidden)
                p,peer_state=peer(xp[:,s:s+1000],peer_state)
                for k,pred in enumerate([a,a+heads_main[0](hidden),a+heads_main[1](hidden),p]):parts[k].append(pred)
            pred=[torch.cat(v,1).cpu().numpy() for v in parts]
            pred=pred[:3]+[np.mean([p,pred[3]],axis=0,dtype=np.float32) for p in pred[:3]]
            for k,p in enumerate(pred):
                for local,seq in enumerate(batches[bi]):
                    mask=batch['is_scored'][local]&need
                    S[k,seq,0]=seq_stats(batch['targets'][local],p[local],mask)
                    S[k,seq,1]=seq_stats(batch['targets'][local],p[local],need)
                    acc[k].add(batch['targets'][local],p[local],mask)
            if (bi+1)%20==0:log(f'EVAL {batches[bi][-1]+1}/1873 elapsed {time.time()-started:.1f}s')
    np.savez_compressed(out/'sequence_stats.npz',stats=S,names=names,stat_names=STAT_NAMES,sets=['scored','all_required'])
    full=wp_from_stats(S.sum(1));held=wp_from_stats(S[:,np.arange(n)%4==0].sum(1))
    assert abs(full[0,0,2]-.6857267)<1e-5 and abs(full[3,0,2]-.6867397)<1e-5
    assert all(abs(full[i,0,2]-a.result()['weighted_pearson'])<1e-12 for i,a in enumerate(acc))
    output=dict(evaluation_seconds=time.time()-started,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                prior_validation_reuse='Full validation and 469 subset repeatedly reused; neither is untouched. No validation targets used for training or choosing this run budget.',
                results=[dict(model=name,scored_wp=float(full[k,0,2]),t0=float(full[k,0,0]),t1=float(full[k,0,1]),all_required_wp=float(full[k,1,2]),
                              diagnostic469_wp=float(held[k,0,2]),delta_vs_incumbent=float(full[k,0,2]-full[3,0,2])) for k,name in enumerate(names)],
                incumbent_reproduced=True,backbone_unchanged=True,training=results)
    (out/'results.json').write_text(json.dumps(output,indent=2)+'\n');log('FINAL RESULTS '+json.dumps(output['results']))


if __name__=='__main__':main()
