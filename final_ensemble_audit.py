"""Fixed final incumbent and two mixed GRU/LSTM triples; one validation pass, sequence-paired bootstrap.
Never trains or tunes. Fresh output directory required. Uses historical metric helpers.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import train_v1 as base
import train_v4m
from spread_v2 import STAT_NAMES, WARMUP, seq_stats, wp_from_stats

ROOT = Path(__file__).resolve().parent
CKPTS = {
    'v18c': Path('/home/jovyan/work/checkpoints/train_v18c_wd05_nodrop_final.pt'),
    'v19a': Path('/home/jovyan/work/checkpoints/train_v19a_wd10_ds1_final.pt'),
    'v24a': Path('/home/jovyan/work/checkpoints/train_v24a_ds2_wd05_final.pt'),
    'v13': Path('/home/jovyan/work/checkpoints/train_v13_lstm_maskw_a05_h128_e5_seed0_best.pt'),
}
PAIRS = [('v18c', 'v19a'), ('v18c', 'v24a', 'v13'), ('v18c', 'v19a', 'v13')]
SEED = 20261114
DRAWS = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tm = base
    tm.set_seed(0)
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(2.5 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    device = torch.device('cuda:0')
    models, checkpoints, stats = {}, {}, {}
    for name, path in CKPTS.items():
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['args'].get('arch','gru')==('lstm' if name=='v13' else 'gru')
        model=(train_v4m.LSTMModel if name=='v13' else base.GRUModel)(hidden=ck['args']['hidden'])
        model.load_state_dict(ck['model']);model.eval()
        assert ck['step'] == 64500, (name, ck['step'])
        models[name] = model.to(device).eval()
        checkpoints[name] = ck
        stats[name] = {k: v.to(device) for k,v in ck['stats'].items()}
        print(f'loaded {name}: {path}, step {ck["step"]}', flush=True)
    valid = tm.load_split('valid')
    n, length = valid['prices'].shape[:2]
    assert n == 1873 and length == 20000
    diagnostic = np.flatnonzero(np.arange(n) % 4 == 0)
    assert len(diagnostic) == 469
    need = np.arange(length) >= WARMUP
    subsets = [(name,) for name in CKPTS] + PAIRS
    names = ['+'.join(s) for s in subsets]
    S = np.zeros((len(subsets), n, 2, 2, 6), np.float64)
    rows = np.zeros((n,2), np.int64)
    acc = [tm.GlobalAccumulator() for _ in PAIRS]
    batches = [np.arange(i, min(i+16, n)) for i in range(0,n,16)]
    start = time.perf_counter()
    with torch.inference_mode():
        for bi, batch in enumerate(tm.Prefetcher(valid, batches, depth=1)):
            prices = torch.from_numpy(batch['prices']).to(device)
            features = torch.from_numpy(batch['features']).to(device)
            preds = {}
            for name, model in models.items():
                # Each checkpoint's own stats, including the different data-seed-1 stats.
                x = tm.build_inputs(prices, features, stats[name])
                state, pieces = None, []  # resets BOTH hidden and EMA for every new sequence batch
                for s in range(0,length,1000):
                    pred, state = model(x[:,s:s+1000], state)
                    pieces.append(pred)
                preds[name] = torch.cat(pieces,1).float().cpu().numpy()
                assert np.isfinite(preds[name]).all()
                del x, state, pieces, pred
            for si, subset in enumerate(subsets):
                # Float32 raw mean, never clip individual model predictions before averaging.
                pred = np.mean([preds[name] for name in subset], axis=0, dtype=np.float32)
                for local, sequence in enumerate(batches[bi]):
                    mask = batch['is_scored'][local] & need
                    S[si,sequence,0] = seq_stats(batch['targets'][local],pred[local],mask)
                    S[si,sequence,1] = seq_stats(batch['targets'][local],pred[local],need)
                    if si >= len(CKPTS):
                        acc[si-len(CKPTS)].add(batch['targets'][local],pred[local],mask)
                    rows[sequence] = [mask.sum(),need.sum()]
            if (bi+1) % 10 == 0 or bi+1 == len(batches):
                print(f'{batches[bi][-1]+1}/{n} sequences; {time.perf_counter()-start:.1f}s; peak GPU {torch.cuda.max_memory_allocated()/2**30:.2f} GiB',flush=True)
    pass_seconds = time.perf_counter()-start
    # Save before reference checks so diagnostics survive any sanity failure.
    np.savez_compressed(out/'sequence_stats.npz',stats=S,subsets=np.array(names),sets=np.array(['scored','all_required']),
                        stat_names=np.array(STAT_NAMES),sequence_index=np.arange(n),diagnostic_subset=diagnostic,
                        rows=rows,ckpt_names=np.array(list(CKPTS)),ckpt_paths=np.array([str(p) for p in CKPTS.values()]))
    full = wp_from_stats(S.sum(1))
    held = wp_from_stats(S[:,diagnostic].sum(1))
    checks = []
    for i,name in enumerate(CKPTS):
        for metric,split in [('valid',0),('valid_all',1)]:
            expected=checkpoints[name][metric]['weighted_pearson']
            delta=float(full[i,split,2]-expected)
            checks.append(dict(check=f'{name} {metric} vs checkpoint',delta=delta,tolerance=1e-5))
            assert abs(delta)<1e-5,checks[-1]
    pair_start=len(CKPTS)
    for i,a in enumerate(acc):
        delta=float(full[pair_start+i,0,2]-a.result()['weighted_pearson'])
        checks.append(dict(check=f'{names[pair_start+i]} vs GlobalAccumulator',delta=delta,tolerance=1e-12))
        assert abs(delta)<1e-12,checks[-1]
    reference_error=float(full[pair_start,0,2]-0.6867397)
    assert abs(reference_error)<1e-5, reference_error
    saved=np.load('/home/jovyan/work/ensemble_check11_stats.npz')
    historical_checks=[]
    for i,subset in enumerate(subsets):
        index=next(j for j,name in enumerate(saved['subsets']) if set(str(name).split('+'))==set(subset))
        historical=wp_from_stats(saved['stats'][index].sum(0))
        error=float(np.max(np.abs(full[i]-historical)))
        assert error<1e-5,(subset,error)
        historical_checks.append(dict(blend='+'.join(subset),max_metric_error=error))
    print(f'Reference {full[pair_start,0,2]:.10f}; delta vs .6867397 {reference_error:+.3e}; all single-model checks pass.',flush=True)
    # Resample entire sequences WITH replacement. Identical sequence multiplicities
    # for every pair in each draw; merge moments then recompute global WP.
    rng=np.random.default_rng(SEED)
    flat=S[pair_start:,:,0].transpose(1,0,2,3).reshape(n,-1)
    boot=[]
    for start_draw in range(0,DRAWS,50):
        index=rng.integers(0,n,size=(min(50,DRAWS-start_draw),n))
        counts=np.stack([np.bincount(row,minlength=n) for row in index])
        moments=(counts@flat).reshape(len(index),len(PAIRS),2,6)
        boot.append(wp_from_stats(moments)[...,2])
    boot=np.concatenate(boot)
    differences=boot-boot[:,:1]
    ci=np.percentile(differences,[2.5,97.5],axis=0).T
    np.savez_compressed(out/'bootstrap.npz',scores=boot,differences_vs_reference=differences,ci95=ci,
                        pair_names=np.array(names[pair_start:]),seed=SEED,resamples=DRAWS)
    result=dict(started_utc=started,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                validation_sequences=n,diagnostic_sequences=len(diagnostic),diagnostic_definition='validation array index % 4 == 0; diagnostic, not untouched test',
                pass_seconds=pass_seconds,peak_gpu_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                averaging='equal float32 mean of raw model outputs, followed by metric clipping [-2,2]',
                statistic_axes=['subset','sequence','row_set','target','moment'],
                bootstrap=dict(seed=SEED,resamples=DRAWS,unit='whole validation sequence',method='paired percentile 95% CI; global WP recomputed from resampled sufficient statistics',
                               limitation='Intervals do not account for training randomness or prior model selection.'),
                reference_expected=.6867397,reference_error=reference_error,sanity_checks=checks,historical_ensemble_check11_checks=historical_checks,
                ckpts={name:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),step=checkpoints[name]['step']) for name,path in CKPTS.items()},pairs=[])
    for i,pair in enumerate(PAIRS):
        j=pair_start+i
        result['pairs'].append(dict(pair='+'.join(pair),scored_wp=float(full[j,0,2]),t0=float(full[j,0,0]),t1=float(full[j,0,1]),
                                    all_required_wp=float(full[j,1,2]),diagnostic469_scored_wp=float(held[j,0,2]),
                                    difference_vs_reference=float(full[j,0,2]-full[pair_start,0,2]),ci95_vs_reference=ci[i].tolist()))
    with (out/'results.json').open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    lines=['Final fixed existing-checkpoint ensemble audit','pair | scored WP | t0 | t1 | all-required WP | diagnostic 469 WP | delta vs reference | paired 95% CI']
    for r in result['pairs']:
        lo,hi=r['ci95_vs_reference']
        lines.append(f"{r['pair']} | {r['scored_wp']:.7f} | {r['t0']:.7f} | {r['t1']:.7f} | {r['all_required_wp']:.7f} | {r['diagnostic469_scored_wp']:.7f} | {r['difference_vs_reference']:+.7f} | [{lo:+.7f}, {hi:+.7f}]")
    lines+=['Reference reproduced within 1e-5.','469-sequence subset is diagnostic, not an untouched test set.',result['bootstrap']['limitation'],f'Bootstrap seed {SEED}, {DRAWS} paired sequence resamples; pass {pass_seconds:.1f}s.']
    with (out/'results.txt').open('x') as f:f.write('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__': main()
