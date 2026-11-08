"""Run either trainer on a tiny matched real-cache smoke; fresh outputs only."""
import argparse
import sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--old',action='store_true')
    a=ap.parse_args(); out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=False)
    if a.old:
        import train_v2m as trainer
        specs=['--name','control','--seed','0']
    else:
        from v33_model import register
        trainer=register()
        specs=['--models','control:0:gru::0,projection:0:v33proj::0,context:0:v33context::0','--data-seed','0','--run','smoke_v33']
    trainer.CKPT_DIR=str(out)
    handler=trainer.logging.FileHandler
    trainer.logging.FileHandler=lambda filename,*args,**kw: handler(out/Path(filename).name,*args,**kw)
    trainer.GPU_MEM_CAP_GB=2.5
    sys.argv=[sys.argv[0],*specs,'--smoke','--max-steps','4','--batch','2','--hidden','128','--wd','0.05','--epochs','5','--n-stats','2','--n-val','1','--n-holdout','300','--mask-p','/home/jovyan/work/cache/train_maskp.npy','--alpha','0.5']
    trainer.main()


if __name__=='__main__': main()
