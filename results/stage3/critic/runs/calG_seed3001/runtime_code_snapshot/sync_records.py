#!/usr/bin/env python
"""Copy selected C records into its isolated parent worktree for normal Git commits."""
from pathlib import Path
import shutil

PROJECT=Path('/soalnas/share/data/zbao7/diffusion_policy')
SOURCE=PROJECT/'results/stage3/critic'
TARGET=PROJECT/'.codex-runtime/worktrees/stage3-critic/results/stage3/critic'
EXCLUDED={'.git','__pycache__','.pytest_cache','tmp','cache','checkpoint','checkpoints','data','datasets'}
BINARY={'.pt','.pth','.ckpt','.npy','.npz','.pyc','.pyo'}

def main():
    copied=0
    for path in sorted(SOURCE.rglob('*')):
        rel=path.relative_to(SOURCE)
        if not path.is_file() or EXCLUDED.intersection(rel.parts) or path.suffix in BINARY or path.name.endswith('.lock'):
            continue
        if path.is_symlink():
            raise ValueError('Study records must be ordinary files: '+str(path))
        dst=TARGET/rel
        if dst.exists() and dst.read_bytes()==path.read_bytes():
            continue
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dst);copied+=1
    print('Copied %d selected ordinary files; no raw data were deleted or modified.'%copied)

if __name__=='__main__':main()
