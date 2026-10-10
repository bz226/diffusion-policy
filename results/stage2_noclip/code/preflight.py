"""Read-only import/config checks, in a separate process from training."""
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from source_git import source_git_command


def inspect(overrides):
    root = Path(os.environ['STUDY_ROOT'])
    source = root / 'dppo'
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    OmegaConf.register_new_resolver('round_up', math.ceil, replace=True)
    OmegaConf.register_new_resolver('round_down', math.floor, replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(source / 'cfg/gym/finetune/halfcheetah-v2')):
        cfg = hydra.compose(config_name='ft_ppo_diffusion_mlp', overrides=overrides)
        OmegaConf.resolve(cfg)
    for key in ('base_policy_path', 'normalization_path'):
        assert Path(cfg[key]).is_file(), 'Missing existing asset: ' + str(cfg[key])
    assert cfg.env.n_envs == 40 and cfg.train.n_steps == 500
    assert Path(cfg.logdir).is_relative_to(root) if sys.version_info >= (3, 9) else str(cfg.logdir).startswith(str(root) + '/')
    assert all(os.environ.get(k) is None for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
    origins = {}
    for name in ('agent.finetune.train_ppo_diffusion_agent', 'model.diffusion.diffusion_ppo',
                 'model.diffusion.diffusion_vpg', 'model.diffusion.mlp_diffusion',
                 'model.common.critic', 'env.gym_utils', 'util.scheduler'):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        assert str(path).startswith(str(source) + '/'), (name, str(path))
        origins[name] = str(path)
    misplaced = {}
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in ('agent', 'model', 'env', 'util', 'cfg', 'script'):
            filename = getattr(module, '__file__', None)
            if filename and not str(Path(filename).resolve()).startswith(str(source) + '/'):
                misplaced[name] = filename
    assert not misplaced, misplaced
    return {'status': 'passed', 'python': sys.executable, 'module_origins': origins,
            'resolved_configuration': OmegaConf.to_container(cfg, resolve=True),
            'source_commit': subprocess.check_output(source_git_command(source, 'rev-parse', 'HEAD'),
                                                     cwd=str(source), text=True).strip(),
            'environment': {k: os.environ.get(k) for k in ('DPPO_DATA_DIR', 'DPPO_LOG_DIR', 'PYTHONPATH', 'LD_LIBRARY_PATH',
                                                         'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}


if __name__ == '__main__':
    output = Path(sys.argv[1])
    result = inspect(sys.argv[2:])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('Preflight passed:', output)
