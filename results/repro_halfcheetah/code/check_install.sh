#!/usr/bin/env bash
set -euxo pipefail
date -u
nvidia-smi
nproc
cat /etc/os-release
df -h "$REPRO_ROOT"
git rev-parse HEAD
git diff --exit-code HEAD --
python -m pip freeze > "$REPRO_ROOT/setup/pip-freeze-${SLURM_JOB_ID:-local}.txt"
python -m pip check
python -c "import torch; assert torch.__version__.split('+')[0] == '2.4.0'; assert torch.cuda.is_available(); print('torch', torch.__version__, 'CUDA runtime', torch.version.cuda, 'GPU', torch.cuda.get_device_name(0))"
python -c "import mujoco_py, d4rl, gym; gym.make('HalfCheetah-v2').reset(); print('ok')"
python -c "import d4rl.gym_mujoco, gym; print('Configured environment registration:', gym.spec('halfcheetah-medium-v2'))"
python -c "import mujoco_py; print('mujoco-py', mujoco_py.__version__, 'MuJoCo', mujoco_py.functions.mj_version())"
