#!/usr/bin/env python
"""No-rollout mechanical check of the new checkpoint sampler/repeat dispatch."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
from omegaconf import OmegaConf
PARENT = Path('/soalnas/share/data/zbao7/diffusion_policy')
SOURCE = PARENT / '.codex-runtime/worktrees/stage3-e1cg/results/stage3/e1cg/dppo'
OUT = PARENT / 'results/stage3/e1cg/runs/gates/ckpt_dispatch'
sys.path.insert(0, str(SOURCE))
import agent.finetune.stage3_modes as modes
from agent.finetune.stage3_support import Stage3Support
modes.diagnostic_pairs = lambda *a, **kw: None
modes.saturation_metrics = lambda *a, **kw: {'x0_sat_frac_mean': 0., 'x0_sat_frac_last': 0.}
firsts = np.zeros((501,40)); firsts[::250] = 1
rewards = np.tile((np.arange(500) % 250)[:,None], (1,40)).astype(np.float64)
checks = []
OUT.mkdir(parents=True,exist_ok=False)
for repeat, choice in [(2,'train'), (2,'eval'), (2,'both'), (6,'train')]:
    folder = OUT / ('%s_%d' % (choice,repeat));folder.mkdir()
    agent = SimpleNamespace(cfg=OmegaConf.create({'train': {'ckpt_segments':True,'ckpt_eval_repeats':repeat,'ckpt_samplers':choice}, 'base_policy_path':'synthetic-no-rollout'}),
                logdir=str(folder), seed=5000,gamma=.99,model=SimpleNamespace(actor_ft=torch.nn.Linear(1,1)),
                _stage3_action_metrics=lambda actions: {'action_oor_frac':0., 'policy_action_oor_frac':0.})
    calls=[]
    def collect(**kwargs):
        calls.append(kwargs['deterministic'])
        return {'raw_rewards':rewards,'firsts':firsts,'actions':np.zeros((500,40,4,6))}
    agent._stage3_collect=collect
    result=modes.run_ckpt_eval(agent)
    expected = ([False,True]*repeat if choice=='both' else [choice=='eval']*repeat)
    assert calls==expected
    assert all(v['J_disc']['n']==80*repeat for v in result['samplers'].values())
    assert all([row['mean'] for row in v['reward_segments']]==[24.5,74.5,124.5,174.5,224.5] for v in result['samplers'].values())
    checks.append({'repeats':repeat,'samplers':choice,'calls':calls,'episodes_per_sampler':80*repeat,'passed':True})
invalid=[]
for post,discount,norm in [(True,False,True),(True,True,False),(False,True,True)]:
    agent=SimpleNamespace(model=SimpleNamespace(norm_adv=norm),actor_single_step=True)
    cfg=OmegaConf.create({'train':{'decision_discount_post_norm':post,'decision_discount':discount}})
    try:
        Stage3Support._stage3_init(agent,cfg)
    except ValueError as exc:
        invalid.append({'post_norm':post,'decision_discount':discount,'norm_adv':norm,'message':str(exc),'passed':True})
    else:
        raise AssertionError('invalid combination accepted')
(OUT/'checks.json').write_text(json.dumps({'passed':True,'scope':'synthetic saved-reward dispatch, no environment or rollout, existing numerical modes mocked only saturation','repeat_cases':checks,'invalid_flag_combinations':invalid},indent=2)+'\n')
print('Checkpoint dispatch and invalid flag combinations passed')
