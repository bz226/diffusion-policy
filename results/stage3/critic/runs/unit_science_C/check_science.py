import ast
import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path('/soalnas/share/data/zbao7/diffusion_policy')
SOURCE = ROOT / '.codex-runtime/worktrees/stage3-critic/results/stage3_critic/dppo'
OUT = ROOT / 'results/stage3/critic/runs/unit_science_C'
sys.path.insert(0, str(SOURCE))
from agent.finetune.critic_baseline import (baseline_diagnostics, configured_actor_coefficients,
    configured_actor_gradient, critic_residual, load_critic_initialization, mc_critic_advantages,
    reward_scaler_divisor)
from agent.finetune.stage3_math import as_episodes, mc_returns
from agent.finetune.grad_mean import _copy, actor_state, scaler_state, state_sha256
from util.reward_scaling import RunningRewardScaler

rows = []
def check(name, value, **details):
    if not value:
        raise AssertionError(name)
    rows.append(dict(name=name, passed=True, **details))

for relative in ('agent/finetune/critic_baseline.py', 'agent/finetune/stage3_support.py', 'agent/finetune/stage3_modes.py'):
    ast.parse((SOURCE / relative).read_text(), filename=relative, feature_version=8)
check('Python38_AST', True, version=sys.version)
rng = np.random.RandomState(1703)
r = rng.normal(size=(500, 4))
firsts = np.zeros((501, 4)); firsts[::250] = 1
values = rng.normal(size=r.shape).astype(np.float32)
c = np.asarray(4.321)
expected = np.empty_like(r)
for t in range(500):
    expected[t] = sum(.99 ** j * r[t+j] for j in range(250 - t % 250)) - float(c) * values[t].astype(np.float64)
actual = mc_critic_advantages(r, firsts, .99, values, c)
relative = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
check('MCcritic_float64_bruteforce', relative <= 1e-12, relative_l2=relative,
      max_abs=float(np.max(np.abs(actual-expected))))
original = [x.copy() for x in (r, firsts, values, c)]
state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
G = mc_returns(r, firsts, .99)
diag = baseline_diagnostics(G, values, c, firsts, .99)
check('diagnostic_input_inert', all(np.array_equal(a,b) for a,b in zip(original,(r,firsts,values,c))))
check('diagnostic_RNG_inert', random.getstate()==state[0] and
      np.array_equal(np.random.get_state()[1],state[1][1]) and torch.equal(torch.get_rng_state(),state[2]))
manual = float(np.mean(np.sum(as_episodes(actual, firsts)**2 * .99**(2*np.arange(250)), axis=1)))
check('residual_energy', abs(manual-diag['baseline_energy_critic']) < 1e-10,
      full=diag['baseline_energy_critic'], sum_bins=sum(diag['baseline_bin_energy_critic']))
check('energy_bin_partition', np.isclose(sum(diag['baseline_bin_energy_critic']),manual,rtol=1e-15))
check('diagnostic_schema', len(diag['baseline_residual_mean_critic'])==250 and
      len(diag['baseline_bin_energy_ratio'])==5)
zero = baseline_diagnostics(np.zeros_like(r), np.zeros_like(r), c, firsts, .99)
check('undefined_ratios_None', zero['baseline_energy_ratio'] is None and
      all(v is None for v in zero['baseline_bin_energy_ratio']))
json.dumps(zero, allow_nan=False)

class ToyModel:
    def __init__(self):
        self.actor_ft = torch.nn.Linear(1, 1, bias=False)
        self.actor_ft.weight.data.fill_(.125)
        self.ft_denoising_steps = 2
        self.actor = torch.nn.Linear(1, 1)
        self.critic = torch.nn.Linear(1, 1)
        self.calls = []
    def loss(self, obs, chains, next_chains, steps, returns, values, advantages, old, **kwargs):
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        logp = self.actor_ft(obs['state']).reshape(-1)
        pg = -(adv * logp).mean()
        self.calls.append(len(advantages))
        return (pg, pg*0, pg*0, 0., 0., 1., 0., 0.)

model = ToyModel()
agent = SimpleNamespace(model=model,device='cpu',batch_size=4,reward_horizon=1,
    use_bc_loss=False,ent_coef=0.,bc_loss_coeff=0.)
batch = dict(chains_k=torch.zeros(4,3,1,1), obs_k={'state':torch.tensor([[1.],[2.],[3.],[7.]])},
    returns=np.zeros(4),values=np.zeros(4),oldlogprobs=torch.zeros(4,2,1,1))
coefficients = torch.tensor([1.,2.,4.,16.]); permutation=torch.arange(8)
model.actor_ft.weight.grad=torch.tensor([[.333]])
before=model.actor_ft.weight.grad.clone()
gradient=configured_actor_gradient(agent,batch,coefficients,permutation)
manual_gradient=0.
for pairs in permutation.reshape(2,4):
    d=pairs//2; adv=coefficients[d]; adv=(adv-adv.mean())/(adv.std()+1e-8)
    manual_gradient += float(-(adv*batch['obs_k']['state'][d].reshape(-1)).mean())*.5
check('calibration_chunk_normalization', gradient.item()==manual_gradient,
      actual=gradient.item(),expected=manual_gradient,calls=model.calls)
full_adv=coefficients.repeat_interleave(2); full_adv=(full_adv-full_adv.mean())/(full_adv.std()+1e-8)
full_gradient=float(-(full_adv*batch['obs_k']['state'].reshape(-1).repeat_interleave(2)).mean())
check('calibration_not_fullbatch_normalized',abs(gradient.item()-full_gradient)>1e-3, fullbatch_gradient=full_gradient)
check('calibration_existing_grad_inert',torch.equal(before,model.actor_ft.weight.grad))

# A minimal frozen payload proves selective loading and scaler restoration.
agent.reward_scale_running=True; agent.running_reward_scaler=RunningRewardScaler(4)
agent.actor_optimizer=torch.optim.SGD(model.actor_ft.parameters(),lr=.01)
agent.critic_optimizer=torch.optim.AdamW(model.critic.parameters(),lr=.001)
agent.logdir=str(OUT)
agent.running_reward_scaler.ret_rms.var=np.asarray(19.)
agent.running_reward_scaler.ret_rms.count=72.
warmcritic={k:torch.ones_like(v)*.75 for k,v in model.critic.state_dict().items()}
payload={'point':'P0','critic':warmcritic,'reward_scaler':scaler_state(agent)}
payload['critic_sha256']=state_sha256(warmcritic)
payload['scaler_sha256']=state_sha256(payload['reward_scaler'])
path=OUT/'toy_frozen.pt'; torch.save(payload,path)
agent.running_reward_scaler.ret_rms.var=np.asarray(2.)
actor_before=state_sha256(actor_state(agent)); optimizer_before=state_sha256(
    {'actor':agent.actor_optimizer.state_dict(),'critic':agent.critic_optimizer.state_dict()})
audit=load_critic_initialization(agent,path)
check('critic_init_selective', actor_before==state_sha256(actor_state(agent)) and
    optimizer_before==state_sha256({'actor':agent.actor_optimizer.state_dict(),'critic':agent.critic_optimizer.state_dict()}))
check('critic_init_exact',state_sha256(model.critic.state_dict())==payload['critic_sha256'] and
    state_sha256(scaler_state(agent))==payload['scaler_sha256'])
check('divisor_epsilon',float(reward_scaler_divisor(agent))==np.sqrt(19.+1e-8))
result={'passed':True,'checks':rows,'scope':'Small CPU software checks only; formal C1-C5 not executed here.'}
(OUT/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
