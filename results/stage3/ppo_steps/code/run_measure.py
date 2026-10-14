#!/usr/bin/env python
"""Bounded worker entry; all environment workers are explicitly reaped."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'A', 'C', 'S1', 'train'), required=True)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--checkpoint-itr', type=int, default=35)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    os.environ.setdefault('D4RL_SUPPRESS_IMPORT_ERROR', '1')
    sys.path.insert(0, str(args.source.resolve()))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent
    from agent.finetune.grad_mean import cleanup_environment
    from agent.finetune import ppo_steps_measure as measurement
    cfg = OmegaConf.load(args.source / 'cfg/gym/finetune/halfcheetah-v2/ft_ppo_diffusion_mlp.yaml')
    cfg.seed = args.seed
    cfg.wandb = None
    cfg.logdir = str(args.output.resolve())
    cfg.base_policy_path = str(args.checkpoint.resolve())
    cfg.train.n_train_itr = 140
    cfg.train.save_model_freq = 35
    if args.mode == 'train':
        cfg.train.ppo_steps_capture = True
    OmegaConf.resolve(cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'config.yaml').exists():
        raise FileExistsError('Refusing to reuse scientific worker output')
    OmegaConf.save(cfg, args.output / 'config.yaml')
    (args.output / 'command.json').write_text(json.dumps({'command': sys.argv, 'commit': args.commit,
                                                       'source': str(args.source.resolve())}, indent=2) + '\n')
    agent = TrainPPODiffusionAgent.__new__(TrainPPODiffusionAgent)
    agent.logdir = str(args.output.resolve())
    agent.venv = SimpleNamespace(processes=[], close=lambda: None)
    primary = None
    try:
        agent.__init__(cfg)
        measurement.validate(agent)
        if args.mode == 'train':
            agent.run()
        elif args.mode == 'prepare':
            measurement.prepare(agent, args.checkpoint, args.commit)
        elif args.mode == 'S1':
            measurement.gate_s1(agent, args.state)
        else:
            measurement.collect(agent, args.mode, args.checkpoint_itr, args.state, args.commit)
    except BaseException as error:
        primary = error
        raise
    finally:
        cleanup_environment(agent, primary_error=primary)


if __name__ == '__main__':
    main()
