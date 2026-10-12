"""Addendum-C-only launcher: native DPPO computation plus explicit env cleanup."""
import os
from pathlib import Path
import runpy
import signal
import sys

SOURCE = Path(os.environ['CRITIC_SOURCE']).resolve()
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / 'script'))

from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent
from agent.finetune.grad_mean import cleanup_environment


def main():
    def interrupted(signum, frame):
        raise InterruptedError('Addendum C job received signal%d' % signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    original_init = TrainPPODiffusionAgent.__init__
    original_run = TrainPPODiffusionAgent.run

    def initialize(self, cfg, *args, **kwargs):
        try:
            original_init(self, cfg, *args, **kwargs)
        except BaseException as error:
            if hasattr(self, 'venv'):
                if not hasattr(self, 'logdir'):
                    self.logdir = str(cfg.logdir)
                cleanup_environment(self, primary_error=error)
            raise

    def run(self, *args, **kwargs):
        primary_error = None
        try:
            return original_run(self, *args, **kwargs)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cleanup_environment(self, primary_error=primary_error)

    TrainPPODiffusionAgent.__init__ = initialize
    TrainPPODiffusionAgent.run = run
    runpy.run_path(str(SOURCE / 'script/run.py'), run_name='__main__')


if __name__ == '__main__':
    main()
