"""Record R0 evaluation red flags without using trajectories as an execution gate."""
import datetime as dt
import json
import math
import pickle
from pathlib import Path
import statistics


def save(path, value):
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class TrajectoryObserver:
    """Read saved eval returns only; never compare training or rounded log values."""
    def __init__(self, study_root, run_dir):
        self.root, self.run_dir = Path(study_root), Path(run_dir)
        self.status_path = self.run_dir / 'trajectory_observations.json'
        if self.status_path.exists():
            raise ValueError('Refusing to overwrite previous trajectory observations')
        references = []
        seeds = []
        for name in ('seed0_handoff', 'seed1', 'seed2'):
            manifest_path = self.root.parent / 'repro_halfcheetah/runs' / name / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('status') != 'complete' or manifest.get('verified_rows') != 140:
                raise ValueError('Stage 1 reference is incomplete: ' + name)
            result_path = Path(manifest['result_path'])
            with result_path.open('rb') as stream:
                rows = pickle.load(stream)
            if len(rows) != 140:
                raise ValueError('Stage 1 reference row count differs from 140: ' + name)
            evaluations = {}
            for itr in range(0, 140, 10):
                row = rows[itr]
                step, reward = row['step'], float(row['eval_episode_reward'])
                if row['itr'] != itr or step != (itr - itr // 10) * 80000 or not math.isfinite(reward):
                    raise ValueError('Stage 1 reference has invalid evaluation: ' + name)
                evaluations[step] = reward
            seeds.append(manifest['seed'])
            references.append({'manifest': str(manifest_path), 'result': str(result_path),
                               'seed': manifest['seed'], 'evaluations': evaluations})
        if set(seeds) != {0, 1, 2}:
            raise ValueError('Stage 1 references must include seeds 0, 1 and 2')
        self.bands = {}
        for step in references[0]['evaluations']:
            values = [item['evaluations'][step] for item in references]
            mean, std = statistics.mean(values), statistics.stdev(values)
            self.bands[step] = {'baseline_seed_values': values, 'mean': mean,
                                'sample_std': std, 'lower': mean - 5 * std,
                                'upper': mean + 5 * std}
        self.observed = set()
        self.report = {'role': 'descriptive_only_no_trajectory_stopping',
                       'criterion': 'eval return outside Stage 1 same-step mean +/- 5 sample standard deviations',
                       'standard_deviation_ddof': 1, 'status': 'observing',
                       'references': [{key: value for key, value in item.items() if key != 'evaluations'}
                                      for item in references],
                       'evaluations': [], 'red_flags': [],
                       'training_or_log_token_comparisons': False}
        self._write()

    def _write(self):
        self.report['updated_utc'] = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
        save(self.status_path, self.report)

    def check(self, records, final=False):
        new_flags = []
        count_before = len(self.observed)
        for row in records:
            if 'eval_episode_reward' not in row or row['itr'] in self.observed:
                continue
            # The supervisor validates structure and finiteness before calling us.
            step, reward = int(row['step']), float(row['eval_episode_reward'])
            band = self.bands[step]
            flag = reward < band['lower'] or reward > band['upper']
            observation = dict(band, iteration=int(row['itr']), step=step,
                               eval_return=reward, red_flag=flag,
                               action='record_only_continue')
            self.report['evaluations'].append(observation)
            self.observed.add(row['itr'])
            if flag:
                self.report['red_flags'].append(observation)
                new_flags.append(observation)
        if final:
            self.report['status'] = 'complete' if len(records) == 140 else 'partial_run_terminal'
        if final or len(self.observed) != count_before:
            self._write()
        return new_flags
