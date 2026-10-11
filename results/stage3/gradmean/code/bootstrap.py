"""Durable, bounded transition from offline gates through live gates to production.

Run on the prepared CPU Slurm controller. This program never starts scientific
work until both the formal gate record and its Slurm exit status pass.
"""
import argparse
import fcntl
from pathlib import Path
import signal
import subprocess
import sys
import time

from campaign import (ROOT, SOURCE, TERMINAL, Campaign, append, atomic, command,
                      read, report_status, utc, validate_gate)


class Bootstrap:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.stop = False
        self.next_monitor = self.started
        (ROOT / 'runs').mkdir(exist_ok=True)
        self.lock = (ROOT / 'runs/bootstrap.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = ROOT / 'runs/bootstrap.json'
        previous = read(self.path, {})
        if previous.get('formal_job') not in (None, args.formal_job):
            raise ValueError('Cannot replace the formal gate job in an existing bootstrap record')
        self.record = dict(previous, formal_job=args.formal_job, offline_gate=str(args.offline),
                           started_utc=utc(), status='running', monitor_status='active',
                           controller_cap_seconds=args.timeout_seconds, monitor_interval_seconds=1800)
        atomic(self.path, self.record)
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

    def handle_signal(self, signum, frame):
        self.stop = True

    def check_cap(self):
        if self.stop or time.monotonic() - self.started >= self.args.timeout_seconds - 60:
            raise RuntimeError('Bootstrap controller interrupted or capped; no further admissions')

    def monitor(self, phase, detail=None, force=False):
        self.record.update(phase=phase, detail=detail, updated_utc=utc())
        atomic(self.path, self.record)
        if force or time.monotonic() >= self.next_monitor:
            append(ROOT / 'runs/bootstrap_monitor_history.jsonl', self.record)
            report = ROOT / 'experiment.md'
            start, end = '<!-- gradmean-bootstrap-start -->', '<!-- gradmean-bootstrap-end -->'
            if report.exists():
                text = report.read_text()
                block = start + '\nGate transition monitor: ' + self.record['updated_utc'] + '. **' + self.record['status'] + '**; phase: ' + phase + '. Monitor ' + self.record['monitor_status'] + '.\n' + end
                if start in text and end in text:
                    text = text[:text.index(start)] + block + text[text.index(end) + len(end):]
                else:
                    text = text.rstrip() + '\n\n' + block + '\n'
                temporary = report.with_name('.experiment.bootstrap.tmp')
                temporary.write_text(text)
                temporary.replace(report)
            self.next_monitor = time.monotonic() + 1800

    def formal_status(self):
        result = command(['sacct', '--allocations', '--noheader', '--parsable2', '--jobs=' + self.args.formal_job,
                          '--format=JobIDRaw,State,ExitCode,ElapsedRaw'])['stdout']
        rows = [line.split('|') for line in result.splitlines() if line.startswith(self.args.formal_job + '|')]
        if not rows:
            return dict(state=None)
        if len(rows) != 1:
            raise ValueError('Formal gate accounting is ambiguous')
        return dict(state=rows[0][1].split()[0].rstrip('+'), exit_code=rows[0][2], elapsed_seconds=int(rows[0][3]))

    def await_offline(self):
        unknown_since = None
        while True:
            self.check_cap()
            status = self.formal_status()
            gate = read(self.args.offline, {})
            self.monitor('await_offline_G1_G2_G5', dict(scheduler=status, gate_status=gate.get('status')))
            if gate.get('status') == 'failed' or (status['state'] in TERMINAL and (status['state'] != 'COMPLETED' or status.get('exit_code') != '0:0')):
                raise RuntimeError('Formal offline gate failed; live collection was not admitted')
            if status['state'] is None:
                unknown_since = unknown_since or time.monotonic()
                if time.monotonic() - unknown_since > 300:
                    raise RuntimeError('Formal gate absent from accounting for five minutes')
            else:
                unknown_since = None
            if status['state'] == 'COMPLETED':
                if gate.get('passed') is not True or gate.get('stage') != 'offline':
                    raise RuntimeError('Formal allocation exited zero without a passed offline gate record')
                if str(gate.get('slurm_job_id')) != self.args.formal_job:
                    raise ValueError('Offline gate record refers to a different allocation')
                validate_gate(self.args.offline, gate['parent_commit'])
                return gate
            time.sleep(30)

    def await_runs(self, campaign, run_ids, phase):
        while True:
            self.check_cap()
            campaign.inspect()
            entries = [campaign.entry(name) for name in run_ids]
            if any(entry and entry.get('terminal_recorded') and entry.get('run_status') != 'complete' for entry in entries):
                raise RuntimeError('Live-gate allocation failed; no replacement or production admitted')
            self.monitor(phase, [dict(run_id=name, entry=entry) for name, entry in zip(run_ids, entries)])
            if all(entry and entry.get('terminal_recorded') and entry.get('run_status') == 'complete' for entry in entries):
                return
            time.sleep(30)

    def admit(self, campaign, names):
        while True:
            self.check_cap()
            campaign.inspect()
            pending = [name for name in names if campaign.entry(name) is None]
            if not pending:
                return
            if campaign.slots() >= len(pending):
                admitted = [campaign.submit(name) for name in pending]
                campaign.state['status'] = 'live_gate_running'
                campaign.record()
                campaign.monitor(reason='bootstrap live-gate admission')
                if all(admitted):
                    return
            self.monitor('await_live_gate_resources', dict(run_ids=pending, ledger=campaign.state['ledger']))
            time.sleep(30)

    def live_gate(self, offline):
        output = self.args.live_output
        summary = output / 'gate_summary.json'
        if output.exists():
            previous = read(summary, {})
            if previous.get('passed') is not True or previous.get('live_gate_passed') is not True:
                raise RuntimeError('Existing live-gate directory did not pass; preserve it and stop')
            validate_gate(summary, offline['parent_commit'], full=True)
            return summary
        self.monitor('verify_G3_G4_clean_exit', force=True)
        argv = [sys.executable, '-B', str(ROOT / 'code/gate_live.py'), '--offline', str(self.args.offline), '--output', str(output)]
        log_path = ROOT / 'runs/gate_live_G1.out'
        if log_path.exists():
            raise ValueError('Existing live-gate console log is preserved; no automatic overwrite')
        with log_path.open('x') as stream:
            completed = subprocess.run(argv, cwd=str(SOURCE), stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        append(ROOT / 'runs/bootstrap_commands.jsonl', dict(utc=utc(), command=argv, returncode=completed.returncode, console_log=str(log_path)))
        if completed.returncode:
            raise RuntimeError('G3/G4 or extended clean-exit gate failed; production stopped')
        validate_gate(summary, offline['parent_commit'], full=True)
        return summary

    def checkpoint_distances(self):
        output = ROOT / 'runs/checkpoint_distances'
        if output.exists():
            if read(output / 'checkpoint_distances.json', {}).get('status') != 'complete':
                raise RuntimeError('Prior checkpoint-distance output is incomplete; preserve it and stop')
            return
        self.check_cap()
        self.monitor('checkpoint_distances_CPU', force=True)
        argv = [sys.executable, '-B', str(ROOT / 'code/checkpoint_distances.py'), '--output', str(output),
                '--gradient-meta', str(ROOT / 'collect/P0_j00/meta.json')]
        log_path = ROOT / 'runs/checkpoint_distances.out'
        with log_path.open('x') as stream:
            result = subprocess.run(argv, cwd=str(SOURCE), stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        append(ROOT / 'runs/bootstrap_commands.jsonl', dict(utc=utc(), command=argv, returncode=result.returncode,
                                                           cpu_cap_seconds=600, console_log=str(log_path)))
        if result.returncode or read(output / 'checkpoint_distances.json', {}).get('status') != 'complete':
            raise RuntimeError('Checkpoint-distance CPU verification failed')

    def run(self):
        campaign = None
        try:
            offline = self.await_offline()
            options = argparse.Namespace(action='submit', gate=self.args.offline, commit=offline['parent_commit'],
                                         timeout_seconds=self.args.timeout_seconds)
            campaign = Campaign(options)
            # Campaign installs signal handlers; keep both cap checks responsive.
            signal.signal(signal.SIGTERM, self.handle_signal)
            signal.signal(signal.SIGINT, self.handle_signal)
            self.admit(campaign, ['prepare_P0'])
            self.await_runs(campaign, ['prepare_P0'], 'prepare_P0')
            self.admit(campaign, ['P0_j00', 'P0_j01'])
            self.await_runs(campaign, ['P0_j00', 'P0_j01'], 'live_gate_two_independent_groups')
            self.checkpoint_distances()
            full = self.live_gate(offline)
            campaign.args.action = 'run'
            campaign.args.gate = full
            campaign.args.timeout_seconds = int(self.args.timeout_seconds - (time.monotonic() - self.started))
            campaign.started = time.monotonic()
            campaign.source = validate_gate(full, offline['parent_commit'], full=True)
            campaign.state.update(gate=str(full), source_provenance=campaign.source)
            campaign.record()
            signal.signal(signal.SIGTERM, campaign.handle_signal)
            signal.signal(signal.SIGINT, campaign.handle_signal)
            self.record.update(status='production', monitor_status='paused')
            self.monitor('full_campaign_controller_active', dict(full_gate=str(full)), force=True)
            result = campaign.run()
            self.record.update(status=campaign.state['status'], monitor_status='paused', ended_utc=utc())
            self.monitor('campaign_terminal', force=True)
            return result
        except BaseException as error:
            self.record.update(status='blocked', monitor_status='paused', ended_utc=utc(), error=type(error).__name__ + ': ' + str(error))
            self.monitor('stopped', force=True)
            g2 = ROOT / 'runs/gate_G2_G1/gate_summary.json'
            g2_note = ' The revised G2 gate passed.' if read(g2, {}).get('passed') is True else ''
            report_status('blocked', str(error) + '. No further collection is being admitted.' + g2_note,
                          [('Bootstrap failure record', self.path), ('Offline gate', self.args.offline),
                           ('Live gate', self.args.live_output / 'gate_summary.json'),
                           ('Live gate log', ROOT / 'runs/gate_live_G1.out'), ('Revised G2', g2)])
            if campaign is not None:
                campaign.state.update(status='blocked', failure_reason=self.record['error'], monitor_status='paused')
                campaign.record()
                campaign.monitor(reason='bootstrap stopped', terminal=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal-job', required=True)
    parser.add_argument('--offline', type=Path, default=ROOT / 'runs/gate_offline_G1/gate_summary.json')
    parser.add_argument('--live-output', type=Path, default=ROOT / 'runs/gate_live_G1')
    parser.add_argument('--timeout-seconds', type=int, default=86400)
    args = parser.parse_args()
    if not args.formal_job.isdigit() or not 60 <= args.timeout_seconds <= 86400:
        raise ValueError('Require a real formal gate job and an at-most24-hour CPU controller')
    if ROOT not in args.live_output.resolve().parents:
        raise ValueError('Live-gate output must stay in the existing study')
    return Bootstrap(args).run()


if __name__ == '__main__':
    sys.exit(main())
