#!/usr/bin/env python
"""Lightweight, bounded gate monitor; execute the approved queue only on success."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/soalnas/share/data/zbao7/diffusion_policy/results/stage3/critic')

def utc():return datetime.now(timezone.utc).isoformat()

def atomic(path,value):
    tmp=path.with_name('.'+path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate-job',required=True)
    parser.add_argument('--cap-seconds',type=int,default=172800)
    args=parser.parse_args()
    if not args.gate_job.isdigit() or not 0<args.cap_seconds<=172800:raise ValueError('Invalid bounded gate continuation')
    record={'gate_job_id':args.gate_job,'started_utc':utc(),'status':'waiting_for_gate','monitor_status':'active','monitor_interval_seconds':1800,'cap_seconds':args.cap_seconds,'pid':os.getpid(),'command':[sys.executable]+sys.argv}
    started=time.monotonic();next_monitor=0
    path=ROOT/'runs/gate_continuation.json'
    if path.exists():raise ValueError('An existing gate continuation record must not be overwritten')
    def interrupt(signum,frame):raise InterruptedError('Gate monitor interrupted by signal%d'%signum)
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        while time.monotonic()-started<min(args.cap_seconds,7200+3600*12):
            command=['sacct','--allocations','--noheader','--parsable2','--jobs='+args.gate_job,'--format=JobIDRaw,State,ExitCode,ElapsedRaw']
            result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=120,check=True)
            rows=[r.split('|') for r in result.stdout.splitlines() if r.startswith(args.gate_job+'|')]
            state=rows[0][1].split()[0].rstrip('+') if rows else None
            gatepath=ROOT/'runs/gate/gate_summary.json'
            try:gate=json.loads(gatepath.read_text()) if gatepath.exists() else {}
            except json.JSONDecodeError:gate={'status':'write_in_progress'}
            record.update(updated_utc=utc(),scheduler_state=state,scheduler_command=command,
                          scheduler_output=result.stdout,gate_status=gate.get('status'),gate_phase=gate.get('active_phase'))
            atomic(path,record)
            if time.monotonic()>=next_monitor:
                with (ROOT/'runs/gate_monitor_history.jsonl').open('a') as out:out.write(json.dumps(record)+'\n')
                next_monitor=time.monotonic()+1800
            if state in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'):
                if state!='COMPLETED' or rows[0][2]!='0:0' or gate.get('passed') is not True:
                    raise RuntimeError('C gate failed or did not exit cleanly; no calibration or training admitted')
                ready=ROOT/'provenance/runtime_ready.json'
                if not ready.exists() or json.loads(ready.read_text()).get('ready') is not True:
                    record.update(status='waiting_for_reviewed_queue',gate_passed=True);atomic(path,record)
                    time.sleep(15);continue
                remaining=int(args.cap_seconds-(time.monotonic()-started))
                if remaining<60:raise RuntimeError('Continuation lifetime exhausted before queue')
                argv=[sys.executable,'-B',str(ROOT/'code/campaign.py'),'run','--controller-cap-seconds',str(remaining)]
                record.update(status='handed_to_campaign',monitor_status='campaign_active',gate_passed=True,
                              ended_utc=utc(),queue_command=argv)
                atomic(path,record)
                os.execv(sys.executable,argv)
            time.sleep(30)
        raise RuntimeError('Gate monitor cap reached; no new jobs admitted')
    except BaseException as error:
        record.update(status='blocked',monitor_status='paused',ended_utc=utc(),error=type(error).__name__+': '+str(error))
        atomic(path,record)
        report=ROOT/'REPORT.md'
        body=report.read_text() if report.exists() else '# Addendum C\n'
        body+='\n\nGate continuation stopped: '+record['error']+'. See [gate evidence](runs/gate/gate_summary.json) and [monitor record](runs/gate_continuation.json). No production was admitted by this continuation.\n'
        report.write_text(body)
        raise

if __name__=='__main__':main()
