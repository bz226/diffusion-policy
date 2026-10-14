"""Write the one approved study plan; does not submit jobs (Python 3.8)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    code = Path(__file__).resolve().parent
    source = code.parent / 'dppo'
    study = project / 'results/stage3/ppo_steps'
    runs = study / 'runs'
    python = project / 'results/repro_halfcheetah/env/bin/python'
    baseline = project / 'results/repro_halfcheetah/dppo'
    original = baseline / 'log/gym-finetune/halfcheetah-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10'
    original0 = original / '2026-10-08_20-01-32_0'
    pretrained = baseline / 'log/gym-pretrain/halfcheetah-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-04-42/checkpoint/state_3000.pt'
    r1 = runs / 'gates/R1/capture_gate.json'
    rerun = runs / 'B_seed0'
    r2 = runs / 'gates/R2'
    s1 = runs / 'gates/S1'
    env = {'PATH': str(python.parent) + ':/usr/local/bin:/usr/bin:/bin',
           'PYTHONPATH': str(source), 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'LD_LIBRARY_PATH': '/sailhome/zbao7/.mujoco/mujoco210/bin:/usr/lib/nvidia',
           'DPPO_DATA_DIR': str(baseline/'data'), 'DPPO_LOG_DIR': str(baseline/'log'),
           'D4RL_SUPPRESS_IMPORT_ERROR': '1', 'OMP_NUM_THREADS': None,
           'MKL_NUM_THREADS': None, 'OPENBLAS_NUM_THREADS': None}
    for key, relative in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('MPLCONFIGDIR','cache/matplotlib'),
                          ('TORCH_HOME','cache/torch'),('TRITON_CACHE_DIR','cache/triton'),
                          ('CUDA_CACHE_PATH','cache/cuda'),('D4RL_DATASET_DIR','cache/d4rl')]:
        path = study/relative
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    def check(path, key, value):
        return {'path': str(path), 'key': key, 'equals': value}
    def measure(mode, out, checkpoint, seed, state=None, itr=35):
        command = [str(python), '-B', str(code/'run_measure.py'), '--source',str(source),
                   '--output',str(out),'--checkpoint',str(checkpoint),'--mode',mode,
                   '--checkpoint-itr',str(itr),'--seed',str(seed),'--commit',args.revision]
        if state:
            command += ['--state',str(state)]
        return command
    jobs = []
    def add(name, kind, command, out, cap, **kwargs):
        job = {'id':name,'kind':kind,'command':command,'cwd':str(source),'env':env,
               'cap_seconds':cap,'source_revision':args.revision,'source_path':str(source),
               'progress_file':str(out/'progress.json'),'progress_keys':['completed'],
               'cpus':40,'memory_gib':64,'gpu_count':1}
        job.update(kwargs)
        jobs.append(job)
        return job
    add('B_seed0','B',measure('train',rerun,pretrained,0),rerun,14400,
        completion_files=[str(rerun/'result.pkl'),str(rerun/'ppo_steps/full_state_139.pt')],
        completion_checks=[check(rerun/'progress.json','completed',140)])
    for p,itr in enumerate((35,70,105)):
        out=runs/('prepare_A%d'%itr)
        add('prepare_A%d'%itr,'prepare',measure('prepare',out,original0/('checkpoint/state_%d.pt'%itr),
                                              2009000+p*10000,itr=itr),out,900,
            completion_files=[str(out/'frozen.pt')],
            completion_checks=[check(out/'prepare.json','critic_gate.passed',True)])
    # Gate readiness does not wait for the rerun to finish.
    add('R2','gate',[str(python),'-B',str(source/'script/replay_ppo_steps.py'),
                    '--run-dir',str(rerun),'--output',str(r2),'--device','cuda:0'],r2,3600,
        ready_files=[str(rerun/'ppo_steps'/name) for name in
                     ('full_state_35.pt','replay_itr36_input.pt','replay_itr36_expected.pt')],
        completion_checks=[check(r2/'replay_result.json','status','passed')])
    add('S1','gate',measure('S1',s1,rerun/'checkpoint/state_35.pt',2190000,
                           rerun/'ppo_steps/full_state_35.pt'),s1,3600,depends_on=['R2'],
        completion_checks=[check(s1/'S1.json','passed',True)])
    # Interleave points across available allocations; no additional batches.
    for j in range(10):
        for p,itr in enumerate((35,70,105)):
            name='A%d_j%02d'%(itr,j);out=runs/name;prepare=runs/('prepare_A%d'%itr)
            add(name,'A',measure('A',out,original0/('checkpoint/state_%d.pt'%itr),
                                2000000+p*10000+j*100,prepare/'frozen.pt',itr),out,1800,
                depends_on=['prepare_A%d'%itr],
                ready_checks=[check(prepare/'prepare.json','critic_gate.passed',True)],
                completion_files=[str(out/'sums_step01.npy'),str(out/'sums_full.npy')],
                completion_checks=[check(out/'meta.json','status','completed'),check(out/'meta.json','completed_batches',50)])
    def analysis_command(out, final):
        cmd=[str(python),'-B',str(code/'analyze.py'),'--groups-root',str(runs),
             '--output-dir',str(out),'--reference-table',str(project/'results/stage3/gradmean/figs_clean/points_table.csv'),
             '--gate-record',str(r1)]
        if final:
            cmd += ['--gate-record',str(r2/'replay_result.json'),'--gate-record',str(s1/'S1.json'),
                    '--rerun-result',str(rerun/'result.pkl')]
            for dirname in ('2026-10-08_20-01-32_0','2026-10-08_22-06-13_1','2026-10-08_22-48-33_2'):
                cmd += ['--original-result',str(original/dirname/'result.pkl')]
        return cmd
    a_ids=[job['id'] for job in jobs if job['kind']=='A']
    out=study/'candidates/part_A'
    add('analysis_A','analysis',analysis_command(out,False),out,3600,gpu_count=0,cpus=4,memory_gib=16,
        depends_on=a_ids,completion_files=[str(out/'analysis_record.json')])
    for j in range(10):
        for p,itr in enumerate((35,70,105)):
            name='C%d_j%02d'%(itr,j);out=runs/name
            add(name,'C',measure('C',out,rerun/('checkpoint/state_%d.pt'%itr),
                                2100000+p*10000+j*100,rerun/('ppo_steps/full_state_%d.pt'%itr),itr),out,5400,
                depends_on=['R2','S1'],ready_files=[str(rerun/('ppo_steps/full_state_%d.pt'%itr))],
                completion_files=[str(out/'sums_step01.npy'),str(out/'sums_G.npy'),str(out/'sums_delta.npy')],
                completion_checks=[check(out/'meta.json','status','completed'),check(out/'meta.json','completed_batches',50)])
    out=study/'candidates/final'
    add('analysis_final','analysis',analysis_command(out,True),out,3600,gpu_count=0,cpus=4,memory_gib=16,
        depends_on=[job['id'] for job in jobs],completion_files=[str(out/'analysis_record.json')],
        completion_checks=[check(out/'analysis_record.json','status','complete')])
    # Start C as soon as its gates and checkpoint are ready, without interrupting A.
    priority = {'B': 0, 'prepare': 1, 'gate': 2, 'analysis': 3, 'C': 4, 'A': 5}
    jobs.sort(key=lambda job: priority[job['kind']])
    plan={'campaign_id':'dppo_ppo_steps_20261014','run_root':str(runs/'campaign'),
          'python':str(python),'max_parallel':9,'budget_gpu_hours':80,'account':'soal','partition':'soal',
          'monitor_seconds':4800,'poll_seconds':30,'controller_cap_seconds':43200,
          'experiment_path':str(study/'experiment.md'),'source_revision':args.revision,
          'global_checks':[check(r1,'status','passed')],'jobs':jobs}
    destination=study/'plan.json'
    with destination.open('x') as stream:
        json.dump(plan,stream,indent=2);stream.write('\n')
    print(json.dumps({'plan':str(destination),'jobs':len(jobs),'maximum_gpu_hours_if_all_hit_caps':
                      sum(job['gpu_count']*job['cap_seconds']/3600 for job in jobs)},indent=2))


if __name__=='__main__':
    main()
