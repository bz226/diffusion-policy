"""Fail-closed launch validation for the user-approved fixed-batch gate."""
import json
import math
from pathlib import Path


COMPARISONS = ('loss_components', 'actor_gradients', 'critic_gradients',
               'actor_parameters', 'critic_parameters', 'model_state',
               'actor_optimizer_state', 'critic_optimizer_state', 'permutations', 'rng_after_update')
INPUTS = ('batch', 'model_state', 'actor_optimizer_state', 'critic_optimizer_state', 'rng')
RNG_STATES = ('python', 'numpy', 'torch_cpu', 'torch_cuda')


def checked_comparison(compared, label, bitwise):
    if compared.get('passed') is not True:
        raise ValueError('Missing or failed ' + label + ' comparison')
    for key in ('max_abs', 'max_rel'):
        value = compared.get(key)
        if key == 'max_rel' and value == 'inf' and not bitwise:
            continue  # A tolerated absolute GPU error against reference zero can have infinite relative error.
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid reported difference for ' + label)
    if bitwise and (compared.get('bitwise_equal') is not True
                    or compared['max_abs'] != 0 or compared['max_rel'] != 0):
        raise ValueError('Fixed-batch comparison is not bitwise equal: ' + label)


def require_file(path, label):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('Missing saved gate evidence: ' + label)


def validate_gate(path, expected_commit):
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if (report.get('passed') is not True or report.get('status') != 'passed'
            or report.get('source_commit') != expected_commit
            or report.get('reference_commit') != 'cc7234ad7ff39a8f32de3af903606723a16f0648'
            or report.get('deterministic_algorithms_enabled') is not False
            or report.get('deterministic_settings_unchanged') is not True
            or not report.get('numeric_settings_before')
            or report.get('numeric_settings_before') != report.get('numeric_settings_after')):
        raise ValueError('Fixed-batch gate is missing, failed, or belongs to different source/settings')
    capture = report.get('capture', {})
    if (capture.get('training_iteration') != 1 or capture.get('training_environment_transitions') != 80000
            or capture.get('evaluation_environment_transitions') != 80000
            or capture.get('env_step_denoising_pairs') != 200000
            or capture.get('optimizer_steps_performed') != 0
            or set(capture.get('batch_shapes', {})) != {'obs_k', 'chains_k', 'returns_k', 'values_k', 'advantages_k', 'logprobs_k'}):
        raise ValueError('Gate did not use the prescribed saved training batch before its update')
    saved_batch = capture.get('saved_batch_path')
    if not isinstance(saved_batch, str):
        raise ValueError('Missing saved training batch path')
    require_file(saved_batch, 'saved training batch')
    for device in ('cpu', 'gpu'):
        result = report.get(device, {})
        update = result.get('full_update', {})
        if (result.get('passed') is not True or update.get('passed') is not True
                or update.get('epochs') != 5 or update.get('minibatches') != 20
                or update.get('actor_steps') != 20 or update.get('critic_steps') != 20):
            raise ValueError('Fixed-batch gate did not verify the full update on ' + device)
        if device == 'cpu' and result.get('required_bitwise') is not True:
            raise ValueError('CPU gate must require bitwise equality')
        if device == 'gpu' and result.get('tolerance') != {'atol': 1e-7, 'rtol': 1e-5}:
            raise ValueError('GPU gate must record the agreed float32 tolerance')
        for category in ('same_saved_batch', 'same_initial_model_state', 'same_initial_optimizer_states', 'same_initial_rng_state'):
            if update.get(category) is not True:
                raise ValueError('Missing identical-input check: %s %s' % (device, category))
        for category in INPUTS:
            checked_comparison(update.get('initial_comparisons', {}).get(category, {}), device + ' initial ' + category, True)
        for category in COMPARISONS:
            compared = update.get('comparisons', {}).get(category, {})
            checked_comparison(compared, device + ' ' + category,
                               device == 'cpu' or category in ('permutations', 'rng_after_update'))
        minibatches = update.get('minibatch_comparisons', [])
        if len(minibatches) != 20:
            raise ValueError('Missing individual minibatch comparisons on ' + device)
        for index, minibatch in enumerate(minibatches):
            if minibatch.get('minibatch') != index or minibatch.get('epoch') != index // 4:
                raise ValueError('Invalid minibatch comparison schedule on ' + device)
            for category in ('loss_components', 'actor_gradients', 'critic_gradients'):
                checked_comparison(minibatch.get(category, {}), '%s minibatch%d %s' % (device, index, category), device == 'cpu')
        for category in ('sampling', 'rng_isolation'):
            if result.get(category, {}).get('passed') is not True:
                raise ValueError('Missing or failed %s %s check' % (device, category))
        sampling = result['sampling']
        if sampling.get('returned_chain_states') != 11 or sampling.get('observed_transition_states') != 20:
            raise ValueError('Sampling did not compare the whole denoising chain on ' + device)
        for category in ('chain', 'trajectories', 'all_20_transition_states'):
            checked_comparison(sampling.get('comparisons', {}).get(category, {}), device + ' sampling ' + category, device == 'cpu')
        isolation = result['rng_isolation']
        if (set(isolation.get('states', {})) != set(RNG_STATES)
                or any(isolation['states'][key] is not True for key in RNG_STATES)
                or isolation.get('cuda_generator_count', 0) < 1):
            raise ValueError('Diagnostics did not preserve all required RNG states on ' + device)
        for filename in ('pristine_update_evidence.pt', 'edited_update_evidence.pt',
                         'sampling_evidence.pt', 'diagnostic_rng_evidence.pt'):
            require_file(path.parent / device / filename, device + '/' + filename)
    return {'path': str(path), 'source_commit': report['source_commit'],
            'passed': True, 'criterion': 'full_update_fixed_batch_cpu_bitwise_gpu_float_tolerance',
            'trajectory_comparison_gate': False}
