"""Protect measured trajectory tracking from command/feedback confusion."""
import hashlib
import json

import numpy as np
import pytest

from phantom.sim.kinematics import JOINT_NAMES
from phantom.sim.measured_replay import MeasuredArmVelocityReference, SampledGripperCommands


def recording():
    return {"joint_names": np.array(JOINT_NAMES),
            "native_arm_qd_t": np.array([-.1, .1, .3]),
            "native_arm_qd": np.array([[0., 1., 2., 3., 4., 5.],
                                        [1., 2., 3., 4., 5., 6.],
                                        [2., 3., 4., 5., 6., 7.]])}


def test_interpolates_native_measured_velocity_without_changing_units_or_order():
    ref = MeasuredArmVelocityReference(recording(), mode="dynamics", duration=.25)
    np.testing.assert_allclose(ref.at(0.), np.arange(6)+.5)
    np.testing.assert_allclose(ref.at(.2), np.arange(6)+1.5)
    assert ref.metadata['units'] == 'rad/s'
    for time in [-.01, .26, float('nan')]:
        with pytest.raises(ValueError, match='outside'):
            ref.at(time)


@pytest.mark.parametrize('mode', ['policy', 'replay', 'command_replay', 'contact_probe'])
def test_measured_velocity_treatment_cannot_become_a_policy_override(mode):
    with pytest.raises(ValueError, match='dynamics mode'):
        MeasuredArmVelocityReference(recording(), mode=mode, duration=.2)


def test_rejects_incomplete_mislabeled_or_nonfinite_native_streams():
    data = recording()
    del data['native_arm_qd']
    with pytest.raises(ValueError, match='requires native'):
        MeasuredArmVelocityReference(data, mode='dynamics', duration=.2)
    data = recording()
    data['joint_names'] = data['joint_names'][::-1]
    with pytest.raises(ValueError, match='ordering'):
        MeasuredArmVelocityReference(data, mode='dynamics', duration=.2)
    with pytest.raises(ValueError, match='full replay'):
        MeasuredArmVelocityReference(recording(), mode='dynamics', duration=.4)
    for bad in [np.array([0., 0., .3]), np.array([-.1, .1, float('nan')])]:
        data = recording()
        data['native_arm_qd_t'] = bad
        with pytest.raises(ValueError, match='finite increasing'):
            MeasuredArmVelocityReference(data, mode='dynamics', duration=.2)


@pytest.fixture
def command_sidecar(tmp_path):
    reference = tmp_path/'replay.npz'
    reference.write_bytes(b'episode-specific measured states')
    path = tmp_path/'gripper.json'
    record = {'format': 'phantom_teleop_gripper_commands_v1', 'acquisition': 'teleop',
              'semantics': 'sampled_last_sent_gripper_target',
              'source_episode': '/recordings/ep_teleop_example',
              'source_meta_sha256': 'a' * 64,
              'sampling_limitations': ['Sampled command echo, not exact socket-write timestamps.'],
              'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
              't0_master': 10., 'timestamps_s': [.1, .2, .3],
              'requested_closure': [.1, .6, .2]}
    path.write_text(json.dumps(record))
    return path, reference, record


def load_commands(path, reference, **kwargs):
    return SampledGripperCommands(path, reference_path=reference, reference_t0=10.,
                                  initial_closure=.05, mode=kwargs.get('mode', 'dynamics'))


def test_sampled_requests_are_causal_quantized_and_distinct_from_feedback(command_sidecar):
    path, reference, _ = command_sidecar
    commands = load_commands(path, reference)
    assert commands.at(0.) == .05  # measured initial pose until a command is known
    assert commands.at(.1) == round(.1*255)/255
    assert commands.at(.19999) == commands.at(.1)  # no future-command interpolation
    assert commands.at(.2) == .6
    assert commands.at(1.) == .2  # persistent last request, not a reset
    with pytest.raises(ValueError, match='nonnegative'):
        commands.at(-.001)


def test_sampled_requests_cannot_be_associated_with_another_episode_or_clock(command_sidecar):
    path, reference, record = command_sidecar
    reference.write_bytes(b'a different demonstration')
    with pytest.raises(ValueError, match='different measured replay'):
        load_commands(path, reference)
    record['reference_sha256'] = hashlib.sha256(reference.read_bytes()).hexdigest()
    record['t0_master'] = 11.
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='time origin'):
        load_commands(path, reference)


def test_sampled_requests_require_teleop_and_cannot_override_policy_mode(command_sidecar):
    path, reference, record = command_sidecar
    with pytest.raises(ValueError, match='dynamics mode'):
        load_commands(path, reference, mode='policy')
    record['acquisition'] = 'teacher'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='teleop last-sent'):
        load_commands(path, reference)


@pytest.mark.parametrize('field,value', [
    ('source_episode', None), ('source_episode', '  '),
    ('source_meta_sha256', 'a' * 63), ('source_meta_sha256', 'z' * 64),
    ('sampling_limitations', []), ('sampling_limitations', 'unstructured text'),
    ('sampling_limitations', [' ']),
])
def test_sampled_requests_require_source_identity_and_explicit_sampling_limits(command_sidecar, field, value):
    path, reference, record = command_sidecar
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='source_episode.*source_meta_sha256.*sampling_limitations'):
        load_commands(path, reference)
