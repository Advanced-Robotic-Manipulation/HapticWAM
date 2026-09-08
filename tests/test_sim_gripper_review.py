"""Review figures must never borrow future force or zero-fill unknown filters."""
import copy
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gripper_review", ROOT / "tools/sim/make_gripper_validation_review.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def rows():
    pad = {"coverage_mode": "manifold_patch_v2", "filter_labels_match_columns": True,
           "populated_unique_contact_count": 0, "observed_filtered_normal_force_n": 0,
           "per_filter_contacts": [], "ignored_contact_normal_force_n": 0}
    return [{"t": t, "normal_force_n": [0, 0], "per_pad": [copy.deepcopy(pad), copy.deepcopy(pad)],
             "filter_paths": [["/World/Waffle"], ["/World/Waffle"]]} for t in [0., .1]]


def test_causal_freshness_and_no_future_contact():
    index, age, fresh = review.causal_indices(np.array([0., .1, .2]), np.array([-.1, 0., .09, .11, .5]), .1)
    assert index.tolist() == [-1, 0, 0, 1, 2]
    assert fresh.tolist() == [False, True, True, True, False]
    assert np.isinf(age[0])
    assert review.pair_text([123., 456.], False) == "unavailable / stale"


def test_confirmed_no_contact_is_zero_but_unknown_schema_is_unavailable():
    data = rows()
    assert np.array_equal(review.contact_arrays(data)[2], np.zeros((2, 2)))
    data[0]["per_pad"][0]["filter_labels_match_columns"] = False
    data[0]["per_pad"][1]["populated_unique_contact_count"] = 2
    data[0]["per_pad"][1]["observed_filtered_normal_force_n"] = 3.
    packet = review.contact_arrays(data)[2]
    assert np.isnan(packet[0]).all()
    assert np.array_equal(packet[1], [0, 0])


def test_packet_force_is_separate_from_all_environment_contacts():
    data = rows()
    data[0]["normal_force_n"] = [8., 0.]
    pad = data[0]["per_pad"][0]
    pad["per_filter_contacts"] = [
        {"filter_path": "/World/Waffle", "gel_compression_n": 2.},
        {"filter_path": "/World/Mat/Base", "gel_compression_n": 6.}]
    _, active_all, packet, _, _ = review.contact_arrays(data)
    assert active_all[0, 0] == 8.
    assert packet[0, 0] == 2.
    del pad["per_filter_contacts"][0]["gel_compression_n"]
    assert np.isnan(review.contact_arrays(data)[2][0, 0])
