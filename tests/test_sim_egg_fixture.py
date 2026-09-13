"""The egg supports must have physical cavities and exact contact labels."""

import numpy as np
import pytest

from phantom.sim.egg_fixture import cup_shell_mesh
from tools.sim.object_support import PacketSupportViews
from tools.sim.robot_environment_contacts import _validate_paths


def test_cup_is_closed_material_with_positive_volume_and_open_cavity():
    vertices, faces = cup_shell_mesh(.012, .022, .032, .0015)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert np.all(counts == 2)
    triangles = vertices[faces]
    volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
    assert 0 < volume < np.pi * .024**2 * .032
    # The only triangles crossing the cavity's central axis are its bottom
    # and underside fans. A filled convex cap would appear at the rim height.
    cross_axis = []
    p = np.array([.001, .001])
    for triangle in triangles:
        a, b, c = triangle
        matrix = np.stack((b[:2] - a[:2], c[:2] - a[:2]), axis=1)
        if abs(np.linalg.det(matrix)) < 1e-12:
            continue
        u, v = np.linalg.solve(matrix, p - a[:2])
        if u >= 0 and v >= 0 and u + v <= 1:
            cross_axis.append(a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2]))
    assert max(cross_axis) == pytest.approx(.0015)


@pytest.mark.parametrize("object_path", ["/World/Egg", "/World/Carton"])
def test_new_object_and_fixture_contact_paths_remain_exact(object_path):
    fixture = "/World/EggFixture/Tray/Cell_1_2/Bowl"
    actor = "/World/Robot/left_pad"
    assert _validate_paths([actor], [object_path, fixture]) == ([actor], [object_path, fixture])
    options = {}

    def capture(**kwargs):
        options.update(kwargs)

    PacketSupportViews(object_path, [fixture], [actor], rigid_prim_cls=capture)
    assert options["prim_paths_expr"] == object_path
    assert options["contact_filter_prim_paths_expr"] == [fixture, actor]
    for bad in ("/World/EggFixture/.*", "/World/EggFixture/Tray/*", "/World/EggFixture/../Robot/left_pad"):
        with pytest.raises(ValueError):
            _validate_paths([actor], [bad])
        with pytest.raises(ValueError):
            PacketSupportViews(object_path, [bad], [actor], rigid_prim_cls=capture)
