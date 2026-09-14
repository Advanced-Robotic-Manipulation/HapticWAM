"""The egg supports must have physical cavities and exact contact labels."""

import numpy as np
import pytest

from phantom.sim.egg_fixture import cup_shell_mesh, tray_web_mesh, support_pad_spec, support_pad_mesh
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


def _vertical_intersections(vertices, faces, point):
    hits = []
    for triangle in vertices[faces]:
        a, b, c = triangle
        matrix = np.stack((b[:2]-a[:2], c[:2]-a[:2]), axis=1)
        if abs(np.linalg.det(matrix)) < 1e-12:
            continue
        u, v = np.linalg.solve(matrix, np.asarray(point)-a[:2])
        if u >= -1e-10 and v >= -1e-10 and u+v <= 1+1e-10:
            hits.append(a[2] + u*(b[2]-a[2]) + v*(c[2]-a[2]))
    return hits


def test_rounded_tray_has_true_diamond_apertures_without_filling_cups():
    pitch = np.array([.055, .048])
    parts = []
    for row in range(5):
        for column in range(2):
            vertices, faces = tray_web_mesh(column, row, 2, 5, pitch, .0235, .032, .0015)
            edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
            _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
            assert np.all(counts == 2)
            triangles = vertices[faces]
            volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()/6
            assert volume > 0
            assert not _vertical_intersections(vertices, faces, [.001, .001])
            center = (np.array([column, row]) - [.5, 2])*pitch
            parts.append((vertices + np.r_[center, 0.], faces))

    def hits(point):
        return [z for vertices, faces in parts for z in _vertical_intersections(vertices, faces, point)]

    for row in range(4):
        center = np.array([0., (row-1.5)*pitch[1]])
        # A real opening, including off-axis interior samples, must pass right
        # through the material. Sampling only a mesh vertex could miss a cap.
        for delta in ([.0003, .0002], [.002, .002], [-.002, -.002], [.0001, .006]):
            assert not hits(center + delta)
        # The narrow bridges beside an opening retain physical support.
        assert max(hits(center + [.014, .0003])) == pytest.approx(.032)
    assert not hits([.054, .119])  # rounded global corner, previously rectangular
    assert max(hits([.054, .010])) == pytest.approx(.032)  # straight outer rim


def test_tray_rounding_or_aperture_must_not_change_bowl_support_geometry():
    with pytest.raises(ValueError, match="intersects a cup rim"):
        tray_web_mesh(0, 1, 2, 5, [.055, .048], .0235, .032, .0015,
                      opening_half_xy=[.025, .025])
    with pytest.raises(ValueError, match="intersects a cup rim"):
        tray_web_mesh(0, 0, 2, 5, [.055, .048], .0235, .032, .0015,
                      corner_radius=.045)


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


def _support_cfg():
    return {
        "columns": 2, "rows": 5, "cell_pitch_xy": [.055, .048],
        "cup_bottom_radius_m": .012, "wall_thickness_m": .0015,
        "height_m": .032,
        "support_pad": {"cell": [1, 3], "footprint_xy_m": [.020, .018],
                        "thickness_m": .020, "max_compression_m": .005, "stiffness_n_m": 300.,
                        "damping_n_s_m": 5.},
    }


def test_support_pad_is_local_to_one_cavity_with_known_rest_surface():
    cfg = _support_cfg()
    spec = support_pad_spec(cfg)
    assert spec["cell"] == [1, 3]
    assert spec["center_local_m"] == pytest.approx([.0275, .048, .0115])
    assert spec["uncompressed_top_local_z_m"] == pytest.approx(.0215)
    assert spec["rigid_floor_local_z_m"] == pytest.approx(.0015)
    assert spec["bottoming_top_local_z_m"] == pytest.approx(.0165)
    assert spec["uncompressed_top_local_z_m"] - spec["rigid_floor_local_z_m"] == pytest.approx(.020)
    # Sample the complete footprint against every side of the exact48-gon floor.
    angle = np.arange(1024)*2*np.pi/1024
    ellipse = np.c_[np.cos(angle)*.010, np.sin(angle)*.009]
    normals = np.c_[np.cos((np.arange(48)+.5)*2*np.pi/48),
                    np.sin((np.arange(48)+.5)*2*np.pi/48)]
    assert np.max(ellipse @ normals.T) < .012*np.cos(np.pi/48)
    path = "/World/EggFixture/Tray/Cell_1_3/SupportPad"
    assert _validate_paths(["/World/Robot/left_pad"], [path])[1] == [path]
    assert support_pad_spec({}) is None
    assert support_pad_spec({"support_pad": {"enabled": False}}) is None


@pytest.mark.parametrize("field,value,match", [
    ("cell", [1, 5], "valid"), ("cell", [.5, 3], "valid"),
    ("cell", [1, float("nan")], "valid"),
    ("footprint_xy_m", [.024, .018], "polygonal"),
    ("footprint_xy_m", [.020, 0.], "polygonal"),
    ("thickness_m", .031, "rim"), ("thickness_m", -.001, "positive"),
    ("max_compression_m", .010, "half"),
    ("stiffness_n_m", 0., "positive"),
    ("damping_n_s_m", float("inf"), "positive"),
    ("color", [1., -.1, .5], "color"),
    ("model", "attached_egg", "Unsupported"),
])
def test_invalid_support_pad_cannot_fill_cavity_or_hide_uncertain_physics(field, value, match):
    cfg = _support_cfg()
    cfg["support_pad"][field] = value
    with pytest.raises(ValueError, match=match):
        support_pad_spec(cfg)


@pytest.mark.parametrize("depth", [0., .002])
def test_opt_in_parabolic_pad_has_real_cavity_shared_watertight_geometry(depth):
    cfg = _support_cfg()
    cfg['support_pad']['thickness_m'] = .022
    cfg['support_pad']['top_profile'] = {'model': 'parabolic_mesh_v1', 'central_depth_m': depth}
    spec = support_pad_spec(cfg)
    vertices, faces = support_pad_mesh(spec)
    assert spec['center_clearance_to_stop_m'] == pytest.approx(.005-depth)
    assert spec['uncompressed_center_top_local_z_m'] == pytest.approx(.0235-depth)
    edges = {}
    for face in faces:
        for a,b in zip(face,np.roll(face,-1)):
            edges.setdefault(tuple(sorted((a,b))),[]).append((a,b))
    assert all(len(e)==2 and e[0]==e[1][::-1] for e in edges.values())
    assert max(_vertical_intersections(vertices,faces,[0.,0.])) == pytest.approx(.011-depth)
    assert max(_vertical_intersections(vertices,faces,[.005,0.])) == pytest.approx(.011-.75*depth)
    triangles=vertices[faces]
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    assert np.linalg.norm(cross,axis=1).min()>1e-10
    volume=np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2])).sum()/6
    assert volume>0
    assert np.all(vertices[:,:2].min(0)==pytest.approx([-.010,-.009]))
    assert np.all(vertices[:,:2].max(0)==pytest.approx([.010,.009]))
    if depth:
        # Convexification must not be used: it would fill this physical2mm dip.
        rim=vertices[1+7*64:1+8*64]
        assert np.allclose(rim[:,2]-vertices[0,2],depth)


@pytest.mark.parametrize("profile,match", [
    ({'model':'parabolic_mesh_v1','central_depth_m':.005},'internal stop'),
    ({'model':'parabolic_mesh_v1','central_depth_m':-.001},'nonnegative'),
    ({'model':'parabolic_mesh_v1','central_depth_m':float('nan')},'nonnegative'),
    ({'model':'parabolic_mesh_v1','segments':7},'segments'),
    ({'model':'parabolic_mesh_v1','segments':64.5},'segments'),
    ({'model':'parabolic_mesh_v1','radial_bands':0},'radial_bands'),
    ({'model':'parabolic_mesh_v1','adhesion':True},'parameter'),
    ({'model':'flat_cylinder_v1','central_depth_m':.002},'no shape'),
    ({'model':'convex_hull'},'Unsupported'),
    ('parabolic_mesh_v1','mapping'),
])
def test_pad_profile_cannot_silently_fill_cavity_or_hide_bottoming(profile,match):
    cfg=_support_cfg();cfg['support_pad']['top_profile']=profile
    with pytest.raises(ValueError,match=match):
        support_pad_spec(cfg)


def test_absent_profile_preserves_old_flat_pad_without_mesh_conversion():
    spec=support_pad_spec(_support_cfg())
    assert spec['top_profile']['model']=='flat_cylinder_v1'
    assert spec['uncompressed_center_top_local_z_m']==spec['uncompressed_top_local_z_m']
    assert spec['center_clearance_to_stop_m']==spec['max_compression_m']
    with pytest.raises(ValueError,match='requires'):
        support_pad_mesh(spec)
