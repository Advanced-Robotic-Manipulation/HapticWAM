"""Convex folded-carton envelopes in metres, shared by visuals and collisions.

A profile is an explicit reconstruction hypothesis, not a material deformation
model. Recorded paper printing is mapped onto the same contact envelope.
"""
from __future__ import annotations

import numpy as np


def carton_mesh(size, profile):
    """Return vertices, outward triangles and per-triangle source-face labels.

    Four or eight vertices form each rectangular or chamfered section;
    z fractions span [-.5,.5]. A side-view octagon can use rectangular
    sections with a wider constant-width body and inset top/bottom ends.
    It does not require cutting the horizontal cross-section corners.
    Section scales are relative to the declared maximum exterior dimensions.
    Concave profiles are rejected rather than silently filled by a convex hull.
    """
    from scipy.spatial import ConvexHull

    size = np.asarray(size, float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Carton size must contain three finite positive dimensions")
    model = profile.get('model')
    if model not in ('chamfered_sections_v1', 'rectangular_sections_v1'):
        raise ValueError("Unsupported carton profile model")
    chamfer = np.asarray(profile.get('corner_cut_xy_m', [0., 0.]), float)
    rectangular = model == 'rectangular_sections_v1'
    if (chamfer.shape != (2,) or not np.isfinite(chamfer).all()
            or (rectangular and np.any(chamfer != 0))
            or (not rectangular and np.any(chamfer <= 0))):
        raise ValueError("Rectangular sections require zero XY corner cuts; chamfered sections require positive cuts")
    sections = np.asarray(profile.get('sections', []), float)
    if (sections.ndim != 2 or sections.shape[1:] != (3,) or len(sections) < 2
            or not np.isfinite(sections).all() or np.any(np.diff(sections[:, 0]) <= 0)
            or not np.isclose(sections[0, 0], -.5, rtol=0, atol=1e-12)
            or not np.isclose(sections[-1, 0], .5, rtol=0, atol=1e-12)
            or np.any(sections[:, 1:] <= 0) or np.any(sections[:, 1:] > 1)
            or not np.allclose(sections[:, 1:].max(axis=0), 1, rtol=0, atol=1e-12)):
        raise ValueError("Carton profile needs increasing z/xy-scale sections spanning full bounds")
    vertices = []
    for z, sx, sy in sections:
        hx, hy = size[:2] * [sx, sy] / 2
        cx, cy = chamfer * [sx, sy]
        if cx >= hx or cy >= hy:
            raise ValueError("Corner cuts must be smaller than each section half-width")
        if rectangular:
            vertices.extend([[hx, hy, z*size[2]], [-hx, hy, z*size[2]],
                             [-hx, -hy, z*size[2]], [hx, -hy, z*size[2]]])
            continue
        vertices.extend([[hx, hy-cy, z*size[2]], [hx-cx, hy, z*size[2]],
                         [-hx+cx, hy, z*size[2]], [-hx, hy-cy, z*size[2]],
                         [-hx, -hy+cy, z*size[2]], [-hx+cx, -hy, z*size[2]],
                         [hx-cx, -hy, z*size[2]], [hx, -hy+cy, z*size[2]]])
    triangles, labels = [], []
    edge_faces = (['back', 'left', 'front', 'right'] if rectangular else
                  ['back', 'back', 'left', 'left', 'front', 'front', 'right', 'right'])
    n = len(edge_faces)
    for row in range(len(sections)-1):
        for i in range(n):
            a, b = row*n+i, row*n+(i+1)%n
            triangles.extend([[a,b,b+n], [a,b+n,a+n]])
            labels.extend([edge_faces[i]]*2)
    for i in range(1, n-1):
        triangles.append([0,i+1,i]);labels.append('bottom')
        offset = (len(sections)-1)*n
        triangles.append([offset,offset+i,offset+i+1]);labels.append('top')
    vertices, triangles = np.asarray(vertices), np.asarray(triangles, dtype=np.int32)
    hull_volume = ConvexHull(vertices).volume
    tri = vertices[triangles]
    signed_volume = np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
    if not np.isclose(signed_volume,hull_volume,rtol=1e-8,atol=1e-14):
        raise ValueError("Carton section envelope must be convex; refusing collision hull that fills a concavity")
    return vertices, triangles, np.asarray(labels)


def carton_envelope_volume(size, profile):
    vertices, faces, _ = carton_mesh(size, profile)
    tri = vertices[faces]
    return float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)


def carton_face_uv(points, size, face):
    """Match historical printed face orientations without floating quads."""
    p = np.asarray(points, float)/np.asarray(size)
    axes = {
        'top': (p[:,0]+.5, p[:,1]+.5),
        'bottom': (p[:,0]+.5, .5-p[:,1]),
        'front': (p[:,0]+.5, p[:,2]+.5),
        'back': (.5-p[:,0], p[:,2]+.5),
        'left': (.5-p[:,1], p[:,2]+.5),
        'right': (p[:,1]+.5, p[:,2]+.5),
    }
    if face not in axes:
        raise ValueError(f"Unknown carton face {face}")
    return np.c_[*axes[face]]
