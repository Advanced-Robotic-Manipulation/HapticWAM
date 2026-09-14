"""Convex folded-carton envelopes in metres, shared by visuals and collisions.

A profile is an explicit reconstruction hypothesis, not a material deformation
model. Recorded paper printing is mapped onto the same contact envelope.
"""
from __future__ import annotations

import numpy as np


def carton_mesh(size, profile):
    """Return vertices, outward triangles and per-triangle source-face labels.

    Eight vertices form each chamfered section; z fractions span [-.5,.5].
    Section scales are relative to the declared maximum exterior dimensions.
    Concave profiles are rejected rather than silently filled by a convex hull.
    """
    from scipy.spatial import ConvexHull

    size = np.asarray(size, float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Carton size must contain three finite positive dimensions")
    if profile.get('model') != 'chamfered_sections_v1':
        raise ValueError("Unsupported carton profile model")
    chamfer = np.asarray(profile.get('corner_cut_xy_m', [0., 0.]), float)
    sections = np.asarray(profile.get('sections', []), float)
    if (chamfer.shape != (2,) or not np.isfinite(chamfer).all() or np.any(chamfer <= 0)
            or sections.ndim != 2 or sections.shape[1:] != (3,) or len(sections) < 2
            or not np.isfinite(sections).all() or np.any(np.diff(sections[:, 0]) <= 0)
            or not np.isclose(sections[0, 0], -.5, rtol=0, atol=1e-12)
            or not np.isclose(sections[-1, 0], .5, rtol=0, atol=1e-12)
            or np.any(sections[:, 1:] <= 0) or np.any(sections[:, 1:] > 1)
            or not np.allclose(sections[:, 1:].max(axis=0), 1, rtol=0, atol=1e-12)):
        raise ValueError("Carton profile needs positive corner cuts and increasing z/xy-scale sections spanning full bounds")
    vertices = []
    for z, sx, sy in sections:
        hx, hy = size[:2] * [sx, sy] / 2
        cx, cy = chamfer * [sx, sy]
        if cx >= hx or cy >= hy:
            raise ValueError("Corner cuts must be smaller than each section half-width")
        vertices.extend([[hx, hy-cy, z*size[2]], [hx-cx, hy, z*size[2]],
                         [-hx+cx, hy, z*size[2]], [-hx, hy-cy, z*size[2]],
                         [-hx, -hy+cy, z*size[2]], [-hx+cx, -hy, z*size[2]],
                         [hx-cx, -hy, z*size[2]], [hx, -hy+cy, z*size[2]]])
    triangles, labels = [], []
    edge_faces = ['back', 'back', 'left', 'left', 'front', 'front', 'right', 'right']
    for row in range(len(sections)-1):
        for i in range(8):
            a, b = row*8+i, row*8+(i+1)%8
            triangles.extend([[a,b,b+8], [a,b+8,a+8]])
            labels.extend([edge_faces[i]]*2)
    for i in range(1, 7):
        triangles.append([0,i+1,i]);labels.append('bottom')
        offset = (len(sections)-1)*8
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
