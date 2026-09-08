# Supplier W2L geometry

Rebuild from the hash-bound local supplier sources:

```bash
uv run --no-project --with cadquery==2.8.0 python tools/sim/build_w2l_assets.py
```

`geometry.json` is the scene integration contract. Every mesh uses metres in a
common sensor frame: X points away from the contact surface, Y across its width,
and Z toward the distal tip. The origin is the CAD wear-layer front-face centroid
in Y/Z and the middle of that layer's depth. The left contact face is at negative
X; rotate the right-side geometry 180 degrees about Z inside its aligned pad link.

The named supplier wear layer is distinct from the transparent backing, rigid
cover, housing, cable and mounting parts. Visual meshes preserve supplier detail.
Collision meshes use separately clipped component hulls; the cover is partitioned
around the optical-area boundary so one hull does not seal its aperture. Hulls
retain CAD vertices, with at most 240 vertices and 0.025 mm support-plane reduction
error. This is in addition to the recorded visual tessellation tolerance.

The supplier reports 131 g for a complete sensor. Component masses and inertias
use explicitly documented CAD-volume allocation; the reference bracket masses
use an assumed density. Gel stiffness, friction and the optical sensing map are
unmeasured. The 27 × 36 mm optical window's centred placement and an elliptical
filter remain separate hypotheses from the physical wear-layer CAD outline.

The sensor/base/reference-adapter holes and mating planes register without
substantial CAD overlap. However, the supplied generic Robotiq adapter's hole
pattern does **not** match the pinned 2F85 stock pad-face pattern. The actual
finger-to-adapter transform and installed gripper variant remain unresolved.
Do not describe the reference assembly as a verified replica of the mounted rig.

`PROVENANCE.json` binds the supplier inputs, generator, named CAD parts, exports
and validation details. Source recordings, the live robot and frozen V10 scene
are not modified by this asset generator.
