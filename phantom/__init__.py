"""PHANTOM: Predictive Haptic ANTicipation with Occlusion-robust Manipulation.

Package halves:
  - hardware side (config, data, drivers, recording, teleop, deploy, eval):
    cosmos-free, runs anywhere;
  - model side (backbone, model, train, inference): imports cosmos_predict2
    from the path configured in configs/paths.yaml.
"""

__version__ = "0.1.0"
