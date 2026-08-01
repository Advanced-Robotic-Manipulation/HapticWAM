"""data_collect — unified Echo teleop + data collection for PHANTOM.

The operator application for recording PHANTOM demonstrations: device-rate
Echo teleop, continuous gripper, a latched DM-Tac force safeguard, a rerun
live view, a web control panel, and session-end offload to an external
drive. See docs/data_collect_app.md for the full write-up and how to run it.

Entry point:  python -m phantom.scripts.collect  [--config configs/data_collect.yaml]

It reuses the hardware-verified layers elsewhere in phantom (drivers,
recording, teleop.echo, viz.rerun_logger) and replaces only the application
layer that drove them.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
