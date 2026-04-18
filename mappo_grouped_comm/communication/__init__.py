"""
Communication abstraction for mappo_grouped_comm.

Public API
----------
  BaseCommunicationModule   — abstract base class
  NoCommunicationModule     — identity (no-op, degrades to mappo_grouped)
  CommNetCommunicationModule — CommNet mean-pooling communication
  build_communication_module — factory function

To add a new communication method:
  1. Subclass BaseCommunicationModule in a new file under this package.
  2. Add an elif branch in factory.py::build_communication_module().
  3. Expose it here if desired.
"""

from .base    import BaseCommunicationModule
from .none    import NoCommunicationModule
from .commnet import CommNetCommunicationModule
from .powernet import PowerNetCommunicationModule
from .factory import build_communication_module

__all__ = [
    "BaseCommunicationModule",
    "NoCommunicationModule",
    "CommNetCommunicationModule",
    "PowerNetCommunicationModule",
    "build_communication_module",
]
