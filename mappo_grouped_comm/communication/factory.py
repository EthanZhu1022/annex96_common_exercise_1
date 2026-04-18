"""
Factory for building communication modules.

All callers should go through build_communication_module() so they remain
agnostic of the concrete comm class.  To add a new communication method:
  1. Implement a subclass of BaseCommunicationModule (e.g. attention.py).
  2. Add an elif branch here.
  3. Expose in __init__.py if desired.

The caller never depends on CommNet-specific details — this is the single
extension point for adding attention-based, GNN, or parameter-sharing comm.
"""

from __future__ import annotations

from .base    import BaseCommunicationModule
from .none    import NoCommunicationModule
from .commnet import CommNetCommunicationModule
from .powernet import PowerNetCommunicationModule


def build_communication_module(
    comm_method:                 str,
    hidden_dim:                  int,
    comm_hidden_dim:             int   = 64,
    comm_rounds:                 int   = 1,
    comm_use_residual:           bool  = True,
    comm_dropout:                float = 0.0,
    comm_share_within_group_only: bool = True,
    **kwargs,
) -> BaseCommunicationModule:
    """Create and return a communication module by name.

    Parameters
    ----------
    comm_method : str
        'none'    → NoCommunicationModule  (identity, no extra params).
        'commnet' → CommNetCommunicationModule.

        Extension point: add new method strings in the elif ladder below.
        Example future methods: 'attention', 'gnn', 'shared_param'.

    hidden_dim : int
        Dimensionality of actor hidden features passed to the module.

    comm_hidden_dim : int
        Internal projection size for CommNet's message MLP.

    comm_rounds : int
        Number of CommNet communication rounds.

    comm_use_residual : bool
        Use residual connection in CommNet updates.

    comm_dropout : float
        Dropout probability on CommNet projected messages.

    comm_share_within_group_only : bool
        If True, communication is limited to agents within the same group.
        Currently only intra-group communication is implemented.
        Reserved for future inter-group communication support.

    **kwargs
        Absorbs unknown keys so that adding new config fields to Config
        does not require updating every call site.

    Returns
    -------
    BaseCommunicationModule — ready to use, parameters registered.
    """
    comm_method = comm_method.lower().strip()

    if comm_method == "none":
        return NoCommunicationModule()

    elif comm_method == "commnet":
        return CommNetCommunicationModule(
            hidden_dim      = hidden_dim,
            comm_hidden_dim = comm_hidden_dim,
            comm_rounds     = comm_rounds,
            use_residual    = comm_use_residual,
            dropout         = comm_dropout,
        )

    elif comm_method == "powernet":
        return PowerNetCommunicationModule(
            hidden_dim=hidden_dim,
            comm_hidden_dim=comm_hidden_dim,
            comm_rounds=comm_rounds,
            use_residual=comm_use_residual,
            dropout=comm_dropout,
            comm_num_agents=kwargs.get("comm_num_agents", 1),
            comm_topology=kwargs.get("comm_topology", "ring"),
            comm_neighbors=kwargs.get("comm_neighbors", 1),
        )

    # ── Extension points ────────────────────────────────────────────────────
    # elif comm_method == "attention":
    #     from .attention import AttentionCommunicationModule
    #     return AttentionCommunicationModule(hidden_dim=hidden_dim, ...)
    #
    # elif comm_method == "gnn":
    #     from .gnn import GNNCommunicationModule
    #     return GNNCommunicationModule(hidden_dim=hidden_dim, ...)
    # ────────────────────────────────────────────────────────────────────────

    else:
        raise ValueError(
            f"Unknown comm_method='{comm_method}'. "
            f"Valid options: 'none', 'commnet', 'powernet'. "
            f"To add a new method, subclass BaseCommunicationModule and "
            f"add an elif branch in communication/factory.py."
        )
