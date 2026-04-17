"""
mappo_grouped_dial - Grouped MAPPO with DIAL-style global communication.

This package keeps the grouped-encoder / grouped-action-head structure from
`mappo_grouped_comm_v2`, but replaces CommNet with a differentiable
communication bottleneck:

  h_i -> message head -> DRU/noisy channel -> aggregate other agents' messages
      -> message decoder -> residual update -> action head

Training uses continuous messages with noise, while evaluation discretises the
messages, following the DIAL centralised-training / decentralised-execution
pattern in simplified form.
"""

