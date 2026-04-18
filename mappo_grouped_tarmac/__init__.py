"""
mappo_grouped_tarmac - Grouped MAPPO with TarMAC-style global communication.

This package keeps the grouped-encoder / grouped-action-head structure from
`mappo_grouped_comm_v2`, but replaces CommNet with TarMAC-style targeted
communication:

  h_i -> q_i, k_i, v_i
  a_ij = softmax_j(q_i^T k_j)
  c_i  = sum_j a_ij v_j
  h_i' = tanh(W[h_i ; c_i])

Multiple communication rounds are supported before action selection.
"""

