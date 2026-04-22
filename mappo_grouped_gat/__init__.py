"""
mappo_grouped_gat - Grouped MAPPO with K-means-derived GAT communication.

This package keeps the grouped-encoder / grouped-action-head structure from
`mappo_grouped_comm_v2`, but replaces global mean/attention communication with
Graph Attention Network style masked self-attention:

  h_i -> W h_i
  e_ij = LeakyReLU(a^T [W h_i || W h_j])
  alpha_ij = softmax over j in N_i
  h_i' = concat_heads(sum_j alpha_ij W h_j)

The graph is constructed from K-means building similarity because CE1 does not
provide a physical feeder or geographic adjacency. Same-cluster buildings are
fully connected, and sparse nearest-cluster edges provide weak inter-group
communication.
"""

