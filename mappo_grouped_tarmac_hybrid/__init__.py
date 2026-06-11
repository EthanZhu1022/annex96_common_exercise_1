"""
mappo_grouped_tarmac_hybrid - Grouped MAPPO with hybrid TarMAC communication.

The training and evaluation pipeline matches `mappo_grouped_tarmac`. The only
model change is the communication update:

  local_i = ReLU(W_local h_i)
  context_i = sum_j attention(i, j) W_value h_j
  delta_i = update([local_i ; context_i])
  h_i' = h_i + delta_i
"""

