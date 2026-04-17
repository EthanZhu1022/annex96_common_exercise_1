"""
mappo_grouped_comm_weighted - Grouped MAPPO with simple weighted communication.

This package keeps the grouped-encoder / grouped-action-head structure from
`mappo_grouped_comm_v2`, but replaces the global communication block with a
same-group vs other-group weighted message rule:

  h_i        = encoder(obs_i)
  msg_same   = mean(h_same_group)
  msg_other  = mean(h_other_groups)
  new_msg    = alpha * msg_same + beta * msg_other
  out_i      = h_i + MLP(new_msg)
"""
