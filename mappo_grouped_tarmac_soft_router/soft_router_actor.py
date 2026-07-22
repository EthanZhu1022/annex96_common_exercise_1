"""
Global communication actor controller with state-conditioned soft routing.

The controller supports two encoder paths:

  legacy:       obs(group_k) -> encoder_k -> global communication
  three-stage:  obs(all agents) -> shared encoder -> global communication

After communication, each building receives a router distribution over all
grouped action heads. The shared path guarantees that every action head sees
one common latent coordinate system. For continuous actions the resulting
policy is a mixture of diagonal Gaussian actor heads.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.algorithms.utils.util import check


class SoftRouterGlobalCommActorController(nn.Module):
    """Run global communication and softly route each agent over action heads."""

    def __init__(
        self,
        actors: Sequence[nn.Module],
        comm: nn.Module,
        group_indices: Sequence[Sequence[int]],
        router_hidden_dim: int = 64,
        router_temperature: float = 1.0,
        router_entropy_scale: float = 0.1,
        router_alpha: float = 0.0,
        router_obs_indices: Optional[Mapping[str, int]] = None,
        building_capacity_norm: Optional[Sequence[float]] = None,
        building_hvac_norm: Optional[Sequence[float]] = None,
        use_capacity_router_features: bool = True,
        deterministic_hard_routing: bool = True,
        shared_encoder: bool = False,
        full_expert_routing: bool = False,
    ) -> None:
        super().__init__()
        self.group_actors = nn.ModuleList(list(actors))
        self.comm = comm
        self.group_indices = [list(map(int, idx)) for idx in group_indices]
        self.group_sizes = [len(idx) for idx in self.group_indices]
        self.n_agents_total = sum(self.group_sizes)
        self.n_experts = len(self.group_actors)

        ref_actor = self.group_actors[0]
        self.hidden_size = ref_actor.hidden_size
        self.tpdv = ref_actor.tpdv
        self._use_policy_active_masks = ref_actor._use_policy_active_masks
        self.router_temperature = float(router_temperature)
        self.router_entropy_scale = float(router_entropy_scale)
        self.use_capacity_router_features = bool(use_capacity_router_features)
        self.deterministic_hard_routing = bool(deterministic_hard_routing)
        # In three-stage training every agent must live in the same latent
        # coordinate system.  The legacy controller selected encoder_k from a
        # fixed cluster and then allowed the router to select any head_j.  That
        # silently fed head_j features produced by an unrelated encoder_k.
        # shared_encoder=True makes actor 0's encoder the common trunk while
        # retaining all K action heads as experts.
        self.shared_encoder = bool(shared_encoder)
        self.full_expert_routing = bool(full_expert_routing)
        if self.shared_encoder and self.full_expert_routing:
            raise ValueError("shared_encoder and full_expert_routing are mutually exclusive.")
        self.router_obs_indices = dict(router_obs_indices or {})

        action_layer = ref_actor.act
        if not getattr(action_layer, "mujoco_box", False):
            raise ValueError("SoftRouterGlobalCommActorController currently supports Box actions only.")

        if building_capacity_norm is None:
            building_capacity_norm = [0.0] * self.n_agents_total
        if building_hvac_norm is None:
            building_hvac_norm = [0.0] * self.n_agents_total
        capacity_tensor = torch.as_tensor(building_capacity_norm, dtype=torch.float32).view(self.n_agents_total, 1)
        hvac_tensor = torch.as_tensor(building_hvac_norm, dtype=torch.float32).view(self.n_agents_total, 1)
        self.register_buffer("building_capacity_norm", capacity_tensor)
        self.register_buffer("building_hvac_norm", hvac_tensor)

        router_extra_dim = 0
        if self.use_capacity_router_features:
            # cap, hvac, soc, charge headroom, discharge headroom, current nsl, current heating
            router_extra_dim = 7
        self.router_extra_dim = router_extra_dim

        self.router = nn.Sequential(
            nn.Linear(self.hidden_size + self.router_extra_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, self.n_experts),
        )
        final_layer = self.router[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

        static_group_ids = torch.empty(self.n_agents_total, dtype=torch.long)
        for group_id, idx_k in enumerate(self.group_indices):
            for agent_id in idx_k:
                static_group_ids[agent_id] = group_id
        static_prior = F.one_hot(static_group_ids, num_classes=self.n_experts).float()
        self.register_buffer("static_group_ids", static_group_ids)
        self.register_buffer("static_prior", static_prior)
        self.register_buffer("router_alpha", torch.tensor(float(router_alpha), dtype=torch.float32))

        self.last_gate_stats: Dict[str, float] = {}
        self.last_gate_balance_loss: Optional[torch.Tensor] = None
        self.last_dynamic_gate: Optional[torch.Tensor] = None

    def set_router_alpha(self, alpha: float) -> None:
        """Set the dynamic-router mixing strength in [0, 1]."""
        alpha = max(0.0, min(1.0, float(alpha)))
        self.router_alpha.fill_(alpha)

    def _encode_group(
        self,
        actor: nn.Module,
        obs,
        rnn_states,
        masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        features = actor.base(obs)
        if actor._use_naive_recurrent_policy or actor._use_recurrent_policy:
            features, rnn_states = actor.rnn(features, rnn_states, masks)

        return features, rnn_states

    def _apply_global_comm(self, global_features: torch.Tensor) -> torch.Tensor:
        if getattr(self.comm, "is_identity", False):
            return global_features
        return self.comm(global_features)

    def _obs_feature(self, obs: torch.Tensor, name: str) -> torch.Tensor:
        idx = self.router_obs_indices.get(name)
        if idx is None or idx < 0 or idx >= obs.shape[-1]:
            return torch.zeros(*obs.shape[:-1], 1, dtype=obs.dtype, device=obs.device)
        return obs[..., idx : idx + 1]

    def _capacity_router_features(self, obs: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_capacity_router_features:
            return None

        cap = self.building_capacity_norm.to(device=obs.device, dtype=obs.dtype)
        hvac = self.building_hvac_norm.to(device=obs.device, dtype=obs.dtype)
        while cap.dim() < obs.dim():
            cap = cap.unsqueeze(0)
            hvac = hvac.unsqueeze(0)
        cap = cap.expand(*obs.shape[:-1], 1)
        hvac = hvac.expand(*obs.shape[:-1], 1)

        soc = self._obs_feature(obs, "electrical_storage_soc").clamp(0.0, 1.0)
        available_charge = cap * (1.0 - soc)
        available_discharge = cap * soc
        nsl = self._obs_feature(obs, "non_shiftable_load")
        heating = self._obs_feature(obs, "heating_demand")

        return torch.cat([cap, hvac, soc, available_charge, available_discharge, nsl, heating], dim=-1)

    def _compute_gates(
        self,
        global_features: torch.Tensor,
        router_extra_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return final gates with shape (..., n_agents, n_experts)."""
        if global_features.shape[-2] != self.n_agents_total:
            raise ValueError(
                f"Expected {self.n_agents_total} agents in global_features, "
                f"got {global_features.shape[-2]}."
            )

        router_inputs = global_features
        if self.router_extra_dim > 0:
            if router_extra_features is None:
                router_extra_features = torch.zeros(
                    *global_features.shape[:-1],
                    self.router_extra_dim,
                    dtype=global_features.dtype,
                    device=global_features.device,
                )
            router_inputs = torch.cat([global_features, router_extra_features], dim=-1)

        temperature = max(self.router_temperature, 1e-6)
        logits = self.router(router_inputs) / temperature
        dynamic_gate = torch.softmax(logits, dim=-1)
        self.last_dynamic_gate = dynamic_gate

        alpha = self.router_alpha.to(device=dynamic_gate.device, dtype=dynamic_gate.dtype)
        static_prior = self.static_prior.to(device=dynamic_gate.device, dtype=dynamic_gate.dtype)
        while static_prior.dim() < dynamic_gate.dim():
            static_prior = static_prior.unsqueeze(0)
        gates = (1.0 - alpha) * static_prior + alpha * dynamic_gate
        gates = gates.clamp_min(1e-8)
        gates = gates / gates.sum(dim=-1, keepdim=True)
        reduce_dims = tuple(range(gates.dim() - 1))
        mean_usage = gates.mean(dim=reduce_dims)
        target_usage = self.static_prior.to(
            device=gates.device, dtype=gates.dtype
        ).mean(dim=0)
        self.last_gate_balance_loss = (mean_usage - target_usage).square().mean()
        self._update_gate_stats(gates)
        return gates

    def _update_gate_stats(self, gates: torch.Tensor) -> None:
        with torch.no_grad():
            flat_gates = gates.reshape(-1, self.n_agents_total, self.n_experts)
            entropy = -(flat_gates * flat_gates.clamp_min(1e-8).log()).sum(dim=-1)
            max_prob = flat_gates.max(dim=-1).values
            static_ids = self.static_group_ids.to(flat_gates.device)
            static_weight = flat_gates.gather(
                -1,
                static_ids.view(1, self.n_agents_total, 1).expand(flat_gates.shape[0], -1, -1),
            ).squeeze(-1)
            expert_means = flat_gates.mean(dim=(0, 1))

            stats: Dict[str, float] = {
                "router_alpha": float(self.router_alpha.item()),
                "gate_entropy": float(entropy.mean().item()),
                "gate_max": float(max_prob.mean().item()),
                "gate_static_weight": float(static_weight.mean().item()),
            }
            for k, value in enumerate(expert_means):
                stats[f"gate_expert_{k}_mean"] = float(value.item())
            self.last_gate_stats = stats

    def _component_normals(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means: List[torch.Tensor] = []
        stds: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        for actor in self.group_actors:
            action_layer = actor.act
            if not getattr(action_layer, "mujoco_box", False):
                raise ValueError("Soft routing currently supports Box actions only.")
            dist = action_layer.action_out(features)
            means.append(dist.mean)
            stds.append(dist.stddev)
            entropies.append(dist.entropy())
        return (
            torch.stack(means, dim=1),
            torch.stack(stds, dim=1),
            torch.stack(entropies, dim=1),
        )

    def _expert_component_normals(
        self,
        expert_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate head_k only on features produced by encoder_k."""
        if expert_features.dim() != 3 or expert_features.shape[1] != self.n_experts:
            raise ValueError(
                "expert_features must have shape (samples, n_experts, hidden_size), "
                f"got {tuple(expert_features.shape)}."
            )
        means: List[torch.Tensor] = []
        stds: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        for k, actor in enumerate(self.group_actors):
            dist = actor.act.action_out(expert_features[:, k, :])
            means.append(dist.mean)
            stds.append(dist.stddev)
            entropies.append(dist.entropy())
        return (
            torch.stack(means, dim=1),
            torch.stack(stds, dim=1),
            torch.stack(entropies, dim=1),
        )

    def _static_component_normal(
        self,
        static_features: torch.Tensor,
        static_expert_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reproduce the original fixed-group policy distribution exactly."""
        ref_dist = self.group_actors[0].act.action_out(static_features)
        means = torch.zeros_like(ref_dist.mean)
        stds = torch.zeros_like(ref_dist.stddev)
        entropies = torch.zeros_like(ref_dist.entropy())
        for k, actor in enumerate(self.group_actors):
            mask = static_expert_ids == k
            if not torch.any(mask):
                continue
            dist = actor.act.action_out(static_features[mask])
            means[mask] = dist.mean
            stds[mask] = dist.stddev
            entropies[mask] = dist.entropy()
        return means, stds, entropies

    def _full_expert_weights(self, dynamic_gate: torch.Tensor) -> torch.Tensor:
        """Weights for [original static policy, K complete dynamic experts]."""
        alpha = self.router_alpha.to(device=dynamic_gate.device, dtype=dynamic_gate.dtype)
        static_weight = torch.ones_like(dynamic_gate[..., :1]) * (1.0 - alpha)
        return torch.cat([static_weight, alpha * dynamic_gate], dim=-1)

    def _full_expert_log_probs_and_entropy(
        self,
        static_features: torch.Tensor,
        expert_features: torch.Tensor,
        static_expert_ids: torch.Tensor,
        dynamic_gate: torch.Tensor,
        actions: torch.Tensor,
        active_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        static_mean, static_std, static_entropy = self._static_component_normal(
            static_features, static_expert_ids
        )
        expert_means, expert_stds, expert_entropies = self._expert_component_normals(
            expert_features
        )
        means = torch.cat([static_mean.unsqueeze(1), expert_means], dim=1)
        stds = torch.cat([static_std.unsqueeze(1), expert_stds], dim=1)
        component_entropies = torch.cat(
            [static_entropy.unsqueeze(1), expert_entropies], dim=1
        )
        weights = self._full_expert_weights(dynamic_gate)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        safe_log_weights = weights.clamp_min(1e-20).log()

        actions_expanded = actions.unsqueeze(1).expand_as(means)
        component_log_probs = torch.distributions.Normal(means, stds).log_prob(
            actions_expanded
        ).sum(dim=-1)
        log_probs = torch.logsumexp(
            safe_log_weights + component_log_probs, dim=-1, keepdim=True
        )

        mixture_entropy = -(weights * safe_log_weights).sum(dim=-1)
        entropy_per_sample = (
            weights * component_entropies
        ).sum(dim=-1) + self.router_entropy_scale * mixture_entropy
        if active_masks is not None:
            dist_entropy = (
                entropy_per_sample * active_masks.squeeze(-1)
            ).sum() / active_masks.sum()
        else:
            dist_entropy = entropy_per_sample.mean()
        return log_probs, dist_entropy

    def _sample_from_full_experts(
        self,
        static_features: torch.Tensor,
        expert_features: torch.Tensor,
        static_expert_ids: torch.Tensor,
        dynamic_gate: torch.Tensor,
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        static_mean, static_std, _ = self._static_component_normal(
            static_features, static_expert_ids
        )
        expert_means, expert_stds, _ = self._expert_component_normals(expert_features)
        means = torch.cat([static_mean.unsqueeze(1), expert_means], dim=1)
        stds = torch.cat([static_std.unsqueeze(1), expert_stds], dim=1)
        weights = self._full_expert_weights(dynamic_gate)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        if deterministic and not self.deterministic_hard_routing:
            actions = (weights.unsqueeze(-1) * means).sum(dim=1)
        else:
            component_ids = (
                weights.argmax(dim=-1)
                if deterministic
                else torch.distributions.Categorical(probs=weights).sample()
            )
            gather_idx = component_ids.view(-1, 1, 1).expand(-1, 1, means.shape[-1])
            selected_means = means.gather(1, gather_idx).squeeze(1)
            if deterministic:
                actions = selected_means
            else:
                selected_stds = stds.gather(1, gather_idx).squeeze(1)
                actions = torch.distributions.Normal(selected_means, selected_stds).sample()

        log_probs, _ = self._full_expert_log_probs_and_entropy(
            static_features=static_features,
            expert_features=expert_features,
            static_expert_ids=static_expert_ids,
            dynamic_gate=dynamic_gate,
            actions=actions,
        )
        return actions, log_probs

    def _mixture_log_probs_and_entropy(
        self,
        features: torch.Tensor,
        gates: torch.Tensor,
        actions: torch.Tensor,
        active_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        means, stds, component_entropies = self._component_normals(features)
        actions_expanded = actions.unsqueeze(1).expand_as(means)
        component_log_probs = torch.distributions.Normal(means, stds).log_prob(actions_expanded).sum(
            dim=-1
        )
        log_probs = torch.logsumexp(gates.clamp_min(1e-8).log() + component_log_probs, dim=-1, keepdim=True)

        gate_entropy = -(gates * gates.clamp_min(1e-8).log()).sum(dim=-1)
        entropy_per_sample = (gates * component_entropies).sum(dim=-1) + self.router_entropy_scale * gate_entropy
        if active_masks is not None:
            dist_entropy = (entropy_per_sample * active_masks.squeeze(-1)).sum() / active_masks.sum()
        else:
            dist_entropy = entropy_per_sample.mean()

        return log_probs, dist_entropy

    def _sample_from_mixture(
        self,
        features: torch.Tensor,
        gates: torch.Tensor,
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        means, stds, _ = self._component_normals(features)
        if deterministic:
            if self.deterministic_hard_routing:
                expert_ids = gates.argmax(dim=-1)
                gather_idx = expert_ids.view(-1, 1, 1).expand(-1, 1, means.shape[-1])
                actions = means.gather(1, gather_idx).squeeze(1)
            else:
                actions = (gates.unsqueeze(-1) * means).sum(dim=1)
        else:
            expert_ids = torch.distributions.Categorical(probs=gates).sample()
            gather_idx = expert_ids.view(-1, 1, 1).expand(-1, 1, means.shape[-1])
            selected_means = means.gather(1, gather_idx).squeeze(1)
            selected_stds = stds.gather(1, gather_idx).squeeze(1)
            actions = torch.distributions.Normal(selected_means, selected_stds).sample()

        log_probs, _ = self._mixture_log_probs_and_entropy(features, gates, actions)
        return actions, log_probs

    def act(
        self,
        obs,
        rnn_states,
        masks,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one rollout / eval step with global communication and soft routing."""
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        n_agents = obs.shape[0]
        if n_agents != self.n_agents_total:
            raise ValueError(f"Expected {self.n_agents_total} agents, got {n_agents}.")

        global_features = torch.zeros(
            n_agents, self.hidden_size, dtype=obs.dtype, device=obs.device
        )
        new_rnn_states = rnn_states.clone()

        if self.shared_encoder:
            global_features, new_rnn_states = self._encode_group(
                actor=self.group_actors[0],
                obs=obs,
                rnn_states=rnn_states,
                masks=masks,
            )
        else:
            for actor, idx_k in zip(self.group_actors, self.group_indices):
                feats_k, rnn_k = self._encode_group(
                    actor=actor,
                    obs=obs[idx_k],
                    rnn_states=rnn_states[idx_k],
                    masks=masks[idx_k],
                )
                global_features[idx_k] = feats_k
                new_rnn_states[idx_k] = rnn_k

        global_features = self._apply_global_comm(global_features.unsqueeze(0)).squeeze(0)
        router_extra = self._capacity_router_features(obs)
        if router_extra is not None:
            router_extra = router_extra.unsqueeze(0)
        gates = self._compute_gates(global_features.unsqueeze(0), router_extra).squeeze(0)
        if self.full_expert_routing:
            expert_features: List[torch.Tensor] = []
            for actor in self.group_actors:
                features_k, _ = self._encode_group(
                    actor=actor,
                    obs=obs,
                    rnn_states=rnn_states,
                    masks=masks,
                )
                expert_features.append(
                    self._apply_global_comm(features_k.unsqueeze(0)).squeeze(0)
                )
            all_expert_features = torch.stack(expert_features, dim=1)
            if self.last_dynamic_gate is None:
                raise RuntimeError("Router did not produce a dynamic gate.")
            dynamic_gate = self.last_dynamic_gate.squeeze(0)
            actions, log_probs = self._sample_from_full_experts(
                static_features=global_features,
                expert_features=all_expert_features,
                static_expert_ids=self.static_group_ids.to(obs.device),
                dynamic_gate=dynamic_gate,
                deterministic=deterministic,
            )
        else:
            actions, log_probs = self._sample_from_mixture(global_features, gates, deterministic)

        return actions, log_probs, new_rnn_states

    def evaluate_actions(
        self,
        group_batches: Sequence[Dict[str, torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Evaluate grouped actions after a single global communication pass."""
        if len(group_batches) != len(self.group_actors):
            raise ValueError("group_batches length must match number of group actors.")

        encoded_groups: List[torch.Tensor] = []
        obs_groups: List[torch.Tensor] = []
        rnn_groups: List[torch.Tensor] = []
        mask_groups: List[torch.Tensor] = []
        timesteps_in_batch: Optional[int] = None

        for actor, idx_k, batch in zip(self.group_actors, self.group_indices, group_batches):
            obs = batch["obs"]
            rnn_states = batch["rnn_states"]
            masks = batch["masks"]

            n_k = len(idx_k)
            if obs.shape[0] % n_k != 0:
                raise ValueError(
                    f"Batch size {obs.shape[0]} is not divisible by group size {n_k}."
                )
            t_mb = obs.shape[0] // n_k
            if timesteps_in_batch is None:
                timesteps_in_batch = t_mb
            elif timesteps_in_batch != t_mb:
                raise ValueError("All group mini-batches must contain the same sampled timesteps.")

            obs_groups.append(obs.view(t_mb, n_k, -1))
            rnn_groups.append(rnn_states.view(t_mb, n_k, *rnn_states.shape[1:]))
            mask_groups.append(masks.view(t_mb, n_k, -1))
            if not self.shared_encoder:
                feats_k, _ = self._encode_group(
                    actor=actor,
                    obs=obs,
                    rnn_states=rnn_states,
                    masks=masks,
                )
                encoded_groups.append(feats_k.view(t_mb, n_k, self.hidden_size))

        assert timesteps_in_batch is not None
        global_obs = torch.zeros(
            timesteps_in_batch,
            self.n_agents_total,
            obs_groups[0].shape[-1],
            dtype=obs_groups[0].dtype,
            device=obs_groups[0].device,
        )
        global_rnn = torch.zeros(
            timesteps_in_batch,
            self.n_agents_total,
            *rnn_groups[0].shape[2:],
            dtype=rnn_groups[0].dtype,
            device=rnn_groups[0].device,
        )
        global_masks = torch.zeros(
            timesteps_in_batch,
            self.n_agents_total,
            mask_groups[0].shape[-1],
            dtype=mask_groups[0].dtype,
            device=mask_groups[0].device,
        )
        for idx_k, obs_k, rnn_k, masks_k in zip(
            self.group_indices, obs_groups, rnn_groups, mask_groups
        ):
            global_obs[:, idx_k, :] = obs_k
            global_rnn[:, idx_k, ...] = rnn_k
            global_masks[:, idx_k, :] = masks_k

        if self.shared_encoder:
            flat_features, _ = self._encode_group(
                actor=self.group_actors[0],
                obs=global_obs.reshape(-1, global_obs.shape[-1]),
                rnn_states=global_rnn.reshape(-1, *global_rnn.shape[2:]),
                masks=global_masks.reshape(-1, global_masks.shape[-1]),
            )
            global_features = flat_features.view(
                timesteps_in_batch, self.n_agents_total, self.hidden_size
            )
        else:
            global_features = torch.zeros(
                timesteps_in_batch,
                self.n_agents_total,
                self.hidden_size,
                dtype=encoded_groups[0].dtype,
                device=encoded_groups[0].device,
            )
            for idx_k, feats_k in zip(self.group_indices, encoded_groups):
                global_features[:, idx_k, :] = feats_k

        global_features = self._apply_global_comm(global_features)
        full_expert_features: Optional[torch.Tensor] = None
        if self.full_expert_routing:
            expert_branches: List[torch.Tensor] = []
            flat_obs = global_obs.reshape(-1, global_obs.shape[-1])
            flat_rnn = global_rnn.reshape(-1, *global_rnn.shape[2:])
            flat_masks = global_masks.reshape(-1, global_masks.shape[-1])
            for actor in self.group_actors:
                branch_features, _ = self._encode_group(
                    actor=actor,
                    obs=flat_obs,
                    rnn_states=flat_rnn,
                    masks=flat_masks,
                )
                branch_features = branch_features.view(
                    timesteps_in_batch, self.n_agents_total, self.hidden_size
                )
                expert_branches.append(self._apply_global_comm(branch_features))
            full_expert_features = torch.stack(expert_branches, dim=2)

        router_extra = None
        if self.router_extra_dim > 0:
            router_extra = self._capacity_router_features(global_obs)
        gates = self._compute_gates(global_features, router_extra)

        action_log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        for static_k, (idx_k, batch) in enumerate(zip(self.group_indices, group_batches)):
            features_k = global_features[:, idx_k, :].reshape(-1, self.hidden_size)
            gates_k = gates[:, idx_k, :].reshape(-1, self.n_experts)
            active_masks = batch.get("active_masks")

            if self.full_expert_routing:
                if full_expert_features is None or self.last_dynamic_gate is None:
                    raise RuntimeError("Full-expert features or dynamic gates are missing.")
                expert_features_k = full_expert_features[:, idx_k, :, :].reshape(
                    -1, self.n_experts, self.hidden_size
                )
                dynamic_gate_k = self.last_dynamic_gate[:, idx_k, :].reshape(
                    -1, self.n_experts
                )
                static_ids = torch.full(
                    (features_k.shape[0],),
                    static_k,
                    dtype=torch.long,
                    device=features_k.device,
                )
                logp_k, entropy_k = self._full_expert_log_probs_and_entropy(
                    static_features=features_k,
                    expert_features=expert_features_k,
                    static_expert_ids=static_ids,
                    dynamic_gate=dynamic_gate_k,
                    actions=batch["action"],
                    active_masks=(active_masks if self._use_policy_active_masks else None),
                )
            else:
                logp_k, entropy_k = self._mixture_log_probs_and_entropy(
                    features=features_k,
                    gates=gates_k,
                    actions=batch["action"],
                    active_masks=(active_masks if self._use_policy_active_masks else None),
                )
            action_log_probs.append(logp_k)
            entropies.append(entropy_k)

        return action_log_probs, entropies
