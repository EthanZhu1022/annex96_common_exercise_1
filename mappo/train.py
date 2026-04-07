"""
Hierarchical MAPPO with Communication — Training Script
=======================================================

Run:
    python -m mappo.train                          # VT, 25 buildings, comm ON
    python -m mappo.train --climate TX             # Texas dataset
    python -m mappo.train --n_clusters 5           # 5 clusters of 5 buildings
    python -m mappo.train --no_comm                # disable communication
    python -m mappo.train --n_episodes 200 --eval_freq 20

Overview
--------
Each episode:
  1. Collect a full trajectory (one CityLearn episode).
     - Build cluster obs by concatenating per-building obs.
     - Run communication: m_i = f_i(cluster_obs_i), M = concat(m_0..m_{K-1}).
     - Each actor_i samples actions from π_i(cluster_obs_i, M).
     - All K agents share the same global reward (sum of building rewards).
     - Centralized critic V(global_obs) bootstraps returns for ALL agents.
  2. Compute GAE(γ, λ) advantages and discounted returns.
  3. PPO update over minibatches for `ppo_epochs` epochs.
     - Actor losses use the clipped surrogate + entropy bonus.
     - Critic loss is MSE against discounted returns.
  4. Log per-episode metrics to W&B and print to stdout.
  5. Periodically save training-curve plots.

See agent.py for CTDE design rationale and utils.py for GAE + KPI details.
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/notebook use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from gymnasium import spaces

# ── Ensure the local CityLearn copy is used (not any pip-installed version) ──
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper

from mappo.agent import Actor, Critic
from mappo.communication import CommunicationNet
from mappo.utils import RolloutBuffer, extract_episode_kpis, get_soc_stats

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
    print("[warn] wandb not installed — W&B logging disabled.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # ── Dataset ──────────────────────────────────────────────────────────
    climate:    str = "VT"  # "VT" (heating-dominated) or "TX" (cooling-dominated)
    n_buildings: int = 25  # how many buildings to include (max 25 in CE1)

    # ── Clustering ────────────────────────────────────────────────────────
    # Buildings are assigned round-robin to clusters to preserve CityLearn
    # ordering and handle cases where n_buildings % n_clusters != 0.
    n_clusters: int = 5     # number of cluster agents (each controls ~5 buildings)

    # ── Episode horizon ───────────────────────────────────────────────────
    # None → use the schema default (full month: 744 h for VT, 720 h for TX).
    # Set an int to override, e.g. 168 for a one-week horizon.
    episode_time_steps: Optional[int] = None

    # ── Network architecture ──────────────────────────────────────────────
    hidden_dim:        int  = 256   # hidden layer width for Actor and Critic
    msg_dim:           int  = 32    # per-agent communication message size
    use_communication: bool = True  # set False to ablate communication

    # ── Training ──────────────────────────────────────────────────────────
    n_episodes:    int   = 100
    gamma:         float = 0.99    # discount factor
    gae_lambda:    float = 0.95    # GAE λ
    clip_eps:      float = 0.2     # PPO clip ε
    entropy_coeff: float = 0.01    # entropy bonus coefficient
    value_coeff:   float = 0.5     # critic loss coefficient
    lr_actor:      float = 3e-4
    lr_critic:     float = 1e-3
    ppo_epochs:    int   = 4       # number of PPO update epochs per episode
    minibatch_size: int  = 64
    max_grad_norm: float = 0.5     # gradient clipping

    # ── Logging & output ──────────────────────────────────────────────────
    eval_freq:     int = 10             # plot/save every N episodes
    seed:          int = 42
    wandb_project: str = "citylearn-mappo"
    wandb_name:    str = "clustered-mappo"
    save_dir:      str = "results/mappo"


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def build_env(cfg: Config) -> Tuple[NormalizedObservationWrapper, CityLearnEnv]:
    """Instantiate and wrap CityLearnEnv.

    We use central_agent=False so that:
      - env.observation_space[i] → Box for building i (easy per-cluster slicing)
      - env.action_space[i]      → Box for building i
      - env.step(actions)        expects List[List[float]], one sublist per building
      - reward from step()       is List[float], one per building

    NormalizedObservationWrapper applies periodic normalization (sin/cos for
    temporal features) and min-max scaling — standard preprocessing for NNs.
    """
    dataset_name = f"annex96_ce1_{cfg.climate.lower()}_neighborhood"
    dataset_dir  = REPO_DIR / "data" / "datasets" / dataset_name
    schema_path  = dataset_dir / "schema.json"

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema not found: {schema_path}\n"
            f"Available datasets: {[d.name for d in (REPO_DIR / 'data' / 'datasets').iterdir()]}"
        )

    env_kwargs: Dict = dict(
        schema=str(schema_path),
        root_directory=str(dataset_dir),
        central_agent=False,                    # per-building obs/actions
        buildings=list(range(cfg.n_buildings)),  # first n_buildings by index
    )
    if cfg.episode_time_steps is not None:
        env_kwargs["episode_time_steps"] = cfg.episode_time_steps

    base_env = CityLearnEnv(**env_kwargs)
    env      = NormalizedObservationWrapper(base_env)
    return env, base_env


def create_clusters(n_buildings: int, n_clusters: int) -> List[List[int]]:
    """Assign buildings to clusters in round-robin order.

    Round-robin (rather than contiguous blocks) spreads the load evenly
    when n_buildings % n_clusters != 0 and keeps cluster sizes balanced.

    Example (25 buildings, 5 clusters):
        cluster 0: [0, 5, 10, 15, 20]
        cluster 1: [1, 6, 11, 16, 21]
        ...
    Preserves CityLearn building index ordering within each cluster so that
    the obs/action slicing in assemble_env_actions() is deterministic.
    """
    clusters: List[List[int]] = [[] for _ in range(n_clusters)]
    for b in range(n_buildings):
        clusters[b % n_clusters].append(b)
    return clusters


# ---------------------------------------------------------------------------
# Obs / action assembly helpers
# ---------------------------------------------------------------------------

def build_cluster_obs(
    obs_list: List[List[float]],
    clusters: List[List[int]],
) -> List[np.ndarray]:
    """Concatenate per-building obs for each cluster.

    Args:
        obs_list: List[n_buildings][obs_dim] — per-building normalized obs.
        clusters: List[n_clusters][building_indices]
    Returns:
        List[n_clusters] of np.ndarray, each shape (cluster_obs_dim_i,)
    """
    return [
        np.concatenate([obs_list[b] for b in cluster], dtype=np.float32)
        for cluster in clusters
    ]


def build_global_obs(obs_list: List[List[float]]) -> np.ndarray:
    """Concatenate ALL buildings' obs into one global state vector.

    This is the input to the centralized critic. Shared features (e.g., hour,
    temperature) appear once per building — redundant but harmless; the critic
    can learn to ignore them.  Keeping it simple avoids any ambiguity around
    which features are "shared" vs "private".
    """
    return np.concatenate(obs_list, dtype=np.float32)


def build_actor_inputs(
    cluster_obs:       List[np.ndarray],
    comm_nets:         List[Optional[CommunicationNet]],
    use_communication: bool,
    device:            torch.device,
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """Compute communication messages and build actor inputs.

    Communication flow:
      1. Each agent computes m_i = comm_net_i(cluster_obs_i).
      2. All messages are concatenated: M = [m_0 | m_1 | ... | m_{K-1}].
         Concatenation (not averaging) preserves individual agent identity.
      3. Actor input for agent i = concat(cluster_obs_i, M).

    When use_communication=False, actor inputs are just the cluster obs.

    Returns:
        actor_inputs: List[n_clusters] of np.ndarray for storing in buffer.
        M_np:         np.ndarray of shape (n_clusters * msg_dim,) or None.
    """
    n_agents = len(cluster_obs)

    if use_communication:
        with torch.no_grad():
            msgs = []
            for i in range(n_agents):
                x = torch.from_numpy(cluster_obs[i]).unsqueeze(0).to(device)
                m = comm_nets[i](x).squeeze(0).cpu().numpy()
                msgs.append(m)
        M_np = np.concatenate(msgs, dtype=np.float32)
        actor_inputs = [
            np.concatenate([cluster_obs[i], M_np], dtype=np.float32)
            for i in range(n_agents)
        ]
    else:
        M_np         = None
        actor_inputs = [cluster_obs[i].copy() for i in range(n_agents)]

    return actor_inputs, M_np


def assemble_env_actions(
    cluster_actions: List[np.ndarray],
    clusters:        List[List[int]],
    act_dims:        List[int],
    action_spaces:   List[spaces.Box],
    n_buildings:     int,
) -> List[List[float]]:
    """Split cluster action vectors back into per-building action lists.

    CityLearn expects actions as List[List[float]] where actions[b] is the
    action vector for building b.  Each cluster actor outputs a *flattened*
    joint action vector in [-1, 1] for all its assigned buildings in CityLearn
    order, so we rescale each per-building slice to that building's action
    space before returning it.

    Example — cluster 0 controls buildings [0, 5, 10], each with act_dim=2:
        cluster_actions[0] = [a0_bat, a0_heat, a5_bat, a5_heat, a10_bat, a10_heat]
        → env_actions[0]  = [a0_bat,  a0_heat]
        → env_actions[5]  = [a5_bat,  a5_heat]
        → env_actions[10] = [a10_bat, a10_heat]

    The policy actions stored in the rollout buffer remain unscaled; only the
    environment-facing copy is transformed here.
    """
    env_actions: List[Optional[List[float]]] = [None] * n_buildings
    for cluster_idx, building_indices in enumerate(clusters):
        vec    = cluster_actions[cluster_idx]
        offset = 0
        for b in building_indices:
            dim = act_dims[b]
            raw_action = vec[offset: offset + dim]
            action_space = action_spaces[b]
            low = action_space.low.astype(np.float32)
            high = action_space.high.astype(np.float32)
            scaled_action = low + (raw_action + 1.0) * 0.5 * (high - low)
            env_actions[b] = np.clip(scaled_action, low, high).tolist()
            offset += dim
    return env_actions  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    actors:        List[Actor],
    critic:        Critic,
    actor_opts:    List[Adam],
    critic_opt:    Adam,
    buffer:        RolloutBuffer,
    cfg:           Config,
    device:        torch.device,
) -> Tuple[float, float, float]:
    """Run PPO_EPOCHS passes over the rollout buffer.

    Each actor is updated independently with its own importance ratio, but
    all actors share the same advantage signal (from the centralized critic).
    The critic is updated via MSE against GAE-bootstrapped returns.

    Returns (avg_actor_loss, avg_critic_loss, avg_entropy) over all updates.
    """
    n_clusters       = len(actors)
    total_actor_loss = 0.0
    total_crit_loss  = 0.0
    total_entropy    = 0.0
    n_batches        = 0

    for _ in range(cfg.ppo_epochs):
        for batch in buffer.get_minibatches(cfg.minibatch_size, device):
            adv = batch["advantages"]

            # ── Actor updates (one per cluster) ─────────────────────────
            for i in range(n_clusters):
                new_lp, entropy = actors[i].evaluate_actions(
                    batch["actor_inputs"][i],
                    batch["actions"][i],
                )
                # Importance-sampling ratio π_new / π_old
                ratio = (new_lp - batch["old_log_probs"][i]).exp()

                # Clipped PPO surrogate objective
                surr1      = ratio * adv
                surr2      = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss = actor_loss - cfg.entropy_coeff * entropy.mean()

                actor_opts[i].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actors[i].parameters(), cfg.max_grad_norm)
                actor_opts[i].step()

                total_actor_loss += actor_loss.item()
                total_entropy    += entropy.mean().item()

            # ── Critic update ────────────────────────────────────────────
            v_pred    = critic(batch["global_obs"])
            crit_loss = cfg.value_coeff * F.mse_loss(v_pred, batch["returns"])

            critic_opt.zero_grad()
            crit_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            critic_opt.step()

            total_crit_loss += crit_loss.item()
            n_batches       += 1

    denom = max(n_batches, 1)
    return (
        total_actor_loss / (denom * n_clusters),
        total_crit_loss  / denom,
        total_entropy    / (denom * n_clusters),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(
    episode_rewards:    List[float],
    actor_losses:       List[float],
    critic_losses:      List[float],
    entropies:          List[float],
    kpis_list:          List[Dict],
    save_dir:           str,
    rbc_kpis:           Optional[pd.DataFrame] = None,
) -> None:
    """Save training-curve and KPI-trend figures.

    Plots:
      Row 0: episode reward | actor+critic losses | policy entropy
      Row 1: ramping KPI    | daily peak KPI      | all-time peak KPI
    A horizontal red dashed line at y=1.0 marks the no-control baseline.
    RBC baseline values (if available) are shown as orange dashed lines.
    """
    eps = list(range(1, len(episode_rewards) + 1))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("MAPPO Training — CityLearn CE1", fontsize=14)

    # Reward curve
    ax = axes[0, 0]
    ax.plot(eps, episode_rewards)
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")

    # Loss curves
    ax = axes[0, 1]
    ax.plot(eps, actor_losses,  label="Actor")
    ax.plot(eps, critic_losses, label="Critic")
    ax.legend()
    ax.set_title("Loss")
    ax.set_xlabel("Episode")

    # Entropy
    ax = axes[0, 2]
    ax.plot(eps, entropies)
    ax.set_title("Policy Entropy")
    ax.set_xlabel("Episode")

    # KPI trends
    kpi_cfg = [
        ("kpi/ramping",     "Ramping (avg)",     "ramping_average"),
        ("kpi/daily_peak",  "Daily Peak (avg)",  "daily_peak_average"),
        ("kpi/all_time_peak","All-Time Peak",    "all_time_peak_average"),
    ]
    for ax, (key, title, rbc_col) in zip(axes[1, :], kpi_cfg):
        vals  = [k.get(key) for k in kpis_list]
        valid = [(e, v) for e, v in zip(eps, vals) if v is not None]
        if valid:
            ep_v, val_v = zip(*valid)
            ax.plot(ep_v, val_v, label="MAPPO")

        # No-control baseline (all KPIs are normalized to this = 1)
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.6, label="No-control (=1)")

        # RBC baseline comparison (optional)
        if rbc_kpis is not None and rbc_col in rbc_kpis.columns:
            try:
                rbc_val = float(rbc_kpis[rbc_col].iloc[0])
                ax.axhline(rbc_val, color="orange", linestyle="--", alpha=0.7, label="RBC")
            except Exception:
                pass

        ax.set_title(f"KPI: {title}")
        ax.set_xlabel("Episode")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  [plot] saved → {out}")


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_models(
    actors:    List[Actor],
    critic:    Critic,
    comm_nets: List[Optional[CommunicationNet]],
    save_dir:  str,
) -> None:
    for i, actor in enumerate(actors):
        torch.save(actor.state_dict(), os.path.join(save_dir, f"actor_{i}.pt"))
    torch.save(critic.state_dict(), os.path.join(save_dir, "critic.pt"))
    for i, cn in enumerate(comm_nets):
        if cn is not None:
            torch.save(cn.state_dict(), os.path.join(save_dir, f"comm_net_{i}.pt"))
    print(f"  [ckpt] models saved → {save_dir}/")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: Config) -> Tuple[List[Actor], Critic]:
    """Full MAPPO training loop.

    Returns the trained actors and critic for further evaluation or export.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(cfg.save_dir, exist_ok=True)

    # ── W&B ──────────────────────────────────────────────────────────────
    # API key is read from the WANDB_API_KEY environment variable or
    # ~/.netrc / wandb login — never hardcoded here.
    use_wandb = _WANDB_OK
    if use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=vars(cfg))

    # ── Environment ───────────────────────────────────────────────────────
    env, base_env = build_env(cfg)
    n_buildings   = len(base_env.buildings)
    obs_dims      = [env.observation_space[i].shape[0] for i in range(n_buildings)]
    act_dims      = [env.action_space[i].shape[0]      for i in range(n_buildings)]
    action_spaces = [base_env.action_space[i] for i in range(n_buildings)]
    global_obs_dim = sum(obs_dims)

    print(
        f"\nEnvironment: {cfg.climate} | buildings: {n_buildings} | "
        f"obs/building: {obs_dims[0]} | act/building: {act_dims[0]} | "
        f"global obs: {global_obs_dim}"
    )

    # ── Clusters ─────────────────────────────────────────────────────────
    clusters         = create_clusters(n_buildings, cfg.n_clusters)
    n_clusters       = len(clusters)
    cluster_obs_dims = [sum(obs_dims[b] for b in cl) for cl in clusters]
    cluster_act_dims = [sum(act_dims[b] for b in cl) for cl in clusters]

    print(f"Clusters ({n_clusters}): {[len(c) for c in clusters]} buildings each")

    # ── Communication networks ────────────────────────────────────────────
    # Concatenated message size M = n_clusters * msg_dim
    total_msg_dim = n_clusters * cfg.msg_dim if cfg.use_communication else 0
    comm_nets: List[Optional[CommunicationNet]] = (
        [CommunicationNet(cluster_obs_dims[i], cfg.msg_dim).to(device)
         for i in range(n_clusters)]
        if cfg.use_communication else [None] * n_clusters
    )

    # ── Actors and Critic ────────────────────────────────────────────────
    # Actor input = cluster obs (+ communication messages if enabled)
    actor_input_dims = [cluster_obs_dims[i] + total_msg_dim for i in range(n_clusters)]
    actors = [
        Actor(actor_input_dims[i], cluster_act_dims[i], cfg.hidden_dim).to(device)
        for i in range(n_clusters)
    ]
    critic = Critic(global_obs_dim, cfg.hidden_dim).to(device)

    # ── Optimizers ────────────────────────────────────────────────────────
    # Each actor's optimizer also covers its comm_net parameters so that
    # the message encoder is trained end-to-end with the policy.
    actor_param_groups = [
        list(actors[i].parameters()) +
        (list(comm_nets[i].parameters()) if comm_nets[i] is not None else [])
        for i in range(n_clusters)
    ]
    actor_opts = [Adam(pg, lr=cfg.lr_actor) for pg in actor_param_groups]
    critic_opt = Adam(critic.parameters(), lr=cfg.lr_critic)

    # ── Rollout buffer ────────────────────────────────────────────────────
    buffer = RolloutBuffer(n_clusters)

    # ── Optional RBC baseline for comparison in plots ─────────────────────
    rbc_kpis: Optional[pd.DataFrame] = None
    rbc_kpi_file = REPO_DIR / "notebooks" / "rbc_baseline_kpi_summary.csv"
    if rbc_kpi_file.exists():
        try:
            rbc_kpis = pd.read_csv(rbc_kpi_file)
            print(f"Loaded RBC baseline from {rbc_kpi_file}")
        except Exception as exc:
            print(f"[warn] Could not load RBC baseline: {exc}")
    else:
        print("[info] RBC baseline file not found — skipping comparison.")

    # ── Metric history ────────────────────────────────────────────────────
    all_rewards:      List[float] = []
    all_actor_losses: List[float] = []
    all_crit_losses:  List[float] = []
    all_entropies:    List[float] = []
    all_kpis:         List[Dict]  = []

    print("=" * 65)
    print(
        f"MAPPO | Climate: {cfg.climate} | comm: {cfg.use_communication} | "
        f"episodes: {cfg.n_episodes}"
    )
    print("=" * 65)

    for episode in range(1, cfg.n_episodes + 1):

        # ================================================================
        # 1. ROLLOUT COLLECTION
        # ================================================================
        obs_list, _ = env.reset()   # List[n_buildings][obs_dim_i]
        buffer.clear()

        episode_reward  = 0.0
        step_rewards:   List[float] = []

        while not base_env.terminated:
            # Build cluster and global observations
            cluster_obs = build_cluster_obs(obs_list, clusters)
            global_obs  = build_global_obs(obs_list)

            # Build actor inputs (includes messages if comm is on)
            actor_inputs, _ = build_actor_inputs(
                cluster_obs, comm_nets, cfg.use_communication, device
            )

            # Sample actions from each cluster's actor
            actions_np:   List[np.ndarray] = []
            log_probs_np: List[float]       = []
            with torch.no_grad():
                for i in range(n_clusters):
                    x = torch.from_numpy(actor_inputs[i]).unsqueeze(0).to(device)
                    a, lp = actors[i].act(x)
                    actions_np.append(a.squeeze(0).cpu().numpy())
                    log_probs_np.append(lp.item())

                # Centralized value estimate for ALL agents
                g_t   = torch.from_numpy(global_obs).unsqueeze(0).to(device)
                value = critic(g_t).item()

            # Assemble per-building action list for CityLearn
            # Each cluster action vector is split back into per-building chunks
            env_actions = assemble_env_actions(
                actions_np, clusters, act_dims, action_spaces, n_buildings
            )

            # Step environment
            # rewards is List[float] with one reward per building
            next_obs_list, rewards, terminated, truncated, _ = env.step(env_actions)

            # Global cooperative reward: all agents share this signal.
            # Maximizing the sum of building rewards is equivalent to
            # minimizing the district's total electricity cost/emissions.
            global_reward = float(sum(rewards))
            done = terminated or truncated

            buffer.add(
                actor_inputs=actor_inputs,
                actions=actions_np,
                log_probs=log_probs_np,
                global_obs=global_obs,
                reward=global_reward,
                done=done,
                value=value,
            )

            episode_reward += global_reward
            step_rewards.append(global_reward)
            obs_list = next_obs_list

        # Bootstrap value for the last state (0 if episode terminated naturally)
        with torch.no_grad():
            last_g_obs = build_global_obs(obs_list)
            last_val   = (
                critic(torch.from_numpy(last_g_obs).unsqueeze(0).to(device)).item()
                if not base_env.terminated else 0.0
            )
        buffer.compute_returns_and_advantages(last_val, cfg.gamma, cfg.gae_lambda)

        # ================================================================
        # 2. PPO UPDATE
        # ================================================================
        actor_loss, crit_loss, entropy = ppo_update(
            actors, critic, actor_opts, critic_opt, buffer, cfg, device
        )

        # ================================================================
        # 3. KPI EXTRACTION & LOGGING
        # ================================================================
        kpis      = extract_episode_kpis(base_env)
        soc_stats = get_soc_stats(base_env)

        all_rewards.append(episode_reward)
        all_actor_losses.append(actor_loss)
        all_crit_losses.append(crit_loss)
        all_entropies.append(entropy)
        all_kpis.append(kpis)

        log_dict = {
            "episode":           episode,
            "episode_reward":    episode_reward,
            "loss_actor":        actor_loss,
            "loss_critic":       crit_loss,
            "entropy":           entropy,
            "step_reward_mean":  float(np.mean(step_rewards)),
            **kpis,
            **soc_stats,
        }
        if use_wandb:
            wandb.log(log_dict, step=episode)

        if episode % 10 == 0:
            print(
                f"Ep {episode:4d} | rew {episode_reward:9.2f} | "
                f"a_loss {actor_loss:7.4f} | c_loss {crit_loss:7.4f} | "
                f"ent {entropy:5.3f} | "
                f"ramp {kpis.get('kpi/ramping', float('nan')):.3f} | "
                f"peak {kpis.get('kpi/daily_peak', float('nan')):.3f}"
            )

        # ================================================================
        # 4. PERIODIC PLOTS & MODEL SAVES
        # ================================================================
        if episode % cfg.eval_freq == 0:
            save_plots(
                all_rewards, all_actor_losses, all_crit_losses,
                all_entropies, all_kpis, cfg.save_dir, rbc_kpis,
            )
            save_models(actors, critic, comm_nets, cfg.save_dir)

    # Final save
    save_plots(
        all_rewards, all_actor_losses, all_crit_losses,
        all_entropies, all_kpis, cfg.save_dir, rbc_kpis,
    )
    save_models(actors, critic, comm_nets, cfg.save_dir)

    if use_wandb:
        wandb.finish()

    print("\nTraining complete.")
    return actors, critic


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Hierarchical MAPPO with Communication for CityLearn CE1"
    )
    parser.add_argument("--climate",            default="VT",  choices=["VT", "TX"])
    parser.add_argument("--n_buildings",        type=int,   default=25)
    parser.add_argument("--n_clusters",         type=int,   default=5)
    parser.add_argument("--episode_time_steps", type=int,   default=None,
                        help="Episode length in hours (default: full month from schema)")
    parser.add_argument("--hidden_dim",         type=int,   default=256)
    parser.add_argument("--msg_dim",            type=int,   default=32)
    parser.add_argument("--no_comm",            action="store_true",
                        help="Disable communication between agents")
    parser.add_argument("--n_episodes",         type=int,   default=100)
    parser.add_argument("--ppo_epochs",         type=int,   default=4)
    parser.add_argument("--minibatch_size",     type=int,   default=64)
    parser.add_argument("--lr_actor",           type=float, default=3e-4)
    parser.add_argument("--lr_critic",          type=float, default=1e-3)
    parser.add_argument("--eval_freq",          type=int,   default=10)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--wandb_project",      default="citylearn-mappo")
    parser.add_argument("--wandb_name",         default="clustered-mappo")
    parser.add_argument("--save_dir",           default="results/mappo")
    args = parser.parse_args()

    return Config(
        climate            = args.climate,
        n_buildings        = args.n_buildings,
        n_clusters         = args.n_clusters,
        episode_time_steps = args.episode_time_steps,
        hidden_dim         = args.hidden_dim,
        msg_dim            = args.msg_dim,
        use_communication  = not args.no_comm,
        n_episodes         = args.n_episodes,
        ppo_epochs         = args.ppo_epochs,
        minibatch_size     = args.minibatch_size,
        lr_actor           = args.lr_actor,
        lr_critic          = args.lr_critic,
        eval_freq          = args.eval_freq,
        seed               = args.seed,
        wandb_project      = args.wandb_project,
        wandb_name         = args.wandb_name,
        save_dir           = args.save_dir,
    )


if __name__ == "__main__":
    train(parse_args())
