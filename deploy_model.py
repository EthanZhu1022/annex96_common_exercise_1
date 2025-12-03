#!/usr/bin/env python3
from pathlib import Path
import os, zipfile, tempfile, json, re, argparse
from typing import Optional

import numpy as np
import pandas as pd
import torch

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.aac_madrl import AAC_MADRL
from citylearn.agents.sac import SAC
from citylearn.agents.marlisa import MARLISA
from citylearn.agents.rbc import PITemperatureController as RBC


# ----------------------------- helpers -----------------------------

def _find_outputs_root(anchor: Path) -> Path:
    """Trova l'antenato 'outputs' (altrimenti usa la cartella data come fallback)."""
    for p in [anchor] + list(anchor.parents):
        if p.name.lower() == "outputs":
            return p
    # Se non trova outputs, usa la cartella data o il parent del dataset
    if anchor.is_file():
        return anchor.parent.parent  # schema.json -> dataset -> data
    else:
        return anchor.parent  # dataset folder -> data


def _find_real_schema_for_env(anchor: Path, outputs_root: Path) -> str:
    """
    Dall'anchor .../outputs/data/<name>/schema.json prova a trovare un vero schema.json per CityLearnEnv:
    1) usa anchor se è un file reale
    2) usa <project_root>/data/<name>/schema.json
    3) usa Path.cwd()/data/<name>/schema.json
    4) altrimenti, passa 'data/<name>/schema.json'
    """
    if anchor.is_file():
        return str(anchor)

    rel = anchor.relative_to(outputs_root)                  # data/<name>/schema.json
    project_root = outputs_root.parent                      # .../Citylearn_dinamics
    cand1 = project_root / rel                              # .../Citylearn_dinamics/data/<name>/schema.json
    if cand1.exists():
        return str(cand1.resolve())

    cand2 = Path.cwd() / rel
    if cand2.exists():
        return str(cand2.resolve())

    return str(Path(*rel.parts))                            # "data/<name>/schema.json"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _torch_cast_state_dict(sd, target_dtype):
    if target_dtype is None:
        return sd
    return {k: (v.to(target_dtype) if torch.is_floating_point(v) else v) for k, v in sd.items()}


def _load_norm_from_extracted(model, tmp_root: Path):
    cfg = tmp_root / "mean_std" / "config.json"
    if cfg.exists():
        params = json.loads(cfg.read_text())
        model.norm_mean = [np.asarray(x) for x in params.get("mean", [])]
        model.norm_std  = [np.asarray(x) for x in params.get("std", [])]


def _generic_load_from_extracted(model, tmp_root: Path, map_location=None, cast_to=None, strict=True):
    _load_norm_from_extracted(model, tmp_root)

    def _indices(pattern):
        rx = re.compile(pattern)
        idx = []
        for p in tmp_root.rglob("*"):
            m = rx.fullmatch(p.name)
            if m:
                idx.append(int(m.group(1)))
        return sorted(set(idx))

    indices = _indices(r"policy_net_(\d+)\.pt") or _indices(r"soft_q_net1_(\d+)\.pt")

    if cast_to:
        cast_to = cast_to.lower()
    tgt_dtype = {"fp16": torch.float16, "float16": torch.float16,
                 "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                 "fp32": torch.float32, "float32": torch.float32}.get(cast_to)

    def _load_into(module, path: Path):
        sd = torch.load(path, map_location=map_location)
        # cast al dtype del modulo, se non forzato
        if tgt_dtype is None:
            try:
                target_dtype = next(module.parameters()).dtype
                sd = _torch_cast_state_dict(sd, target_dtype)
            except StopIteration:
                pass
        else:
            sd = _torch_cast_state_dict(sd, tgt_dtype)
        module.load_state_dict(sd, strict=strict)

    soft_q_net1 = getattr(model, "soft_q_net1", None)
    soft_q_net2 = getattr(model, "soft_q_net2", None)
    target_soft_q_net1 = getattr(model, "target_soft_q_net1", None)
    target_soft_q_net2 = getattr(model, "target_soft_q_net2", None)
    policy_net = getattr(model, "policy_net", None)

    for i in indices:
        p = tmp_root / f"soft_q_net1_{i}.pt"
        if soft_q_net1 is not None and len(soft_q_net1) > i and p.exists():
            _load_into(soft_q_net1[i], p)
        p = tmp_root / f"soft_q_net2_{i}.pt"
        if soft_q_net2 is not None and len(soft_q_net2) > i and p.exists():
            _load_into(soft_q_net2[i], p)
        p = tmp_root / f"target_soft_q_net1_{i}.pt"
        if target_soft_q_net1 is not None and len(target_soft_q_net1) > i and p.exists():
            _load_into(target_soft_q_net1[i], p)
        p = tmp_root / f"target_soft_q_net2_{i}.pt"
        if target_soft_q_net2 is not None and len(target_soft_q_net2) > i and p.exists():
            _load_into(target_soft_q_net2[i], p)
        p = tmp_root / f"policy_net_{i}.pt"
        if policy_net is not None and len(policy_net) > i and p.exists():
            _load_into(policy_net[i], p)

    if hasattr(model, "critic") and (tmp_root / "critic.pt").exists():
        _load_into(model.critic, tmp_root / "critic.pt")
    if hasattr(model, "target_critic") and (tmp_root / "target_critic.pt").exists():
        _load_into(model.target_critic, tmp_root / "target_critic.pt")


def load_from_zip(model, zip_path: Path, map_location: Optional[str] = None, cast_to: Optional[str] = None, strict: bool = True):
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip non trovato: {zip_path}")

    try:
        return model.load_models(zip_path=str(zip_path), map_location=map_location, cast_to=cast_to, strict=strict)
    except TypeError:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(td)
            try:
                return model.load_models(directory=td)
            except Exception:
                _generic_load_from_extracted(model, Path(td), map_location=map_location, cast_to=cast_to, strict=strict)


def derive_weights_zip(dataset_name: str, model_type: str, beta: float, lr: float, gamma: Optional[float] = None) -> Path:
    """
    Costruisce il path al file .zip dei pesi salvati in outputs/<dataset_name>/save_models/...
    """
    m = model_type.lower()
    zip_name = {"sac": "sac.zip",
                "sac_centralized": "sac.zip",
                "marlisa": "marlisa.zip",
                "aac_madrl": "aac_madrl.zip"}.get(m, f"{m}.zip")
    
    # Se gamma è None o 1.0, usa solo beta; altrimenti usa beta_gamma
    if gamma is None or gamma == 1.0:
        beta_folder = f"beta={beta}"
    else:
        beta_folder = f"beta={beta}_gamma={gamma}"
    
    return Path.cwd() / "outputs" / dataset_name / "save_models" / m / beta_folder / f"lr={lr}" / zip_name


def run_model_and_save_obs(
    dataset_anchor: Path,
    model_type: str,
    lr: float,
    beta: float,
    gamma: float,
    sim_start: Optional[int] = None,
    sim_end: Optional[int] = None,
    period_name: Optional[str] = None,
    aac_classes: Optional[dict] = None,
):
    # Se dataset_anchor è solo un nome (es. "TX_100_dynamics"), costruisci il path completo
    if not dataset_anchor.exists():
        # Prova a costruire il path: data/{dataset_anchor}/schema.json
        potential_path = Path.cwd() / "data" / str(dataset_anchor) / "schema.json"
        if potential_path.exists():
            dataset_anchor = potential_path
            print(f"[INFO] Using schema at: {dataset_anchor}")
        else:
            raise FileNotFoundError(f"Schema not found at {dataset_anchor} or {potential_path}")
    
    # outputs_root e dataset_root (cartella "chiave dataset" sotto outputs)
    outputs_root = _find_outputs_root(dataset_anchor)
    dataset_root = dataset_anchor if dataset_anchor.is_dir() else dataset_anchor.parent
    ensure_dir(dataset_root)
    
    # Directory per salvare gli output (in outputs invece che in data)
    # Estrae il nome del dataset (es. TX_25_dynamics) dal path
    dataset_name = dataset_root.name
    output_save_dir = Path.cwd() / "outputs" / dataset_name
    ensure_dir(output_save_dir)

    # path effettivo allo schema per CityLearnEnv
    dataset_arg = _find_real_schema_for_env(dataset_anchor, outputs_root)

    # Env kwargs
    central = (model_type == "SAC_CENTRALIZED")
    env_kwargs = {"central_agent": central}
    if sim_start is not None:
        env_kwargs["simulation_start_time_step"] = sim_start
    if sim_end is not None:
        env_kwargs["simulation_end_time_step"] = sim_end
    reward_kwargs = {}
    if beta is not None:
        reward_kwargs["beta"] = beta
    if gamma is not None:
        reward_kwargs["gamma"] = gamma
    if reward_kwargs:
        env_kwargs["reward_function_kwargs"] = reward_kwargs

    # setta CITYLEARN_DATA_DIR → <project_root>/data (se non presente)
    os.environ.setdefault("CITYLEARN_DATA_DIR", str(outputs_root.parent / "data"))

    print(f"[INFO] Creating environment from: {dataset_arg}")
    print(f"[INFO] Environment kwargs: {env_kwargs}")
    env = CityLearnEnv(dataset_arg, **env_kwargs)
    print(f"[INFO] Environment created with {len(env.buildings)} buildings")

    # Modello
    if model_type == "AAC_MADRL":
        model = AAC_MADRL(env, classes=aac_classes or {}, attend_heads=1, lr=lr, sample=False)
    elif model_type in ("SAC", "SAC_CENTRALIZED"):
        model = SAC(env, lr=lr)
    elif model_type == "MARLISA":
        model = MARLISA(env, lr=lr)
    elif model_type == "RBC":
        model = RBC(env, ki=0.05, kp=1/100)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # Carica pesi (se non RBC)
    if model_type != "RBC":
        weights_zip = derive_weights_zip(dataset_name, model_type, beta, lr, gamma)
        print(f"[INFO] Using weights from: {weights_zip}")
        load_from_zip(model, weights_zip, map_location="cpu", cast_to=None, strict=True)

    # Rollout deterministico
    print(f"[INFO] Starting rollout with {model_type}...")
    model.learn(deterministic=True)
    print(f"[INFO] Rollout completed!")

    # Dove salvare obs (usa output_save_dir invece di dataset_root)
    algo_dir = model_type.lower()
    if model_type == "RBC":
        if period_name:
            obs_dir = output_save_dir / "obs" / algo_dir / period_name
        else:
            obs_dir = output_save_dir / "obs" / algo_dir
    else:
        # Se gamma è None o 1.0, usa solo beta; altrimenti usa beta_gamma
        if gamma is None or gamma == 1.0:
            beta_folder = f"beta={beta}"
        else:
            beta_folder = f"beta={beta}_gamma={gamma}"
        if period_name:
            obs_dir = output_save_dir / "obs" / algo_dir / period_name / beta_folder / f"lr={lr}"
        else:
            obs_dir = output_save_dir / "obs" / algo_dir / beta_folder / f"lr={lr}"
    ensure_dir(obs_dir)

    # DISTRICT
    pos_el = np.where(model.env.electrical_storage_electricity_consumption > 0,
                      model.env.electrical_storage_electricity_consumption, 0)
    neg_el = np.where(model.env.electrical_storage_electricity_consumption < 0,
                      model.env.electrical_storage_electricity_consumption, 0)

    pos_dhw = np.where(model.env.dhw_storage_electricity_consumption > 0,
                       model.env.dhw_storage_electricity_consumption, 0)
    neg_dhw = np.where(model.env.dhw_storage_electricity_consumption < 0,
                       model.env.dhw_storage_electricity_consumption, 0)

    nec = np.array(model.env.net_electricity_consumption)
    pos_nec = np.where(nec > 0, nec, 0)
    neg_nec = np.where(nec < 0, nec, 0)

    df_obs_dist = pd.DataFrame({
        "cooling demand": model.env.cooling_demand,
        "cooling electricity consumption": model.env.cooling_electricity_consumption,
        "heating electricity consumption": model.env.heating_electricity_consumption,
        "dhw demand": model.env.dhw_demand,
        "dhw electricity consumption": model.env.dhw_electricity_consumption,
        "dhw storage electricity consumption": model.env.dhw_storage_electricity_consumption,
        "positive dhw storage electricity consumption": pos_dhw,
        "negative dhw storage electricity consumption": neg_dhw,
        "electrical storage electricity consumption": model.env.electrical_storage_electricity_consumption,
        "positive electrical storage electricity consumption": pos_el,
        "negative electrical storage electricity consumption": neg_el,
        "energy from cooling device": model.env.energy_from_cooling_device,
        "energy from dhw device": model.env.energy_from_dhw_storage,
        "energy from dhw device to dhw storage": model.env.energy_from_dhw_device_to_dhw_storage,
        "energy from dhw storage": model.env.energy_from_dhw_storage,
        "energy from electrical storage": model.env.energy_from_electrical_storage,
        "energy from heating device": model.env.energy_from_heating_device,
        "energy to electrical storage": model.env.energy_to_electrical_storage,
        "energy to non shiftable load": model.env.energy_to_non_shiftable_load,
        "net electricity consumption": model.env.net_electricity_consumption,
        "positive net electricity consumption": pos_nec,
        "negative net electricity consumption": neg_nec,
        "net electricity consumption without storage": model.env.net_electricity_consumption_without_storage,
        "net electricity consumption without storage and pv": model.env.net_electricity_consumption_without_storage_and_pv,
        "net electricity consumption without storage and partial load": model.env.net_electricity_consumption_without_storage_and_partial_load,
        "net electricity consumption without storage and partial load and pv": model.env.net_electricity_consumption_without_storage_and_partial_load_and_pv,
        "non shiftable load": model.env.non_shiftable_load,
        "solar generation": model.env.solar_generation,
    })
    df_obs_dist.to_csv(obs_dir / "district_obs.csv", index=False)

    # PER-BUILDING (tutti gli edifici)
    for i, b in enumerate(model.env.buildings):
        pos_el_b = np.where(b.electrical_storage_electricity_consumption > 0,
                            b.electrical_storage_electricity_consumption, 0)
        neg_el_b = np.where(b.electrical_storage_electricity_consumption < 0,
                            b.electrical_storage_electricity_consumption, 0)

        pos_dhw_b = np.where(b.dhw_storage_electricity_consumption > 0,
                             b.dhw_storage_electricity_consumption, 0)
        neg_dhw_b = np.where(b.dhw_storage_electricity_consumption < 0,
                             b.dhw_storage_electricity_consumption, 0)

        nec_b = np.array(b.net_electricity_consumption)
        pos_nec_b = np.where(nec_b > 0, nec_b, 0)
        neg_nec_b = np.where(nec_b < 0, nec_b, 0)

        df_obs_b = pd.DataFrame({
            "Outdoor_Temperature": b.weather.outdoor_dry_bulb_temperature,
            "occupant_count": b.occupant_count,
            "cooling demand": b.cooling_demand,
            "heating demand": b.heating_demand,
            "cooling electricity consumption": b.cooling_electricity_consumption,
            "heating electricity consumption": b.heating_electricity_consumption,
            "dhw demand": b.dhw_demand,
            "dhw electricity consumption": b.dhw_electricity_consumption,
            "dhw storage electricity consumption": b.dhw_storage_electricity_consumption,
            "positive dhw storage electricity consumption": pos_dhw_b,
            "negative dhw storage electricity consumption": neg_dhw_b,
            "dhw soc storage": b.dhw_storage.soc if hasattr(b, "dhw_storage") else None,
            "dhw capacity": b.dhw_storage.capacity if hasattr(b, "dhw_storage") else None,
            "dhw stored energy (?)": (b.dhw_storage.soc * b.dhw_storage.capacity) if hasattr(b, "dhw_storage") else None,
            "electrical storage electricity consumption": b.electrical_storage_electricity_consumption,
            "positive electrical storage electricity consumption": pos_el_b,
            "negative electrical storage electricity consumption": neg_el_b,
            "electrical storage soc": b.electrical_storage.soc if hasattr(b, "electrical_storage") else None,
            "electrical capacity history": b.electrical_storage.capacity_history if hasattr(b, "electrical_storage") else None,
            "stored electrical energy": (b.electrical_storage.soc * b.electrical_storage.capacity_history) if hasattr(b, "electrical_storage") else None,
            "energy from cooling device": b.energy_from_cooling_device,
            "energy from dhw device": b.energy_from_dhw_storage,
            "energy from dhw device to dhw storage": b.energy_from_dhw_device_to_dhw_storage,
            "energy from dhw storage": b.energy_from_dhw_storage,
            "energy from electrical storage": b.energy_from_electrical_storage,
            "energy from heating device": b.energy_from_heating_device,
            "energy to electrical storage": b.energy_to_electrical_storage,
            "energy to non shiftable load": b.energy_to_non_shiftable_load,
            "net electricity consumption": b.net_electricity_consumption,
            "positive net electricity consumption": pos_nec_b,
            "negative net electricity consumption": neg_nec_b,
            "net electricity consumption without storage": b.net_electricity_consumption_without_storage,
            "net electricity consumption without storage and pv": b.net_electricity_consumption_without_storage_and_pv,
            "non shiftable load": b.non_shiftable_load,
            "solar generation": b.solar_generation,
            "indoor_temperature": b.indoor_dry_bulb_temperature,
            "outdoor_temperature": b.weather.outdoor_dry_bulb_temperature,  # Added outdoor temperature
            "heating_sp": b.indoor_dry_bulb_temperature_heating_set_point,
            "cooling_sp": b.indoor_dry_bulb_temperature_cooling_set_point,
            "comfort_band": b.comfort_band
        })
        df_obs_b.to_csv(obs_dir / f"obs_building_{i}.csv", index=False)

        # azioni (se non centralizzato e disponibili)
        if model_type != "SAC_CENTRALIZED" and hasattr(model, "actions") and hasattr(b, "active_actions"):
            try:
                actions_name = b.active_actions
                actions_value = model.actions[i]
                dictionary = {actions_name[act]: [row[act] for row in actions_value[:-1]]
                              for act in range(len(actions_name))}
                pd.DataFrame(dictionary).to_csv(obs_dir / f"action_building_{i}.csv", index=False)
            except Exception:
                pass

    print("[OK] Osservazioni salvate in:", obs_dir)


# ----------------------------- CLI ---------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Deploy CityLearn agent: carica zip pesi da outputs/<dataset>/save_models/... e salva obs in outputs/<dataset>/obs/..."
    )
    p.add_argument(
        "--dataset-anchor",
        type=Path,
        default=Path(r"data\TX_25_dynamics\schema.json"),
        help="Path to schema.json file or dataset folder.",
    )
    p.add_argument("--model-type", choices=["AAC_MADRL", "MARLISA", "SAC", "SAC_CENTRALIZED", "RBC"], required=True)
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate usato in training.")
    p.add_argument("--beta", type=float, default=0.5, help="Beta per la reward (e per la cartella dei pesi).")
    p.add_argument("--gamma", type=float, default=1.0, help="Gamma per la reward (e per la cartella dei pesi).")
    p.add_argument("--sim-start", type=int, default=None, help="simulation_start_time_step.")
    p.add_argument("--sim-end", type=int, default=None, help="simulation_end_time_step.")
    p.add_argument("--period-name", type=str, default=None, help="Nome del periodo (es. 'july', 'august') per organizzare gli output.")
    # Nuovi argomenti per le classi delle azioni
    p.add_argument("--dhw-storage", type=int, default=21, help="Numero di classi per dhw_storage.")
    p.add_argument("--electrical-storage", type=int, default=21, help="Numero di classi per electrical_storage.")
    p.add_argument("--cooling-or-heating-device", type=int, default=21, help="Numero di classi per cooling_or_heating_device.")
    p.add_argument("--cooling-device", type=int, default=20, help="Numero di classi per cooling_device.")
    p.add_argument("--heating-device", type=int, default=10, help="Numero di classi per heating_device.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Costruisci il dizionario delle classi per AAC_MADRL
    aac_classes = None
    if args.model_type == "AAC_MADRL":
        aac_classes = {
            "dhw_storage": args.dhw_storage,
            "electrical_storage": args.electrical_storage,
            "cooling_or_heating_device": args.cooling_or_heating_device,
            "cooling_device": args.cooling_device,
            "heating_device": args.heating_device,
        }
    run_model_and_save_obs(
        dataset_anchor=args.dataset_anchor,
        model_type=args.model_type,
        lr=args.lr,
        beta=args.beta if args.beta is not None else 1.0,
        gamma=args.gamma if args.gamma is not None else 1.0,
        sim_start=args.sim_start,
        sim_end=args.sim_end,
        period_name=args.period_name,
        aac_classes=aac_classes,
    )





