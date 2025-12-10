import pandas as pd
import numpy as np
from pathlib import Path

# --- NEW DISTRICT TARGET GENERATION: same day's target is mean of the same day's district net load ---
import argparse
import json

DATA_DIR = Path.cwd()
N_BUILDINGS = 25
dataset_key = "annex96_ce1_tx_neighborhood"

def load_building_profiles(period_name, start_date, end_date):
    profiles = {}
    for bid in range(N_BUILDINGS):
        obs_path = DATA_DIR / "outputs" / dataset_key / "obs" /  "rbc"  / f"obs_building_{bid}.csv"
        if not obs_path.exists():
            print(f"[WARN] Building {bid} obs file not found at {obs_path}")
            continue
        df = pd.read_csv(obs_path)
        idx = pd.date_range(start=start_date, periods=len(df), freq="h")
        df.index = idx
        profiles[bid] = df
    return profiles

def extract_net(df):
    # try case-insensitive match for expected column
    candidates = [c for c in df.columns if 'positive net electricity consumption' in c.lower()]
    if candidates:
        return pd.to_numeric(df[candidates[0]], errors="coerce").fillna(0)
    # fallback: try looser match
    candidates2 = [c for c in df.columns if 'net' in c.lower() and 'consum' in c.lower()]
    if candidates2:
        return pd.to_numeric(df[candidates2[0]], errors="coerce").fillna(0)
    raise ValueError("Column 'positive net electricity consumption' not found")

def main():
    parser = argparse.ArgumentParser(description="Generate district target: same-day mean of district net load")
    parser.add_argument("--sim-start", dest="sim_start", default=3600, type=int)
    parser.add_argument("--sim-end", dest="sim_end", default=4343, type=int)
    args = parser.parse_args()

    sim_start_timestep = args.sim_start
    sim_end_timestep = args.sim_end
    base_date = pd.Timestamp("2023-01-01 00:00:00")
    start_date = base_date + pd.Timedelta(hours=sim_start_timestep)
    end_date = base_date + pd.Timedelta(hours=sim_end_timestep)

    # Determine the period_name by majority of days in the range
    all_days = pd.date_range(start=start_date, end=end_date, freq='D')
    if len(all_days) == 0:
        print("[ERROR] Empty date range.")
        return
    month_counts = all_days.month.value_counts()
    month_map = {1:'january', 2:'february', 3:'march', 4:'april', 5:'may', 6:'june',
                 7:'july', 8:'august', 9:'september', 10:'october', 11:'november', 12:'december'}
    majority_month = month_map[month_counts.idxmax()]
    period_name = majority_month

    print(f"[INFO] Using sim_start_timestep: {sim_start_timestep}")
    print(f"[INFO] Using sim_end_timestep: {sim_end_timestep}")
    print(f"[INFO] Calculated START_DATE: {start_date}")
    print(f"[INFO] Calculated END_DATE: {end_date}")
    print(f"[INFO] Inferred period_name: {period_name}")

    profiles = load_building_profiles(period_name, start_date, end_date)
    if not profiles:
        print("[ERROR] No building profiles loaded.")
        return

    # Build a DataFrame with all buildings' net load
    all_nets = []
    for bid, df in profiles.items():
        try:
            net = extract_net(df)
            all_nets.append(net)
        except Exception as e:
            print(f"[WARN] building {bid}: {e}")
    if not all_nets:
        print("[ERROR] No net series available.")
        return

    # Align and sum
    district_net = pd.concat(all_nets, axis=1).sum(axis=1)
    district_net = district_net.loc[start_date:end_date]

    # Group by day (daily energy)
    daily = district_net.resample('D').sum()
    days = daily.index

    # NEW: target for each day is the mean of the SAME day
    target_rows = []
    for this_day in days:
        # daily energy / 24 => average hourly
        same_day_mean = daily.loc[this_day] / 24.0
        for h in range(24):
            dt = pd.Timestamp(this_day) + pd.Timedelta(hours=h)
            target_rows.append({
                'datetime': dt,
                'district_load_target': float(same_day_mean)
            })

    # Truncate or pad to exact sim window length
    sim_window_len = sim_end_timestep - sim_start_timestep + 1
    if len(target_rows) < sim_window_len:
        # Pad end with zeros if needed
        if target_rows:
            last_dt = target_rows[-1]['datetime']
        else:
            last_dt = start_date
        for i in range(sim_window_len - len(target_rows)):
            target_rows.append({
                'datetime': last_dt + pd.Timedelta(hours=i+1),
                'district_load_target': 0.0
            })
    target_rows = target_rows[:sim_window_len]

    # Save to CSV (full with datetime)
    target_df = pd.DataFrame(target_rows)
    out_dir = DATA_DIR / "outputs" / "datasets"/ dataset_key / "target_profiles_per_building" / period_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "district_target.csv"
    target_df.to_csv(out_path, index=False)
    print(f"[OK] Saved district target to {out_path}")

    # Save the generated target values in the middle, zeros before and after to fill 8760 rows
    data_dir = DATA_DIR / "datasets"/ dataset_key 
    data_dir.mkdir(parents=True, exist_ok=True)
    simple_target_path = data_dir / "district_target.csv"
    HOURS_PER_YEAR = 8760
    annual_target = np.zeros(HOURS_PER_YEAR)
    # Find the correct start index from the first datetime in target_df
    if not target_df.empty:
        first_dt = pd.to_datetime(target_df['datetime'].iloc[0])
        base_dt = pd.Timestamp("2023-01-01 00:00:00")
        start_idx = int((first_dt - base_dt).total_seconds() // 3600)
        fill_len = len(target_df)
        if start_idx + fill_len <= HOURS_PER_YEAR:
            annual_target[start_idx:start_idx+fill_len] = target_df['district_load_target'].values
        else:
            # clip if target extends beyond 8760
            end_idx = min(HOURS_PER_YEAR, start_idx + fill_len)
            annual_target[start_idx:end_idx] = target_df['district_load_target'].values[:end_idx-start_idx]
    pd.DataFrame({'district_load_target': annual_target}).to_csv(simple_target_path, index=False)
    print(f"[OK] Saved simple district target to {simple_target_path} (zeros before and after, generated values at correct timesteps)")


if __name__ == "__main__":
    main()