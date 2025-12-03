import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# --- NEW DISTRICT TARGET GENERATION: next day's target is mean of previous day's district net load ---
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json

DATA_DIR = Path.cwd()
N_BUILDINGS = 25
dataset_key = "TX_25_dynamics"

def load_building_profiles(period_name, start_date, end_date):
    profiles = {}
    for bid in range(N_BUILDINGS):
        obs_path = DATA_DIR / "outputs" / dataset_key / "obs" /  "rbc" / period_name / f"obs_building_{bid}.csv"
        if not obs_path.exists():
            print(f"[WARN] Building {bid} obs file not found at {obs_path}")
            continue
        df = pd.read_csv(obs_path)
        idx = pd.date_range(start=start_date, periods=len(df), freq="h")
        df.index = idx
        profiles[bid] = df
    return profiles

def extract_net(df):
    if "positive net electricity consumption" in df.columns:
        return pd.to_numeric(df["positive net electricity consumption"], errors="coerce").fillna(0)
    raise ValueError("Column 'positive net electricity consumption' not found")

def main():
   
   
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description="Generate district target: next day = mean of previous day's district net load")
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
    month_counts = all_days.month.value_counts()
    # Map month number to name
    month_map = {1:'january', 2:'february', 3:'march', 4:'april', 5:'may', 6:'june', 7:'july', 8:'august', 9:'september', 10:'october', 11:'november', 12:'december'}
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
        net = extract_net(df)
        all_nets.append(net)
    # Align and sum
    district_net = pd.concat(all_nets, axis=1).sum(axis=1)
    district_net = district_net.loc[start_date:end_date]

    # Group by day
    daily = district_net.resample('D').sum()
    days = daily.index

    # Il district target di ogni giorno è costante e pari alla media del giorno precedente
    target_rows = []
    for i in range(1, len(days)):
        prev_day = days[i-1]
        this_day = days[i]
        prev_mean = daily[prev_day] / 24.0
        for h in range(24):
            dt = pd.Timestamp(this_day) + pd.Timedelta(hours=h)
            target_rows.append({
                'datetime': dt,
                'district_load_target': prev_mean
            })

    # Truncate or pad to exact sim window length
    sim_window_len = sim_end_timestep - sim_start_timestep + 1
    if len(target_rows) < sim_window_len:
        # Pad end with zeros if needed
        last_dt = target_rows[-1]['datetime']
        for i in range(sim_window_len - len(target_rows)):
            target_rows.append({
                'datetime': last_dt + pd.Timedelta(hours=i+1),
                'district_load_target': 0.0
            })
    target_rows = target_rows[:sim_window_len]

    # Save to CSV (full with datetime)
    target_df = pd.DataFrame(target_rows)
    out_dir = DATA_DIR / "outputs" / dataset_key / "target_profiles_per_building" / period_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "district_target.csv"
    target_df.to_csv(out_path, index=False)
    print(f"[OK] Saved district target to {out_path}")

    # Save the generated target values in the middle, zeros before and after to fill 8760 rows
    data_dir = DATA_DIR / "data" / dataset_key
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
    pd.DataFrame({'district_load_target': annual_target}).to_csv(simple_target_path, index=False)
    print(f"[OK] Saved simple district target to {simple_target_path} (zeros before and after, generated values at correct timesteps)")

    # Plot the district target for the month
        # Filtra solo il mese di June (fino al 30 June)

        # Supponiamo che district_target sia un array orario con datetimes
    # Filtro district dal 1 June 00:00 al 30 June 23:00
    dt_start_district = pd.Timestamp('2023-06-01 00:00:00')
    dt_end_district = pd.Timestamp('2023-06-30 23:00:00')
    datetimes = pd.to_datetime(target_df['datetime'])
    mask_district = (datetimes >= dt_start_district) & (datetimes <= dt_end_district)
    district_target_june = np.array(target_df['district_load_target'])[mask_district]
    datetimes_june = datetimes[mask_district]

    # Filtro RBC dal 31 maggio 00:00 al 29 June 23:00
    dt_start_rbc = pd.Timestamp('2023-05-31 00:00:00')
    dt_end_rbc = pd.Timestamp('2023-06-29 23:00:00')
    rbc_df = pd.read_csv('outputs/TX_25_dynamics_noBESS/obs/flat/rbc/june/district_obs.csv')
    rbc_datetimes = pd.date_range(start=dt_start_rbc, end=dt_end_rbc, freq='h')
    col_candidates = [c for c in rbc_df.columns if 'positive net electricity consumption' in c.lower()]
    if col_candidates:
        rbc_load = rbc_df[col_candidates[0]].values
        # Assumo che la lunghezza sia uguale a rbc_datetimes
        rbc_load_june = rbc_load[:len(rbc_datetimes)]
    else:
        print('[WARN] Colonna "positive net electricity consumption" non trovata in district_obs.csv')
        rbc_load_june = np.full(len(rbc_datetimes), np.nan)

    # Carica il load RBC dal file district_obs.csv e la feature 'positive net electricity consumption'
    rbc_load = None
    rbc_df = pd.read_csv('outputs/TX_25_dynamics_noBESS/obs/flat/rbc/june/district_obs.csv')
    # Cerca la colonna giusta (case insensitive)
    col_candidates = [c for c in rbc_df.columns if 'positive net electricity consumption' in c.lower()]
    if col_candidates:
        rbc_load = rbc_df[col_candidates[0]].values
    else:
        print('[WARN] Colonna "positive net electricity consumption" non trovata in district_obs.csv')

    # Ora calcolo area su questi intervalli
    if len(district_target_june) == len(rbc_load_june):
        area_target = np.sum(district_target_june)
        area_rbc = np.sum(rbc_load_june)
        print(f'Area sotto district target (1-30 June) [somma]: {area_target:.2f}')
        print(f'Area sotto RBC load (31 maggio-29 June) [somma]: {area_rbc:.2f}')
        with open('outputs/area_target_vs_rbc.txt', 'w') as f:
            f.write(f'Area sotto district target (1-30 June) [somma]: {area_target:.2f}\n')
            f.write(f'Area sotto RBC load (31 maggio-29 June) [somma]: {area_rbc:.2f}\n')
    else:
        print('[WARN] Impossibile calcolare area: lunghezze non compatibili.')

    # Confronto area giorno per giorno tra district target (1-30 June) e RBC (31 maggio-29 June, giorno precedente)
    # Creo DataFrame per district target (1-30 June)
    df_target = pd.DataFrame({
        'datetime': datetimes_june,
        'district_target': district_target_june
    })
    df_target['date'] = df_target['datetime'].dt.date
    area_target_days = df_target.groupby('date')['district_target'].sum()

    # Creo DataFrame per RBC (31 maggio-29 June)
    rbc_dates = pd.date_range(start=pd.Timestamp('2023-05-31 00:00:00'), end=pd.Timestamp('2023-06-29 23:00:00'), freq='h')
    df_rbc = pd.DataFrame({
        'datetime': rbc_dates,
        'rbc_load': rbc_load_june
    })
    df_rbc['date'] = df_rbc['datetime'].dt.date
    area_rbc_days = df_rbc.groupby('date')['rbc_load'].sum()

    # Confronta area target del giorno d (1-30 June) con area RBC del giorno d-1 (31 maggio-29 June)
    target_dates = sorted(area_target_days.index)
    with open('outputs/area_giornaliera_target_vs_rbc.txt', 'w') as f:
        for d in target_dates:
            d_prev = d - pd.Timedelta(days=1)
            at = area_target_days[d]
            ar = area_rbc_days.get(d_prev, np.nan)
            print(f"{d}: area target={at:.2f}, area rbc giorno prima={ar:.2f}")
            f.write(f"{d}: area target={at:.2f}, area rbc giorno prima={ar:.2f}\n")
            if not np.isclose(at, ar, rtol=1e-3):
                print(f"[WARN] Area diversa per il giorno {d}: target={at:.2f}, rbc giorno prima={ar:.2f}")
                f.write(f"[WARN] Area diversa per il giorno {d}: target={at:.2f}, rbc giorno prima={ar:.2f}\n")

    # --- RIMOSSO USO DI rbc_load_june_shifted, USO DIRETTAMENTE rbc_load_june ---
    # Calcolo area già fatto sopra

    # Plot district target e RBC load
        plt.figure(figsize=(14, 7))
        # Shift RBC di -24 ore (valori del giorno precedente)
        rbc_load_june_shifted = np.roll(rbc_load_june, -24)
        rbc_load_june_shifted[-24:] = np.nan
        # Plot linee principali
        plt.plot(datetimes_june, district_target_june, label='District Target', color='#222222', linewidth=2.5)
        plt.plot(datetimes_june, rbc_load_june_shifted, label='RBC District Load', color='#0072B2', linewidth=2.5, alpha=0.85)
        # Area sotto le curve
        plt.fill_between(datetimes_june, 0, district_target_june, color='#222222', alpha=0.18, label='Area District Target')
        plt.fill_between(datetimes_june, 0, rbc_load_june_shifted, color='#0072B2', alpha=0.18, label='Area RBC Load Shiftato')
        # Griglia e ottimizzazione
        plt.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.5)
        plt.tick_params(axis='both', labelsize=14)
        # Annotazioni intervalli temporali
        ax = plt.gca()
        boxprops = dict(facecolor='white', edgecolor='#222222', alpha=0.97)
        # Migliora legenda
        leg = plt.legend(fontsize=14, frameon=True)
        leg.get_frame().set_alpha(0.92)
        leg.get_frame().set_edgecolor('#222222')
        plt.xlabel('Datetime', fontsize=16)
        plt.ylabel('Load (kW)', fontsize=16)
        plt.title('District Target vs RBC Load Shiftato (June)', fontsize=18, fontweight='bold')
        plt.tight_layout(pad=2.0)

        # --- PLOT SOLO PER UN GIORNO (District Target 4 June, RBC 3 June, RBC = dati orari reali) ---
        giorno = pd.Timestamp('2023-06-04')
        giorno_rbc = giorno - pd.Timedelta(days=1)
        ore = np.arange(24)
        # District Target del 4 June (costante)
        mask_target = (datetimes_june.dt.date == giorno.date())
        target_giorno = district_target_june[mask_target]
        # RBC del 3 June (dati orari reali)
        mask_rbc = (datetimes_june.dt.date == giorno_rbc.date())
        rbc_giorno = rbc_load_june_shifted[mask_rbc]
        plt.figure(figsize=(10, 5))
        plt.plot(ore, target_giorno, label='District Target (4 June)', color='#222222', linewidth=2.5)
        plt.plot(ore, rbc_giorno, label='RBC District Load (3 June)', color='#0072B2', linewidth=2.5, alpha=0.85)
        plt.fill_between(ore, 0, target_giorno, color='#222222', alpha=0.18)
        plt.fill_between(ore, 0, rbc_giorno, color='#0072B2', alpha=0.18)
        plt.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.5)
        plt.tick_params(axis='both', labelsize=13)
        leg = plt.legend(fontsize=13, frameon=True, loc='lower right')
        leg.get_frame().set_alpha(0.92)
        leg.get_frame().set_edgecolor('#222222')
        plt.xlabel('Hour', fontsize=15)
        plt.ylabel('Load', fontsize=15)
        plt.title('District Target (4 June) vs RBC Load (3 June)', fontsize=16, fontweight='bold')
        plt.tight_layout(pad=2.0)
        # Calcolo area per il giorno selezionato
        area_target_giorno = np.sum(target_giorno)
        area_rbc_giorno = np.sum(rbc_giorno)
        ax = plt.gca()
        boxprops = dict(facecolor='white', edgecolor='black', alpha=1.0)
        y_max = ax.get_ylim()[1]
        x_min = ax.get_xlim()[0]
        ax.text(x_min, y_max,
            f'Area District Target (4 June): {area_target_giorno:.2f}', color='black', fontsize=12,
            ha='left', va='top', fontweight='bold', bbox=boxprops)
        ax.text(x_min, y_max - (y_max*0.07),
            f'Area RBC Load (3 June, reale): {area_rbc_giorno:.2f}', color='blue', fontsize=12,
            ha='left', va='top', fontweight='bold', bbox=boxprops)
        ax.text(x_min, y_max - (y_max*0.14),
            f'Diff: {area_target_giorno-area_rbc_giorno:.2f}', color='red', fontsize=12,
            ha='left', va='top', bbox=boxprops)
        plt.savefig('outputs/district_target_vs_rbc_4June_vs_3June_reale.png')
        plt.close()
    # Scrivo i valori delle aree solo se sono definiti
    if area_target is not None and area_rbc is not None:
        ax = plt.gca()
        boxprops = dict(facecolor='white', edgecolor='black', alpha=1.0)
        y_max = ax.get_ylim()[1]
        x_min = ax.get_xlim()[0]
        """
        ax.text(x_min, y_max,
            f'Area District Target: {area_target:.2f}', color='black', fontsize=12,
            ha='left', va='top', fontweight='bold', bbox=boxprops)
        ax.text(x_min, y_max - (y_max*0.07),
            f'Area RBC Load: {area_rbc:.2f}', color='blue', fontsize=12,
            ha='left', va='top', fontweight='bold', bbox=boxprops)
        ax.text(x_min, y_max - (y_max*0.14),
            f'Diff: {area_target-area_rbc:.2f}', color='red', fontsize=12,
            ha='left', va='top', bbox=boxprops)
        """
    plt.ylabel('Load (kW)')
    plt.title('District Target vs RBC Load Shiftato')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/district_target_vs_rbc_june.png')
    plt.close()
    
if __name__ == "__main__":
    main()