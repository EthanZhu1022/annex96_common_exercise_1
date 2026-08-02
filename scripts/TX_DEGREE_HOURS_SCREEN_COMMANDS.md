# VT/TX degree-hours experiment commands

Run these commands from the repository root on the Linux experiment machine:

```bash
chmod +x scripts/run_tarmac_3fA_tx_degree_hours_3seeds_parallel.sh
chmod +x scripts/start_tarmac_3fA_tx_degree_hours_screen.sh
bash scripts/start_tarmac_3fA_tx_degree_hours_screen.sh
```

The launcher creates a detached `screen` session named `tx_3fA_degree_hours`.
All seeds `0 1 2` start concurrently and each seed is pinned to two CPU cores.

Useful commands:

```bash
screen -ls
screen -r tx_3fA_degree_hours
```

Inside `screen`, detach without stopping the experiment by pressing `Ctrl-a`, then `d`.

To follow the individual seed logs from another shell:

```bash
tail -f experiment_queue_logs/tx_3fA_degree_hours_*/*.stdout.log
```

Optional overrides must be supplied to the screen launcher, for example:

```bash
EPISODES=500 USE_GPU=0 bash scripts/start_tarmac_3fA_tx_degree_hours_screen.sh
```

Run VT alone (three seeds, six CPU cores):

```bash
chmod +x scripts/run_tarmac_3fA_vt_degree_hours_3seeds_parallel.sh
chmod +x scripts/start_tarmac_3fA_vt_degree_hours_screen.sh
bash scripts/start_tarmac_3fA_vt_degree_hours_screen.sh
```

Run VT and TX together (six seeds, twelve non-overlapping CPU cores):

```bash
chmod +x scripts/run_tarmac_3fA_{vt,tx}_degree_hours_3seeds_parallel.sh
chmod +x scripts/start_tarmac_3fA_{vt,tx}_degree_hours_screen.sh
chmod +x scripts/start_tarmac_3fA_vt_tx_degree_hours_screens.sh
bash scripts/start_tarmac_3fA_vt_tx_degree_hours_screens.sh
```
