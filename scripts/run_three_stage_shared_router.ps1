$ErrorActionPreference = "Stop"

$PythonExe = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location -LiteralPath $RepoDir

$SaveDir = "results/mappo_grouped_tarmac_soft_router_three_stage_shared_3f_vt_1500_seed42"
$RunArgs = @(
    "--climate", "VT",
    "--seed", "42",
    "--train_month", "1",
    "--test_month", "2",
    "--group_k_candidates", "4", "5",
    "--cluster_seed", "0",
    "--cluster_retries", "10",
    "--grouping_method", "agglomerative",
    "--grouping_feature_columns", "bes_capacity_kwh", "heating_mean", "nsl_mean",
    "--comm_fusion_mode", "linear",
    "--training_schedule", "three_stage",
    "--static_actor_episodes", "500",
    "--router_only_episodes", "500",
    "--dynamic_actor_episodes", "500",
    "--dynamic_actor_router_freeze_episodes", "100",
    "--router_temperature", "0.5",
    "--router_prior_start", "0.5",
    "--router_warmup_episodes", "150",
    "--router_entropy_scale", "0.05",
    "--router_balance_coef", "0.01",
    "--router_only_lr", "1e-4",
    "--dynamic_actor_lr", "3e-5",
    "--router_finetune_lr", "2e-5",
    "--checkpoint_keep_every", "50",
    "--wandb_name", (Split-Path -Leaf $SaveDir),
    "--save_dir", $SaveDir
)

& $PythonExe -m mappo_grouped_tarmac_soft_router.train @RunArgs
exit $LASTEXITCODE
