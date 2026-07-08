param(
    [string]$PythonExe = "python",
    [switch]$NoSkipCompleted
)

$ErrorActionPreference = "Continue"

$repo = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $repo

$logDir = Join-Path $repo "experiment_queue_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$transcript = Join-Path $logDir "priority_experiment_queue_$stamp.log"

Start-Transcript -Path $transcript | Out-Null

Write-Host "Repository: $repo"
Write-Host "Python: $PythonExe"
Write-Host "Queue log: $transcript"
Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

$experiments = @(
    @{
        Name = "soft_router_agglomerative_5f_capacity_router"
        Module = "mappo_grouped_tarmac_soft_router.train"
        SaveDir = "results/mappo_grouped_tarmac_soft_router_agglomerative_5f_capacity_router_vt_500_final"
        Args = @(
            "--climate", "VT",
            "--n_episodes", "500",
            "--train_month", "1",
            "--test_month", "2",
            "--seed", "42",
            "--group_k_candidates", "4", "5",
            "--cluster_seed", "0",
            "--cluster_retries", "10",
            "--grouping_method", "agglomerative",
            "--grouping_feature_columns", "bes_capacity_kwh", "hvac_total_kw", "heating_mean", "nsl_mean", "comfort_lower_excess_mean",
            "--comm_fusion_mode", "linear",
            "--router_temperature", "0.7",
            "--router_prior_end", "0.7",
            "--router_warmup_episodes", "100",
            "--router_entropy_scale", "0.02",
            "--wandb_name", "mappo_grouped_tarmac_soft_router_agglomerative_5f_capacity_router_vt_500_final",
            "--save_dir", "results/mappo_grouped_tarmac_soft_router_agglomerative_5f_capacity_router_vt_500_final"
        )
    },
    @{
        Name = "soft_router_agglomerative_5f_capacity_router_prior05"
        Module = "mappo_grouped_tarmac_soft_router.train"
        SaveDir = "results/mappo_grouped_tarmac_soft_router_agglomerative_5f_capacity_router_prior05_vt_500_final"
        Args = @(
            "--climate", "VT",
            "--n_episodes", "500",
            "--train_month", "1",
            "--test_month", "2",
            "--seed", "42",
            "--group_k_candidates", "4", "5",
            "--cluster_seed", "0",
            "--cluster_retries", "10",
            "--grouping_method", "agglomerative",
            "--grouping_feature_columns", "bes_capacity_kwh", "hvac_total_kw", "heating_mean", "nsl_mean", "comfort_lower_excess_mean",
            "--comm_fusion_mode", "linear",
            "--router_temperature", "0.7",
            "--router_prior_end", "0.5",
            "--router_warmup_episodes", "100",
            "--router_entropy_scale", "0.02",
            "--wandb_name", "mappo_grouped_tarmac_soft_router_agglomerative_5f_capacity_router_prior05_vt_500_final",
            "--save_dir", "results/mappo_grouped_tarmac_soft_router_agglomerative_5f_capacity_router_prior05_vt_500_final"
        )
    },
    @{
        Name = "soft_router_agglomerative_5f_capacity_router_temp05"
        Module = "mappo_grouped_tarmac_soft_router.train"
        SaveDir = "results/mappo_grouped_tarmac_soft_router_agglomerative_5f_capacity_router_temp05_vt_500_final"
        Args = @(
            "--climate", "VT",
            "--n_episodes", "500",
            "--train_month", "1",
            "--test_month", "2",
            "--seed", "42",
            "--group_k_candidates", "4", "5",
            "--cluster_seed", "0",
            "--cluster_retries", "10",
            "--grouping_method", "agglomerative",
            "--grouping_feature_columns", "bes_capacity_kwh", "hvac_total_kw", "heating_mean", "nsl_mean", "comfort_lower_excess_mean",
            "--comm_fusion_mode", "linear",
            "--router_temperature", "0.5",
            "--router_prior_end", "0.7",
            "--router_warmup_episodes", "100",
            "--router_entropy_scale", "0.02",
            "--wandb_name", "mappo_grouped_tarmac_soft_router_agglomerative_5f_capacity_router_temp05_vt_500_final",
            "--save_dir", "results/mappo_grouped_tarmac_soft_router_agglomerative_5f_capacity_router_temp05_vt_500_final"
        )
    },
    @{
        Name = "soft_router_agglomerative_5f_no_capacity_router"
        Module = "mappo_grouped_tarmac_soft_router.train"
        SaveDir = "results/mappo_grouped_tarmac_soft_router_agglomerative_5f_no_capacity_router_vt_500_final"
        Args = @(
            "--climate", "VT",
            "--n_episodes", "500",
            "--train_month", "1",
            "--test_month", "2",
            "--seed", "42",
            "--group_k_candidates", "4", "5",
            "--cluster_seed", "0",
            "--cluster_retries", "10",
            "--grouping_method", "agglomerative",
            "--grouping_feature_columns", "bes_capacity_kwh", "hvac_total_kw", "heating_mean", "nsl_mean", "comfort_lower_excess_mean",
            "--comm_fusion_mode", "linear",
            "--router_temperature", "0.7",
            "--router_prior_end", "0.7",
            "--router_warmup_episodes", "100",
            "--router_entropy_scale", "0.02",
            "--no_router_capacity_features",
            "--wandb_name", "mappo_grouped_tarmac_soft_router_agglomerative_5f_no_capacity_router_vt_500_final",
            "--save_dir", "results/mappo_grouped_tarmac_soft_router_agglomerative_5f_no_capacity_router_vt_500_final"
        )
    },
    @{
        Name = "tarmac_hybrid_agglomerative_3f"
        Module = "mappo_grouped_tarmac_hybrid_grouping.train"
        SaveDir = "results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final"
        Args = @(
            "--climate", "VT",
            "--n_episodes", "500",
            "--train_month", "1",
            "--test_month", "2",
            "--seed", "42",
            "--group_k_candidates", "4", "5",
            "--cluster_seed", "0",
            "--cluster_retries", "10",
            "--grouping_method", "agglomerative",
            "--grouping_feature_columns", "bes_capacity_kwh", "heating_mean", "nsl_mean",
            "--comm_fusion_mode", "linear",
            "--wandb_name", "mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final",
            "--save_dir", "results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final"
        )
    },
    @{
        Name = "tarmac_hybrid_agglomerative_4f"
        Module = "mappo_grouped_tarmac_hybrid_grouping.train"
        SaveDir = "results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_4f_linear_vt_500_final"
        Args = @(
            "--climate", "VT",
            "--n_episodes", "500",
            "--train_month", "1",
            "--test_month", "2",
            "--seed", "42",
            "--group_k_candidates", "4", "5",
            "--cluster_seed", "0",
            "--cluster_retries", "10",
            "--grouping_method", "agglomerative",
            "--grouping_feature_columns", "bes_capacity_kwh", "hvac_total_kw", "heating_mean", "nsl_mean",
            "--comm_fusion_mode", "linear",
            "--wandb_name", "mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_4f_linear_vt_500_final",
            "--save_dir", "results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_4f_linear_vt_500_final"
        )
    },
    @{
        Name = "powernet_global_agglomerative_3f"
        Module = "mappo_grouped_powernet_global_grouping.train"
        SaveDir = "results/mappo_grouped_powernet_global_agglomerative_capacity_load_3f_vt_500_final"
        Args = @(
            "--climate", "VT",
            "--n_episodes", "500",
            "--train_month", "1",
            "--test_month", "2",
            "--seed", "42",
            "--group_k_candidates", "4", "5",
            "--cluster_seed", "0",
            "--cluster_retries", "10",
            "--grouping_method", "agglomerative",
            "--grouping_feature_columns", "bes_capacity_kwh", "heating_mean", "nsl_mean",
            "--wandb_name", "mappo_grouped_powernet_global_agglomerative_capacity_load_3f_vt_500_final",
            "--save_dir", "results/mappo_grouped_powernet_global_agglomerative_capacity_load_3f_vt_500_final"
        )
    },
    @{
        Name = "powernet_global_agglomerative_4f"
        Module = "mappo_grouped_powernet_global_grouping.train"
        SaveDir = "results/mappo_grouped_powernet_global_agglomerative_capacity_load_4f_vt_500_final"
        Args = @(
            "--climate", "VT",
            "--n_episodes", "500",
            "--train_month", "1",
            "--test_month", "2",
            "--seed", "42",
            "--group_k_candidates", "4", "5",
            "--cluster_seed", "0",
            "--cluster_retries", "10",
            "--grouping_method", "agglomerative",
            "--grouping_feature_columns", "bes_capacity_kwh", "hvac_total_kw", "heating_mean", "nsl_mean",
            "--wandb_name", "mappo_grouped_powernet_global_agglomerative_capacity_load_4f_vt_500_final",
            "--save_dir", "results/mappo_grouped_powernet_global_agglomerative_capacity_load_4f_vt_500_final"
        )
    }
)

$failures = @()
$completed = @()
$skipped = @()

for ($i = 0; $i -lt $experiments.Count; $i++) {
    $exp = $experiments[$i]
    $index = $i + 1
    $saveDir = Join-Path $repo $exp.SaveDir
    $metricsPath = Join-Path $saveDir "latest_metrics.json"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[$index/$($experiments.Count)] $($exp.Name)"
    Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "SaveDir: $($exp.SaveDir)"
    Write-Host "============================================================"

    if ((-not $NoSkipCompleted) -and (Test-Path -LiteralPath $metricsPath)) {
        Write-Host "Skipping completed experiment: $metricsPath"
        $skipped += $exp.Name
        continue
    }

    & $PythonExe -m $exp.Module @($exp.Args)
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        Write-Host "Completed: $($exp.Name) at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        $completed += $exp.Name
    }
    else {
        Write-Host "FAILED: $($exp.Name) exit_code=$exitCode at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        $failures += "$($exp.Name) exit_code=$exitCode"
    }
}

Write-Host ""
Write-Host "================ Queue Summary ================"
Write-Host "Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Completed: $($completed.Count)"
$completed | ForEach-Object { Write-Host "  OK   $_" }
Write-Host "Skipped: $($skipped.Count)"
$skipped | ForEach-Object { Write-Host "  SKIP $_" }
Write-Host "Failed: $($failures.Count)"
$failures | ForEach-Object { Write-Host "  FAIL $_" }
Write-Host "Log: $transcript"

Stop-Transcript | Out-Null

if ($failures.Count -gt 0) {
    exit 1
}

exit 0
