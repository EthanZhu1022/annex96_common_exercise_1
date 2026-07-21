$ErrorActionPreference = "Continue"

$PythonExe = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$NoSkipCompleted = if ($env:NO_SKIP_COMPLETED) { $env:NO_SKIP_COMPLETED } else { "0" }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location -LiteralPath $RepoDir

$LogDir = Join-Path $RepoDir "experiment_queue_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "twostage_router_followup_queue_$Stamp.log"

Start-Transcript -Path $LogFile -Append | Out-Null

Write-Host "Repository: $RepoDir"
Write-Host "Python: $PythonExe"
Write-Host "Queue log: $LogFile"
Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

$Completed = New-Object System.Collections.Generic.List[string]
$Skipped = New-Object System.Collections.Generic.List[string]
$Failed = New-Object System.Collections.Generic.List[string]

$Common = @(
    "--climate", "VT",
    "--n_episodes", "500",
    "--train_month", "1",
    "--test_month", "2",
    "--group_k_candidates", "4", "5",
    "--cluster_seed", "0",
    "--cluster_retries", "10",
    "--grouping_method", "agglomerative",
    "--grouping_feature_columns", "bes_capacity_kwh", "heating_mean", "nsl_mean",
    "--comm_fusion_mode", "linear"
)

function Run-Experiment {
    param(
        [string]$Name,
        [string]$SaveDir,
        [string[]]$ArgsList
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Name
    Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "SaveDir: $SaveDir"
    Write-Host "============================================================"

    $MetricsPath = Join-Path $SaveDir "latest_metrics.json"
    if ($NoSkipCompleted -ne "1" -and (Test-Path -LiteralPath $MetricsPath)) {
        Write-Host "Skipping completed experiment: $MetricsPath"
        $Skipped.Add($Name)
        return
    }

    & $PythonExe -m mappo_grouped_tarmac_soft_router.train @ArgsList
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -eq 0) {
        Write-Host "Completed: $Name at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        $Completed.Add($Name)
    } else {
        Write-Host "FAILED: $Name exit_code=$ExitCode at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        $Failed.Add("$Name exit_code=$ExitCode")
    }
}

function Run-RouterOnly500 {
    param([int]$Seed)
    $ExpertDir = "results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_seed$Seed"
    $SaveDir = "results/mappo_grouped_tarmac_soft_router_twostage_3f_expert_routeronly500_temp05_prior1_vt_500_seed$Seed"
    $ArgsList = @(
        $Common +
        @(
            "--seed", "$Seed",
            "--expert_checkpoint_dir", $ExpertDir,
            "--router_temperature", "0.5",
            "--router_prior_start", "0.5",
            "--router_prior_end", "1.0",
            "--router_warmup_episodes", "150",
            "--router_entropy_scale", "0.005",
            "--router_freeze_experts_episodes", "500",
            "--router_only_lr", "1e-4",
            "--router_finetune_lr", "5e-5",
            "--wandb_name", (Split-Path -Leaf $SaveDir),
            "--save_dir", $SaveDir
        )
    )
    Run-Experiment "twostage_routeronly500_temp05_prior1_seed$Seed" $SaveDir $ArgsList
}

function Run-NoCapacityFreeze200 {
    param([int]$Seed)
    $ExpertDir = "results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_seed$Seed"
    $SaveDir = "results/mappo_grouped_tarmac_soft_router_twostage_3f_expert_freeze200_temp05_prior1_no_capacity_vt_500_seed$Seed"
    $ArgsList = @(
        $Common +
        @(
            "--seed", "$Seed",
            "--expert_checkpoint_dir", $ExpertDir,
            "--router_temperature", "0.5",
            "--router_prior_start", "0.5",
            "--router_prior_end", "1.0",
            "--router_warmup_episodes", "150",
            "--router_entropy_scale", "0.005",
            "--router_freeze_experts_episodes", "200",
            "--router_only_lr", "1e-4",
            "--router_finetune_lr", "5e-5",
            "--no_router_capacity_features",
            "--wandb_name", (Split-Path -Leaf $SaveDir),
            "--save_dir", $SaveDir
        )
    )
    Run-Experiment "twostage_freeze200_temp05_prior1_no_capacity_seed$Seed" $SaveDir $ArgsList
}

foreach ($Seed in @(0, 1, 2, 3)) {
    Run-RouterOnly500 $Seed
}

foreach ($Seed in @(0, 1, 2, 3)) {
    Run-NoCapacityFreeze200 $Seed
}

Write-Host ""
Write-Host "================ Queue Summary ================"
Write-Host "Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Completed: $($Completed.Count)"
$Completed | ForEach-Object { Write-Host "  OK   $_" }
Write-Host "Skipped: $($Skipped.Count)"
$Skipped | ForEach-Object { Write-Host "  SKIP $_" }
Write-Host "Failed: $($Failed.Count)"
$Failed | ForEach-Object { Write-Host "  FAIL $_" }
Write-Host "Log: $LogFile"

Stop-Transcript | Out-Null

if ($Failed.Count -gt 0) {
    exit 1
}

exit 0
