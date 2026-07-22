$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Resolve-Path (Join-Path $ScriptDir "..")
$PowerShellExe = (Get-Process -Id $PID).Path
$PythonExe = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }

$Experiments = @(
    [PSCustomObject]@{
        Name = "Route 1/2: shared encoder, 3-stage training"
        Script = Join-Path $ScriptDir "run_three_stage_shared_router.ps1"
    },
    [PSCustomObject]@{
        Name = "Route 2/2: pretrained full experts, router + dynamic actor"
        Script = Join-Path $ScriptDir "run_three_stage_full_expert_router.ps1"
    }
)

Set-Location -LiteralPath $RepoDir

& $PythonExe --version
if ($LASTEXITCODE -ne 0) {
    throw "Python is unavailable. Set PYTHON_EXE to the required interpreter before running this script."
}

$ExpertCheckpointDir = Join-Path $RepoDir "results\mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final"
if (-not (Test-Path -LiteralPath $ExpertCheckpointDir -PathType Container)) {
    throw "Required pretrained checkpoint directory was not found: $ExpertCheckpointDir"
}

$QueueStartedAt = Get-Date
Write-Host "Sequential experiment queue started at $QueueStartedAt"
Write-Host "Repository: $RepoDir"
Write-Host "Python: $PythonExe"

for ($Index = 0; $Index -lt $Experiments.Count; $Index++) {
    $Experiment = $Experiments[$Index]
    if (-not (Test-Path -LiteralPath $Experiment.Script -PathType Leaf)) {
        throw "Experiment script was not found: $($Experiment.Script)"
    }

    $StartedAt = Get-Date
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Starting $($Experiment.Name)"
    Write-Host "Script: $($Experiment.Script)"
    Write-Host "Started at: $StartedAt"
    Write-Host "============================================================"

    # Each child script calls exit, so run it in an isolated PowerShell process.
    & $PowerShellExe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $Experiment.Script
    $ExperimentExitCode = $LASTEXITCODE

    $FinishedAt = Get-Date
    $Elapsed = $FinishedAt - $StartedAt
    if ($ExperimentExitCode -ne 0) {
        throw "$($Experiment.Name) failed with exit code $ExperimentExitCode after $Elapsed. The remaining experiment was not started."
    }

    Write-Host "Completed $($Experiment.Name) at $FinishedAt (elapsed: $Elapsed)"
}

$QueueFinishedAt = Get-Date
$QueueElapsed = $QueueFinishedAt - $QueueStartedAt
Write-Host ""
Write-Host "All sequential experiments completed successfully."
Write-Host "Finished at: $QueueFinishedAt"
Write-Host "Total elapsed: $QueueElapsed"

