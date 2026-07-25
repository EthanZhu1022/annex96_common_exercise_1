$ErrorActionPreference = "Stop"

$PythonExe = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location -LiteralPath $RepoDir

$SourceCheckpoint = "results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final"
$SocDir = "results/mappo_grouped_tarmac_soc_regrouping_source_3f_vt_january"
$SocStats = Join-Path $SocDir "soc_statistics.csv"
$SocMetadata = Join-Path $SocDir "soc_collection_metadata.json"
$Seeds = @(42, 0, 1)

if ((Test-Path -LiteralPath $SocStats -PathType Leaf) -and
    (Test-Path -LiteralPath $SocMetadata -PathType Leaf)) {
    Write-Host "Reusing completed SOC collection in $SocDir"
}
else {
    & $PythonExe -m mappo_grouped_tarmac_soc_regrouping.collect_soc `
        --checkpoint $SourceCheckpoint `
        --output_dir $SocDir `
        --climate VT `
        --n_buildings 25 `
        --collection_month 1 `
        --seed 42
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-SocRegroupingVariant {
    param(
        [Parameter(Mandatory = $true)]
        [string]$GroupingMode,
        [Parameter(Mandatory = $true)]
        [string]$ResultPrefix
    )

    foreach ($Seed in $Seeds) {
        $SaveDir = "results/${ResultPrefix}_seed${Seed}"
        $CheckpointPath = Join-Path $SaveDir "checkpoint.pt"
        $MetricsPath = Join-Path $SaveDir "latest_metrics.json"
        if ((Test-Path -LiteralPath $CheckpointPath -PathType Leaf) -and
            (Test-Path -LiteralPath $MetricsPath -PathType Leaf)) {
            Write-Host "Skipping completed $GroupingMode, seed=${Seed}: $SaveDir"
            continue
        }
        $WandbName = Split-Path -Leaf $SaveDir
        Write-Host "Starting $GroupingMode, seed=$Seed, save_dir=$SaveDir"

        & $PythonExe -m mappo_grouped_tarmac_soc_regrouping.train `
            --soc_statistics_path $SocStats `
            --soc_grouping_mode $GroupingMode `
            --climate VT `
            --n_buildings 25 `
            --group_k_candidates 4 5 `
            --cluster_seed 0 `
            --cluster_retries 10 `
            --grouping_method agglomerative `
            --grouping_feature_month 1 `
            --n_episodes 500 `
            --train_month 1 `
            --test_month 2 `
            --seed $Seed `
            --comm_fusion_mode linear `
            --wandb_name $WandbName `
            --save_dir $SaveDir
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

Invoke-SocRegroupingVariant `
    -GroupingMode "soc6f" `
    -ResultPrefix "mappo_grouped_tarmac_soc6f_agglomerative_linear_vt_500"

Invoke-SocRegroupingVariant `
    -GroupingMode "energy4f" `
    -ResultPrefix "mappo_grouped_tarmac_energy4f_agglomerative_linear_vt_500"
