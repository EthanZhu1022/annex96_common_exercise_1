$ErrorActionPreference = "Stop"

$PythonExe = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location -LiteralPath $RepoDir

$SourceCheckpoint = "results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final"
$SocDir = "results/mappo_grouped_tarmac_soc_regrouping_source_3f_vt_january"
$SocStats = Join-Path $SocDir "soc_statistics.csv"
$Soc6fSaveDir = "results/mappo_grouped_tarmac_soc6f_agglomerative_linear_vt_500_seed42"
$Energy4fSaveDir = "results/mappo_grouped_tarmac_energy4f_agglomerative_linear_vt_500_seed42"

& $PythonExe -m mappo_grouped_tarmac_soc_regrouping.collect_soc `
    --checkpoint $SourceCheckpoint `
    --output_dir $SocDir `
    --climate VT `
    --n_buildings 25 `
    --collection_month 1 `
    --seed 42
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe -m mappo_grouped_tarmac_soc_regrouping.train `
    --soc_statistics_path $SocStats `
    --soc_grouping_mode soc6f `
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
    --seed 42 `
    --comm_fusion_mode linear `
    --wandb_name (Split-Path -Leaf $Soc6fSaveDir) `
    --save_dir $Soc6fSaveDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe -m mappo_grouped_tarmac_soc_regrouping.train `
    --soc_statistics_path $SocStats `
    --soc_grouping_mode energy4f `
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
    --seed 42 `
    --comm_fusion_mode linear `
    --wandb_name (Split-Path -Leaf $Energy4fSaveDir) `
    --save_dir $Energy4fSaveDir
exit $LASTEXITCODE
