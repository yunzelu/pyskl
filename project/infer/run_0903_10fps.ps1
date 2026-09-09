param(
    [string]$Python = 'C:/Users/elmbo/.conda/envs/har/python.exe',
    [string]$InputRoot = 'D:/lu/project/auto_labeling_pipeline/outputs/0903',
    [string]$OutputDir = 'data/project/inference/0903_10fps',
    [string]$Device = 'cpu',
    [int]$BatchSize = 128,
    [int]$ChunkSize = 10000,
    [int]$NumThreads = 2,
    [int]$MaxWindows = 0,
    [switch]$ValidateOnly,
    [switch]$Overwrite
)

$ErrorActionPreference = 'Stop'
$projectRepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$projectPreviousPythonPath = $env:PYTHONPATH
$projectRuntimeCache = Join-Path $projectRepoRoot '.cache/project_infer_mmcv1'
$projectArguments = @(
    'project/infer/infer_csv.py',
    (Join-Path $InputRoot 'pose_2026-06-23_ds_cf.csv'),
    (Join-Path $InputRoot 'pose_2026-07-20_ds_cf.csv'),
    '--output-dir', $OutputDir,
    '--device', $Device,
    '--batch-size', $BatchSize,
    '--chunk-size', $ChunkSize,
    '--num-threads', $NumThreads
)
if ($ValidateOnly) { $projectArguments += '--validate-only' }
if ($Overwrite) { $projectArguments += '--overwrite' }
if ($MaxWindows -gt 0) { $projectArguments += @('--max-windows', $MaxWindows) }
if ($MaxWindows -lt 0) { throw 'MaxWindows must be nonnegative' }

# This optional local cache supplies MMCV 1.x without changing the har environment.
# A compatible training environment can run infer_csv.py directly without this cache.
Push-Location $projectRepoRoot
try {
    if (Test-Path -LiteralPath (Join-Path $projectRuntimeCache 'mmcv')) {
        $env:PYTHONPATH = $projectRuntimeCache
        if ($projectPreviousPythonPath) {
            $env:PYTHONPATH += [System.IO.Path]::PathSeparator + $projectPreviousPythonPath
        }
    }
    & $Python @projectArguments
    if ($LASTEXITCODE -ne 0) { throw "Inference exited with code $LASTEXITCODE" }
}
finally {
    $env:PYTHONPATH = $projectPreviousPythonPath
    Pop-Location
}
