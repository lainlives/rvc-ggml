# RVC-Train-GGML - practical training pipeline driver (ASCII only)
#
# Why a script: rvc-prep (ggml backend) and rvc-train (traincpp backend) MUST run
# in separate processes (project rule R28), so a single exe cannot do prep+train.
# This driver wires them with YOUR paths and parameters.
#
# Typical usage:
#   powershell -ExecutionPolicy Bypass -File scripts\train_pipeline.ps1 `
#       -Audio "D:\myvoice" -Work "work\myvoice" -Mode finetune `
#       -G work\models\f0G40k.train.gguf -D work\models\f0D40k.train.gguf `
#       -Epochs 50 -ExportEvery 5 -Infer -InferInput input\test.wav `
#       -InferModelDir <dir with hubert.gguf / rmvpe.gguf / your model gguf>
#
# Reuse an existing cache (skip prep):
#   ... -Cache work\myvoice -SkipPrep ...
#
# Smoke / dry run:
#   ... -DryRun          (print the exact commands, do nothing)
#
# All parameters have sane defaults; override only what you need.

[CmdletBinding()]
param(
    # ---- data / directories (choose anything) ----
    [string]$Audio = "",            # raw wav DIR -> rvc-prep builds a cache
    [string]$Cache = "",            # an EXISTING cache dir (dataset.json); skips prep
    [string]$Work  = "",            # cache dir to create from -Audio (default work\rvc_project)
    [string]$Out   = "",            # training output dir (default work\train_<cache name>)
    [string]$Config = "",           # base train config (default configs\v1-40k-f0.json)

    # ---- model source ----
    [ValidateSet("finetune", "scratch")]
    [string]$Mode = "finetune",     # finetune: load G/D; scratch: random init
    [string]$G = "",                # training-state G gguf (finetune)
    [string]$D = "",                # training-state D gguf (finetune)

    # ---- preprocessing options (only used when -Audio is given) ----
    [int]$Sr = 40000,
    [string]$Version = "v1",
    [int]$HasF0 = 1,
    [string]$Hubert = "",           # default <root>\..\rvc-ggml\models\hubert.gguf
    [string]$Rmvpe = "",            # default <root>\..\rvc-ggml\models\rmvpe.gguf
    [int]$PrepDevice = 0,           # rvc-prep device: 0=Vulkan, 99=CPU

    # ---- training options (CLI overrides; unknown ones go in -Config) ----
    [int]$Epochs = 20,
    [int]$Batch = 1,
    [int]$Bucket = 384,             # graph T; must be >= segment_frames
    [int]$DiscLen = 12800,          # D input len; 0 = official whole segment_size
    [int]$GradAccum = 1,            # effective batch = Batch * GradAccum
    [int]$SaveInterval = 1,         # epoch snapshot interval (0 = only final)
    [int]$ExportEvery = 1,          # export inference gguf every N epochs (0 = off)
    [int]$SnapshotKeep = 0,         # keep N epoch snapshots (0 = keep all)
    [int]$Device = 1,               # rvc-train device: 1=Vulkan, 0=CPU (debug)
    [string]$Resume = "",           # resume from a *.ckpt.gguf (with its .meta.json)
    [int]$MaxSteps = 0,             # >0: stop after N optimizer steps

    # ---- config-only hyperparameters (NaN / -1 = keep base config) ----
    [double]$Lr = [double]::NaN,
    [double]$LrDecay = [double]::NaN,
    [double]$CMel = [double]::NaN,
    [double]$CKl = [double]::NaN,
    [double]$Beta1 = [double]::NaN,
    [double]$Beta2 = [double]::NaN,
    [double]$Eps = [double]::NaN,
    [double]$WeightDecay = [double]::NaN,
    [int]$Seed = -1,

    # ---- steps ----
    [switch]$PrepOnly,
    [switch]$SkipPrep,
    [switch]$Export,                # additionally export an inference gguf after training
    [switch]$BuildIndex,            # build a retrieval index from the cache
    [string]$IndexOut = "",          # default <Out>\index.gguf

    # ---- optional closed-loop inference (rvc-prep-independent, ggml backend) ----
    [switch]$Infer,
    [string]$InferInput = "",
    [string]$InferModelDir = "",    # dir with hubert.gguf / rmvpe.gguf
    [string]$RvcGgml = "",          # default <root>\build\bin\RVC-GGML.exe
    [int]$InferDevice = 0,          # RVC-GGML device: 0=Vulkan, 99=CPU
    [int]$Pitch = 0,
    [string]$Index = "",
    [double]$IndexRate = 0.75,

    [switch]$DryRun                 # print commands, do nothing
)

$ErrorActionPreference = "Continue"   # native exe stderr must not abort the script

function Fail([string]$msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

function Run-Tool {
    param([string]$Exe, [string[]]$ArgList, [string]$Log)
    if (-not (Test-Path -LiteralPath $Exe)) { Fail "not found: $Exe" }
    Write-Host ""
    Write-Host ("[cmd] " + $Exe + " " + ($ArgList -join " ")) -ForegroundColor DarkGray
    if ($DryRun) { Write-Host "[dry-run] skipped" -ForegroundColor DarkGray; return 0 }
    & $Exe @ArgList 2>&1 | Tee-Object -FilePath $Log | Out-Host
    $rc = $LASTEXITCODE
    if ($rc -ne 0) { Write-Host "[WARN] exit=$rc (log: $Log)" -ForegroundColor Yellow }
    return $rc
}

# ---------------------------------------------------------------- resolve root/tools
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Root      = Split-Path -Parent $ScriptDir
$TrainExe  = Join-Path $Root "build\bin\rvc-train.exe"
$PrepExe   = Join-Path $Root "build\bin\rvc-prep.exe"
if (-not $Config)     { $Config = Join-Path $Root "configs\v1-40k-f0.json" }
if (-not $RvcGgml)    { $RvcGgml = Join-Path $Root "build\bin\RVC-GGML.exe" }
if (-not $Hubert)     { $Hubert = Join-Path $Root "..\rvc-ggml\models\hubert.gguf" }
if (-not $Rmvpe)      { $Rmvpe = Join-Path $Root "..\rvc-ggml\models\rmvpe.gguf" }
if (-not $G)          { $G = Join-Path $Root "work\models\f0G40k.train.gguf" }
if (-not $D)          { $D = Join-Path $Root "work\models\f0D40k.train.gguf" }

# ---------------------------------------------------------------- interactive fallback
if (-not $Audio -and -not $Cache) {
    $Audio = Read-Host "Raw audio DIR (wav; press Enter to use an existing cache instead)"
}
if (-not $Audio -and -not $Cache) {
    $Cache = Read-Host "Existing cache DIR (must contain dataset.json)"
}
if (-not $Audio -and -not $Cache) { Fail "need -Audio <dir> or -Cache <dir>" }

if (-not $Work) { $Work = Join-Path $Root "work\rvc_project" }
$CachePath = if ($Cache) { $Cache } else { $Work }
$doPrep = (-not $SkipPrep) -and ([string]::IsNullOrEmpty($Cache)) -and (-not [string]::IsNullOrEmpty($Audio))
if ($Cache) { $doPrep = $false }

if (-not $Out) {
    $leaf = Split-Path -Leaf $CachePath
    $Out = Join-Path $Root ("work\train_" + $leaf)
}

Write-Host "==== RVC-Train-GGML training pipeline ====" -ForegroundColor Cyan
Write-Host "  root        : $Root"
Write-Host "  audio       : $(if ($Audio) { $Audio } else { '(none)' })"
Write-Host "  cache       : $CachePath"
Write-Host "  out         : $Out"
Write-Host "  config      : $Config"
Write-Host "  mode        : $Mode"
if ($Mode -eq "finetune") {
    Write-Host "  G / D       : $G`n                $D"
}
Write-Host "  train       : device=$Device epochs=$Epochs batch=$Batch bucket=$Bucket disc=$DiscLen grad=$GradAccum save=$SaveInterval export=$ExportEvery"

if (-not (Test-Path -LiteralPath $TrainExe) -and -not $DryRun) { Fail "rvc-train.exe not found: $TrainExe" }
if ($doPrep -and -not (Test-Path -LiteralPath $PrepExe) -and -not $DryRun) { Fail "rvc-prep.exe not found: $PrepExe" }
if (-not (Test-Path -LiteralPath $Config) -and -not $DryRun) { Fail "config not found: $Config" }
if ($Mode -eq "finetune") {
    if (-not (Test-Path -LiteralPath $G) -and -not $DryRun) { Fail "G gguf not found: $G (use -Mode scratch for from-scratch)" }
    if (-not (Test-Path -LiteralPath $D) -and -not $DryRun) { Fail "D gguf not found: $D (use -Mode scratch for from-scratch)" }
}

New-Item -ItemType Directory -Force -Path $Out | Out-Null
$logDir = Join-Path $Out "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# ---------------------------------------------------------------- step 1: prepare cache
if ($doPrep) {
    $cacheDir = $CachePath
    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
    Write-Host "`n[1/5] preprocessing: rvc-prep" -ForegroundColor Cyan
    $prepArgs = @("--input", $Audio, "--sr", "$Sr", "--version", $Version, "--has-f0", "$HasF0",
                  "--exp", $cacheDir, "--rmvpe", $Rmvpe, "--hubert", $Hubert, "--device", "$PrepDevice")
    $rc = Run-Tool -Exe $PrepExe -ArgList $prepArgs -Log (Join-Path $logDir "prep.log")
    if ($rc -ne 0 -and -not $DryRun) { Fail "rvc-prep failed (exit=$rc)" }
    Write-Host "[1/5] validate cache" -ForegroundColor Cyan
    $rc = Run-Tool -Exe $PrepExe -ArgList @("--validate-cache", $cacheDir) -Log (Join-Path $logDir "validate_cache.log")
    if ($rc -ne 0 -and -not $DryRun) { Fail "cache validation failed (exit=$rc)" }
} else {
    if (-not (Test-Path -LiteralPath (Join-Path $CachePath "dataset.json")) -and -not $DryRun) {
        Fail "cache has no dataset.json: $CachePath"
    }
    Write-Host "`n[1/5] preprocessing: skipped (using existing cache)" -ForegroundColor Cyan
}

# ---------------------------------------------------------------- step 2: optional index
$indexPath = ""
if ($BuildIndex) {
    if (-not $IndexOut) { $IndexOut = Join-Path $Out "index.gguf" }
    Write-Host "`n[2/5] build retrieval index: $IndexOut" -ForegroundColor Cyan
    $rc = Run-Tool -Exe $PrepExe -ArgList @("--index", "--exp", $CachePath, "--out", $IndexOut,
                  "--version", $Version, "--seed", "1234") -Log (Join-Path $logDir "index.log")
    if ($rc -ne 0 -and -not $DryRun) { Fail "index build failed (exit=$rc)" }
    $indexPath = $IndexOut
} else {
    Write-Host "`n[2/5] retrieval index: skipped (use -BuildIndex to create one)" -ForegroundColor Cyan
}

# ---------------------------------------------------------------- optional config overrides
$configToUse = $Config
$overrides = @{}
if (-not [double]::IsNaN($Lr))          { $overrides["learning_rate"] = $Lr }
if (-not [double]::IsNaN($LrDecay))     { $overrides["lr_decay"]     = $LrDecay }
if (-not [double]::IsNaN($CMel))        { $overrides["c_mel"]        = $CMel }
if (-not [double]::IsNaN($CKl))         { $overrides["c_kl"]         = $CKl }
if (-not [double]::IsNaN($Eps))         { $overrides["eps"]          = $Eps }
if (-not [double]::IsNaN($WeightDecay)) { $overrides["weight_decay"] = $WeightDecay }
if ($Seed -ge 0)                        { $overrides["seed"]         = $Seed }
$betasGiven = (-not [double]::IsNaN($Beta1)) -and (-not [double]::IsNaN($Beta2))
if ($overrides.Count -gt 0 -or $betasGiven) {
    $obj = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
    if ($null -eq $obj.train) { $obj | Add-Member -NotePropertyName train -NotePropertyValue ([pscustomobject]@{}) -Force }
    foreach ($k in $overrides.Keys) {
        $obj.train | Add-Member -NotePropertyName $k -NotePropertyValue $overrides[$k] -Force
    }
    if ($betasGiven) { $obj.train | Add-Member -NotePropertyName betas -NotePropertyValue @($Beta1, $Beta2) -Force }
    $configToUse = Join-Path $Out "train_config.json"
    $json = $obj | ConvertTo-Json -Depth 12
    [System.IO.File]::WriteAllText($configToUse, $json)   # UTF-8 no BOM
    Write-Host "[cfg] wrote derived config (overrides): $configToUse" -ForegroundColor DarkGray
}

if ($PrepOnly) {
    Write-Host "`n[done] -PrepOnly: cache ready at $CachePath" -ForegroundColor Green
    exit 0
}

# ---------------------------------------------------------------- step 3: training
Write-Host "`n[3/5] training: rvc-train" -ForegroundColor Cyan
$trainArgs = @("--train", $configToUse, "--exp", $CachePath, "--out", $Out, "--device", "$Device",
               "--epochs", "$Epochs", "--batch", "$Batch", "--bucket", "$Bucket",
               "--disc-len", "$DiscLen", "--grad-accum", "$GradAccum",
               "--save-interval", "$SaveInterval", "--export-every", "$ExportEvery",
               "--snapshot-keep", "$SnapshotKeep",
               "--export-run-json", (Join-Path $Out "run.json"))
if ($Mode -eq "scratch") { $trainArgs += "--scratch" }
else { $trainArgs += @("--weights", $G, "--d-weights", $D) }
if ($Resume)   { $trainArgs += @("--resume", $Resume) }
if ($MaxSteps -gt 0) { $trainArgs += @("--steps", "$MaxSteps") }
$rc = Run-Tool -Exe $TrainExe -ArgList $trainArgs -Log (Join-Path $logDir "train.log")
if ($rc -ne 0 -and -not $DryRun) { Fail "training failed (exit=$rc)" }

# ---------------------------------------------------------------- step 4: export inference gguf
$latestExport = ""
$exports = @(Get-ChildItem -LiteralPath $Out -Filter "export.ep*.gguf" -ErrorAction SilentlyContinue | Sort-Object Name)
if ($exports.Count -gt 0) { $latestExport = $exports[-1].FullName }

if ($Export) {
    Write-Host "`n[4/5] export inference gguf" -ForegroundColor Cyan
    $snap = $null
    $snaps = @(Get-ChildItem -LiteralPath $Out -Filter "*.snapshot.gguf" -ErrorAction SilentlyContinue | Sort-Object Name)
    if ($snaps.Count -gt 0) { $snap = $snaps[-1].FullName }
    if ($latestExport) {
        Write-Host "[4/5] training already exported: $latestExport" -ForegroundColor DarkGray
    } elseif ($snap) {
        $expOut = Join-Path $Out "export.gguf"
        $rc = Run-Tool -Exe $TrainExe -ArgList @("--export-gguf", $snap, "--out", $expOut) -Log (Join-Path $logDir "export.log")
        if ($rc -eq 0) { $latestExport = $expOut }
    } else {
        Write-Host "[4/5] no snapshot to export (set -ExportEvery >= 1 and run enough epochs)" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n[4/5] export: skipped (training exports go to $Out\export.ep*.gguf; use -Export for offline)" -ForegroundColor Cyan
}
if (-not $latestExport) {
    $exports = @(Get-ChildItem -LiteralPath $Out -Filter "export.ep*.gguf" -ErrorAction SilentlyContinue | Sort-Object Name)
    if ($exports.Count -gt 0) { $latestExport = $exports[-1].FullName }
}

# ---------------------------------------------------------------- step 5: optional inference
if ($Infer) {
    Write-Host "`n[5/5] inference (RVC-GGML)" -ForegroundColor Cyan
    if (-not $latestExport)      { Write-Host "[skip] no exported model" -ForegroundColor Yellow }
    elseif (-not $InferInput)    { Write-Host "[skip] -InferInput not set" -ForegroundColor Yellow }
    elseif (-not $InferModelDir) { Write-Host "[skip] -InferModelDir not set (dir with hubert/rmvpe)" -ForegroundColor Yellow }
    else {
        $outWav = Join-Path $Out "converted.wav"
        $inferArgs = @("--input", $InferInput, "--model", $latestExport,
                       "--model-dir", $InferModelDir, "--output", $outWav,
                       "--rms-mix-rate", "0.75", "--pitch", "$Pitch", "--device", "$InferDevice")
        if ($Index)      { $inferArgs += @("--index", $Index, "--index-rate", "$IndexRate") }
        elseif ($indexPath) { $inferArgs += @("--index", $indexPath, "--index-rate", "$IndexRate") }
        $rc = Run-Tool -Exe $RvcGgml -ArgList $inferArgs -Log (Join-Path $logDir "inference.log")
        if ($rc -eq 0 -and -not $DryRun) { Write-Host "[ok] wrote $outWav" -ForegroundColor Green }
    }
} else {
    Write-Host "`n[5/5] inference: skipped (use -Infer -InferInput <wav> -InferModelDir <dir>)" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "==== done ====" -ForegroundColor Green
Write-Host "  cache    : $CachePath"
Write-Host "  output   : $Out"
Write-Host "  checkpoints: $Out\*.ckpt.gguf  (+ .meta.json)"
Write-Host "  exports  : $Out\export.ep*.gguf"
if ($latestExport) { Write-Host "  latest   : $latestExport" }
Write-Host "  logs     : $logDir"
