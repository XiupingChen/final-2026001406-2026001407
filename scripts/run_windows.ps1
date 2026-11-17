param(
    [string]$Python = '.venv\Scripts\python.exe',
    [string]$Rscript = 'Rscript',
    [string]$Rtools = '',
    [string]$Output = 'results/windows-20261117'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:RSCRIPT = (Get-Command $Rscript -ErrorAction Stop).Source
$env:LC_ALL = 'C'
$env:LANG = 'C'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
if ($Rtools) {
    $env:RTOOLS43_HOME = $Rtools
    $env:PATH = "$Rtools\usr\bin;$Rtools\x86_64-w64-mingw32.static.posix\bin;$env:PATH"
}
function Invoke-Python {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw "Python failed ($LASTEXITCODE): $args" }
}
if (Test-Path $Output) { throw "Choose a new output directory: $Output" }
New-Item -ItemType Directory -Path $Output | Out-Null
if (!(Test-Path 'data/cleaned/cleaned.csv')) {
    Invoke-Python -m creditbench.clean --input data/raw/default_of_credit_card_clients.xls --output data/cleaned
}
Copy-Item data/cleaned/audit.json "$Output/cleaning_audit.json"
Invoke-Python -m pytest -q -ra --junitxml "$Output/tests.xml"
foreach ($stage in @('smoke', 'phase1_prediction', 'phase2_parallel', 'phase3_scaling')) {
    $prepared = "data/prepared/$(Split-Path $Output -Leaf)-$stage"
    Invoke-Python -m creditbench.data --config "configs/$stage.json" --cleaned data/cleaned/cleaned.csv --output $prepared
    Invoke-Python -m creditbench.run --config "configs/$stage.json" --prepared $prepared --engines sklearn ranger --output "$Output/$stage"
}
Invoke-Python -m creditbench.benchmark_kernel "$Output/phase2_parallel"
Invoke-Python -m creditbench.educational "$Output/phase1_prediction"
Invoke-Python -m creditbench.sensitivity --cleaned data/cleaned/cleaned.csv --output "$Output/sensitivity" --phase1 "$Output/phase1_prediction"
Invoke-Python scripts/verify_windows.py $Output
Write-Output "Windows experiments completed: $Output"
