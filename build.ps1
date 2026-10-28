param(
    [int]$Samples = 3200,
    [UInt64]$Seed = 11
)

$ErrorActionPreference = 'Stop'

gcc -O2 -std=c99 -Wall -Wextra -o build_dataset.exe build_dataset.c -lm
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
gcc -O2 -std=c99 -Wall -Wextra -o train_model.exe train_model.c -lm
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

.\build_dataset.exe --synthetic $Samples --seed $Seed --out dataset.csv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.\train_model.exe --data dataset.csv --header model_params.h --seed $Seed
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host 'Build complete. Run: python simulator.py --data dataset.csv'
