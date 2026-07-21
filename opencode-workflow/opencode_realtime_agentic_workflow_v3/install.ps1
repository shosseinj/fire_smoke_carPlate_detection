param(
    [string]$Target = ".",
    [string]$OptionsExcel = "",
    [string]$Sheet = "",
    [switch]$Force,
    [switch]$ReplaceCatalogProgress
)

$ErrorActionPreference = "Stop"
$Installer = Join-Path $PSScriptRoot "installer\install.py"
$ArgsList = @($Installer, "--target", $Target)
if ($OptionsExcel) { $ArgsList += @("--options-excel", $OptionsExcel) }
if ($Sheet) { $ArgsList += @("--sheet", $Sheet) }
if ($Force) { $ArgsList += "--force" }
if ($ReplaceCatalogProgress) { $ArgsList += "--replace-catalog-progress" }
python @ArgsList
