$ErrorActionPreference = "Stop"

$Source = Split-Path -Parent $MyInvocation.MyCommand.Path
$Destination = (Get-Location).Path

Write-Host "Installing decode-tee live-branch workflow into: $Destination"

function Merge-Directory {
    param(
        [Parameter(Mandatory = $true)][string]$From,
        [Parameter(Mandatory = $true)][string]$To
    )

    if (-not (Test-Path $From)) { return }
    New-Item -ItemType Directory -Path $To -Force | Out-Null

    Get-ChildItem -LiteralPath $From -Force | ForEach-Object {
        $target = Join-Path $To $_.Name
        if ($_.PSIsContainer) {
            Merge-Directory -From $_.FullName -To $target
        }
        else {
            Copy-Item -LiteralPath $_.FullName -Destination $target -Force
        }
    }
}

Merge-Directory -From (Join-Path $Source ".opencode") -To (Join-Path $Destination ".opencode")
Merge-Directory -From (Join-Path $Source ".agentic") -To (Join-Path $Destination ".agentic")
Merge-Directory -From (Join-Path $Source "scripts") -To (Join-Path $Destination "scripts")

Write-Host "Installed agents:"
Get-ChildItem (Join-Path $Destination ".opencode\agents") -Filter "live-branch-*.md" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
Write-Host "Installed commands:"
Get-ChildItem (Join-Path $Destination ".opencode\commands") -Filter "*live-branch*.md" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
Write-Host "Done. Start OpenCode and run /implement-decode-tee-live-branch"
