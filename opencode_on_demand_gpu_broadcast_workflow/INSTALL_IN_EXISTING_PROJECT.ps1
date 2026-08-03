$ErrorActionPreference = "Stop"
$Source = Split-Path -Parent $MyInvocation.MyCommand.Path
$Destination = (Get-Location).Path
$Python = "C:/Users/jafari.h/Desktop/ai_project/.venv/Scripts/python.exe"

function Merge-Directory {
    param(
        [Parameter(Mandatory = $true)][string]$SourceDirectory,
        [Parameter(Mandatory = $true)][string]$DestinationDirectory
    )

    if (-not (Test-Path -LiteralPath $SourceDirectory -PathType Container)) {
        throw "Workflow source directory not found: $SourceDirectory"
    }

    New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
    $sourceRoot = (Get-Item -LiteralPath $SourceDirectory).FullName

    Get-ChildItem -LiteralPath $sourceRoot -Recurse -Force | ForEach-Object {
        $relativePath = $_.FullName.Substring($sourceRoot.Length).TrimStart('\', '/')
        $targetPath = Join-Path $DestinationDirectory $relativePath

        if ($_.PSIsContainer) {
            New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
        }
        else {
            $targetParent = Split-Path -Parent $targetPath
            New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $targetPath -Force
        }
    }
}

Write-Host "Installing OpenCode workflow into: $Destination"
Merge-Directory -SourceDirectory (Join-Path $Source ".opencode") -DestinationDirectory (Join-Path $Destination ".opencode")
Merge-Directory -SourceDirectory (Join-Path $Source "docs") -DestinationDirectory (Join-Path $Destination "docs")
Merge-Directory -SourceDirectory (Join-Path $Source "scripts") -DestinationDirectory (Join-Path $Destination "scripts")
Merge-Directory -SourceDirectory (Join-Path $Source ".agentic\broadcast") -DestinationDirectory (Join-Path $Destination ".agentic\broadcast")

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Configured Python interpreter not found: $Python"
}

& $Python (Join-Path $Destination "scripts\validate_broadcast_workflow.py")
Write-Host "Run OpenCode and enter: /implement-on-demand-gpu-broadcast"
