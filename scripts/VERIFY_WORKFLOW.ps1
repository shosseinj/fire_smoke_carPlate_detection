$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Required = @(
    ".opencode\agents\live-branch-orchestrator.md",
    ".opencode\agents\live-branch-architect.md",
    ".opencode\agents\live-branch-ai-guardian.md",
    ".opencode\agents\live-branch-gpu-implementer.md",
    ".opencode\agents\live-branch-frontend-integrator.md",
    ".opencode\agents\live-branch-browser-tester.md",
    ".opencode\agents\live-branch-runtime-validator.md",
    ".opencode\agents\live-branch-final-reviewer.md",
    ".opencode\commands\implement-decode-tee-live-branch.md",
    ".opencode\commands\audit-decode-tee-live-branch.md",
    ".opencode\commands\test-live-branch-dashboard.md",
    ".agentic\live-branch\spec.md"
)

$Missing = @()
foreach ($Item in $Required) {
    if (-not (Test-Path (Join-Path $Root $Item))) { $Missing += $Item }
}
if ($Missing.Count -gt 0) {
    throw "Missing workflow files: $($Missing -join ', ')"
}
Write-Host "Workflow structure verified: $($Required.Count) required files present."
