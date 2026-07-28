param(
    [int]$FirstSourceIndex = 0,
    [int]$SecondSourceIndex = 6,
    [int]$GstDurationSeconds = 65,
    [int]$ChildDurationSeconds = 70
)

$ErrorActionPreference = "Stop"
$service = "video-ai-router"
$container = "merged_video_ai_router-video-ai-router-1"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outputDirectory = Join-Path $PSScriptRoot "..\artifacts\rtsp-offline-$stamp"
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

$confirmation = Read-Host "Disconnect Internet and connect the camera LAN. Type YES to continue"
if ($confirmation -ne "YES") {
    throw "Test cancelled. Disconnect Internet before running this script."
}

function Invoke-GstProbe {
    param([int]$Index)

    $discoverPath = Join-Path $outputDirectory "source-$Index-discover.txt"
    & docker exec $container python3 /workspace/scripts/gst_rtsp_probe.py `
        --index $Index --duration 15 --codec discover *>&1 |
        Tee-Object -FilePath $discoverPath

    $discoverText = Get-Content -LiteralPath $discoverPath -Raw
    if ($discoverText -match "encoding-name=\(string\)H265|encoding-name=\(string\)HEVC") {
        $codec = "h265"
    }
    elseif ($discoverText -match "encoding-name=\(string\)H264") {
        $codec = "h264"
    }
    else {
        throw "No H264/H265 caps found for source index $Index. Send file: $discoverPath"
    }

    $streamPath = Join-Path $outputDirectory "source-$Index-$codec-$GstDurationSeconds-sec.txt"
    & docker exec $container python3 /workspace/scripts/gst_rtsp_probe.py `
        --index $Index --duration $GstDurationSeconds --codec $codec *>&1 |
        Tee-Object -FilePath $streamPath
    return $codec
}

function Invoke-ChildProbe {
    param([int]$Index)

    $redactedId = (
        & docker exec $container python3 /workspace/scripts/gst_rtsp_probe.py `
            --index $Index --redacted-only
    ).Trim()
    if (-not $redactedId.StartsWith("rtsp")) {
        throw "Could not resolve the redacted source ID for index $Index."
    }

    $env:VIDEO_ONLY_MODE = "true"
    $env:DEEPSTREAM_MAX_ACTIVE_SOURCES = "1"
    $env:DEEPSTREAM_SOURCE_OPEN_STAGGER_SECONDS = "2"
    $env:DEEPSTREAM_SOURCE_ALLOWLIST = $redactedId
    docker compose up -d --force-recreate $service |
        Tee-Object -FilePath (Join-Path $outputDirectory "source-$Index-compose.txt")

    $probeOutput = Join-Path $outputDirectory "source-$Index-websocket.txt"
    $probeError = Join-Path $outputDirectory "source-$Index-websocket-error.txt"
    $probe = Start-Process -FilePath "python" -WindowStyle Hidden -PassThru `
        -ArgumentList @(
            "scripts/video_ws_probe.py",
            "ws://127.0.0.1:9999/api/v1/video-wall/ws?wall=true&batch=true",
            "--duration",
            "$ChildDurationSeconds"
        ) `
        -RedirectStandardOutput $probeOutput `
        -RedirectStandardError $probeError

    $healthPath = Join-Path $outputDirectory "source-$Index-health.jsonl"
    $deadline = (Get-Date).AddSeconds($ChildDurationSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $health = Invoke-RestMethod -TimeoutSec 4 -Uri "http://127.0.0.1:9999/health"
            $sample = [ordered]@{
                timestamp = (Get-Date).ToUniversalTime().ToString("o")
                video_ingestor = $health.video_ingestor
                frontend_frame_worker = $health.frontend_frame_worker
            }
            $sample | ConvertTo-Json -Depth 12 -Compress |
                Add-Content -LiteralPath $healthPath -Encoding UTF8
        }
        catch {
            [ordered]@{
                timestamp = (Get-Date).ToUniversalTime().ToString("o")
                health_error = $_.Exception.GetType().Name
            } | ConvertTo-Json -Compress |
                Add-Content -LiteralPath $healthPath -Encoding UTF8
        }
        Start-Sleep -Seconds 5
    }

    if (-not $probe.HasExited) {
        $probe.WaitForExit(10000) | Out-Null
    }
    docker compose ps --all |
        Out-File (Join-Path $outputDirectory "source-$Index-compose-ps.txt") -Encoding UTF8
    docker inspect $container --format "{{json .State}}" |
        Out-File (Join-Path $outputDirectory "source-$Index-container-state.json") -Encoding UTF8
    docker compose logs --tail 400 $service |
        Out-File (Join-Path $outputDirectory "source-$Index-service.log") -Encoding UTF8
}

docker compose up -d $service | Out-Null
docker exec $container gst-launch-1.0 --version |
    Out-File (Join-Path $outputDirectory "gstreamer-version.txt") -Encoding UTF8
docker exec $container gst-inspect-1.0 rtspsrc |
    Out-File (Join-Path $outputDirectory "rtspsrc-inspect.txt") -Encoding UTF8
docker exec $container gst-inspect-1.0 rtph264depay |
    Out-File (Join-Path $outputDirectory "rtph264depay-inspect.txt") -Encoding UTF8
docker exec $container gst-inspect-1.0 rtph265depay |
    Out-File (Join-Path $outputDirectory "rtph265depay-inspect.txt") -Encoding UTF8

$firstCodec = Invoke-GstProbe -Index $FirstSourceIndex
Invoke-ChildProbe -Index $FirstSourceIndex

$secondCodec = Invoke-GstProbe -Index $SecondSourceIndex
Invoke-ChildProbe -Index $SecondSourceIndex

Remove-Item Env:DEEPSTREAM_SOURCE_ALLOWLIST -ErrorAction SilentlyContinue
Write-Host "All diagnostic output was saved in:"
Write-Host (Resolve-Path $outputDirectory)
Write-Host "codec source $FirstSourceIndex = $firstCodec"
Write-Host "codec source $SecondSourceIndex = $secondCodec"
