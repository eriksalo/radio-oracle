# Radio Oracle — Download Knowledge Bases (Windows)
# Usage:
#   .\scripts\download_knowledge.ps1              # download all
#   .\scripts\download_knowledge.ps1 -DryRun      # preview only
#   .\scripts\download_knowledge.ps1 -Source wiki  # download one source
#
# Downloads ~60GB total. Run from the radio-oracle repo root.

param(
    [switch]$DryRun,
    [string]$Source = "all",
    [string]$OutDir = "data\knowledge"
)

$ErrorActionPreference = "Stop"

# ── Source definitions ──────────────────────────────────────────────────────

$Sources = [ordered]@{
    wiki = @{
        Name = "Wikipedia EN (text, no images)"
        Url  = "https://download.kiwix.org/zim/wikipedia/wikipedia_en_all_nopic_latest.zim"
        Size = "22 GB"
    }
    ifixit = @{
        Name = "iFixit repair guides"
        Url  = "https://download.kiwix.org/zim/other/ifixit_en_all_latest.zim"
        Size = "2.5 GB"
    }
    wikibooks = @{
        Name = "Wikibooks"
        Url  = "https://download.kiwix.org/zim/wikibooks/wikibooks_en_all_latest.zim"
        Size = "2 GB"
    }
    wikimed = @{
        Name = "WikiMed medical encyclopedia"
        Url  = "https://download.kiwix.org/zim/wikipedia/wikipedia_en_medicine_nopic_latest.zim"
        Size = "1 GB"
    }
}

# ── Models ──────────────────────────────────────────────────────────────────

$Models = [ordered]@{
    whisper = @{
        Name = "Whisper small.en (STT)"
        Url  = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin"
        File = "models\whisper-small.en.bin"
        Size = "460 MB"
    }
    piper = @{
        Name = "Piper lessac-medium (TTS)"
        Url  = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
        File = "models\en_US-lessac-medium.onnx"
        Size = "75 MB"
    }
    piper_json = @{
        Name = "Piper lessac-medium config"
        Url  = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
        File = "models\en_US-lessac-medium.onnx.json"
        Size = "<1 MB"
    }
}

# ── Helper ──────────────────────────────────────────────────────────────────

function Download-File {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$Label,
        [string]$Size
    )

    if (Test-Path $Destination) {
        Write-Host "  SKIP  $Label — already exists at $Destination" -ForegroundColor Yellow
        return
    }

    if ($DryRun) {
        Write-Host "  WOULD DOWNLOAD  $Label (~$Size)" -ForegroundColor Cyan
        Write-Host "    $Url"
        Write-Host "    -> $Destination"
        return
    }

    $dir = Split-Path $Destination -Parent
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    $tempFile = "$Destination.downloading"
    Write-Host "  DOWNLOADING  $Label (~$Size)..." -ForegroundColor Green
    Write-Host "    $Url"

    try {
        # Use BITS for large files (supports resume), fall back to Invoke-WebRequest
        if ($Size -match "^\d+ GB$") {
            # Large file — use BITS transfer (resumable)
            try {
                Import-Module BitsTransfer -ErrorAction Stop
                Start-BitsTransfer -Source $Url -Destination $tempFile -DisplayName $Label
            } catch {
                Write-Host "    BITS unavailable, using WebRequest (no resume)..." -ForegroundColor Yellow
                $ProgressPreference = 'SilentlyContinue'  # speeds up Invoke-WebRequest significantly
                Invoke-WebRequest -Uri $Url -OutFile $tempFile -UseBasicParsing
                $ProgressPreference = 'Continue'
            }
        } else {
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $Url -OutFile $tempFile -UseBasicParsing
            $ProgressPreference = 'Continue'
        }

        Move-Item $tempFile $Destination -Force
        $fileSize = (Get-Item $Destination).Length / 1MB
        Write-Host "    Done — $([math]::Round($fileSize, 1)) MB" -ForegroundColor Green
    } catch {
        Write-Host "    FAILED: $_" -ForegroundColor Red
        if (Test-Path $tempFile) { Remove-Item $tempFile -Force }
    }
}

# ── Main ────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "  Radio Oracle — Knowledge Base Downloader (Windows)" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor White
Write-Host ""

if ($DryRun) {
    Write-Host "  [DRY RUN MODE — nothing will be downloaded]" -ForegroundColor Cyan
    Write-Host ""
}

# Download models
Write-Host "── Models ──────────────────────────────────────────────" -ForegroundColor White
Write-Host ""

foreach ($key in $Models.Keys) {
    $m = $Models[$key]
    Download-File -Url $m.Url -Destination $m.File -Label $m.Name -Size $m.Size
}

Write-Host ""

# Download knowledge bases
Write-Host "── Knowledge Bases ─────────────────────────────────────" -ForegroundColor White
Write-Host ""

$toDownload = if ($Source -eq "all") { $Sources.Keys } else { @($Source) }

foreach ($key in $toDownload) {
    if (-not $Sources.ContainsKey($key)) {
        Write-Host "  Unknown source: $key" -ForegroundColor Red
        Write-Host "  Available: $($Sources.Keys -join ', ')" -ForegroundColor Red
        continue
    }

    $s = $Sources[$key]
    $filename = [System.IO.Path]::GetFileName([System.Uri]::new($s.Url).LocalPath)
    $destination = Join-Path $OutDir $filename

    Download-File -Url $s.Url -Destination $destination -Label $s.Name -Size $s.Size
}

# Summary
Write-Host ""
Write-Host "── Next Steps ──────────────────────────────────────────" -ForegroundColor White
Write-Host ""
Write-Host "  1. Manual downloads (not automated):" -ForegroundColor White
Write-Host "     - Project Gutenberg: https://www.gutenberg.org/robot/harvest"
Write-Host "     - Stack Exchange:    https://archive.org/details/stackexchange"
Write-Host "     - Army field manuals from archive.org"
Write-Host ""
Write-Host "  2. Transfer everything to the Jetson:" -ForegroundColor White
Write-Host "     scp -r data\knowledge\ user@jetson:/opt/radio-oracle/data/knowledge/"
Write-Host "     scp -r models\ user@jetson:/opt/radio-oracle/models/"
Write-Host ""
Write-Host "     Or use WinSCP / rsync via WSL for resumable transfers."
Write-Host ""
Write-Host "  3. On the Jetson (or workstation), run ingestion:" -ForegroundColor White
Write-Host "     python scripts/ingest_wikipedia.py data/knowledge/wikipedia_en_all_nopic_latest.zim"
Write-Host "     python scripts/ingest_generic_zim.py data/knowledge/ifixit_en_all_latest.zim --collection ifixit"
Write-Host "     python scripts/ingest_generic_zim.py data/knowledge/wikibooks_en_all_latest.zim --collection wikibooks"
Write-Host "     python scripts/ingest_generic_zim.py data/knowledge/wikipedia_en_medicine_nopic_latest.zim --collection wikimed"
Write-Host ""
