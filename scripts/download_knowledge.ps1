# Radio Oracle - Download Knowledge Bases (Windows)
# Usage:
#   .\download_knowledge.ps1              # download all
#   .\download_knowledge.ps1 -DryRun      # preview only
#   .\download_knowledge.ps1 -Source wiki  # download one source
#
# Run from wherever you want files saved.

param(
    [switch]$DryRun,
    [string]$Source = "all",
    [string]$OutDir = ".\knowledge"
)

$ErrorActionPreference = "Stop"

# -- Source definitions -------------------------------------------------------

$Sources = @{
    "wiki" = @{
        Name = "Wikipedia EN (text, no images)"
        Url  = "https://download.kiwix.org/zim/wikipedia/wikipedia_en_all_nopic_latest.zim"
        Size = "22 GB"
    }
    "ifixit" = @{
        Name = "iFixit repair guides"
        Url  = "https://download.kiwix.org/zim/other/ifixit_en_all_latest.zim"
        Size = "2.5 GB"
    }
    "wikibooks" = @{
        Name = "Wikibooks"
        Url  = "https://download.kiwix.org/zim/wikibooks/wikibooks_en_all_latest.zim"
        Size = "2 GB"
    }
    "wikimed" = @{
        Name = "WikiMed medical encyclopedia"
        Url  = "https://download.kiwix.org/zim/wikipedia/wikipedia_en_medicine_nopic_latest.zim"
        Size = "1 GB"
    }
    "crashcourse" = @{
        Name = "CrashCourse educational videos"
        Url  = "https://download.kiwix.org/zim/other/crashcourse_en_all_latest.zim"
        Size = "44 GB"
    }
    "gutenberg" = @{
        Name = "Project Gutenberg books"
        Url  = "https://download.kiwix.org/zim/gutenberg/gutenberg_mul_all_latest.zim"
        Size = "75 GB"
    }
}

# -- Models -------------------------------------------------------------------

$Models = @{
    "whisper" = @{
        Name = "Whisper small.en (STT)"
        Url  = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin"
        File = "models\whisper-small.en.bin"
        Size = "460 MB"
    }
    "piper" = @{
        Name = "Piper lessac-medium (TTS)"
        Url  = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
        File = "models\en_US-lessac-medium.onnx"
        Size = "75 MB"
    }
    "piper_json" = @{
        Name = "Piper lessac-medium config"
        Url  = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
        File = "models\en_US-lessac-medium.onnx.json"
        Size = "1 MB"
    }
}

# -- Helper -------------------------------------------------------------------

function Download-File {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$Label,
        [string]$Size
    )

    if (Test-Path $Destination) {
        Write-Host "  SKIP  $Label - already exists at $Destination" -ForegroundColor Yellow
        return
    }

    if ($DryRun) {
        Write-Host "  WOULD DOWNLOAD  $Label (~$Size)" -ForegroundColor Cyan
        Write-Host "    $Url"
        Write-Host "    -> $Destination"
        return
    }

    $dir = Split-Path $Destination -Parent
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    $tempFile = "$Destination.downloading"
    Write-Host "  DOWNLOADING  $Label (~$Size)..." -ForegroundColor Green
    Write-Host "    $Url"

    try {
        # Use BITS for large files (resumable), fall back to Invoke-WebRequest
        $isLarge = $Size -match "GB"
        if ($isLarge) {
            try {
                Import-Module BitsTransfer -ErrorAction Stop
                Start-BitsTransfer -Source $Url -Destination $tempFile -DisplayName $Label
            }
            catch {
                Write-Host "    BITS failed, using WebRequest (no resume)..." -ForegroundColor Yellow
                $ProgressPreference = 'SilentlyContinue'
                Invoke-WebRequest -Uri $Url -OutFile $tempFile -UseBasicParsing
                $ProgressPreference = 'Continue'
            }
        }
        else {
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $Url -OutFile $tempFile -UseBasicParsing
            $ProgressPreference = 'Continue'
        }

        Move-Item $tempFile $Destination -Force
        $fileSize = [math]::Round((Get-Item $Destination).Length / 1MB, 1)
        Write-Host "    Done - $fileSize MB" -ForegroundColor Green
    }
    catch {
        Write-Host "    FAILED: $_" -ForegroundColor Red
        if (Test-Path $tempFile) { Remove-Item $tempFile -Force }
    }
}

# -- Main ---------------------------------------------------------------------

Write-Host ""
Write-Host "========================================================" -ForegroundColor White
Write-Host "  Radio Oracle - Knowledge Base Downloader (Windows)"     -ForegroundColor White
Write-Host "========================================================" -ForegroundColor White
Write-Host ""

if ($DryRun) {
    Write-Host "  [DRY RUN MODE - nothing will be downloaded]" -ForegroundColor Cyan
    Write-Host ""
}

# Download models
Write-Host "-- Models --------------------------------------------------" -ForegroundColor White
Write-Host ""

foreach ($key in $Models.Keys) {
    $m = $Models[$key]
    Download-File -Url $m.Url -Destination $m.File -Label $m.Name -Size $m.Size
}

Write-Host ""

# Download knowledge bases
Write-Host "-- Knowledge Bases -----------------------------------------" -ForegroundColor White
Write-Host ""

if ($Source -eq "all") {
    $toDownload = $Sources.Keys
}
else {
    $toDownload = @($Source)
}

foreach ($key in $toDownload) {
    if (-not $Sources.ContainsKey($key)) {
        Write-Host "  Unknown source: $key" -ForegroundColor Red
        $available = $Sources.Keys -join ", "
        Write-Host "  Available: $available" -ForegroundColor Red
        continue
    }

    $s = $Sources[$key]
    $uri = New-Object System.Uri($s.Url)
    $filename = [System.IO.Path]::GetFileName($uri.LocalPath)
    $destination = Join-Path $OutDir $filename

    Download-File -Url $s.Url -Destination $destination -Label $s.Name -Size $s.Size
}

# Summary
Write-Host ""
Write-Host "-- Next Steps ----------------------------------------------" -ForegroundColor White
Write-Host ""
Write-Host "  Transfer to the Jetson:" -ForegroundColor White
Write-Host "    scp -r knowledge\ user@jetson:/opt/radio-oracle/data/knowledge/"
Write-Host "    scp -r models\ user@jetson:/opt/radio-oracle/models/"
Write-Host ""
Write-Host "  Or use WinSCP for resumable transfers (recommended for large files)."
Write-Host ""
Write-Host "  Then on the Jetson, run ingestion (see docs/SETUP.md Part 3)."
Write-Host ""
