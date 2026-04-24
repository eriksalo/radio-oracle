# Radio Oracle - Download Knowledge Bases (Windows)
# Usage:
#   .\download_knowledge.ps1              # download all
#   .\download_knowledge.ps1 -DryRun      # preview only
#   .\download_knowledge.ps1 -Source wiki  # download one source
#
# Drop this script into your knowledge folder and run it there.
# Downloads the latest versions of all ZIM files to the current directory.
# Already-downloaded files (exact same filename) are skipped.

param(
    [switch]$DryRun,
    [string]$Source = "all"
)

$ErrorActionPreference = "Stop"

# -- Source definitions -------------------------------------------------------
# URLs verified against Kiwix directory listings, April 2026

$Sources = @{
    "wiki" = @{
        Name = "Wikipedia EN (text, no images)"
        Url  = "https://download.kiwix.org/zim/wikipedia/wikipedia_en_all_nopic_2026-03.zim"
        Size = "48 GB"
    }
    "ifixit" = @{
        Name = "iFixit repair guides"
        Url  = "https://download.kiwix.org/zim/ifixit/ifixit_en_all_2025-12.zim"
        Size = "3.3 GB"
    }
    "wikibooks" = @{
        Name = "Wikibooks (text, no images)"
        Url  = "https://download.kiwix.org/zim/wikibooks/wikibooks_en_all_nopic_2026-01.zim"
        Size = "2.9 GB"
    }
    "wikimed" = @{
        Name = "WikiMed medical encyclopedia"
        Url  = "https://download.kiwix.org/zim/wikipedia/wikipedia_en_medicine_maxi_2026-04.zim"
        Size = "454 MB"
    }
    "crashcourse" = @{
        Name = "CrashCourse educational videos"
        Url  = "https://download.kiwix.org/zim/other/crashcourse_en_all_2026-02.zim"
        Size = "21 GB"
    }
    "gutenberg" = @{
        Name = "Project Gutenberg books"
        Url  = "https://download.kiwix.org/zim/gutenberg/gutenberg_mul_all_2025-11.zim"
        Size = "236 GB"
    }
}

# -- Models -------------------------------------------------------------------

$Models = @{
    "whisper" = @{
        Name = "Whisper small.en (STT)"
        Url  = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin"
        File = "whisper-small.en.bin"
        Size = "460 MB"
    }
    "piper" = @{
        Name = "Piper lessac-medium (TTS)"
        Url  = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
        File = "en_US-lessac-medium.onnx"
        Size = "75 MB"
    }
    "piper_json" = @{
        Name = "Piper lessac-medium config"
        Url  = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
        File = "en_US-lessac-medium.onnx.json"
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
        $existSize = [math]::Round((Get-Item $Destination).Length / 1MB, 1)
        Write-Host "  SKIP  $Label - $Destination already exists ($existSize MB)" -ForegroundColor Yellow
        return
    }

    if ($DryRun) {
        Write-Host "  WOULD DOWNLOAD  $Label (~$Size)" -ForegroundColor Cyan
        Write-Host "    $Url"
        Write-Host "    -> $Destination"
        return
    }

    $tempFile = "$Destination.downloading"

    # Resume if a partial download exists
    if (Test-Path $tempFile) {
        $partialMB = [math]::Round((Get-Item $tempFile).Length / 1MB, 1)
        Write-Host "  RESUMING  $Label ($partialMB MB already downloaded)..." -ForegroundColor Magenta
    }
    else {
        Write-Host "  DOWNLOADING  $Label (~$Size)..." -ForegroundColor Green
    }
    Write-Host "    $Url"

    try {
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
        Write-Host "    Partial file kept at: $tempFile" -ForegroundColor Red
        Write-Host "    Re-run the script to retry." -ForegroundColor Red
    }
}

# -- Main ---------------------------------------------------------------------

Write-Host ""
Write-Host "========================================================" -ForegroundColor White
Write-Host "  Radio Oracle - Knowledge Base Downloader (Windows)"     -ForegroundColor White
Write-Host "========================================================" -ForegroundColor White
Write-Host "  Running in: $(Get-Location)"                            -ForegroundColor Gray
Write-Host ""

if ($DryRun) {
    Write-Host "  [DRY RUN MODE - nothing will be downloaded]" -ForegroundColor Cyan
    Write-Host ""
}

# Scan existing files
Write-Host "-- Existing Files ------------------------------------------" -ForegroundColor White
Write-Host ""
$existingZims = Get-ChildItem -Path . -Filter "*.zim" -ErrorAction SilentlyContinue
if ($existingZims) {
    foreach ($f in $existingZims) {
        $sizeGB = [math]::Round($f.Length / 1GB, 1)
        Write-Host "  FOUND  $($f.Name) ($sizeGB GB)" -ForegroundColor Green
    }
}
else {
    Write-Host "  No existing .zim files found" -ForegroundColor Gray
}
Write-Host ""

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
    Download-File -Url $s.Url -Destination $filename -Label $s.Name -Size $s.Size
}

# Summary
Write-Host ""
Write-Host "-- Summary -------------------------------------------------" -ForegroundColor White
Write-Host ""

$allZims = Get-ChildItem -Path . -Filter "*.zim" -ErrorAction SilentlyContinue
if ($allZims) {
    $totalGB = [math]::Round(($allZims | Measure-Object -Property Length -Sum).Sum / 1GB, 1)
    Write-Host "  $($allZims.Count) ZIM files, $totalGB GB total" -ForegroundColor White
    Write-Host ""
    Write-Host "  Old 2022 files can be deleted after you verify the new ones:" -ForegroundColor Gray
    foreach ($f in $allZims) {
        if ($f.Name -match "2022") {
            Write-Host "    del `"$($f.Name)`"" -ForegroundColor Gray
        }
    }
}

Write-Host ""
Write-Host "-- Next Steps ----------------------------------------------" -ForegroundColor White
Write-Host ""
Write-Host "  Transfer this entire folder to the Jetson:" -ForegroundColor White
Write-Host "    scp -r `"$(Get-Location)`" user@jetson:/opt/radio-oracle/data/knowledge/"
Write-Host ""
Write-Host "  Or use WinSCP for resumable transfers (recommended)."
Write-Host ""
Write-Host "  Then on the Jetson, run ingestion (see docs/SETUP.md Part 3)."
Write-Host ""
