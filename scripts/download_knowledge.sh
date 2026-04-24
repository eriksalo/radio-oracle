#!/usr/bin/env bash
# Download knowledge base files for offline RAG.
# Usage: ./scripts/download_knowledge.sh [--dry-run]
#
# Run on a workstation with good bandwidth. Files are large.

set -euo pipefail

DRY_RUN=false
DATA_DIR="data/knowledge"

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Would download the following:"
    echo ""
fi

mkdir -p "$DATA_DIR"

declare -A SOURCES=(
    # Wikipedia EN (text only, no pictures)
    ["wikipedia"]="https://download.kiwix.org/zim/wikipedia/wikipedia_en_all_nopic_latest.zim|22GB"
    # iFixit repair guides
    ["ifixit"]="https://download.kiwix.org/zim/other/ifixit_en_all_latest.zim|2.5GB"
    # Wikibooks
    ["wikibooks"]="https://download.kiwix.org/zim/wikibooks/wikibooks_en_all_latest.zim|2GB"
    # WikiMed medical
    ["wikimed"]="https://download.kiwix.org/zim/wikipedia/wikipedia_en_medicine_nopic_latest.zim|1GB"
)

for name in "${!SOURCES[@]}"; do
    IFS='|' read -r url size <<< "${SOURCES[$name]}"
    filename="$DATA_DIR/$(basename "$url")"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  $name: $url (~$size)"
    else
        if [[ ! -f "$filename" ]]; then
            echo "Downloading $name (~$size)..."
            wget -q --show-progress -O "$filename" "$url"
        else
            echo "$name already exists: $filename"
        fi
    fi
done

echo ""
echo "Additional sources to download manually:"
echo "  - Project Gutenberg (text): https://www.gutenberg.org/robot/harvest"
echo "  - Stack Exchange data dump: https://archive.org/details/stackexchange"
echo "  - Army field manuals: collect PDFs/text from public sources"
echo "  - CrashCourse transcripts: youtube-dl + whisper transcription"
echo ""
echo "After downloading, run the ingest scripts to build the ChromaDB collections."
