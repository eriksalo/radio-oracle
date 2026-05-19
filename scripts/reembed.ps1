# Wrapper around reembed_collection.py that avoids the long-path
# quoting/wrapping issue when pasting commands into PowerShell.
#
# Usage:
#   .\scripts\reembed.ps1 gutenberg
#   .\scripts\reembed.ps1 wikimed -EncodeBatchSize 128

param(
    [Parameter(Mandatory)] [string]$Name,
    [string]$Model = 'nomic-ai/nomic-embed-text-v1.5',
    [int]$Workers = 12,
    [int]$BatchSize = 2000,
    [int]$EncodeBatchSize = 256,
    [int]$MaxSeqLength = 512,
    [int]$Dim = 768
)

$py     = 'C:\Users\erik\Desktop\Huge Information Stores\.venv\Scripts\python.exe'
$dbPath = 'C:\Users\erik\Desktop\Huge Information Stores\data\chroma'
$embDir = 'C:\Users\erik\Desktop\Huge Information Stores\data\embeddings'

$env:PYTHONPATH = '.'

& $py scripts\reembed_collection.py `
    --source $Name --target $Name `
    --model $Model `
    --db-path $dbPath `
    --out-dir $embDir `
    --dim $Dim `
    --workers $Workers `
    --batch-size $BatchSize `
    --encode-batch-size $EncodeBatchSize `
    --max-seq-length $MaxSeqLength
