# Wrapper around build_faiss_ivfpq.py that avoids the long-path
# quoting/wrapping issue when pasting commands into PowerShell.
#
# Usage:
#   .\scripts\build_faiss.ps1 gutenberg
#   .\scripts\build_faiss.ps1 wikipedia -Nlist 4096

param(
    [Parameter(Mandatory)] [string]$Name,
    [int]$Dim = 768,
    [int]$Nlist = 0,
    [int]$PqM = 64
)

$py       = 'C:\Users\erik\Desktop\Huge Information Stores\.venv\Scripts\python.exe'
$embDir   = 'C:\Users\erik\Desktop\Huge Information Stores\data\embeddings'
$faissDir = 'C:\Users\erik\Desktop\Huge Information Stores\data\faiss'

$env:PYTHONPATH = '.'

& $py scripts\build_faiss_ivfpq.py `
    --name $Name `
    --in-dir $embDir `
    --out-dir $faissDir `
    --dim $Dim `
    --nlist $Nlist `
    --pq-m $PqM
