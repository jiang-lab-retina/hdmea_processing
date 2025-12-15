# PowerShell script to copy .py and .ipynb files while maintaining folder structure
# Usage: .\copy_python_files.ps1 -SourceFolder "path/to/source" -DestFolder "path/to/destination"

param(
    [Parameter(Mandatory=$true)]
    [string]$SourceFolder,
    
    [Parameter(Mandatory=$true)]
    [string]$DestFolder
)

# Get all .py and .ipynb files recursively
Get-ChildItem -Path $SourceFolder -Include *.py,*.ipynb -Recurse | ForEach-Object {
    # Calculate destination path maintaining structure
    $relativePath = $_.FullName.Substring($SourceFolder.Length).TrimStart('\')
    $destPath = Join-Path $DestFolder $relativePath
    $destDir = Split-Path $destPath -Parent
    
    # Create destination directory if it doesn't exist
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    
    # Copy the file
    Copy-Item $_.FullName -Destination $destPath -Force
    Write-Host "Copied: $relativePath"
}

Write-Host "Copy completed!"

