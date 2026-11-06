# extract_for_kaggle.ps1
# ======================
# Run this on your Windows machine after manually downloading the datasets.
# It extracts, trims to NROWS, and creates a zip ready to upload as a Kaggle Dataset.
#
# Prerequisites:
#   1. Download Thunderbird.tar.gz and BGL.tar.gz from:
#      https://zenodo.org/records/8196385
#   2. Place both files in the same folder (e.g. E:\Downloads\log_datasets)
#   3. Run this script.
#
# Output:
#   kaggle_upload_logdlr_data.zip  -> Upload this to Kaggle as a Dataset named "logdlr-data"

# EDIT THIS: path to your downloaded .tar.gz files
$downloadDir = "E:\Downloads\log_datasets"

# EDIT THIS: how many lines to keep per dataset
$NROWS = 1000000   # pilot size. Change to 10000000 (or larger) for full runs.

# ---------------------------------------------------------------------------
# Do not edit below this line
# ---------------------------------------------------------------------------
$outputZip   = "$PSScriptRoot\kaggle_upload_logdlr_data.zip"
$finalDir    = "$PSScriptRoot\LogDLR\dataset"

New-Item -ItemType Directory -Path $finalDir -Force | Out-Null

foreach ($name in @("Thunderbird","BGL")) {
    Write-Host "`n========================================"
    Write-Host "Processing $name"
    Write-Host "========================================"

    # Find archive
    $archive = Get-ChildItem -Path $downloadDir -Filter "*$name*.tar.gz" | Select-Object -First 1
    if (-not $archive) {
        Write-Error "Archive not found for $name in $downloadDir"
        continue
    }
    Write-Host "  Found: $($archive.Name)"

    # Extract
    $extractDir = "$PSScriptRoot\_temp\$name"
    New-Item -ItemType Directory -Path $extractDir -Force | Out-Null
    tar -xzf $archive.FullName -C $extractDir

    # Find .log file (might be nested)
    $logFile = Get-ChildItem -Path $extractDir -Recurse -Filter "*.log" | Select-Object -First 1
    if (-not $logFile) {
        Write-Error "No .log found inside $name archive"
        continue
    }
    Write-Host "  Source: $($logFile.FullName)"

    # Trim to first NROWS lines
    $destDir  = "$finalDir\$name"
    $destLog  = "$destDir\$name.log"
    New-Item -ItemType Directory -Path $destDir -Force | Out-Null

    $reader = [System.IO.StreamReader]::new($logFile.FullName)
    $writer = [System.IO.StreamWriter]::new($destLog)
    $count = 0
    while ($count -lt $NROWS -and $null -ne ($line = $reader.ReadLine())) {
        $writer.WriteLine($line)
        $count++
    }
    $reader.Close()
    $writer.Close()

    $sizeMB = [math]::Round((Get-Item $destLog).Length / 1MB, 1)
    Write-Host "  Saved $count lines -> $destLog ($sizeMB MB)"
}

# Zip
Write-Host "`nCreating zip: $outputZip ..."
if (Test-Path $outputZip) { Remove-Item -Force $outputZip }
Compress-Archive -Path "$PSScriptRoot\LogDLR" -DestinationPath $outputZip -Force

# Cleanup
Remove-Item -Recurse -Force "$PSScriptRoot\_temp" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$PSScriptRoot\LogDLR" -ErrorAction SilentlyContinue

$zipSize = if (Test-Path $outputZip) { [math]::Round((Get-Item $outputZip).Length / 1MB, 1) } else { 0 }
Write-Host "`n========================================"
Write-Host "Done!"
Write-Host "Output: $outputZip ($zipSize MB)"
Write-Host "========================================"
Write-Host "Next steps:"
Write-Host "  1. Go to https://www.kaggle.com/datasets"
Write-Host "  2. Click 'New Dataset', title it: logdlr-data"
Write-Host "  3. Upload the zip file above"
Write-Host "  4. In your Notebook, attach 'logdlr-data' as an Input"
Write-Host "  5. Run the notebook with DATA_ROOT=/kaggle/input/logdlr-data/LogDLR/dataset"
