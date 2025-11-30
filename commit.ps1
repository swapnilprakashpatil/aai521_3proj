# Maximum number of files to stage per batch
$batchSize = 500

function Stage-Commit-Push {

    # Get up to $batchSize changed or untracked files
    $files = git status --porcelain | Select-Object -First $batchSize

    if ($files.Count -eq 0) {
        Write-Host "No more files to commit. Finished!"
        return
    }

    Write-Host "`nStaging $($files.Count) files..."

    # Extract paths
    $paths = $files | ForEach-Object {
        $_.Substring(3)  # Skip status chars
    }

    # Stage files
    foreach ($path in $paths) {
        git add -- "$path"
    }

    # Commit
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $commitMessage = "Batch commit ($($paths.Count) files) - $timestamp"

    git commit -m "$commitMessage"
    Write-Host "Committed $($paths.Count) files."

    # Push
    Write-Host "Pushing batch..."
    git push
    Write-Host "Push complete."

    # Continue recursively
    Stage-Commit-Push
}

Stage-Commit-Push
