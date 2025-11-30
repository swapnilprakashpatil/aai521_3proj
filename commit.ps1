# Maximum number of files to stage per batch
$batchSize = 500

function Stage-And-Commit {
    # Get up to $batchSize changed or untracked files
    $files = git status --porcelain | Select-Object -First $batchSize

    if ($files.Count -eq 0) {
        Write-Host "No more files to commit. Finished!"
        return
    }

    Write-Host "Staging $($files.Count) files..."

    # Extract file paths from 'git status --porcelain' output
    $paths = $files | ForEach-Object {
        $_.Substring(3)   # skip two status chars + space
    }

    # Stage files
    foreach ($path in $paths) {
        git add -- "$path"
    }

    # Commit the batch
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $commitMessage = "Batch commit ($($paths.Count) files) - $timestamp"

    git commit -m "$commitMessage"

    Write-Host "Committed $($paths.Count) files. Continuing..."
    
    # Recursively process more files
    Stage-And-Commit
}

Stage-And-Commit
