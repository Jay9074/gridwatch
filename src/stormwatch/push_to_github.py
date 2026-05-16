"""
GridWatch Storm Watch - GitHub Auto-Push
Commits and pushes the latest Storm Watch output to GitHub
so the live dashboard and API see fresh data.

Run after run_pipeline.py.
Will be called automatically by the updated run_pipeline.py.
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Files to commit (only the active "snapshot" files, not timestamped history)
FILES_TO_PUSH = [
    "data/stormwatch/storms/active_storms.csv",
    "data/stormwatch/predictions/active_predictions.csv",
    "data/stormwatch/predictions/prediction_log.csv",
    "data/stormwatch/validation/accuracy_scorecard.json",
    "data/stormwatch/validation/validation_results.csv",
    "data/stormwatch/heartbeat.log",
]


def run(cmd, check=True):
    """Run a shell command, return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def main():
    print("=" * 60)
    print("GridWatch - Push Storm Watch data to GitHub")
    print("=" * 60)
    
    repo_root = Path(__file__).parent.parent.parent
    print(f"Repo: {repo_root}")
    
    # Check git is available
    code, _, _ = run("git --version")
    if code != 0:
        print("ERROR: git not installed or not in PATH")
        return 1
    
    # Move to repo root
    import os
    os.chdir(repo_root)
    
    # Check current branch
    _, branch, _ = run("git rev-parse --abbrev-ref HEAD")
    print(f"Branch: {branch}")
    
    # Stage each file if it exists
    staged = []
    for f in FILES_TO_PUSH:
        p = repo_root / f
        if not p.exists():
            print(f"Skip (not found): {f}")
            continue
        
        code, _, err = run(f'git add "{f}"')
        if code == 0:
            staged.append(f)
            print(f"Staged: {f}")
        else:
            print(f"Could not stage {f}: {err}")
    
    if not staged:
        print("Nothing to push")
        return 0
    
    # Check if anything actually changed
    code, status, _ = run("git diff --cached --name-only")
    if not status:
        print("No changes to commit (files already up to date)")
        return 0
    
    # Commit
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"Storm Watch auto-update {ts}"
    code, out, err = run(f'git commit -m "{msg}"')
    if code != 0:
        print(f"Commit failed: {err}")
        return 1
    print(f"Committed: {msg}")
    
    # Push
    code, out, err = run(f"git push origin {branch}")
    if code != 0:
        print(f"Push failed: {err}")
        print("\nMost common cause: git credentials not configured.")
        print("Run once manually: git push origin main")
        print("It will prompt for username/PAT and remember it.")
        return 1
    
    print(f"Pushed to {branch}")
    print(f"\nDashboard refreshes within 2 minutes:")
    print(f"  https://gridwatch-dashboard.streamlit.app/")
    print(f"  https://gridwatch-y8nu.onrender.com/api/v1/health")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
