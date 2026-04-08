# movie-recommender-system

## Git helper

Use `git_sync.py` to stage, commit, and optionally push changes.

Example:

    python git_sync.py --all -m "Update recommendation logic" --push

Use `--files` to stage only specific files:

    python git_sync.py --files src/predictor.py README.md -m "Fix predictor doc" --push

Use `--dry-run` to preview commands without applying changes:

    python git_sync.py --all -m "Test" --dry-run
