#!/usr/bin/env python
"""Git helper for staging, committing, and pushing repository changes."""

import argparse
import os
import subprocess
import sys


def run_git_command(args, repo_dir):
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"Git command failed: git {' '.join(args)}\n{message}") from exc


def repo_root(path):
    path = os.path.abspath(path)
    while path and path != os.path.dirname(path):
        if os.path.isdir(os.path.join(path, ".git")):
            return path
        path = os.path.dirname(path)
    return None


def get_current_branch(repo_dir):
    return run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)


def stage_paths(repo_dir, paths):
    if not paths:
        return
    run_git_command(["add", "--"] + paths, repo_dir)


def stage_all(repo_dir):
    run_git_command(["add", "-A"], repo_dir)


def commit_changes(repo_dir, message):
    if not message:
        raise RuntimeError("Commit message is required.")
    run_git_command(["commit", "-m", message], repo_dir)


def push_changes(repo_dir, remote, branch, force):
    args = ["push", remote, branch]
    if force:
        args.insert(1, "--force")
    run_git_command(args, repo_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage, commit, and push repository changes to GitHub."
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="Stage all tracked and untracked changes (git add -A).",
    )
    group.add_argument(
        "--files",
        nargs="+",
        metavar="PATH",
        help="Stage only the specified file paths.",
    )

    parser.add_argument(
        "-m",
        "--message",
        required=True,
        help="Commit message to use for git commit.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the commit to the remote branch after committing.",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Remote name to push to (default: origin).",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Branch name to push to (default: current branch).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force push the branch to the remote.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the commands without executing them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    work_dir = os.getcwd()
    repo_dir = repo_root(work_dir)
    if repo_dir is None:
        print("Error: This script must be run from inside a Git repository.", file=sys.stderr)
        sys.exit(1)

    branch = args.branch or get_current_branch(repo_dir)
    if args.dry_run:
        print(f"Repository root: {repo_dir}")
        if args.all:
            print("Would run: git add -A")
        elif args.files:
            print(f"Would run: git add -- {' '.join(args.files)}")
        print(f"Would run: git commit -m '{args.message}'")
        if args.push:
            push_cmd = f"git push {'--force ' if args.force else ''}{args.remote} {branch}"
            print(f"Would run: {push_cmd}")
        return

    try:
        if args.all:
            stage_all(repo_dir)
        elif args.files:
            stage_paths(repo_dir, args.files)
        else:
            print("No paths were selected to stage. Use --all or --files.", file=sys.stderr)
            sys.exit(1)

        commit_changes(repo_dir, args.message)
        print(f"Committed changes on branch '{branch}'.")

        if args.push:
            push_changes(repo_dir, args.remote, branch, args.force)
            print(f"Pushed branch '{branch}' to remote '{args.remote}'.")

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
