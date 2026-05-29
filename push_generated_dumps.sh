#!/usr/bin/env bash
set -euo pipefail

remote="origin"
branch=""
message="Add generated matrix dumps"
max_mb=500
allow_large=0
yes=0
dry_run=0
no_push=0
include_code=0
dump_paths=()

usage() {
  cat <<'EOF'
Usage:
  ./push_generated_dumps.sh [options] [dump_dir ...]

Stages, commits, and pushes generated dump or sanity-check artifact directories.

If no dump_dir is given, the script uses any existing default artifact folders:
  data matrix_dumps matrix_dumps_full condition_window_checks

Options:
  -m, --message TEXT       Commit message.
  --remote NAME            Git remote to push to. Default: origin.
  --branch NAME            Branch to push. Default: current branch.
  --max-mb N               Refuse if total dump size exceeds N MB. Default: 500.
  --allow-large            Allow total dump size above --max-mb.
  --include-code           Also stage dump/reproduction scripts and readme.
  --dry-run                Show what would be staged, without committing.
  --no-push                Commit locally but do not push.
  -y, --yes                Do not ask for confirmation.
  -h, --help               Show this help.

Examples:
  ./push_generated_dumps.sh --dry-run
  ./push_generated_dumps.sh -y data
  ./push_generated_dumps.sh -y --include-code matrix_dumps
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--message)
      [[ $# -ge 2 ]] || die "--message requires text"
      message="$2"
      shift 2
      ;;
    --remote)
      [[ $# -ge 2 ]] || die "--remote requires a name"
      remote="$2"
      shift 2
      ;;
    --branch)
      [[ $# -ge 2 ]] || die "--branch requires a name"
      branch="$2"
      shift 2
      ;;
    --max-mb)
      [[ $# -ge 2 ]] || die "--max-mb requires a number"
      max_mb="$2"
      shift 2
      ;;
    --allow-large)
      allow_large=1
      shift
      ;;
    --include-code)
      include_code=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --no-push)
      no_push=1
      shift
      ;;
    -y|--yes)
      yes=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --scope)
      die "This script takes dump directories, not dump-generation options. Generate first, then run this script."
      ;;
    --*)
      die "Unknown option: $1"
      ;;
    *)
      dump_paths+=("$1")
      shift
      ;;
  esac
done

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git repository"

if [[ ${#dump_paths[@]} -eq 0 ]]; then
  for candidate in data matrix_dumps matrix_dumps_full condition_window_checks; do
    [[ -d "$candidate" ]] && dump_paths+=("$candidate")
  done
fi

[[ ${#dump_paths[@]} -gt 0 ]] || die "no dump directories found. Run dump_plot_matrices.py first or pass a dump directory."

for path in "${dump_paths[@]}"; do
  [[ -d "$path" ]] || die "dump directory does not exist: $path"
done

if [[ -z "$branch" ]]; then
  branch="$(git branch --show-current)"
fi
[[ -n "$branch" ]] || die "could not infer current branch; pass --branch NAME"

total_kb=0
for path in "${dump_paths[@]}"; do
  path_kb="$(du -sk "$path" | awk '{print $1}')"
  total_kb=$((total_kb + path_kb))
done
total_mb=$(( (total_kb + 1023) / 1024 ))

printf 'Dump paths:\n'
for path in "${dump_paths[@]}"; do
  du -sh "$path"
done
printf 'Total dump size: %s MB\n' "$total_mb"

if [[ "$allow_large" -eq 0 && "$total_mb" -gt "$max_mb" ]]; then
  die "dump size exceeds ${max_mb} MB. Use --allow-large only if you really want this in git."
fi

oversized_files="$(find "${dump_paths[@]}" -type f -size +95M -print)"
if [[ -n "$oversized_files" ]]; then
  printf '\nFiles larger than 95 MB were found:\n%s\n\n' "$oversized_files" >&2
  die "GitHub rejects files near/above 100 MB. Use a release asset, Zenodo/OSF, or Git LFS instead."
fi

stage_paths=("${dump_paths[@]}")
if [[ "$include_code" -eq 1 ]]; then
  for path in dump_plot_matrices.py push_generated_dumps.sh qgl_reproduce.py run_seeded_reproduction.py sanity_check_window.py readme.md requirements.txt; do
    [[ -e "$path" ]] && stage_paths+=("$path")
  done
fi

printf '\nTarget remote/branch: %s %s\n' "$remote" "$branch"
printf 'Commit message: %s\n' "$message"
printf 'Paths to stage:\n'
printf '  %s\n' "${stage_paths[@]}"

if [[ "$dry_run" -eq 1 ]]; then
  printf '\nDry run: git add --dry-run output\n'
  git add --dry-run -- "${stage_paths[@]}"
  exit 0
fi

if [[ "$yes" -eq 0 ]]; then
  if [[ -t 0 ]]; then
    printf '\nProceed with git add, commit, and push? [y/N] '
    read -r reply
    [[ "$reply" == "y" || "$reply" == "Y" ]] || die "aborted"
  else
    die "non-interactive shell. Re-run with --yes if you want to proceed."
  fi
fi

git add -- "${stage_paths[@]}"

if git diff --cached --quiet; then
  printf 'Nothing new to commit.\n'
  exit 0
fi

git commit -m "$message"

if [[ "$no_push" -eq 1 ]]; then
  printf 'Committed locally. Skipping push because --no-push was set.\n'
  exit 0
fi

git push "$remote" "$branch"
printf 'Pushed generated dumps to %s/%s.\n' "$remote" "$branch"
