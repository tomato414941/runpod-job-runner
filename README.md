# runpod-job-runner

Small generic runner for disposable RunPod jobs.

This repository owns RunPod pod lifecycle plumbing only. Project-specific
wrappers decide what to sync and what remote commands to run.

## Requirements

- `runpodctl`
- `curl`
- `ssh`
- `rsync`
- RunPod API key via `RUNPOD_API_KEY` or `RUNPOD_API_KEY_FILE`
- SSH key paths via `RUNPOD_SSH_KEY` and `RUNPOD_SSH_PUBLIC_KEY`

## Example

```sh
RUNPOD_API_KEY_FILE="$HOME/.secrets/runpod" \
RUNPOD_SSH_KEY="$HOME/.runpod/ssh/RunPod-Key-Go" \
RUNPOD_SSH_PUBLIC_KEY="$HOME/.runpod/ssh/RunPod-Key-Go.pub" \
RUNPODCTL="$HOME/bin/runpodctl" \
python3 scripts/run_job.py \
  --repo-root /path/to/project \
  --name example-job \
  --gpu-type "NVIDIA GeForce RTX 5090" \
  --sync scripts/setup_runpod.sh \
  --setup-command 'cd "$REMOTE_DIR"; bash scripts/setup_runpod.sh' \
  --remote 'cd "$REMOTE_DIR"; python -c "print(\"hello\")"; mkdir -p runs/example; echo ok > runs/example/out.txt' \
  --output runs/example
```

`--output` values are `--repo-root` relative paths. Absolute paths and paths
containing `..` are rejected because the same value is used for both the remote
path under `$REMOTE_DIR` and the local copy destination.

By default the runner uses the official RunPod PyTorch 2.8 template:

```text
runpod-torch-v280
```

The runner deletes the created pod after success or failure unless
`--keep-pod` or `--keep-pod-on-failure` is set.

Use `--dry-run` before paid jobs to print the pod payload, sync plan, remote
commands, output copy commands, and cleanup command without requiring a RunPod
API key or creating a pod.

Use repeatable `--local` commands for local preflight checks. Local commands
run before any pod is created; if one fails, the job stops without starting paid
compute. In `--dry-run` mode, local commands are printed but not executed.

Use `--timings-output path/to/timings.json` to write job step timings for later
inspection. The timings file records step names, timestamps, durations, and
status without storing command text or secrets.
