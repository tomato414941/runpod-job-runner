# Project Agent Policy

このプロジェクトでは日本語で応答し、コードコメントは英語で書く。

## Purpose

`runpod-job-runner` is a small operations repository for disposable RunPod jobs.
It owns generic pod lifecycle mechanics:

- create a pod,
- wait for SSH,
- sync files,
- run setup and job commands,
- copy outputs back,
- delete the pod.

Domain-specific training, arena evaluation, model code, and benchmark records
belong in the repositories that use this runner.

## Boundaries

- Do not add project-specific training or shogi arena logic here.
- Do not commit secrets, API keys, SSH keys, generated run outputs, or local
  machine paths.
- Keep the runner generic enough for multiple repositories, but avoid adding
  speculative framework abstractions.

## Development Rules

- Keep changes small and verified.
- Prefer explicit CLI arguments over hidden project defaults.
- Preserve secret redaction when changing logging.
