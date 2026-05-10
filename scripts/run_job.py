#!/usr/bin/env python3
"""Run one disposable RunPod job."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time


DEFAULT_TEMPLATE_ID = "runpod-torch-v280"
DEFAULT_REMOTE_DIR = "/root/job"
DEFAULT_SYNC: tuple[str, ...] = ()


@dataclass(frozen=True)
class Connection:
    host: str
    port: int
    user: str = "root"


class TimingRecorder:
    def __init__(self, output_path: Path | None, *, pod_name: str, dry_run: bool) -> None:
        self.output_path = output_path
        self.data: dict[str, object] = {
            "pod_name": pod_name,
            "dry_run": dry_run,
            "status": "running",
            "started_at": utc_timestamp(),
            "finished_at": None,
            "total_seconds": None,
            "pod_id": None,
            "steps": [],
        }
        self._started = time.monotonic()

    def set_pod_id(self, pod_id: str) -> None:
        self.data["pod_id"] = pod_id

    def step(self, name: str, callback):
        started_at = utc_timestamp()
        started = time.monotonic()
        record: dict[str, object] = {"name": name, "started_at": started_at}
        try:
            result = callback()
            record["status"] = "passed"
            return result
        except Exception:
            record["status"] = "failed"
            raise
        finally:
            record["finished_at"] = utc_timestamp()
            record["duration_seconds"] = round(time.monotonic() - started, 3)
            steps = self.data["steps"]
            assert isinstance(steps, list)
            steps.append(record)

    def finish(self, status: str) -> None:
        self.data["status"] = status
        self.data["finished_at"] = utc_timestamp()
        self.data["total_seconds"] = round(time.monotonic() - self._started, 3)
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def command_text(command: list[str]) -> str:
    return " ".join(quote(part) for part in command)


def redact(text: str, secrets: list[str]) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = re.sub(r"ssh-(rsa|ed25519) [^\n\"]+", "[REDACTED_SSH_PUBLIC_KEY]", redacted)
    return redacted


def run(
    command: list[str],
    *,
    cwd: Path,
    secrets: list[str],
    env: dict[str, str] | None = None,
    capture: bool = False,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"$ {redact(command_text(command), secrets)}")
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=os.environ | (env or {}),
        text=True,
        capture_output=capture,
        check=False,
        timeout=timeout,
    )
    if capture and completed.stdout:
        print(redact(completed.stdout, secrets), end="")
    if capture and completed.stderr:
        print(redact(completed.stderr, secrets), end="", file=sys.stderr)
    if check and completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}")
    return completed


def dry_run(command: list[str], *, secrets: list[str]) -> None:
    print(f"$ {redact(command_text(command), secrets)}")


def parse_json(text: str) -> object:
    stripped = text.strip()
    if not stripped:
        return []
    return json.loads(stripped)


def normalize_pod(raw: dict[str, object]) -> dict[str, str]:
    machine = raw.get("machine") if isinstance(raw.get("machine"), dict) else {}
    ssh = raw.get("ssh") if isinstance(raw.get("ssh"), dict) else {}
    return {
        "id": str(raw.get("id") or raw.get("ID") or ""),
        "name": str(raw.get("name") or raw.get("NAME") or ""),
        "status": str(raw.get("desiredStatus") or raw.get("status") or raw.get("STATUS") or ""),
        "ports": format_ports(raw.get("ports") or raw.get("PORTS") or raw.get("portMappings") or ""),
        "gpu": str(machine.get("gpuDisplayName") or raw.get("gpuDisplayName") or ""),
        "location": str(machine.get("location") or raw.get("location") or ""),
        "ssh_error": str(ssh.get("error") or ""),
    }


def format_ports(raw: object) -> str:
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return str(raw or "")
    parts = []
    for item in raw:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        host = item.get("ip") or item.get("host") or item.get("publicIp")
        public_port = item.get("publicPort") or item.get("externalPort") or item.get("port")
        private_port = item.get("privatePort") or item.get("containerPort") or item.get("internalPort")
        protocol = str(item.get("type") or item.get("protocol") or "tcp").lower()
        is_public = item.get("isIpPublic")
        label = f"{'pub' if is_public is True else 'prv' if is_public is False else ''},{protocol}".strip(",")
        if host and public_port and private_port:
            parts.append(f"{host}:{public_port}->{private_port} ({label})")
        else:
            parts.append(str(item))
    return ",".join(parts)


def normalize_pods(raw: object) -> list[dict[str, str]]:
    if isinstance(raw, dict):
        for key in ("pods", "data", "items"):
            value = raw.get(key)
            if isinstance(value, list):
                return [normalize_pod(item) for item in value if isinstance(item, dict)]
        return [normalize_pod(raw)]
    if isinstance(raw, list):
        return [normalize_pod(item) for item in raw if isinstance(item, dict)]
    return []


def list_pods(args: argparse.Namespace, secrets: list[str]) -> list[dict[str, str]]:
    completed = run([args.runpodctl, "pod", "list", "-o", "json"], cwd=args.repo_root, secrets=secrets, capture=True)
    return normalize_pods(parse_json(completed.stdout))


def active_pods(args: argparse.Namespace, secrets: list[str]) -> list[dict[str, str]]:
    inactive = {"EXITED", "TERMINATED", "STOPPED"}
    return [pod for pod in list_pods(args, secrets) if pod["status"].upper() not in inactive]


def pod_payload(args: argparse.Namespace, public_key: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": args.pod_name,
        "gpuTypeIds": [args.gpu_type],
        "gpuCount": args.gpu_count,
        "containerDiskInGb": args.container_disk_size,
        "volumeInGb": args.volume_size,
        "volumeMountPath": args.remote_volume,
        "ports": ["22/tcp"],
        "cloudType": "SECURE" if args.secure_cloud else "COMMUNITY",
    }
    if args.image:
        payload["imageName"] = args.image
    elif args.template_id:
        payload["templateId"] = args.template_id
    else:
        raise ValueError("set --template-id or --image")
    if args.allowed_cuda_version:
        payload["allowedCudaVersions"] = args.allowed_cuda_version
    if args.data_center_ids:
        payload["dataCenterIds"] = [item.strip() for item in args.data_center_ids.split(",") if item.strip()]
    if public_key:
        payload["env"] = {"PUBLIC_KEY": public_key}
    if not args.secure_cloud:
        payload["supportPublicIp"] = True
    return payload


def create_pod(args: argparse.Namespace, secrets: list[str], api_key: str, public_key: str) -> None:
    payload = pod_payload(args, public_key)
    run(
        [
            "curl",
            "--fail-with-body",
            "--silent",
            "--show-error",
            "--request",
            "POST",
            "--url",
            "https://rest.runpod.io/v1/pods",
            "--header",
            f"Authorization: Bearer {api_key}",
            "--header",
            "Content-Type: application/json",
            "--data",
            json.dumps(payload, separators=(",", ":")),
        ],
        cwd=args.repo_root,
        secrets=secrets,
        capture=True,
    )


def pod_connection_from_ports(ports: str) -> Connection | None:
    pattern = re.compile(r"([A-Za-z0-9.-]+):(\d+)->22\s*\(([^)]*)\)")
    for host, port, label in pattern.findall(ports):
        if "tcp" in label.lower() and "prv" not in label.lower():
            return Connection(host=host, port=int(port))
    return None


def parse_ssh_info(text: str) -> Connection | None:
    try:
        raw = parse_json(text)
    except json.JSONDecodeError:
        raw = None
    if isinstance(raw, dict):
        host = raw.get("host") or raw.get("hostname") or raw.get("ip") or raw.get("publicIp")
        port = raw.get("port") or raw.get("sshPort")
        user = raw.get("user") or raw.get("username") or "root"
        if host and port:
            return Connection(host=str(host), port=int(port), user=str(user))
        command = raw.get("command") or raw.get("sshCommand")
        if command:
            return parse_ssh_command(str(command))
    return parse_ssh_command(text)


def parse_ssh_command(text: str) -> Connection | None:
    match = re.search(r"ssh\s+(?:-i\s+\S+\s+)?(?P<target>[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+)(?:\s+-p\s+(?P<port1>\d+))?", text)
    if not match:
        match = re.search(r"ssh\s+(?:-p\s+(?P<port2>\d+)\s+)?(?P<target>[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+)", text)
    if not match:
        return None
    user, host = match.group("target").split("@", 1)
    port = match.groupdict().get("port1") or match.groupdict().get("port2") or "22"
    return Connection(host=host, port=int(port), user=user)


def wait_for_connection(args: argparse.Namespace, pod_id: str, secrets: list[str]) -> Connection:
    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        pods = [pod for pod in list_pods(args, secrets) if pod["id"] == pod_id]
        if pods:
            connection = pod_connection_from_ports(pods[0]["ports"])
            if connection:
                return connection
        completed = run(
            [args.runpodctl, "ssh", "info", pod_id, "-o", "json"],
            cwd=args.repo_root,
            secrets=secrets,
            capture=True,
            check=False,
        )
        if completed.returncode == 0:
            connection = parse_ssh_info(completed.stdout)
            if connection:
                return connection
        time.sleep(10)
    raise TimeoutError(f"pod did not expose SSH within {args.wait_seconds} seconds: {pod_id}")


def ssh_base(args: argparse.Namespace, connection: Connection) -> list[str]:
    return [
        "ssh",
        "-i",
        str(args.ssh_key),
        "-p",
        str(connection.port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
        f"{connection.user}@{connection.host}",
    ]


def ssh(args: argparse.Namespace, connection: Connection, remote_command: str) -> list[str]:
    return [*ssh_base(args, connection), remote_command]


def rsync_ssh(args: argparse.Namespace, connection: Connection) -> str:
    return command_text(ssh_base(args, connection)[:-1])


def wait_for_ssh(args: argparse.Namespace, connection: Connection, secrets: list[str]) -> None:
    deadline = time.monotonic() + args.ssh_wait_seconds
    while time.monotonic() < deadline:
        completed = run(ssh(args, connection, "true"), cwd=args.repo_root, secrets=secrets, capture=True, check=False)
        if completed.returncode == 0:
            return
        time.sleep(5)
    raise TimeoutError(f"SSH did not become ready within {args.ssh_wait_seconds} seconds")


def rsync_to_remote(args: argparse.Namespace, connection: Connection, secrets: list[str]) -> None:
    sources = [source for source in [*DEFAULT_SYNC, *args.sync] if (args.repo_root / source).exists()]
    if not sources:
        raise RuntimeError("no sync sources found")
    relative_sources = [f"./{source}" for source in sources]
    run(
        [
            "rsync",
            "-az",
            "--relative",
            "--timeout",
            "30",
            "-e",
            rsync_ssh(args, connection),
            *relative_sources,
            f"{connection.user}@{connection.host}:{args.remote_dir}/",
        ],
        cwd=args.repo_root,
        secrets=secrets,
    )


def rsync_from_remote(args: argparse.Namespace, connection: Connection, secrets: list[str]) -> None:
    for output in args.output:
        local = args.repo_root / output
        local.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "rsync",
                "-az",
                "--timeout",
                "30",
                "-e",
                rsync_ssh(args, connection),
                f"{connection.user}@{connection.host}:{args.remote_dir}/{output.rstrip('/')}/",
                str(local),
            ],
            cwd=args.repo_root,
            secrets=secrets,
        )


def remote_dir_command(args: argparse.Namespace, command: str) -> str:
    return f"REMOTE_DIR={quote(args.remote_dir)}; {command}"


def split_remote(command: str) -> str:
    if not command.strip():
        raise ValueError("remote command must not be empty")
    return command


def local_shell_command(command: str) -> list[str]:
    if not command.strip():
        raise ValueError("local command must not be empty")
    return ["bash", "-lc", command]


def timestamped_name(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty file: {path}")
    return text


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--name", default="runpod-job")
    parser.add_argument("--pod-name")
    parser.add_argument("--runpodctl", default=os.environ.get("RUNPODCTL", "runpodctl"))
    parser.add_argument("--secret-path", type=Path, default=env_path("RUNPOD_API_KEY_FILE"))
    parser.add_argument("--ssh-key", type=Path, default=env_path("RUNPOD_SSH_KEY"))
    parser.add_argument("--ssh-public-key", type=Path, default=env_path("RUNPOD_SSH_PUBLIC_KEY"))
    parser.add_argument("--template-id", default=DEFAULT_TEMPLATE_ID)
    parser.add_argument("--image")
    parser.add_argument("--allowed-cuda-version", action="append")
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--secure-cloud", action="store_true")
    parser.add_argument("--container-disk-size", type=int, default=20)
    parser.add_argument("--volume-size", type=int, default=20)
    parser.add_argument("--remote-volume", default="/workspace")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--data-center-ids", default="")
    parser.add_argument("--wait-seconds", type=int, default=600)
    parser.add_argument("--ssh-wait-seconds", type=int, default=180)
    parser.add_argument("--max-runtime-minutes", type=int, default=420)
    parser.add_argument("--allow-existing-pods", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timings-output", type=Path)
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument("--keep-pod-on-failure", action="store_true")
    parser.add_argument("--local", action="append", default=[])
    parser.add_argument("--sync", action="append", default=[])
    parser.add_argument("--setup-command", required=True)
    parser.add_argument("--remote", action="append", default=[])
    parser.add_argument("--output", action="append", required=True)
    return parser.parse_args()


def find_created(before: list[dict[str, str]], after: list[dict[str, str]], name: str) -> dict[str, str]:
    before_ids = {pod["id"] for pod in before}
    created = [pod for pod in after if pod["id"] not in before_ids and pod["name"] == name]
    if len(created) != 1:
        raise RuntimeError(f"could not identify created pod named {name}")
    return created[0]


def sync_sources(args: argparse.Namespace) -> list[str]:
    return [source for source in [*DEFAULT_SYNC, *args.sync] if (args.repo_root / source).exists()]


def print_dry_run_plan(args: argparse.Namespace, secrets: list[str], public_key: str) -> None:
    connection = Connection(host="dry-run.runpod.local", port=22)
    print("dry-run: no RunPod API calls, SSH connections, rsync, or remote commands will run")
    print("pod payload:")
    print(redact(json.dumps(pod_payload(args, public_key), indent=2, sort_keys=True), secrets))
    print("sync sources:")
    for source in sync_sources(args):
        print(f"- {source}")
    if args.local:
        print("local preflight commands:")
        for command in args.local:
            dry_run(local_shell_command(command), secrets=secrets)
    print("commands:")
    dry_run([args.runpodctl, "pod", "list", "-o", "json"], secrets=secrets)
    dry_run(["curl", "--request", "POST", "--url", "https://rest.runpod.io/v1/pods", "--data", json.dumps(pod_payload(args, public_key), separators=(",", ":"))], secrets=secrets)
    dry_run(ssh(args, connection, f"mkdir -p {quote(args.remote_dir)}"), secrets=secrets)
    dry_run(
        [
            "rsync",
            "-az",
            "--relative",
            "--timeout",
            "30",
            "-e",
            rsync_ssh(args, connection),
            *[f"./{source}" for source in sync_sources(args)],
            f"{connection.user}@{connection.host}:{args.remote_dir}/",
        ],
        secrets=secrets,
    )
    dry_run(ssh(args, connection, remote_dir_command(args, args.setup_command)), secrets=secrets)
    for command in args.remote:
        dry_run(ssh(args, connection, remote_dir_command(args, split_remote(command))), secrets=secrets)
    for output in args.output:
        dry_run(
            [
                "rsync",
                "-az",
                "--timeout",
                "30",
                "-e",
                rsync_ssh(args, connection),
                f"{connection.user}@{connection.host}:{args.remote_dir}/{output.rstrip('/')}/",
                str(args.repo_root / output),
            ],
            secrets=secrets,
        )
    dry_run([args.runpodctl, "pod", "delete", "dry-run-pod"], secrets=secrets)


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.pod_name = args.pod_name or timestamped_name(args.name)
    timings = TimingRecorder(args.timings_output, pod_name=args.pod_name, dry_run=args.dry_run)
    status = "failed"
    pod_id: str | None = None
    success = False
    try:
        if args.dry_run and args.ssh_key is None:
            args.ssh_key = Path("dry-run-ssh-key")
        if args.dry_run and args.ssh_public_key is None:
            args.ssh_public_key = Path("dry-run-ssh-key.pub")
        if not args.dry_run:
            for index, command in enumerate(args.local, start=1):
                timings.step(
                    f"local_preflight_{index}",
                    lambda command=command: run(local_shell_command(command), cwd=args.repo_root, secrets=[]),
                )
        if not args.dry_run and shutil.which(args.runpodctl) is None:
            raise FileNotFoundError(f"runpodctl command not found: {args.runpodctl}")
        if shutil.which("rsync") is None:
            raise FileNotFoundError("rsync command not found")
        if shutil.which("ssh") is None:
            raise FileNotFoundError("ssh command not found")
        if shutil.which("curl") is None:
            raise FileNotFoundError("curl command not found")
        if args.ssh_key is None:
            raise ValueError("set RUNPOD_SSH_KEY or pass --ssh-key")
        if args.ssh_public_key is None:
            raise ValueError("set RUNPOD_SSH_PUBLIC_KEY or pass --ssh-public-key")
        if not args.dry_run and not args.ssh_key.exists():
            raise FileNotFoundError(args.ssh_key)
        if not args.dry_run and not args.ssh_public_key.exists():
            raise FileNotFoundError(args.ssh_public_key)

        api_key = os.environ.get("RUNPOD_API_KEY", "")
        if not api_key and not args.dry_run:
            if args.secret_path is None:
                raise ValueError("set RUNPOD_API_KEY, set RUNPOD_API_KEY_FILE, or pass --secret-path")
            api_key = load_text(args.secret_path)
        public_key = load_text(args.ssh_public_key) if args.ssh_public_key.exists() else ""
        secrets = [api_key, public_key]
        if args.dry_run:
            if not sync_sources(args):
                raise RuntimeError("no sync sources found")
            print_dry_run_plan(args, secrets, public_key)
            status = "dry-run"
            return 0
        if not args.allow_existing_pods:
            pods = timings.step("active_pods_check", lambda: active_pods(args, secrets))
            if pods:
                names = ", ".join(f"{pod['name']}:{pod['id']}" for pod in pods)
                raise RuntimeError(f"RunPod account already has active pods: {names}")
        before = timings.step("pod_list_before", lambda: list_pods(args, secrets))
        timings.step("pod_create", lambda: create_pod(args, secrets, api_key, public_key))
        after = timings.step("pod_list_after", lambda: list_pods(args, secrets))
        pod = find_created(before, after, args.pod_name)
        pod_id = pod["id"]
        timings.set_pod_id(pod_id)
        print(f"created pod: {pod_id}")
        connection = timings.step("ssh_info_wait", lambda: wait_for_connection(args, pod_id, secrets))
        print(f"ssh: {connection.user}@{connection.host}:{connection.port}")
        timings.step("ssh_ready_wait", lambda: wait_for_ssh(args, connection, secrets))
        timings.step(
            "transport_setup",
            lambda: run(
                ssh(
                    args,
                    connection,
                    "set -euo pipefail; if ! command -v rsync >/dev/null 2>&1; then apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y rsync; fi",
                ),
                cwd=args.repo_root,
                secrets=secrets,
            ),
        )
        timings.step(
            "remote_mkdir",
            lambda: run(ssh(args, connection, f"mkdir -p {quote(args.remote_dir)}"), cwd=args.repo_root, secrets=secrets),
        )
        timings.step("repo_sync", lambda: rsync_to_remote(args, connection, secrets))
        timings.step(
            "setup",
            lambda: run(ssh(args, connection, remote_dir_command(args, args.setup_command)), cwd=args.repo_root, secrets=secrets),
        )
        deadline = time.monotonic() + args.max_runtime_minutes * 60 if args.max_runtime_minutes > 0 else None
        for index, command in enumerate(args.remote, start=1):
            timeout = None if deadline is None else max(1, deadline - time.monotonic())
            timings.step(
                f"remote_{index}",
                lambda command=command, timeout=timeout: run(
                    ssh(args, connection, remote_dir_command(args, split_remote(command))),
                    cwd=args.repo_root,
                    secrets=secrets,
                    timeout=timeout,
                ),
            )
        timings.step("output_sync", lambda: rsync_from_remote(args, connection, secrets))
        success = True
        status = "passed"
        return 0
    finally:
        if pod_id and not args.keep_pod and (success or not args.keep_pod_on_failure):
            timings.step(
                "pod_delete",
                lambda: run([args.runpodctl, "pod", "delete", pod_id], cwd=args.repo_root, secrets=[], check=False),
            )
        timings.finish(status)


if __name__ == "__main__":
    raise SystemExit(main())
