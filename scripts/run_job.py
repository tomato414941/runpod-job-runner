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
import base64
import textwrap


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
        except BaseException:
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


def run_capture(
    command: list[str],
    *,
    cwd: Path,
    secrets: list[str],
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout: float | None = None,
    print_command: bool = True,
) -> subprocess.CompletedProcess[str]:
    if print_command:
        print(f"$ {redact(command_text(command), secrets)}")
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=os.environ | (env or {}),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        if completed.stdout:
            print(redact(completed.stdout, secrets), end="")
        if completed.stderr:
            print(redact(completed.stderr, secrets), end="", file=sys.stderr)
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
        "computeType": args.compute_type,
        "containerDiskInGb": args.container_disk_size,
        "volumeMountPath": args.remote_volume,
        "ports": ["22/tcp"],
        "cloudType": "SECURE" if args.secure_cloud else "COMMUNITY",
    }
    if args.network_volume_id:
        payload["networkVolumeId"] = args.network_volume_id
    else:
        payload["volumeInGb"] = args.volume_size
    if args.image:
        payload["imageName"] = args.image
    elif args.template_id:
        payload["templateId"] = args.template_id
    else:
        raise ValueError("set --template-id or --image")
    if args.compute_type == "GPU":
        payload["gpuTypeIds"] = [args.gpu_type]
        payload["gpuCount"] = args.gpu_count
        if args.allowed_cuda_version:
            payload["allowedCudaVersions"] = args.allowed_cuda_version
        if args.min_vcpu_per_gpu is not None:
            payload["minVCPUPerGPU"] = args.min_vcpu_per_gpu
    elif args.compute_type == "CPU":
        if args.cpu_flavor_id:
            payload["cpuFlavorIds"] = args.cpu_flavor_id
        payload["cpuFlavorPriority"] = args.cpu_flavor_priority
        payload["vcpuCount"] = args.vcpu_count
    else:
        raise ValueError(f"unsupported compute type: {args.compute_type}")
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
            "--no-owner",
            "--no-group",
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
                "--no-owner",
                "--no-group",
                "-e",
                rsync_ssh(args, connection),
                f"{connection.user}@{connection.host}:{args.remote_dir}/{output.rstrip('/')}/",
                str(local),
            ],
            cwd=args.repo_root,
            secrets=secrets,
        )


def primary_output(args: argparse.Namespace) -> str:
    return str(args.output[0]).rstrip("/")


def remote_dir_command(args: argparse.Namespace, command: str) -> str:
    return f"REMOTE_DIR={quote(args.remote_dir)}; {command}"


def split_remote(command: str) -> str:
    if not command.strip():
        raise ValueError("remote command must not be empty")
    return command


def detached_state_dir(remote_dir: str, index: int) -> str:
    return f"{remote_dir.rstrip('/')}/.runpod-job-runner/remote-{index:03d}"


def install_detached_remote_script(
    args: argparse.Namespace,
    connection: Connection,
    secrets: list[str],
    *,
    index: int,
    command: str,
) -> None:
    state_dir = detached_state_dir(args.remote_dir, index)
    script_text = f"#!/usr/bin/env bash\nREMOTE_DIR={quote(args.remote_dir)}\n{split_remote(command)}\n"
    encoded = base64.b64encode(script_text.encode("utf-8")).decode("ascii")
    remote_command = "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {quote(state_dir)}",
            "python3 - <<'PY'",
            "from pathlib import Path",
            "import base64",
            f"path = Path({state_dir!r}) / 'command.sh'",
            f"path.write_bytes(base64.b64decode({encoded!r}))",
            "path.chmod(0o755)",
            "PY",
            f"rm -f {quote(state_dir + '/exit_code')} {quote(state_dir + '/pid')} {quote(state_dir + '/job.log')}",
        ]
    )
    run(ssh(args, connection, remote_command), cwd=args.repo_root, secrets=secrets)


def start_detached_remote(
    args: argparse.Namespace,
    connection: Connection,
    secrets: list[str],
    *,
    index: int,
) -> None:
    state_dir = detached_state_dir(args.remote_dir, index)
    remote_command = " ".join(
        [
            "set -euo pipefail;",
            f"state_dir={quote(state_dir)};",
            'nohup bash -c \'set +e; bash "$0/command.sh"; code=$?; printf "%s\\n" "$code" > "$0/exit_code"; exit "$code"\' "$state_dir"',
            '> "$state_dir/job.log" 2>&1 < /dev/null &',
            'printf "%s\\n" "$!" > "$state_dir/pid"',
        ]
    )
    run(ssh(args, connection, remote_command), cwd=args.repo_root, secrets=secrets)


def install_resource_monitor(
    args: argparse.Namespace,
    connection: Connection,
    secrets: list[str],
) -> None:
    script_text = r'''
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import time


running = True


def stop(_signum: int, _frame: object) -> None:
    global running
    running = False


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def cpu_totals() -> tuple[int, int] | None:
    try:
        parts = Path('/proc/stat').read_text(encoding='utf-8').splitlines()[0].split()[1:]
    except Exception:
        return None
    values = [int(part) for part in parts]
    idle = values[3] + values[4]
    return sum(values), idle


def memory_sample() -> dict[str, float | None]:
    try:
        values: dict[str, int] = {}
        for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
            key, value = line.split(':', 1)
            values[key] = int(value.strip().split()[0])
    except Exception:
        return {'ram_used_mib': None, 'ram_total_mib': None}
    total = values.get('MemTotal')
    available = values.get('MemAvailable')
    if total is None or available is None:
        return {'ram_used_mib': None, 'ram_total_mib': None}
    return {'ram_used_mib': float((total - available) / 1024), 'ram_total_mib': float(total / 1024)}


def gpu_sample() -> dict[str, float | None]:
    command = [
        'nvidia-smi',
        '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw',
        '--format=csv,noheader,nounits',
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip().splitlines()[0]
        gpu_util, memory_used, memory_total, power_draw = [float(part.strip()) for part in output.split(',')]
    except Exception:
        return {
            'gpu_util_percent': None,
            'gpu_memory_used_mib': None,
            'gpu_memory_total_mib': None,
            'gpu_power_draw_w': None,
        }
    return {
        'gpu_util_percent': gpu_util,
        'gpu_memory_used_mib': memory_used,
        'gpu_memory_total_mib': memory_total,
        'gpu_power_draw_w': power_draw,
    }


def summarize(samples: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        'sample_count': len(samples),
        'started_at': samples[0]['timestamp'] if samples else None,
        'finished_at': samples[-1]['timestamp'] if samples else None,
    }
    numeric_keys = [
        'cpu_util_percent',
        'ram_used_mib',
        'ram_total_mib',
        'gpu_util_percent',
        'gpu_memory_used_mib',
        'gpu_memory_total_mib',
        'gpu_power_draw_w',
    ]
    for key in numeric_keys:
        values = [float(sample[key]) for sample in samples if isinstance(sample.get(key), int | float)]
        summary[f'{key}_avg'] = sum(values) / len(values) if values else None
        summary[f'{key}_max'] = max(values) if values else None
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--interval-seconds', type=float, default=1.0)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    samples_path = Path(args.samples)
    summary_path = Path(args.summary)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []
    previous_cpu = cpu_totals()
    with samples_path.open('a', encoding='utf-8') as file:
        while running:
            time.sleep(max(args.interval_seconds, 0.1))
            current_cpu = cpu_totals()
            cpu_util = None
            if previous_cpu is not None and current_cpu is not None:
                total_delta = current_cpu[0] - previous_cpu[0]
                idle_delta = current_cpu[1] - previous_cpu[1]
                if total_delta > 0:
                    cpu_util = 100.0 * (1.0 - idle_delta / total_delta)
            previous_cpu = current_cpu
            sample: dict[str, object] = {'timestamp': utc_timestamp(), 'cpu_util_percent': cpu_util}
            sample.update(memory_sample())
            sample.update(gpu_sample())
            samples.append(sample)
            file.write(json.dumps(sample, sort_keys=True) + '\n')
            file.flush()
    summary_path.write_text(json.dumps(summarize(samples), indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
'''
    state_dir = f"{args.remote_dir.rstrip('/')}/.runpod-job-runner"
    encoded = base64.b64encode(textwrap.dedent(script_text).encode("utf-8")).decode("ascii")
    remote_command = "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {quote(state_dir)}",
            "python3 - <<'PY'",
            "from pathlib import Path",
            "import base64",
            f"path = Path({state_dir!r}) / 'resource_monitor.py'",
            f"path.write_bytes(base64.b64decode({encoded!r}))",
            "path.chmod(0o755)",
            "PY",
        ]
    )
    run(ssh(args, connection, remote_command), cwd=args.repo_root, secrets=secrets)


def start_resource_monitor(args: argparse.Namespace, connection: Connection, secrets: list[str]) -> None:
    output = primary_output(args)
    state_dir = f"{args.remote_dir.rstrip('/')}/.runpod-job-runner"
    output_dir = f"{args.remote_dir.rstrip('/')}/{output}"
    samples_path = f"{output_dir}/resource_samples.jsonl"
    summary_path = f"{output_dir}/resource_summary.json"
    remote_command = " ".join(
        [
            "set -euo pipefail;",
            f"mkdir -p {quote(output_dir)};",
            f"state_dir={quote(state_dir)};",
            f"python3 {quote(state_dir + '/resource_monitor.py')}",
            f"--samples {quote(samples_path)}",
            f"--summary {quote(summary_path)}",
            f"--interval-seconds {quote(str(args.resource_monitor_interval_seconds))}",
            '> "$state_dir/resource_monitor.log" 2>&1 < /dev/null &',
            'printf "%s\\n" "$!" > "$state_dir/resource_monitor.pid"',
        ]
    )
    run(ssh(args, connection, remote_command), cwd=args.repo_root, secrets=secrets)


def stop_resource_monitor(args: argparse.Namespace, connection: Connection, secrets: list[str]) -> None:
    state_dir = f"{args.remote_dir.rstrip('/')}/.runpod-job-runner"
    remote_command = "\n".join(
        [
            "set +e",
            f"state_dir={quote(state_dir)}",
            'pid_file="$state_dir/resource_monitor.pid"',
            'if [ ! -f "$pid_file" ]; then exit 0; fi',
            'pid=$(cat "$pid_file")',
            'kill "$pid" >/dev/null 2>&1 || true',
            'for _ in $(seq 1 20); do',
            '  kill -0 "$pid" >/dev/null 2>&1 || exit 0',
            '  sleep 0.25',
            'done',
            'kill -9 "$pid" >/dev/null 2>&1 || true',
            "exit 0",
        ]
    )
    run(ssh(args, connection, remote_command), cwd=args.repo_root, secrets=secrets, check=False)


@dataclass(frozen=True)
class DetachedRemoteStatus:
    output: str
    log_size: int
    exit_code: int | None
    running: bool


def poll_detached_remote(
    args: argparse.Namespace,
    connection: Connection,
    secrets: list[str],
    *,
    index: int,
    log_offset: int,
) -> DetachedRemoteStatus | None:
    state_dir = detached_state_dir(args.remote_dir, index)
    remote_command = "\n".join(
        [
            "set +e",
            f"state_dir={quote(state_dir)}",
            'log="$state_dir/job.log"',
            'exit_code_file="$state_dir/exit_code"',
            'pid_file="$state_dir/pid"',
            f"offset={log_offset}",
            'size=0',
            'if [ -f "$log" ]; then',
            '  size=$(wc -c < "$log" | tr -d " ")',
            '  if [ "$size" -gt "$offset" ]; then',
            '    tail -c +"$((offset + 1))" "$log"',
            '  fi',
            'fi',
            'printf "\\n__RUNPOD_DETACHED_LOG_SIZE__=%s\\n" "$size"',
            'if [ -f "$exit_code_file" ]; then',
            '  code=$(cat "$exit_code_file")',
            '  printf "__RUNPOD_DETACHED_EXIT_CODE__=%s\\n" "$code"',
            '  exit 0',
            'fi',
            'if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then',
            '  printf "__RUNPOD_DETACHED_RUNNING__=1\\n"',
            '  exit 0',
            'fi',
            'printf "__RUNPOD_DETACHED_EXIT_CODE__=255\\n"',
        ]
    )
    completed = run_capture(
        ssh(args, connection, remote_command),
        cwd=args.repo_root,
        secrets=secrets,
        check=False,
        print_command=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"ssh exited {completed.returncode}"
        print(f"detached remote poll failed: {redact(message, secrets)}", file=sys.stderr)
        return None
    text = completed.stdout
    size_match = re.search(r"^__RUNPOD_DETACHED_LOG_SIZE__=(\d+)$", text, flags=re.MULTILINE)
    exit_match = re.search(r"^__RUNPOD_DETACHED_EXIT_CODE__=(\d+)$", text, flags=re.MULTILINE)
    running = re.search(r"^__RUNPOD_DETACHED_RUNNING__=1$", text, flags=re.MULTILINE) is not None
    output = re.sub(r"^__RUNPOD_DETACHED_(?:LOG_SIZE|EXIT_CODE|RUNNING)__=.*\n?", "", text, flags=re.MULTILINE)
    if output:
        print(redact(output, secrets), end="" if output.endswith("\n") else "\n", flush=True)
    log_size = int(size_match.group(1)) if size_match else log_offset
    exit_code = int(exit_match.group(1)) if exit_match else None
    return DetachedRemoteStatus(output=output, log_size=log_size, exit_code=exit_code, running=running)


def run_detached_remote(
    args: argparse.Namespace,
    connection: Connection,
    secrets: list[str],
    *,
    index: int,
    command: str,
    deadline: float | None,
) -> None:
    install_detached_remote_script(args, connection, secrets, index=index, command=command)
    start_detached_remote(args, connection, secrets, index=index)
    log_offset = 0
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"detached remote command exceeded max runtime: remote_{index}")
        status = poll_detached_remote(args, connection, secrets, index=index, log_offset=log_offset)
        if status is None:
            time.sleep(args.remote_poll_seconds)
            continue
        log_offset = max(log_offset, status.log_size)
        if status.exit_code is not None:
            if status.exit_code != 0:
                raise RuntimeError(f"detached remote command failed with exit code {status.exit_code}")
            return
        time.sleep(args.remote_poll_seconds)


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
    parser.add_argument("--compute-type", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--min-vcpu-per-gpu", type=int)
    parser.add_argument("--cpu-flavor-id", action="append", default=[])
    parser.add_argument("--cpu-flavor-priority", choices=("availability", "custom"), default="availability")
    parser.add_argument("--vcpu-count", type=int, default=16)
    parser.add_argument("--secure-cloud", action="store_true")
    parser.add_argument("--container-disk-size", type=int, default=20)
    parser.add_argument("--volume-size", type=int, default=20)
    parser.add_argument("--network-volume-id")
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
    parser.add_argument("--detached-remote", action="store_true")
    parser.add_argument("--remote-poll-seconds", type=int, default=30)
    parser.add_argument("--disable-resource-monitor", action="store_true")
    parser.add_argument("--resource-monitor-interval-seconds", type=float, default=1.0)
    parser.add_argument("--output", action="append", required=True)
    return parser.parse_args()


def validate_relative_path(value: str, *, option: str) -> None:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{option} must be relative to --repo-root: {value}")
    if not value.strip() or any(part == ".." for part in path.parts):
        raise ValueError(f"{option} must stay within --repo-root: {value}")


def validate_args(args: argparse.Namespace) -> None:
    for output in args.output:
        validate_relative_path(output, option="--output")
    if args.network_volume_id and not args.secure_cloud:
        raise ValueError("--network-volume-id requires --secure-cloud")


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
            "--no-owner",
            "--no-group",
            "-e",
            rsync_ssh(args, connection),
            *[f"./{source}" for source in sync_sources(args)],
            f"{connection.user}@{connection.host}:{args.remote_dir}/",
        ],
        secrets=secrets,
    )
    dry_run(ssh(args, connection, remote_dir_command(args, args.setup_command)), secrets=secrets)
    if not args.disable_resource_monitor:
        print("resource monitor:")
        dry_run(ssh(args, connection, "install resource monitor script"), secrets=secrets)
        dry_run(ssh(args, connection, "start resource monitor"), secrets=secrets)
    for command in args.remote:
        if args.detached_remote:
            print("dry-run detached remote command:")
            dry_run(ssh(args, connection, "install detached remote script"), secrets=secrets)
            dry_run(ssh(args, connection, "start detached remote script"), secrets=secrets)
            dry_run(ssh(args, connection, "poll detached remote status until exit"), secrets=secrets)
        else:
            dry_run(ssh(args, connection, remote_dir_command(args, split_remote(command))), secrets=secrets)
    if not args.disable_resource_monitor:
        dry_run(ssh(args, connection, "stop resource monitor"), secrets=secrets)
    for output in args.output:
        dry_run(
            [
                "rsync",
                "-az",
                "--timeout",
                "30",
                "--no-owner",
                "--no-group",
                "-e",
                rsync_ssh(args, connection),
                f"{connection.user}@{connection.host}:{args.remote_dir}/{output.rstrip('/')}/",
                str(args.repo_root / output),
            ],
            secrets=secrets,
        )
    if not args.keep_pod:
        dry_run([args.runpodctl, "pod", "delete", "dry-run-pod"], secrets=secrets)


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.timings_output is not None and not args.timings_output.is_absolute():
        args.timings_output = args.repo_root / args.timings_output
    args.pod_name = args.pod_name or timestamped_name(args.name)
    timings = TimingRecorder(args.timings_output, pod_name=args.pod_name, dry_run=args.dry_run)
    status = "failed"
    pod_id: str | None = None
    connection: Connection | None = None
    resource_monitor_started = False
    output_synced = False
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
        if not args.disable_resource_monitor:
            timings.step("resource_monitor_install", lambda: install_resource_monitor(args, connection, secrets))
            timings.step("resource_monitor_start", lambda: start_resource_monitor(args, connection, secrets))
            resource_monitor_started = True
        deadline = time.monotonic() + args.max_runtime_minutes * 60 if args.max_runtime_minutes > 0 else None
        for index, command in enumerate(args.remote, start=1):
            timeout = None if deadline is None else max(1, deadline - time.monotonic())
            if args.detached_remote:
                timings.step(
                    f"remote_{index}",
                    lambda command=command: run_detached_remote(
                        args,
                        connection,
                        secrets,
                        index=index,
                        command=command,
                        deadline=deadline,
                    ),
                )
            else:
                timings.step(
                    f"remote_{index}",
                    lambda command=command, timeout=timeout: run(
                        ssh(args, connection, remote_dir_command(args, split_remote(command))),
                        cwd=args.repo_root,
                        secrets=secrets,
                        timeout=timeout,
                    ),
                )
        if resource_monitor_started:
            timings.step("resource_monitor_stop", lambda: stop_resource_monitor(args, connection, secrets))
            resource_monitor_started = False
        timings.step("output_sync", lambda: rsync_from_remote(args, connection, secrets))
        output_synced = True
        success = True
        status = "passed"
        return 0
    finally:
        if resource_monitor_started and connection is not None:
            try:
                timings.step("resource_monitor_stop", lambda: stop_resource_monitor(args, connection, secrets))
            except Exception as exc:
                print(f"resource monitor stop failed: {exc}", file=sys.stderr)
        if pod_id and connection is not None and not output_synced and not args.dry_run:
            try:
                timings.step("output_sync_after_failure", lambda: rsync_from_remote(args, connection, secrets))
            except Exception as exc:
                print(f"output sync after failure failed: {exc}", file=sys.stderr)
        if pod_id and not args.keep_pod and (success or not args.keep_pod_on_failure):
            timings.step(
                "pod_delete",
                lambda: run([args.runpodctl, "pod", "delete", pod_id], cwd=args.repo_root, secrets=[], check=False),
            )
        timings.finish(status)


if __name__ == "__main__":
    raise SystemExit(main())
