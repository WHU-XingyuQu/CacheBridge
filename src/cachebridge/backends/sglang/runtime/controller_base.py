from datetime import datetime, timezone
import hashlib, json, multiprocessing as mp, os, subprocess, time
from multiprocessing.connection import wait
from pathlib import Path
from cachebridge.backends.sglang.runtime.service import service


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_snapshot():
    applications = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        "compute_apps": [line for line in applications.splitlines() if line.strip()],
        "gpus": [line for line in gpus.splitlines() if line.strip()],
    }


def primary_worker_devices(receipts):
    snapshot = gpu_snapshot()
    uuid_to_index = {}
    for line in snapshot["gpus"]:
        fields = [field.strip() for field in line.split(",")]
        uuid_to_index[fields[1]] = int(fields[0])
    allocations = {}
    for line in snapshot["compute_apps"]:
        uuid, pid, memory = [field.strip() for field in line.split(",")]
        allocations.setdefault(int(pid), []).append((int(memory), uuid_to_index[uuid]))
    result = []
    for receipt in receipts:
        pid = int(receipt["pid"])
        if pid not in allocations:
            raise RuntimeError(f"worker {pid} has no visible GPU allocation")
        result.append(max(allocations[pid])[1])
    return result


class Services:
    def __init__(self, root, socket_root):
        self.root = Path(root)
        self.socket_root = Path(socket_root)
        self.context = mp.get_context("spawn")
        self.children = {}
        self.pipes = {}
        self.sequence = 0
        self.controller_pid = os.getpid()

    def assert_isolation(self):
        snapshot = gpu_snapshot()
        parents = {
            int(fields[0]): int(fields[1])
            for line in subprocess.check_output(
                ["ps", "-eo", "pid=,ppid="], text=True
            ).splitlines()
            if (fields := line.split())
        }

        def ours(pid):
            visited = set()
            while pid > 1 and pid not in visited:
                if pid == self.controller_pid:
                    return True
                visited.add(pid)
                pid = parents.get(pid, 0)
            return False

        foreign = [
            int(line.split(",")[1])
            for line in snapshot["compute_apps"]
            if not ours(int(line.split(",")[1]))
        ]
        if foreign:
            raise RuntimeError(f"foreign GPU processes detected: {foreign}")
        return snapshot

    def receive(self, role, deadline):
        pipe = self.pipes[role]
        process = self.children[role]
        while not pipe.poll(0.25):
            if not process.is_alive():
                raise RuntimeError(f"{role} service exited")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{role} service response timed out")
        value = pipe.recv()
        if not value.get("ok", value.get("ready", False)):
            raise RuntimeError(value)
        return value

    def start(self):
        deadline = time.monotonic() + 1200
        for role in ("source", "target"):
            parent, child = self.context.Pipe()
            process = self.context.Process(
                target=service,
                args=(role, self.root / role, child, self.socket_root),
                name=f"qwen14-32-{role}",
            )
            process.start()
            child.close()
            self.children[role], self.pipes[role] = process, parent
        return {role: self.receive(role, deadline) for role in ("source", "target")}

    def call(self, role, command, timeout=600):
        self.pipes[role].send(command)
        return self.receive(role, time.monotonic() + timeout)

    def rpc(self, role, action, phase, **parameters):
        return self.call(
            role,
            {
                "op": "rpc",
                "parameters": {"action": action, "phase": phase, **parameters},
            },
        )["result"]

    def status(self, role, phase, histories=None):
        self.sequence += 1
        label = f"{phase}-{self.sequence}"
        self.rpc(
            role,
            "concurrency_status",
            label,
            history_ids_list=histories or [],
        )
        rows = [
            json.loads(
                (
                    self.root / role / f"{role}-concurrency-status-rank{rank}.json"
                ).read_text()
            )
            for rank in range(4)
        ]
        if not all(
            row["phase"] == label and row["rank"] == rank
            for rank, row in enumerate(rows)
        ):
            raise RuntimeError("stale concurrency status receipt")
        return rows

    def reset(self, role, phase):
        self.rpc(role, "flush", phase)
        self.rpc(role, "concurrency_clear_records", phase)
        rows = self.status(role, phase)
        if not all(row["idle"] and not row["quarantined"] for row in rows):
            raise RuntimeError(f"{role} did not return to a clean idle state")
        if not all(row["tree_tokens"] == 0 for row in rows):
            raise RuntimeError(f"{role} radix cache is not empty")
        return rows

    def generate(self, role, ids, sampling=None):
        return self.call(
            role,
            {
                "op": "generate",
                "ids": ids,
                "sampling": sampling
                or {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
            },
        )["result"]

    def setup_source(self, requests, phase):
        self.reset("source", phase)
        for request in requests:
            output = self.generate("source", request["history_ids"])
            if output["meta_info"]["completion_tokens"] != 1:
                raise RuntimeError("source population probe did not finish")
        histories = [request["history_ids"] for request in requests]
        rows = self.status("source", phase, histories)
        for row in rows:
            if [item["resident_tokens"] for item in row["histories"]] != [
                len(ids) for ids in histories
            ]:
                raise RuntimeError("source histories are not simultaneously resident")
        return rows

    def handoff(self, request, method, phase, execution="double"):
        parameters = {
            "action": "concurrency_run",
            "phase": phase,
            "request_id": phase,
            "history_ids": request["history_ids"],
            "mapper_kind": method,
            "execution": execution,
        }
        for role in ("target", "source"):
            self.pipes[role].send({"op": "rpc", "parameters": parameters})
        pending = {self.pipes[role]: role for role in ("target", "source")}
        replies = {}
        deadline = time.monotonic() + 600
        while pending:
            if time.monotonic() >= deadline:
                raise TimeoutError("handoff response timed out")
            for pipe in wait(list(pending), timeout=0.25):
                replies[pending.pop(pipe)] = pipe.recv()
        failures = {
            role: value for role, value in replies.items() if not value.get("ok")
        }
        if failures:
            raise RuntimeError(failures)
        return replies

    def generate_concurrent(self, requests, common_arrival_ns):
        payload = [
            {
                "request_id": request["request_id"],
                "ids": request["history_ids"] + request["query_ids"],
                "sampling": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
            }
            for request in requests
        ]
        return self.call(
            "target",
            {
                "op": "generate_concurrent",
                "requests": payload,
                "common_arrival_ns": common_arrival_ns,
            },
        )["result"]

    def stop(self):
        for role, process in self.children.items():
            if process.is_alive():
                self.pipes[role].send({"op": "stop"})
        for role, process in self.children.items():
            if process.is_alive():
                try:
                    self.receive(role, time.monotonic() + 120)
                except BaseException:
                    pass
            process.join(timeout=120)
