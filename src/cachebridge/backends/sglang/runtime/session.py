import json, os, time
from pathlib import Path
from cachebridge.backends.sglang.runtime import config as _config
from cachebridge.backends.sglang.runtime import controller_base as base


class Services(base.Services):
    def stop(self, grace_seconds=120):
        """Drain both endpoints, then clean up only verified owned descendants."""
        import psutil

        deadline = time.monotonic() + grace_seconds
        for role, process in self.children.items():
            if process.is_alive():
                try:
                    self.pipes[role].send(dict(op="stop"))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for process in self.children.values():
            process.join(timeout=max(0, deadline - time.monotonic()))
        forced = []
        for role, process in self.children.items():
            if not process.is_alive():
                continue
            try:
                parent = psutil.Process(process.pid)
                if parent.ppid() != os.getpid() or parent.uids().real != os.getuid():
                    raise RuntimeError(
                        "Cannot verify endpoint ownership; leave it untouched"
                    )
                # psutil Process instances retain creation identity across signals.
                candidates = parent.children(recursive=True) + [parent]
                family = []
                for child in candidates:
                    try:
                        if child.uids().real != os.getuid():
                            raise RuntimeError("Endpoint descendant ownership changed")
                        family.append(child)
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                process.join(timeout=1)
                continue
            for child in reversed(family):
                try:
                    child.terminate()
                    forced.append(child.pid)
                except psutil.NoSuchProcess:
                    pass
            # multiprocessing owns waitpid for its direct child. psutil must not
            # reap that PID or multiprocessing can lose the child's exit status.
            _, alive = psutil.wait_procs(
                [p for p in family if p.pid != process.pid], timeout=10
            )
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.join(timeout=10)
            if process.is_alive():
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass
                process.join(timeout=10)
            if process.is_alive():
                raise RuntimeError("Owned endpoint did not exit")
        self.root.mkdir(parents=True, exist_ok=True)
        base.write(
            self.root / "shutdown_receipt.json",
            dict(
                status="passed",
                forced_owned_pids=forced,
                endpoints={
                    role: dict(pid=p.pid, exitcode=p.exitcode)
                    for role, p in self.children.items()
                },
            ),
        )
        for pipe in self.pipes.values():
            pipe.close()

    def rpc(self, role, action, phase, **kwargs):
        if action in ("source_producer_probe", "online_stage_probe"):
            kwargs["probe_completion_backend"] = getattr(
                self, "probe_completion_backend", "default"
            )
        return super().rpc(role, action, phase, **kwargs)

    def call(self, role, command, timeout=600):
        if command["op"] == "trace_open" and getattr(
            self, "system_capture_progress", False
        ):
            command = dict(command, capture_chunk_progress=True)
        if command["op"] == "trace_open":
            command = dict(
                command, offload_control_rpc=getattr(self, "system_offload_rpc", False)
            )
        started = time.perf_counter_ns()
        reply = super().call(role, command, timeout=timeout)
        if hasattr(self, "rpc_observations"):
            self.rpc_observations.append(
                dict(
                    role=role,
                    op=command["op"],
                    action=command.get("parameters", {}).get("action"),
                    phase=command.get("parameters", {}).get("phase"),
                    controller_started_ns=started,
                    controller_finished_ns=time.perf_counter_ns(),
                    service_received_ns=reply.get("service_received_ns"),
                    service_finished_ns=reply.get("service_finished_ns"),
                )
            )
        return reply

    def reference_handoff(self, request, phase, block_tokens):
        from multiprocessing.connection import wait

        parameters = dict(
            action="concurrency_run",
            phase=phase,
            request_id=phase,
            history_ids=request["history_ids"],
            mapper_kind="CacheBridge",
            execution="serial",
            reference_block_tokens=block_tokens,
        )
        for role in ("target", "source"):
            self.pipes[role].send(dict(op="rpc", parameters=parameters))
        pending = {self.pipes[role]: role for role in ("target", "source")}
        deadline = time.monotonic() + 600
        while pending:
            if time.monotonic() > deadline:
                raise TimeoutError("reference handoff")
            for pipe in wait(list(pending), timeout=0.25):
                role = pending.pop(pipe)
                reply = pipe.recv()
                if not reply.get("ok"):
                    raise RuntimeError({role: reply})

    def assert_isolation(self):
        snapshot = base.gpu_snapshot()
        selected = {int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")}
        uuid_to_index = {
            line.split(",")[1].strip(): int(line.split(",")[0])
            for line in snapshot["gpus"]
        }
        foreign = []
        group = os.getpgrp()
        for line in snapshot["compute_apps"]:
            uuid, pid, _ = [x.strip() for x in line.split(",")]
            if uuid_to_index[uuid] not in selected:
                continue
            try:
                if os.getpgid(int(pid)) != group:
                    foreign.append(int(pid))
            except ProcessLookupError:
                pass
        if foreign:
            raise RuntimeError(f"foreign processes on selected GPUs: {foreign}")
        return snapshot

    def status(self, role, phase, histories=None):
        self.sequence += 1
        label = f"{phase}-{self.sequence}"
        self.rpc(role, "concurrency_status", label, history_ids_list=histories or [])
        rows = [
            json.loads(
                (
                    self.root / role / f"{role}-concurrency-status-rank{rank}.json"
                ).read_text()
            )
            for rank in range(_config.TARGET_TP)
        ]
        if any(
            row["phase"] != label or row["rank"] != rank
            for rank, row in enumerate(rows)
        ):
            raise RuntimeError("stale status receipt")
        return rows
