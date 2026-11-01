"""Actual threaded callback drivers; CUDA operations supplied by mixed_cuda."""

from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
from cachebridge.backends.sglang.runtime.mixed_channel import RAW, MAPPED, blocks


def source_pipeline(link, operations):
    plan = blocks(link.contract)
    futures = {}
    prepared = set()
    decisions = []
    tail = plan[-1] if plan[-1].mode == MAPPED else None
    transport = ThreadPoolExecutor(max_workers=1)

    def prepare(b, cause):
        key = b.bank, b.slot
        if key in futures:
            raise RuntimeError("mixed prepare would overwrite unacknowledged export")
        outstanding = [x for x, f in futures.items() if x[0] == RAW and not f.done()]
        operations.prepare(b)
        prepared.add(b)
        if b.mode == MAPPED:
            decisions.append(
                dict(
                    start=b.start,
                    stop=b.stop,
                    cause=cause,
                    raw_ACK_futures_outstanding=len(outstanding),
                    prepared_returned_ns=time.perf_counter_ns(),
                )
            )

    try:
        for b in plan:
            key = b.bank, b.slot
            previous = futures.get(key)
            if previous is not None:
                if (
                    tail is not None
                    and tail not in prepared
                    and b.mode == RAW
                    and not previous.done()
                ):
                    prepare(tail, "observed_raw_slot_reuse_backpressure")
                previous.result()
                del futures[key]
            if b not in prepared:
                prepare(b, "ordered_submission")
            futures[key] = transport.submit(link.offer, b)
        for future in futures.values():
            future.result()
        link.finish()
        return dict(
            blocks=len(plan),
            logical_delivery_bytes=sum(
                2 * (40 if b.mode == RAW else 64) * (b.stop - b.start) * 2 * 128 * 2
                for b in plan
            ),
            mixed_preparation=decisions,
            all_read_ACKs_complete=True,
        )
    finally:
        transport.shutdown(wait=True)
        # On failure no remote completion is inferred; outer Source lifetime
        # quarantines and retains the source pin, exactly as before.
        link.close()


def receiver_pipeline(listener, contract, operations, receiver_class):
    plan = blocks(contract)
    ready = queue.Queue(maxsize=2)
    result = {"error": None, "safe": True, "bytes_by_mode": {RAW: 0, MAPPED: 0}}

    def reader():
        link = None
        try:
            with operations.device_context():
                link = receiver_class(listener, contract, operations.codecs)
                for b in plan:
                    if operations.cancelled():
                        raise RuntimeError("mixed mapping cancelled")
                    value = link.next(b)
                    link.acknowledge()
                    ready.put((b, value), timeout=60)
                link.finish()
                result["bytes_by_mode"] = dict(link.bytes_by_mode)
        except BaseException as e:
            result["error"] = repr(e)
            try:
                operations.synchronize_copies()
                if link is not None:
                    link.abort()
            except BaseException as cleanup:
                result["safe"] = False
                result["error"] += "; copy_cleanup=" + repr(cleanup)
            try:
                ready.put(("error", result["error"]), timeout=1)
            except queue.Full:
                pass

    thread = threading.Thread(
        target=reader, daemon=True, name="cachebridge-mixed-reader"
    )
    thread.start()
    installed = []
    error = None
    try:
        with operations.device_context():
            for b in plan:
                item = ready.get(timeout=120)
                if item[0] == "error":
                    raise RuntimeError(item[1])
                observed, value = item
                if observed != b:
                    raise RuntimeError("mixed local installation sequence")
                if b.mode == RAW:
                    operations.map_install(b, value)
                # Mapped copy already completed direct installation before ACK.
                installed.append(b)
    except BaseException as e:
        error = repr(e)
    finally:
        try:
            operations.finish_compute_and_cancel()
        except BaseException as cleanup:
            result["safe"] = False
            error = (error or "") + "; compute_cleanup=" + repr(cleanup)
        thread.join(timeout=120)
        if thread.is_alive():
            result["safe"] = False
            error = (error or "") + "; reader_not_exited"
    result["error"] = error or result["error"]
    result["blocks"] = len(installed)
    if result["error"] is None and result["safe"]:
        expected = {
            mode: sum(
                2 * (40 if mode == RAW else 64) * (b.stop - b.start) * 2 * 128 * 2
                for b in plan
                if b.mode == mode
            )
            for mode in (RAW, MAPPED)
        }
        if result["bytes_by_mode"] != expected or installed != list(plan):
            result["error"] = "mixed bytes/full-H completion join mismatch"
    return result
