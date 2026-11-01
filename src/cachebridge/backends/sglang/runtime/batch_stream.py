"""Host-visible streaming batch endpoints; chunks are not engine token timings."""

import copy


class BatchStream:
    def __init__(self, requests, arrival_ns, submitted_ns):
        if (
            not requests
            or len(requests) > 16
            or type(arrival_ns) is not int
            or type(submitted_ns) is not int
            or arrival_ns > submitted_ns
        ):
            raise ValueError("Arrived bounded batch and real host timestamps required")
        if len({r["request_id"] for r in requests}) != len(requests):
            raise ValueError("Duplicate logical request ID")
        if any(
            r.get("arrival_offset_ms", 0) != 0
            or type(r["max_new_tokens"]) is not int
            or r["max_new_tokens"] <= 0
            or r["sampling"]["max_new_tokens"] != r["max_new_tokens"]
            for r in requests
        ):
            raise ValueError(
                "Only already-arrived common-epoch requests with matched token budgets"
            )
        self.requests = copy.deepcopy(requests)
        self.arrival_ns = arrival_ns
        self.submitted_ns = submitted_ns
        self.rows = [
            dict(
                first=None, last=None, count=0, engine_id=None, result=None, progress=[]
            )
            for _ in requests
        ]
        self.raw_chunks = []
        self.last_observed_ns = submitted_ns

    def observe(self, value, observed_ns):
        index = value.get("index")
        if type(index) is not int or not 0 <= index < len(self.rows):
            raise ValueError("Streaming batch result has unknown index")
        if type(observed_ns) is not int or observed_ns < self.last_observed_ns:
            raise ValueError("Host observation time moved backwards")
        meta = value["meta_info"]
        count = meta["completion_tokens"]
        engine_id = meta["id"]
        row = self.rows[index]
        request = self.requests[index]
        if (
            type(count) is not int
            or not row["count"] <= count <= request["max_new_tokens"]
            or not isinstance(engine_id, str)
            or not engine_id
        ):
            raise ValueError("Invalid completion count or engine identity")
        if row["engine_id"] is not None and row["engine_id"] != engine_id:
            raise ValueError("Engine ID changed for one batch index")
        if any(
            i != index and r["engine_id"] == engine_id for i, r in enumerate(self.rows)
        ):
            raise ValueError("One engine ID maps to multiple logical requests")
        if count > 0 and row["first"] is None:
            row["first"] = observed_ns
        row.update(
            engine_id=engine_id,
            count=count,
            last=observed_ns,
            result=copy.deepcopy(value),
        )
        row["progress"].append(
            dict(host_observed_ns=observed_ns, completion_tokens=count)
        )
        self.raw_chunks.append(
            dict(index=index, host_observed_ns=observed_ns, value=copy.deepcopy(value))
        )
        self.last_observed_ns = observed_ns

    def finish(self):
        outputs = []
        for request, row in zip(self.requests, self.rows):
            if (
                row["count"] != request["max_new_tokens"]
                or row["first"] is None
                or row["result"] is None
            ):
                raise ValueError("Incomplete streaming batch output")
            outputs.append(
                dict(
                    request_id=request["request_id"],
                    arrival_ns=self.arrival_ns,
                    submitted_ns=self.submitted_ns,
                    host_first_token_ns=row["first"],
                    host_finished_ns=row["last"],
                    frontend_queue_ms=(self.submitted_ns - self.arrival_ns) / 1e6,
                    ttft_ms=(row["first"] - self.arrival_ns) / 1e6,
                    completion_ms=(row["last"] - self.arrival_ns) / 1e6,
                    stream_chunks=len(row["progress"]),
                    chunk_progress=copy.deepcopy(row["progress"]),
                    result=copy.deepcopy(row["result"]),
                )
            )
        return dict(
            outputs=outputs,
            raw_chunks=copy.deepcopy(self.raw_chunks),
            first_token_endpoint="first host observation of positive completion_tokens",
            token_intervals_are_host_chunk_observations_not_engine_TBT=True,
        )


async def run(engine, requests, arrival_ns, clock):
    # All requested arrivals are already present. No future arrival batching.
    submitted_ns = clock()
    collector = BatchStream(requests, arrival_ns, submitted_ns)
    stream = await engine.async_generate(
        input_ids=[r["ids"] for r in requests],
        sampling_params=[r["sampling"] for r in requests],
        stream=True,
    )
    async for value in stream:
        observed_ns = clock()
        collector.observe(value, observed_ns)
    return collector.finish()
