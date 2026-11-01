"""SGLang RPC and scheduler-receipt adapter for generic handoff coordination."""

import json
from ...online.handoff import HandoffRequest, run_handoff


class SGLangHandoffEndpoint:
    def __init__(self, client, target_ranks):
        self.client = client
        self.target_ranks = target_ranks

    def start(self, role, parameters):
        self.client.pipes[role].send(dict(op="rpc", parameters=parameters))

    def receive(self, role, deadline):
        return self.client.receive(role, deadline)

    def try_publish(self, label, parameters):
        self.client.rpc(
            "target",
            "online_stage_try_publish",
            label,
            request_id=parameters["request_id"],
            transaction_phase=parameters["phase"],
            cache_namespace=parameters["cache_namespace"],
        )
        return [
            json.loads(
                (
                    self.client.root
                    / "target"
                    / f"target-ready-publication-rank{k}.json"
                ).read_text()
            )
            for k in range(self.target_ranks)
        ]


def transfer(
    client, request, phase, *, namespace=None, target_ranks=4, block_tokens=2048
):
    transaction = HandoffRequest(
        request["request_id"],
        phase,
        tuple(request["history_ids"]),
        namespace if namespace is not None else client.episode_namespace + "-agent",
        target_ranks,
        block_tokens,
    )
    return run_handoff(SGLangHandoffEndpoint(client, target_ranks), transaction)
