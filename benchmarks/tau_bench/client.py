import json
from pathlib import Path
from cachebridge.backends.sglang.runtime.session import Services, base


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class Client(Services):
    def __init__(self, root, sockets, output):
        super().__init__(root, sockets)
        self.output = output
        self.phase = "startup"

    def call(self, role, command, timeout=600):
        self.assert_isolation()
        reply = super().call(role, command, timeout)
        with (self.output / "raw_replies.jsonl").open("a") as stream:
            stream.write(
                json.dumps(dict(role=role, command=command, reply=reply)) + "\n"
            )
        self.assert_isolation()
        return reply
