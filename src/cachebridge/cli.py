"""CacheBridge fitting, artifacts, backend preparation and serving commands."""

import argparse
from dataclasses import asdict
import json
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    dispatch = {
        "fit": "cachebridge.offline.cli",
        "verify": "cachebridge.artifacts",
        "serve": "cachebridge.backends.sglang.launcher",
        "backend": "cachebridge.backends.sglang.dependency",
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[*dispatch, "recipes"])
    if not argv or argv[0] not in dispatch:
        args = parser.parse_args(argv)
        if args.command == "recipes":
            from .offline.recipes import TRAJECTORY

            print(
                json.dumps(
                    dict(**asdict(TRAJECTORY), stages=TRAJECTORY.stages), indent=2
                )
            )
        return
    from importlib import import_module

    return import_module(dispatch[argv[0]]).main(argv[1:])
