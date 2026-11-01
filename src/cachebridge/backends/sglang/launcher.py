"""Launch workloads with explicit config and a verified prepared SGLang tree."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from .config import RuntimeConfig
from .dependency import verify_engine


def launch_environment(config, engine, devices, environ=None):
    devices = devices.split(",")
    if (
        len(devices) != 8
        or len(set(devices)) != 8
        or any(not d.isdigit() for d in devices)
    ):
        raise ValueError("Supply eight distinct visible GPU indices")
    engine = Path(engine).expanduser().resolve()
    verify_engine(engine)
    env = dict(os.environ if environ is None else environ)
    # Ensure subprocesses and spawned workers find this installed package too.
    package_parent = Path(__file__).resolve().parents[3]
    paths = [str(package_parent), str(engine / "python")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env.update(
        CACHEBRIDGE_CONFIG=str(config.config_path),
        CUDA_VISIBLE_DEVICES=",".join(devices),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CB_SYSTEM_TP=str(config.tensor_parallel_size),
        CACHEBRIDGE_RECEIVER_DECODER_OVERLAP="1",
        CACHEBRIDGE_GENERATED_BG_OVERLAP="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=os.pathsep.join(paths),
        TRITON_CACHE_DIR=env.get(
            "TRITON_CACHE_DIR",
            str(Path("/tmp") / ("cachebridge-triton-" + str(os.getuid()))),
        ),
    )
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--engine",
        type=Path,
        help="Prepared engine directory; otherwise engine_path in config",
    )
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    config = RuntimeConfig.load(args.config)
    engine = args.engine or config.engine_path
    if engine is None:
        candidate = Path.cwd() / ".cache/cachebridge/sglang"
        if candidate.is_dir():
            engine = candidate
    if engine is None:
        parser.error(
            "Supply --engine or engine_path; prepare it with cachebridge backend prepare"
        )
    environment = launch_environment(config, engine, args.devices)
    if args.check:
        command = [
            sys.executable,
            "-c",
            "import json; from importlib.metadata import version; "
            "from cachebridge.backends.sglang.runtime import config,service,session,serving_handoff_adapter; "
            "from cachebridge.backends.sglang.runtime.peer_runtime import PeerRuntime; "
            'print(json.dumps(dict(status="imports_passed",model_pair=config.validate_model_pair(),'
            'torch=version("torch"),transformers=version("transformers"),'
            "cuda_runtime=PeerRuntime().version,config=str(config.CONFIG_PATH),service=service.__file__)))",
        ]
    else:
        command = args.command
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            parser.error("Provide --check or a workload command after --")
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
