"""Prepare the pinned SGLang submodule without changing its working tree."""

import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from ...artifacts import file_sha256

RESOURCES = Path(__file__).parent / "resources"
UPSTREAM_URL = "https://github.com/sgl-project/sglang.git"
RECEIPT = "cachebridge-engine.json"


def pin():
    value = json.loads((RESOURCES / "engine_provenance.json").read_text())
    if file_sha256(RESOURCES / "engine.patch") != value["patch_sha256"]:
        raise ValueError("SGLang compatibility patch hash mismatch")
    return value


def _git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def _files(root):
    return {
        str(p.relative_to(root)): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and p.name != RECEIPT
        and not any(
            part in {".git", "__pycache__", "build", "dist", ".pytest_cache"}
            or part.endswith(".egg-info")
            for part in p.relative_to(root).parts
        )
        and p.suffix not in {".pyc", ".pyo"}
    }


def verify_engine(output):
    output = Path(output).expanduser().resolve()
    expected = pin()
    record = json.loads((output / RECEIPT).read_text())
    if any(record.get(k) != expected[k] for k in ("base_commit", "patch_sha256")):
        raise ValueError(
            "Prepared SGLang version does not match this CacheBridge package"
        )
    if _files(output) != record["files"]:
        raise ValueError(
            "Prepared SGLang files changed; prepare a fresh output directory"
        )
    for name, digest in expected["files"].items():
        if record["files"].get(name) != digest:
            raise ValueError(f"SGLang compatibility file mismatch: {name}")
    return record


def prepare_engine(source, output, *, fetch=False):
    source, output = (Path(p).expanduser().resolve() for p in (source, output))
    expected = pin()
    if (
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError("Prepared engine and submodule must be separate directories")
    if output.exists():
        verify_engine(output)
        return output
    if not source.exists():
        if not fetch:
            raise FileNotFoundError(
                "Initialize third_party/sglang with git submodule update --init, or use --fetch"
            )
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "init", str(source)], check=True, stdout=subprocess.DEVNULL
        )
        _git(source, "remote", "add", "origin", UPSTREAM_URL)
        _git(source, "fetch", "--depth", "1", "origin", expected["base_commit"])
        _git(source, "checkout", "--detach", "FETCH_HEAD")
    if _git(source, "rev-parse", "HEAD") != expected["base_commit"]:
        raise ValueError("SGLang submodule is not at the pinned commit")
    if _git(source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(
            "SGLang submodule has tracked changes; keep patches in CacheBridge resources"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".sglang-prepare-", dir=output.parent
    ) as scratch:
        temp = Path(scratch) / "engine"
        temp.mkdir()
        archive = subprocess.check_output(
            ["git", "-C", str(source), "archive", expected["base_commit"]]
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(temp, filter="data")
        subprocess.run(
            ["git", "init", str(temp)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(temp),
                "apply",
                "--check",
                str(RESOURCES / "engine.patch"),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(temp), "apply", str(RESOURCES / "engine.patch")],
            check=True,
        )
        shutil.rmtree(temp / ".git")
        shutil.copyfile(
            RESOURCES / "CACHEBRIDGE_PATCH.md", temp / "CACHEBRIDGE_PATCH.md"
        )
        files = _files(temp)
        for name, digest in expected["files"].items():
            if files.get(name) != digest:
                raise ValueError(
                    f"Prepared engine does not reproduce pinned file: {name}"
                )
        record = dict(
            schema=1,
            upstream=UPSTREAM_URL,
            base_commit=expected["base_commit"],
            patch_sha256=expected["patch_sha256"],
            files=files,
        )
        (temp / RECEIPT).write_text(json.dumps(record, indent=2) + "\n")
        temp.rename(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source", type=Path, default=Path("third_party/sglang"))
    prepare.add_argument(
        "--output", type=Path, default=Path(".cache/cachebridge/sglang")
    )
    prepare.add_argument(
        "--fetch",
        action="store_true",
        help="Explicitly fetch the pinned upstream commit when source is absent",
    )
    verify = sub.add_parser("verify")
    verify.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(prepare_engine(args.source, args.output, fetch=args.fetch))
    else:
        record = verify_engine(args.directory)
        print(
            json.dumps(
                dict(
                    status="passed",
                    base_commit=record["base_commit"],
                    files=len(record["files"]),
                )
            )
        )


if __name__ == "__main__":
    main()
