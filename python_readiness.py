# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aiohttp>=3.10",
#     "packaging>=24",
# ]
# ///
from __future__ import annotations

import argparse
import asyncio
import collections
import email.parser
import email.policy
import enum
import functools
import hashlib
import importlib.metadata
import io
import json
import pathlib
import re
import sys
import sysconfig
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import aiohttp
import packaging.tags
import packaging.utils
import packaging.version
from packaging.requirements import Requirement
from packaging.specifiers import Specifier
from packaging.version import Version


# ==============================
# aiohttp caching
# ==============================


def _cache_key(url: str, **kwargs) -> str:
    key = json.dumps((url, kwargs), sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()


class CachedResponse:
    def __init__(self, body: bytes, status: int) -> None:
        self.body = body
        self.status = status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP status {self.status}")

    def json(self) -> Any:
        return json.loads(self.body)


class CachedSession:
    def __init__(self) -> None:
        self.session = aiohttp.ClientSession()
        self.cache_dir = Path(tempfile.gettempdir()) / "python_readiness_cache"

    async def get(self, url: str, **kwargs) -> CachedResponse:
        cache_file = self.cache_dir / _cache_key(url, **kwargs)
        if cache_file.is_dir():
            fetch_time = json.loads((cache_file / "fetch").read_text())
            if fetch_time > time.time() - 900:
                return CachedResponse(
                    body=(cache_file / "body").read_bytes(),
                    status=int((cache_file / "status").read_text()),
                )

        async with self.session.get(url, **kwargs) as resp:
            ret = CachedResponse(body=await resp.read(), status=resp.status)

        cache_file.mkdir(parents=True, exist_ok=True)
        (cache_file / "fetch").write_text(json.dumps(time.time()))
        (cache_file / "body").write_bytes(ret.body)
        (cache_file / "status").write_text(str(ret.status))
        return ret

    async def close(self) -> None:
        await self.session.close()


# ==============================
# latest python release
# ==============================


async def latest_python_release(session: CachedSession) -> tuple[int, int]:
    resp = await session.get("https://endoflife.date/api/python.json/")
    resp.raise_for_status()
    data = resp.json()
    latest = max(tuple(map(int, x["cycle"].split("."))) for x in data)
    assert len(latest) == 2
    return latest

# ==============================
# tag munging
# ==============================


@functools.cache
def interpreter_value(python_version: tuple[int, int]) -> str:
    nodot = "".join(map(str, python_version))
    assert sys.implementation.name == "cpython"
    return f"cp{nodot}"


@functools.cache
def valid_interpreter_abi_set(python_version: tuple[int, int]) -> set[tuple[str, str]]:
    assert sys.implementation.name == "cpython"
    # Based on logic in packaging.tags.sys_tags
    tags = set[packaging.tags.Tag]()
    # Note these values can be a little system dependent, but at least we mostly strip
    # platform dependence
    tags.update(packaging.tags.cpython_tags(python_version=python_version, abis=None))
    tags.update(
        packaging.tags.compatible_tags(
            python_version=python_version, interpreter=interpreter_value(python_version)
        )
    )
    return {(t.interpreter, t.abi) for t in tags}


def tag_viable_for_python(tag: packaging.tags.Tag, python_version: tuple[int, int]) -> bool:
    return (tag.interpreter, tag.abi) in valid_interpreter_abi_set(python_version)


# ==============================
# determining support
# ==============================


class PythonSupport(enum.IntEnum):
    unsupported = 0
    totally_unknown = 1
    has_viable_wheel = 2
    has_explicit_wheel = 3
    has_classifier = 4


async def support_from_wheels(
    session: CachedSession, wheels: list[dict[str, Any]], python_version: tuple[int, int]
) -> PythonSupport:
    if not wheels:
        return PythonSupport.totally_unknown

    support = PythonSupport.unsupported
    best_wheel = None

    for file in wheels:
        _, _, _, tags = packaging.utils.parse_wheel_filename(file["filename"])
        for tag in tags:
            # If we have a wheel specifically for this version, we're definitely supported
            if tag.interpreter == interpreter_value(python_version):
                support = PythonSupport.has_explicit_wheel
                if best_wheel is None or file.get("core-metadata"):
                    best_wheel = file
            # If we have a wheel that works for this version, we're maybe supported
            if tag_viable_for_python(tag, python_version):
                if support < PythonSupport.has_viable_wheel:
                    support = PythonSupport.has_viable_wheel
                    if best_wheel is None or file.get("core-metadata"):
                        best_wheel = file

    assert support <= PythonSupport.has_explicit_wheel
    if support == PythonSupport.unsupported:
        # We have no wheels that work for this version (and there are other wheels)
        # (don't bother to check if there is a classifier if we'd have to build sdist for support)
        return support
    assert support >= PythonSupport.has_viable_wheel

    # We have wheels that are at least viable for this version — time to check the classifiers!
    assert best_wheel is not None

    if best_wheel.get("core-metadata"):
        url = best_wheel["url"] + ".metadata"
        resp = await session.get(url)
        resp.raise_for_status()
        content = io.BytesIO(resp.body)
        parser = email.parser.BytesParser(policy=email.policy.compat32)
        metadata = parser.parse(content)
    else:
        url = best_wheel["url"]
        resp = await session.get(url)
        resp.raise_for_status()
        body = io.BytesIO(resp.body)
        with zipfile.ZipFile(body) as zf:
            metadata_file = next(
                n
                for n in zf.namelist()
                if Path(n).name == "METADATA" and Path(n).parent.suffix == ".dist-info"
            )
            with zf.open(metadata_file) as f:
                parser = email.parser.BytesParser(policy=email.policy.compat32)
                metadata = parser.parse(f)

    classifiers = set(metadata.get_all("Classifier", []))
    python_version_str = ".".join(str(v) for v in python_version)
    if f"Programming Language :: Python :: {python_version_str}" in classifiers:
        return PythonSupport.has_classifier
    return support


async def dist_support(
    session: CachedSession, name: str, python_version: tuple[int, int]
) -> tuple[Version | None, PythonSupport]:
    headers = {"Accept": "application/vnd.pypi.simple.v1+json"}

    resp = await session.get(f"https://pypi.org/simple/{name}/", headers=headers)
    if resp.status == 404:
        return None, PythonSupport.totally_unknown
    resp.raise_for_status()
    data = resp.json()

    version_wheels = collections.defaultdict[Version, list[dict[str, Any]]](list)

    for file in data["files"]:
        if not file["filename"].endswith(".whl"):
            continue
        _, version, _, _ = packaging.utils.parse_wheel_filename(file["filename"])
        if version.is_prerelease:
            continue
        version_wheels[version].append(file)

    all_versions = sorted((safe_version(v) for v in data["versions"]), reverse=True)
    all_versions = [v for v in all_versions if not v.is_prerelease]
    if not all_versions:
        return None, PythonSupport.totally_unknown
    latest_version = all_versions[0]

    support = await support_from_wheels(session, version_wheels[latest_version], python_version)
    if support <= PythonSupport.has_viable_wheel:
        return None, support

    # Try to figure out which version added the classifier / explicit wheel
    # Just do a dumb linear search
    earliest_supported_version = latest_version
    for version in all_versions:
        if version == latest_version:
            continue
        version_support = await support_from_wheels(
            session, version_wheels[version], python_version
        )
        if version_support < support:
            return earliest_supported_version, support
        earliest_supported_version = version

    return earliest_supported_version, support


# ==============================
# packaging utilities
# ==============================


def safe_version(v: str) -> Version:
    try:
        return Version(v)
    except packaging.version.InvalidVersion:
        return Version("0")


def parse_requirements_txt(req_file: str) -> list[str]:
    def strip_comments(s: str) -> str:
        try:
            return s[: s.index("#")].strip()
        except ValueError:
            return s.strip()

    with open(req_file) as f:
        contents = f.read().replace("\\\n", "")

    entries = []
    for line in contents.splitlines():
        entry = strip_comments(line)
        entry = entry.split("--")[0].strip()
        if entry:
            entries.append(entry)
    return entries


def approx_min_satisfying_version(r: Requirement) -> Version:
    def inner(spec: Specifier) -> Version:
        if spec.operator == "==":
            return Version(spec.version.removesuffix(".*"))
        if spec.operator == "~=":
            return Version(spec.version)
        if spec.operator == "!=":
            return Version("0")
        if spec.operator == "<=":
            return Version("0")
        if spec.operator == ">=":
            return Version(spec.version)
        if spec.operator == "<":
            return Version("0")
        if spec.operator == ">":
            return Version(spec.version)
        raise ValueError(f"Unknown operator {spec.operator}")

    return max((inner(spec) for spec in r.specifier), default=Version("0"))


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def combine_reqs(reqs: list[Requirement]) -> Requirement:
    assert reqs
    combined = Requirement(str(reqs[0]))
    for req in reqs[1:]:
        assert canonical_name(combined.name) == canonical_name(req.name)
        # It would be nice if there was an officially sanctioned way of combining these
        if combined.url and req.url and combined.url != req.url:
            raise RuntimeError(f"Conflicting URLs for {combined.name}: {combined.url} vs {req.url}")
        combined.url = combined.url or req.url
        combined.extras.update(req.extras)
        combined.specifier &= req.specifier
        if combined.marker and req.marker:
            # Note that if a marker doesn't pan out, it can still contribute its version specifier
            # to the combined requirement
            combined.marker._markers = [combined.marker._markers, "or", req.marker._markers]
        else:
            # If one of markers is None, that is an unconditional install
            combined.marker = None
    return combined


def deduplicate_reqs(reqs: list[Requirement]) -> list[Requirement]:
    simplified: dict[str, list[Requirement]] = {}
    for req in reqs:
        simplified.setdefault(canonical_name(req.name), []).append(req)
    return [combine_reqs(reqs) for reqs in simplified.values()]


@functools.lru_cache()
def sysconfig_purelib() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


# ==============================
# main
# ==============================


async def main() -> None:
    assert sys.version_info >= (3, 9)

    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=None)
    parser.add_argument("-p", "--package", action="append", default=[])
    parser.add_argument("-r", "--requirement", action="append", default=[])
    args = parser.parse_args()

    session = CachedSession()

    python_version: tuple[int, int]
    if args.python is None:
        python_version = await latest_python_release(session)
    else:
        python_version = tuple(map(int, args.python.split(".")))  # type: ignore
        if len(python_version) != 2:
            parser.error("Python version must be a major and minor version")

    for pkg in args.package:
        if re.fullmatch(r"(python)?[23]\.\d{1,2}", pkg):
            parser.error(f"Did you mean to use '--python {pkg}'? (Use -p to specify a package)")

    previous = [Requirement(r) for r in args.package]

    for req_file in args.requirement:
        previous.extend(Requirement(req) for req in parse_requirements_txt(req_file))

    if not previous:
        # Default to pulling "requirements" from the current environment
        venv_versions = {}
        for dist in importlib.metadata.distributions():
            if (
                isinstance(dist, importlib.metadata.PathDistribution)
                and (dist_path := getattr(dist, "_path", None))
                and isinstance(dist_path, pathlib.Path)
                and not dist_path.is_relative_to(sysconfig_purelib())
            ):
                continue
            metadata = dist.metadata
            venv_versions[metadata["Name"]] = Version(metadata["Version"]).base_version
        previous = [Requirement(f"{name}>={version}") for name, version in venv_versions.items()]

    previous = deduplicate_reqs(previous)

    supports: list[tuple[Version | None, PythonSupport]] = await asyncio.gather(
        *(dist_support(session, p.name, python_version) for p in previous)
    )
    package_support = dict(zip(previous, supports, strict=True))
    for previous_req, (version, support) in sorted(
        package_support.items(), key=lambda x: (-x[1][1].value, x[0].name)
    ):
        assert (version is None) == (support <= PythonSupport.has_viable_wheel)

        package = previous_req.name
        if version is None:
            new_req = Requirement(package)
        else:
            new_req = Requirement(f"{package}>={version}")

        PAD = 40
        previous_req_min = approx_min_satisfying_version(previous_req)
        if previous_req_min in new_req.specifier:
            if support > PythonSupport.has_viable_wheel:
                print(
                    f"{str(previous_req):<{PAD}}  # {support.name} (existing requirement ensures support)"
                )
            elif support >= PythonSupport.has_viable_wheel:
                print(f"{str(previous_req):<{PAD}}  # {support.name} (cannot ensure support)")
            else:
                print(f"{str(previous_req):<{PAD}}  # {support.name}")
        elif previous_req.specifier:
            print(f"{str(new_req):<{PAD}}  # {support.name} (previously: {str(previous_req)})")
        else:
            print(f"{str(new_req):<{PAD}}  # {support.name}")

    await session.close()


if __name__ == "__main__":
    asyncio.run(main())
