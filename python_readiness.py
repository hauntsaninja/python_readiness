# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "aiohttp>=3.10",
#     "packaging>=24",
# ]
# ///
from __future__ import annotations

import argparse
import asyncio
import collections
import email.message
import email.parser
import email.policy
import enum
import functools
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import pathlib
import re
import subprocess
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


def _cache_key(url: str, **kwargs: Any) -> str:
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
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(connect=15, total=60))
        self.cache_dir = Path(tempfile.gettempdir()) / "python_readiness_cache"
        self.skip_cache = os.environ.get("PYTHON_READINESS_SKIP_CACHE")

    async def get(self, url: str, **kwargs: Any) -> CachedResponse:
        cache_file = self.cache_dir / _cache_key(url, **kwargs, cache_version=1)
        if not self.skip_cache:
            try:
                fetch_time = json.loads((cache_file / "fetch").read_text())
                if fetch_time > time.time() - 3600:
                    await asyncio.sleep(0)
                    status = int((cache_file / "status").read_text())
                    with gzip.open(cache_file / "body.gz", "rb") as f:
                        body = f.read()
                    return CachedResponse(body=body, status=status)
            except (FileNotFoundError, json.JSONDecodeError):
                pass

        retries = 3
        for i in range(retries):
            async with self.session.get(url, **kwargs) as resp:
                if i == retries - 1 or not (500 <= resp.status < 600):
                    ret = CachedResponse(body=await resp.read(), status=resp.status)
                    break
            await asyncio.sleep(0.1)

        cache_file.mkdir(parents=True, exist_ok=True)
        (cache_file / "fetch").write_text(json.dumps(time.time()))
        (cache_file / "status").write_text(str(ret.status))
        with gzip.open(cache_file / "body.gz", "wb") as f:
            f.write(ret.body)
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


def previous_minor_python_version(python_version: tuple[int, int]) -> tuple[int, int]:
    major, minor = python_version
    if minor == 0:
        raise ValueError("No previous minor version")
    return major, minor - 1


# ==============================
# tag munging
# ==============================


@functools.cache
def interpreter_value(python_version: tuple[int, int]) -> str:
    nodot = "".join(map(str, python_version))
    assert sys.implementation.name == "cpython"
    return f"cp{nodot}"


@functools.cache
def _valid_interpreter_abi_set(python_version: tuple[int, int]) -> set[tuple[str, str]]:
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
    return (tag.interpreter, tag.abi) in _valid_interpreter_abi_set(python_version)


# ==============================
# determining support
# ==============================


class PythonSupport(enum.IntEnum):
    unsupported = 0
    totally_unknown = 1
    has_viable_wheel = 2
    has_explicit_wheel = 3
    has_classifier = 4
    has_classifier_and_explicit_wheel = 5


async def metadata_from_wheel(
    session: CachedSession, wheel: dict[str, Any]
) -> email.message.Message:
    if wheel.get("core-metadata"):
        url = wheel["url"] + ".metadata"
        resp = await session.get(url)
        resp.raise_for_status()
        content = io.BytesIO(resp.body)
        parser = email.parser.BytesParser(policy=email.policy.compat32)
        metadata = parser.parse(content)
        return metadata

    url = wheel["url"]
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
    return metadata


def support_from_wheel_tags_helper(
    wheels: list[dict[str, Any]], python_version: tuple[int, int]
) -> tuple[dict[str, Any] | None, PythonSupport]:
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
    return best_wheel, support


async def support_from_files(
    session: CachedSession, files: list[dict[str, Any]], python_version: tuple[int, int]
) -> tuple[PythonSupport, dict[str, Any] | None]:

    wheels = []
    for file in files:
        if not file["filename"].endswith(".whl"):
            continue
        wheels.append(file)

    if not wheels:
        # We could check if the sdist's PKG-INFO has a classifier, but anyone shipping only sdists
        # probably doesn't care enough about packaging to add classifiers.
        return PythonSupport.totally_unknown, None

    best_wheel, support = support_from_wheel_tags_helper(wheels, python_version)
    assert support <= PythonSupport.has_explicit_wheel
    if support == PythonSupport.unsupported:
        # We have no wheels that work for this version (and we know there are other wheels)
        # In theory, the sdist could declare a classifier, but that's going to suck compared
        # to a wheel and we know this package has wheels.
        return PythonSupport.unsupported, None

    assert support >= PythonSupport.has_viable_wheel

    unsupported_without_classifier = False
    if support == PythonSupport.has_viable_wheel:
        # If we have a viable wheel, check if we have an explicit wheel for the previous Python
        # minor version. If we do, we're only supported if we have a classifier.
        # This results in better behaviour for cases like mypyc-compiled wheels, where yes, there
        # is a pure Python wheel, but upstream probably hasn't tested on the new Python and also
        # you're probably running much slower without the compiled extension.
        _, prev_support = support_from_wheel_tags_helper(
            wheels, previous_minor_python_version(python_version)
        )
        if prev_support == PythonSupport.has_explicit_wheel:
            unsupported_without_classifier = True

    # We have wheels that are at least viable for this version — time to check the classifiers!
    assert best_wheel is not None
    metadata = await metadata_from_wheel(session, best_wheel)

    classifiers = set(metadata.get_all("Classifier", []))
    python_version_str = ".".join(str(v) for v in python_version)
    if f"Programming Language :: Python :: {python_version_str}" in classifiers:
        # It's worth distinguishing these two cases, because it's often much more urgent to
        # upgrade dependencies that have an explicit wheel. For pure Python packages that just
        # bump classifiers, sure, it's great to know upstream is testing against newer Python, but
        # old versions will usually work fine.
        if support == PythonSupport.has_explicit_wheel:
            return PythonSupport.has_classifier_and_explicit_wheel, best_wheel
        elif support == PythonSupport.has_viable_wheel:
            # Interestingly, it's possible for unsupported_without_classifier to be True here
            # charset_normalizer has a release where they only have pure Python wheels for 3.12
            # but they do have the classifier. We just trust the upstream.
            # If a later release ships an explicit wheel, that will mark a higher level of support.
            return PythonSupport.has_classifier, best_wheel
        else:
            raise AssertionError

    if unsupported_without_classifier:
        return PythonSupport.unsupported, None
    return support, best_wheel


async def dist_support(
    session: CachedSession, name: str, python_version: tuple[int, int]
) -> tuple[Version | None, PythonSupport, dict[str, Any] | None]:
    headers = {"Accept": "application/vnd.pypi.simple.v1+json"}

    resp = await session.get(f"https://pypi.org/simple/{name}/", headers=headers)
    if resp.status == 404:
        return None, PythonSupport.totally_unknown, None
    resp.raise_for_status()
    data = resp.json()

    version_files = collections.defaultdict[Version, list[dict[str, Any]]](list)
    for file in data["files"]:
        if file.get("yanked"):
            continue
        if file["filename"].endswith(".whl"):
            _, version, _, _ = packaging.utils.parse_wheel_filename(file["filename"])
        elif file["filename"].endswith(("tar.gz", ".zip")):
            try:
                _, version = packaging.utils.parse_sdist_filename(file["filename"])
            except packaging.utils.InvalidSdistFilename:
                continue
        else:
            # There's a bunch of things we don't care about
            # .egg .exe .tar.bz2 .msi .tgz .rpm .ZIP
            continue
        version_files[version].append(file)

    all_versions = sorted((safe_version(v) for v in data["versions"]), reverse=True)
    # Note we check version_files[v] to filter out yanked releases (that would otherwise be treated
    # as totally unknown)
    all_versions = [v for v in all_versions if not v.is_prerelease and version_files[v]]
    if not all_versions:
        return None, PythonSupport.totally_unknown, None
    latest_version = all_versions[0]

    support, best_file = await support_from_files(
        session, version_files[latest_version], python_version
    )
    if support <= PythonSupport.has_viable_wheel:
        return None, support, None

    left = 0
    right = len(all_versions)
    while left < right:
        left_release = all_versions[left].release
        right_release = all_versions[right - 1].release
        left_release = left_release + (0,) * (len(right_release) - len(left_release))
        right_release = right_release + (0,) * (len(left_release) - len(right_release))
        # Say left_release = (2, 8, 4) and right_release = (2, 1, 1)
        # We'll want to test the largest version < (2, 5, 0)
        if left_release != right_release:
            shared_prefix_len = next(
                i for i, (a, b) in enumerate(zip(left_release, right_release)) if a != b
            )
            mid_release = left_release[:shared_prefix_len] + (
                (left_release[shared_prefix_len] + right_release[shared_prefix_len] + 1) // 2,
            )
            assert left_release >= mid_release >= right_release
            mid = left + next(
                i for i, v in enumerate(all_versions[left:right]) if v.release < mid_release
            )
        else:
            mid = (left + right) // 2

        version_support, version_best_file = await support_from_files(
            session, version_files[all_versions[mid]], python_version
        )
        if version_support < support:
            right = mid
        else:
            left = mid + 1
            best_file = version_best_file

    return all_versions[right - 1], support, best_file


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
            # We might drag in a few unnecessary specifiers or markers, but that's the cost
            # of combining requirements
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


def requirements_from_current_environment() -> list[Requirement]:
    purelib = sysconfig_purelib()
    venv_versions = {}
    for dist in importlib.metadata.distributions():
        if (
            isinstance(dist, importlib.metadata.PathDistribution)
            and (dist_path := getattr(dist, "_path", None))
            and isinstance(dist_path, pathlib.Path)
            and not dist_path.is_relative_to(purelib)
        ):
            continue
        metadata = dist.metadata
        venv_versions[metadata["Name"]] = Version(metadata["Version"]).base_version
    return [Requirement(f"{name}>={version}") for name, version in venv_versions.items()]


def requirements_from_ext_environment(env_path: str) -> list[Requirement]:
    python_exe = r"Scripts\python.exe" if sys.platform == "win32" else "bin/python"
    python_path = str(Path(env_path) / python_exe)

    code = """
import importlib.metadata
import json
import sysconfig

from pathlib import Path

purelib = Path(sysconfig.get_paths()["purelib"])
venv_versions = {}
for dist in importlib.metadata.distributions():
    if (
        isinstance(dist, importlib.metadata.PathDistribution)
        and (dist_path := getattr(dist, '_path', None))
        and isinstance(dist_path, Path)
        and not dist_path.is_relative_to(purelib)
    ):
        continue
    metadata = dist.metadata
    venv_versions[metadata["Name"]] = metadata["Version"]

print(json.dumps(venv_versions))
"""

    try:
        result = subprocess.run(
            [python_path, "-"], input=code.encode(), capture_output=True, check=False
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not find Python environment at {env_path}") from e

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to read environment data from {env_path}. Error:\n"
            f"{result.stderr.decode()}"
        )

    venv_versions = json.loads(result.stdout)
    venv_versions = {name: Version(version).base_version for name, version in venv_versions.items()}
    return [Requirement(f"{name}>={version}") for name, version in venv_versions.items()]


# ==============================
# main
# ==============================


async def python_readiness(
    packages: list[Requirement],
    *,
    python_version: tuple[int, int] | None,
    req_files: list[str],
    ignore_existing_requirements: bool,
    envs: list[str] | None = None,
) -> str:

    for req_file in req_files:
        packages.extend(Requirement(req) for req in parse_requirements_txt(req_file))

    if envs:
        for env in envs:
            packages.extend(requirements_from_ext_environment(env))

    if not packages:
        # Default to pulling "requirements" from the current environment
        packages = requirements_from_current_environment()
    if ignore_existing_requirements:
        packages = [Requirement(r.name) for r in packages]
    packages = deduplicate_reqs(packages)

    session = CachedSession()

    if python_version is None:
        python_version = await latest_python_release(session)
    assert len(python_version) == 2

    tasks = [asyncio.create_task(dist_support(session, p.name, python_version)) for p in packages]
    pending = set(tasks)
    while pending:
        _done, pending = await asyncio.wait(pending, timeout=1)
        if pending:
            print(
                f"Determined support for {len(tasks) - len(pending)}/{len(tasks)} packages...",
                file=sys.stderr,
            )

    out = []
    package_support = {p: await t for p, t in zip(packages, tasks)}
    for previous_req, (version, support, file_proof) in sorted(
        package_support.items(), key=lambda x: (-x[1][1].value, x[0].name)
    ):
        assert (version is None) == (support <= PythonSupport.has_viable_wheel)
        assert (file_proof is None) == (support <= PythonSupport.has_viable_wheel)
        # file_proof["upload-time"] is interesting data

        package = previous_req.name
        if version is None:
            new_req = Requirement(package)
        else:
            new_req = Requirement(f"{package}>={version}")

        PAD = 40
        previous_req_min = approx_min_satisfying_version(previous_req)
        if previous_req_min in new_req.specifier:
            if support > PythonSupport.has_viable_wheel:
                out.append(
                    f"{str(previous_req):<{PAD}}  # {support.name} (existing requirement ensures support)"
                )
            elif support >= PythonSupport.has_viable_wheel:
                assert support == PythonSupport.has_viable_wheel
                out.append(f"{str(previous_req):<{PAD}}  # {support.name} (cannot ensure support)")
            else:
                out.append(f"{str(previous_req):<{PAD}}  # {support.name}")
        elif previous_req.specifier:
            out.append(f"{str(new_req):<{PAD}}  # {support.name} (previously: {str(previous_req)})")
        else:
            out.append(f"{str(new_req):<{PAD}}  # {support.name}")

    await session.close()
    return "\n".join(out)


def main() -> None:
    assert sys.version_info >= (3, 9)

    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=None)
    parser.add_argument(
        "-e", "--env", action="append", default=[], help="Path to a virtual environment"
    )
    parser.add_argument("-p", "--package", action="append", default=[])
    parser.add_argument("-r", "--requirement", action="append", default=[])
    parser.add_argument("--ignore-existing-requirements", action="store_true")
    args = parser.parse_args()

    python_version: tuple[int, int] | None = None
    if args.python is not None:
        python_version = tuple(map(int, args.python.split(".")))  # type: ignore
        assert isinstance(python_version, tuple)
        if len(python_version) != 2:
            parser.error("Python version must be a major and minor version")

    for package in args.package:
        if re.fullmatch(r"(python)?[23]\.\d{1,2}", package):
            parser.error(f"Did you mean to use '--python {package}'? (Use -p to specify a package)")

    out = asyncio.run(
        python_readiness(
            [Requirement(p) for p in args.package],
            python_version=python_version,
            req_files=args.requirement,
            ignore_existing_requirements=args.ignore_existing_requirements,
            envs=args.env,
        )
    )
    print(out)


if __name__ == "__main__":
    main()
