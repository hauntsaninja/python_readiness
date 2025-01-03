import asyncio
import functools
import sys
from pathlib import Path
from typing import Any, Callable, Coroutine

import pytest
from packaging.requirements import Requirement
from packaging.tags import Tag
from packaging.version import Version

from python_readiness import (
    CachedSession,
    PythonSupport,
    approx_min_satisfying_version,
    canonical_name,
    deduplicate_reqs,
    dist_support,
    interpreter_value,
    latest_python_release,
    parse_requirements_txt,
    previous_minor_python_version,
    python_readiness,
    requirements_from_current_environment,
    requirements_from_ext_environment,
    safe_version,
    support_from_wheel_tags_helper,
    tag_viable_for_python,
)


def test_previous_minor_python_version() -> None:
    assert previous_minor_python_version((3, 9)) == (3, 8)
    with pytest.raises(ValueError):
        previous_minor_python_version((3, 0))


def test_interpreter_value() -> None:
    assert interpreter_value((3, 9)) == "cp39"
    assert interpreter_value((3, 10)) == "cp310"
    assert interpreter_value((3, 11)) == "cp311"


def test_tag_viable_for_python() -> None:
    tag = Tag("cp39", "cp39", "manylinux1_x86_64")
    assert tag_viable_for_python(tag, (3, 9)) is True
    assert tag_viable_for_python(tag, (3, 10)) is False

    tag = Tag("cp38", "cp38", "manylinux1_x86_64")
    assert tag_viable_for_python(tag, (3, 8)) is True
    assert tag_viable_for_python(tag, (3, 9)) is False
    assert tag_viable_for_python(tag, (3, 10)) is False

    tag = Tag("py3", "none", "any")
    assert tag_viable_for_python(tag, (3, 9)) is True
    assert tag_viable_for_python(tag, (3, 10)) is True

    tag = Tag("cp39", "abi3", "manylinux1_x86_64")
    assert tag_viable_for_python(tag, (3, 8)) is False
    assert tag_viable_for_python(tag, (3, 9)) is True
    assert tag_viable_for_python(tag, (3, 10)) is True

    tag = Tag("cp39", "none", "any")
    assert tag_viable_for_python(tag, (3, 8)) is False
    assert tag_viable_for_python(tag, (3, 9)) is True
    assert tag_viable_for_python(tag, (3, 10)) is False

    tag = Tag("cp310", "cp310", "manylinux1_x86_64")
    assert tag_viable_for_python(tag, (3, 9)) is False
    assert tag_viable_for_python(tag, (3, 10)) is True

    tag = Tag("cp27", "cp27mu", "manylinux1_x86_64")
    assert tag_viable_for_python(tag, (2, 7)) is True
    assert tag_viable_for_python(tag, (3, 9)) is False
    assert tag_viable_for_python(tag, (3, 10)) is False


def test_support_from_wheel_tags_helper() -> None:
    wheels = [
        {"filename": "testpkg-1.0.0-cp310-cp310-manylinux_2_17_x86_64.whl"},
        {"filename": "testpkg-1.0.0-py3-none-any.whl"},
        {"filename": "testpkg-1.0.0-cp39-cp39-win_amd64.whl"},
    ]

    assert support_from_wheel_tags_helper(wheels, (3, 11))[1] == PythonSupport.has_viable_wheel
    assert support_from_wheel_tags_helper(wheels, (3, 10))[1] == PythonSupport.has_explicit_wheel
    assert support_from_wheel_tags_helper(wheels, (3, 9))[1] == PythonSupport.has_explicit_wheel
    assert support_from_wheel_tags_helper(wheels, (3, 8))[1] == PythonSupport.has_viable_wheel
    assert support_from_wheel_tags_helper(wheels, (2, 7))[1] == PythonSupport.unsupported


def test_safe_version() -> None:
    assert safe_version("1.0.0") == Version("1.0.0")
    assert safe_version("invalid_version") == Version("0")
    assert safe_version("2.0.0b1") == Version("2.0.0b1")


def test_parse_requirements_txt(tmp_path: Path) -> None:
    content = """
    # Comment line
    packageA>=1.0
    packageB==2.0  # Another comment
    packageC<=3.0 --hash=sha256:1234567890abcdefb
    """
    req_file = tmp_path / "requirements.txt"
    req_file.write_text(content)

    reqs = parse_requirements_txt(str(req_file))
    assert reqs == ["packageA>=1.0", "packageB==2.0", "packageC<=3.0"]


def test_approx_min_satisfying_version() -> None:
    assert approx_min_satisfying_version(Requirement("package==1.4.*")) == Version("1.4")
    assert approx_min_satisfying_version(Requirement("package~=1.5.2")) == Version("1.5.2")
    assert approx_min_satisfying_version(Requirement("package!=0.1")) == Version("0")
    assert approx_min_satisfying_version(Requirement("package<=2.0")) == Version("0")
    assert approx_min_satisfying_version(Requirement("package>=1.2.3")) == Version("1.2.3")
    assert approx_min_satisfying_version(Requirement("package>1.2.3")) == Version("1.2.3")
    assert approx_min_satisfying_version(Requirement("package")) == Version("0")

    assert approx_min_satisfying_version(Requirement("package>=0.5,==1")) == Version("1")
    assert approx_min_satisfying_version(Requirement("package>=0.5,<2")) == Version("0.5")


def test_deduplicate_reqs() -> None:
    reqs = [
        Requirement("package1>=1.0"),
        Requirement("package1<2.0"),
        Requirement("package2==3.0"),
        Requirement("package1!=1.5"),
        Requirement("pAckaGe2>=3.0"),
        Requirement("package2[extra]"),
    ]
    deduped = deduplicate_reqs(reqs)
    assert len(deduped) == 2

    for req in deduped:
        if canonical_name(req.name) == "package1":
            assert str(req) == "package1!=1.5,<2.0,>=1.0"
        elif canonical_name(req.name) == "package2":
            assert str(req) == "package2[extra]==3.0,>=3.0"
        else:
            raise AssertionError

    reqs = [
        Requirement("package @ https://example.com/package-1.0.tar.gz"),
        Requirement("package @ https://example.com/package-2.0.tar.gz"),
    ]
    with pytest.raises(RuntimeError):
        deduplicate_reqs(reqs)

    reqs = [
        Requirement("package; python_version<'3.8'"),
        Requirement("package; python_version>'3.6'"),
    ]
    deduped = deduplicate_reqs(reqs)
    assert len(deduped) == 1
    assert str(deduped[0]) == 'package; python_version < "3.8" or python_version > "3.6"'

    reqs = [
        Requirement("package; python_version<'3.8'"),
        Requirement("package>=3; python_version>'3.6'"),
        Requirement("package"),
    ]
    deduped = deduplicate_reqs(reqs)
    assert len(deduped) == 1
    assert str(deduped[0]) == "package>=3"


def test_requirements_from_current_environment() -> None:
    reqs = {canonical_name(r.name): r for r in requirements_from_current_environment()}
    assert reqs["aiohttp"]
    assert Version("3.9") not in reqs["aiohttp"].specifier
    assert Version("9999") in reqs["aiohttp"].specifier

    assert reqs["pytest"]
    assert Version("5") not in reqs["pytest"].specifier
    assert Version("9999") in reqs["pytest"].specifier


def test_requirements_from_ext_environment() -> None:
    # Both methods should give the same result for the active venv
    from_current_env = sorted(requirements_from_current_environment(), key=str)
    from_ext_env = sorted(requirements_from_ext_environment(sys.prefix), key=str)
    assert from_current_env == from_ext_env

    with pytest.raises(RuntimeError):
        _ = requirements_from_ext_environment(str(Path(sys.prefix) / "not_a_venv"))


def we_have_pytest_asyncio_at_home(
    fn: Callable[[], Coroutine[Any, Any, None]]
) -> Callable[[], None]:
    @functools.wraps(fn)
    def wrapper() -> None:
        asyncio.run(fn())

    return wrapper


DEFAULT_EXCLUDE_NEWER = "2024-10-01"


@we_have_pytest_asyncio_at_home
async def test_dist_support() -> None:
    session = CachedSession()

    for monotonic_support in [False, True]:
        get_support = functools.partial(
            dist_support,
            session,
            monotonic_support=monotonic_support,
            exclude_newer=DEFAULT_EXCLUDE_NEWER,
        )

        version, support, file_proof = await get_support("mypy", (3, 11))
        assert version == Version("0.990")
        assert support == PythonSupport.has_classifier_and_explicit_wheel
        assert file_proof is not None
        assert file_proof["filename"] == "mypy-0.990-cp311-cp311-win_amd64.whl"

        version, support, file_proof = await get_support("mypy", (3, 12))
        assert version == Version("1.10.0")
        assert support == PythonSupport.has_classifier_and_explicit_wheel
        assert file_proof is not None
        assert file_proof["filename"] == "mypy-1.10.0-cp312-cp312-win_amd64.whl"

        # This will eventually fail, but packaging.tags gets really slow for
        # implausibly high minor versions
        version, support, file_proof = await get_support("mypy", (3, 20))
        assert version is None
        assert support == PythonSupport.has_viable_wheel
        assert file_proof is None

        version, support, file_proof = await get_support("mypy", (4, 0))
        assert version is None
        assert support == PythonSupport.unsupported
        assert file_proof is None

        version, support, file_proof = await get_support("typing-extensions", (3, 11))
        assert version == Version("4.5.0")
        assert support == PythonSupport.has_classifier
        assert file_proof is not None
        assert file_proof["filename"] == "typing_extensions-4.5.0-py3-none-any.whl"

        version, support, file_proof = await get_support("typing-extensions", (3, 12))
        assert version == Version("4.7.0")
        assert support == PythonSupport.has_classifier
        assert file_proof is not None
        assert file_proof["filename"] == "typing_extensions-4.7.0-py3-none-any.whl"

        version, support, file_proof = await get_support("typing-extensions", (3, 20))
        assert version is None
        assert support == PythonSupport.has_viable_wheel
        assert file_proof is None

        version, support, file_proof = await get_support("typing-extensions", (4, 0))
        assert version is None
        assert support == PythonSupport.unsupported
        assert file_proof is None

        version, support, file_proof = await get_support("charset-normalizer", (3, 12))
        assert version == Version("3.3.0")
        assert support == PythonSupport.has_classifier_and_explicit_wheel
        assert file_proof is not None
        assert file_proof["filename"] == "charset_normalizer-3.3.0-cp312-cp312-win_amd64.whl"

        version, support, file_proof = await get_support("ansiconv", (3, 10))
        assert version is None
        assert support == PythonSupport.totally_unknown
        assert file_proof is None

        version, support, file_proof = await get_support("torch", (3, 11))
        assert version == Version("1.13.0")
        assert support == PythonSupport.has_classifier_and_explicit_wheel
        assert file_proof is not None
        assert file_proof["filename"] == "torch-1.13.0-cp311-cp311-manylinux1_x86_64.whl"

        version, support, file_proof = await get_support("psygnal", (3, 11))
        assert version == Version("0.6.0.post0")
        assert support == PythonSupport.has_classifier_and_explicit_wheel
        assert file_proof is not None
        assert file_proof["filename"] == "psygnal-0.6.0.post0-cp311-cp311-win_amd64.whl"

        version, support, file_proof = await get_support("cryptography", (3, 9))
        assert version == Version("42.0.0")
        assert support == PythonSupport.has_classifier_and_explicit_wheel
        assert file_proof is not None
        assert file_proof["filename"] == "cryptography-42.0.0-cp39-abi3-win_amd64.whl"

        version, support, file_proof = await get_support("cryptography", (3, 10))
        assert version == Version("36.0.0")
        assert support == PythonSupport.has_classifier
        assert file_proof is not None
        assert file_proof["filename"] == "cryptography-36.0.0-cp36-abi3-macosx_10_10_universal2.whl"

    await session.close()


@we_have_pytest_asyncio_at_home
async def test_dist_support_requires_python() -> None:
    session = CachedSession()

    get_support = functools.partial(
        dist_support, session, monotonic_support=False, exclude_newer=DEFAULT_EXCLUDE_NEWER
    )

    version, support, file_proof = await get_support("k8", (3, 11))
    assert version == Version("29.0.0")
    assert support == PythonSupport.is_requires_python_lower_bound
    assert file_proof is not None
    assert file_proof["filename"] == "k8-29.0.0-py3-none-any.whl"

    version, support, file_proof = await get_support("aiopath", (3, 11))
    assert version is None
    assert support == PythonSupport.unsupported
    assert file_proof is None

    version, support, file_proof = await get_support("aiopath", (3, 11), exclude_newer="2023-01")
    assert version is None
    assert support == PythonSupport.has_viable_wheel
    assert file_proof is None

    await session.close()


@we_have_pytest_asyncio_at_home
async def test_dist_support_yanked() -> None:
    session = CachedSession()

    for monotonic_support in [False, True]:
        get_support = functools.partial(
            dist_support,
            session,
            monotonic_support=monotonic_support,
            exclude_newer=DEFAULT_EXCLUDE_NEWER,
        )

        version, support, file_proof = await get_support("memray", (3, 11))
        assert version == Version("1.3.0")
        assert support == PythonSupport.has_classifier_and_explicit_wheel
        assert file_proof is not None
        assert (
            file_proof["filename"]
            == "memray-1.3.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        )

    await session.close()


@we_have_pytest_asyncio_at_home
async def test_dist_support_exclude_newer() -> None:
    session = CachedSession()

    get_support = functools.partial(dist_support, session, monotonic_support=True)

    version, support, file_proof = await get_support("mypy", (3, 12), exclude_newer="2024-04-25")
    assert version == Version("1.10.0")
    assert support == PythonSupport.has_classifier_and_explicit_wheel
    assert file_proof is not None
    assert file_proof["filename"] == "mypy-1.10.0-cp312-cp312-win_amd64.whl"

    version, support, file_proof = await get_support("mypy", (3, 12), exclude_newer="2024-04-24")
    assert version == Version("1.5.1")
    assert support == PythonSupport.has_explicit_wheel
    assert file_proof is not None
    assert file_proof["filename"] == "mypy-1.5.1-cp312-cp312-win_amd64.whl"

    version, support, file_proof = await get_support("mypy", (3, 12), exclude_newer="2023-08-17")
    assert version == Version("1.5.1")
    assert support == PythonSupport.has_explicit_wheel
    assert file_proof is not None
    assert file_proof["filename"] == "mypy-1.5.1-cp312-cp312-win_amd64.whl"

    version, support, file_proof = await get_support("mypy", (3, 12), exclude_newer="2023-08-16")
    assert version is None
    assert support == PythonSupport.unsupported
    assert file_proof is None

    await session.close()


@we_have_pytest_asyncio_at_home
async def test_dist_support_large() -> None:
    # Test cases with lots of candidates where bisection really does matter for performance
    session = CachedSession()

    for monotonic_support in [False, True]:
        get_support = functools.partial(
            dist_support,
            session,
            monotonic_support=monotonic_support,
            exclude_newer=DEFAULT_EXCLUDE_NEWER,
        )

        version, _, _ = await get_support("boto3", (3, 8))
        assert version == Version("1.13.16")
        version, _, _ = await get_support("boto3", (3, 9))
        assert version == Version("1.18.48")
        version, _, _ = await get_support("boto3", (3, 10))
        assert version == Version("1.18.49")
        version, _, _ = await get_support("boto3", (3, 11))
        assert version == Version("1.24.14")
        version, _, _ = await get_support("boto3", (3, 12))
        assert version == Version("1.28.27")

        version, _, _ = await get_support("botocore", (3, 8))
        assert version == Version("1.13.45")
        version, _, _ = await get_support("botocore", (3, 9))
        assert version == Version("1.21.49")
        version, _, _ = await get_support("botocore", (3, 10))
        assert version == Version("1.21.49")
        version, _, _ = await get_support("botocore", (3, 11))
        assert version == Version("1.27.13")
        version, _, _ = await get_support("botocore", (3, 12))
        assert version == Version("1.31.27")

        version, _, _ = await get_support("openai", (3, 8))
        assert version == Version("1.0.0")
        version, _, _ = await get_support("openai", (3, 12))
        assert version == Version("1.0.0")

        version, _, _ = await get_support("hypothesis", (3, 8))
        assert version == Version("4.44.3")
        version, _, _ = await get_support("hypothesis", (3, 12))
        assert version == Version("6.91.0")

    await session.close()


@we_have_pytest_asyncio_at_home
async def test_dist_support_bisection_differences() -> None:
    # Test cases where bisection can identify an earlier version than linear search
    session = CachedSession()

    get_support = functools.partial(dist_support, session, exclude_newer=DEFAULT_EXCLUDE_NEWER)

    version, _, _ = await get_support("sqlalchemy", (3, 12), monotonic_support=False)
    assert version == Version("1.4.51")
    version, _, _ = await get_support("sqlalchemy", (3, 12), monotonic_support=True)
    assert version == Version("2.0.24")

    version, _, _ = await get_support("greenlet", (3, 11), monotonic_support=False)
    assert version == Version("1.1.3")
    version, _, _ = await get_support("greenlet", (3, 11), monotonic_support=True)
    assert version == Version("2.0.0.post0")

    version, _, _ = await get_support("rapidfuzz", (3, 12), monotonic_support=False)
    assert version == Version("2.15.2")
    version, _, _ = await get_support("rapidfuzz", (3, 12), monotonic_support=True)
    assert version == Version("3.3.1")

    version, _, _ = await get_support("pydantic", (3, 12), monotonic_support=False)
    assert version == Version("1.10.17")
    version, _, _ = await get_support("pydantic", (3, 12), monotonic_support=True)
    assert version == Version("2.5.0")

    version, _, _ = await get_support("black", (3, 11), monotonic_support=False)
    assert version == Version("22.10.0")
    version, _, _ = await get_support("black", (3, 11), monotonic_support=True)
    assert version == Version("23.9.1")

    version, _, _ = await get_support("regex", (3, 11), monotonic_support=False)
    assert version == Version("2022.10.31")
    version, _, _ = await get_support("regex", (3, 11), monotonic_support=True)
    assert version == Version("2023.5.2")

    version, _, _ = await get_support("coverage", (3, 11), monotonic_support=False)
    assert version == Version("6.1.2")
    version, _, _ = await get_support("coverage", (3, 11), monotonic_support=True)
    assert version == Version("6.4.4")

    version, _, _ = await get_support("aiofile", (3, 10), monotonic_support=False)
    assert version == Version("3.7.4")
    version, _, _ = await get_support("aiofile", (3, 10), monotonic_support=True)
    assert version == Version("3.8.3")

    version, _, _ = await get_support("pytest", (3, 9), monotonic_support=False)
    assert version == Version("4.6.10")
    version, _, _ = await get_support("pytest", (3, 9), monotonic_support=True)
    assert version == Version("5.4.2")

    await session.close()


@we_have_pytest_asyncio_at_home
async def test_latest_python_release() -> None:
    session = CachedSession()
    assert await latest_python_release(session) >= (3, 12)
    await session.close()


@we_have_pytest_asyncio_at_home
async def test_python_readiness_e2e() -> None:
    readiness = await python_readiness(
        packages=[
            Requirement("aiohttp"),
            Requirement("black>=23"),
            Requirement("mypy>=1"),
            Requirement("typing-extensions>=4"),
            Requirement("blobfile"),
        ],
        monotonic_support=True,
        exclude_newer=DEFAULT_EXCLUDE_NEWER,
        req_files=[],
        python_version=(3, 11),
        ignore_existing_requirements=True,
        envs=[],
    )
    assert (
        readiness
        == """\
aiohttp>=3.9.4                # has_classifier_and_explicit_wheel
black>=23.9.1                 # has_classifier_and_explicit_wheel
mypy>=0.990                   # has_classifier_and_explicit_wheel
typing-extensions>=4.5.0      # has_classifier
blobfile                      # has_viable_wheel (cannot ensure support)"""
    )

    readiness = await python_readiness(
        packages=[
            Requirement("aiohttp"),
            Requirement("black>=23"),
            Requirement("mypy>=1"),
            Requirement("typing-extensions>=4"),
            Requirement("blobfile>=0.1"),
            Requirement("ansiconv>=0.1"),
            Requirement("opentelemetry-util-http>=0.47b0"),
        ],
        monotonic_support=False,
        exclude_newer=DEFAULT_EXCLUDE_NEWER,
        req_files=[],
        python_version=(3, 11),
        ignore_existing_requirements=False,
        envs=[],
    )
    print(readiness)
    assert (
        readiness
        == """\
aiohttp>=3.9.4                            # has_classifier_and_explicit_wheel
black>=23                                 # has_classifier_and_explicit_wheel (existing requirement ensures support)
mypy>=1                                   # has_classifier_and_explicit_wheel (existing requirement ensures support)
opentelemetry-util-http>=0.47b0           # has_classifier (existing requirement ensures support)
typing-extensions>=4.5.0                  # has_classifier (previously: typing-extensions>=4)
blobfile>=0.1                             # has_viable_wheel (cannot ensure support)
ansiconv>=0.1                             # totally_unknown"""
    )
