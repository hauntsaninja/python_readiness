import asyncio
import functools

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
    safe_version,
    support_from_wheel_tags_helper,
    tag_viable_for_python,
    requirements_from_environment,
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


def test_parse_requirements_txt(tmp_path) -> None:
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


def test_requirements_from_environment() -> None:
    reqs = {canonical_name(r.name): r for r in requirements_from_environment()}
    assert reqs["aiohttp"]
    assert Version("3.9") not in reqs["aiohttp"].specifier
    assert Version("9999") in reqs["aiohttp"].specifier

    assert reqs["pytest"]
    assert Version("5") not in reqs["pytest"].specifier
    assert Version("9999") in reqs["pytest"].specifier


def we_have_pytest_asyncio_at_home(fn):
    @functools.wraps(fn)
    def wrapper():
        asyncio.run(fn())

    return wrapper


@we_have_pytest_asyncio_at_home
async def test_python_readiness() -> None:
    session = CachedSession()

    version, support = await dist_support(session, "mypy", (3, 11))
    assert version == Version("0.990")
    assert support == PythonSupport.has_classifier_and_explicit_wheel

    version, support = await dist_support(session, "mypy", (3, 12))
    assert version == Version("1.10.0")
    assert support == PythonSupport.has_classifier_and_explicit_wheel

    # this will eventually fail, but packaging.tags gets really slow for
    # implausibly high minor versions
    version, support = await dist_support(session, "mypy", (3, 20))
    assert version is None
    assert support == PythonSupport.has_viable_wheel

    version, support = await dist_support(session, "mypy", (4, 0))
    assert version is None
    assert support == PythonSupport.unsupported

    version, support = await dist_support(session, "typing-extensions", (3, 11))
    assert version == Version("4.5.0")
    assert support == PythonSupport.has_classifier

    version, support = await dist_support(session, "typing-extensions", (3, 12))
    assert version == Version("4.7.0")
    assert support == PythonSupport.has_classifier

    version, support = await dist_support(session, "typing-extensions", (3, 20))
    assert version is None
    assert support == PythonSupport.has_viable_wheel

    version, support = await dist_support(session, "typing-extensions", (4, 0))
    assert version is None
    assert support == PythonSupport.unsupported

    version, support = await dist_support(session, "charset-normalizer", (3, 12))
    assert version == Version("3.3.0")
    assert support == PythonSupport.has_classifier_and_explicit_wheel

    version, support = await dist_support(session, "ansiconv", (3, 10))
    assert version is None
    assert support == PythonSupport.totally_unknown

    await session.close()


@we_have_pytest_asyncio_at_home
async def test_latest_python_release():
    session = CachedSession()
    assert await latest_python_release(session) >= (3, 12)
    await session.close()
