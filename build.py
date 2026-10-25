# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""The build script: builds the executables of Gyro Doctor for the system it runs on,
at the pins of the manifest ``dist-manifest.txt`` beside the script.

The script derives the name of the system; tests that the Python it runs on and every
package that the build requires are at the versions that the manifest pins; tests that PyAV has the
encoder ``libvpx-vp9`` and the decoders of H.264 and H.265; builds one single file
executable of each command with PyInstaller into ``build/pyinstaller/dist/`` of the
repository; runs ``--help`` and ``--version`` of each executable; and prints the minimum
system of the executables and the path of each. It installs nothing: the package and its
optional dependency group ``build`` are installed at the pins before it runs.

PyAV and packaging are imported inside the functions that use them, so that the script
reports a Python other than the pinned one before it imports either."""

import concurrent.futures
import os
import platform
import re
import subprocess
import sys
import sysconfig
import tomllib
from importlib import metadata
from pathlib import Path
from typing import NamedTuple

# The repository root is the nearest folder above this script that holds
# pyproject.toml.
REPOSITORY = next(
    folder
    for folder in Path(__file__).resolve().parents
    if (folder / "pyproject.toml").is_file()
)
MANIFEST = Path(__file__).resolve().parent / "dist-manifest.txt"
LAUNCHERS = Path(__file__).resolve().parent
BUILD = REPOSITORY / "build"
PYINSTALLER_FOLDER = BUILD / "pyinstaller"

COMMANDS = {
    "gyro-doctor": "gyro_doctor_launcher.py",
    "gyro-doctor-chart": "gyro_doctor_chart_launcher.py",
    "gyro-doctor-render": "gyro_doctor_render_launcher.py",
}
EXECUTABLE_SUFFIX = ".exe" if sys.platform == "win32" else ""

# The modules that each executable leaves out. Every import of them in the bundle catches
# ImportError or runs only on a path the commands never take: the TkAgg backend of
# matplotlib, the line editing of interactive input, the opening of a URL, or a test
# helper of scipy. The test helpers import pytest, and the bundle reaches pygments and
# setuptools only through pytest. Without _hashlib, hashlib uses the hashes built into
# Python, and no executable bundles OpenSSL.
EXCLUDED_MODULES = (
    "tkinter",
    "_tkinter",
    "readline",
    "ssl",
    "certifi",
    "_hashlib",
    "pytest",
    "_pytest",
    "pygments",
    "setuptools",
)

# The name of each system that has a zip, by sys.platform and platform.machine().
SYSTEMS = {
    ("win32", "AMD64"): "windows-x64",
    ("darwin", "arm64"): "macos-arm64",
    ("darwin", "x86_64"): "macos-x64",
    ("linux", "x86_64"): "linux-x64",
}

# Each codec that the product uses, by its kind and its FFmpeg name, and the mode in
# which PyAV looks up a codec of each kind.
REQUIRED_CODECS = (("encoder", "libvpx-vp9"), ("decoder", "h264"), ("decoder", "hevc"))
CODEC_MODES = {"encoder": "w", "decoder": "r"}

# The oldest Windows that Python 3.14 supports, as "Supported Windows versions" of
# https://docs.python.org/3.14/using/windows.html states it.
WINDOWS_MINIMUM = "Windows 10"


class BuildError(Exception):
    """A failure of the build, carrying the WHAT FAILED: WHY part of its error line."""


class Manifest(NamedTuple):
    """The records of the manifest: the pinned Python, and each pinned package as a
    requirement."""

    python: str
    packages: list[str]


def read_manifest() -> Manifest:
    python = None
    packages = []
    for number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kind, _, value = line.partition(" ")
        match kind:
            case "python":
                python = value
            case "package":
                packages.append(value)
            case _:
                raise BuildError(
                    f"cannot read {MANIFEST.name}: unknown record {kind} on line {number}"
                )
    if python is None:
        raise BuildError(f"cannot read {MANIFEST.name}: no python record")
    return Manifest(python, packages)


def package_version() -> str:
    with (REPOSITORY / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]["version"]


def relative(path: Path) -> str:
    return path.relative_to(REPOSITORY).as_posix()


def system_name() -> str:
    key = (sys.platform, platform.machine())
    if key not in SYSTEMS:
        raise BuildError(
            f"cannot build executables: unsupported system {' '.join(key)}"
        )
    return SYSTEMS[key]


def executable_path(command: str) -> Path:
    return PYINSTALLER_FOLDER / "dist" / f"{command}{EXECUTABLE_SUFFIX}"


def test_pins(manifest: Manifest) -> None:
    """Stops on the first package that the build requires, the package with its group
    ``build`` and every package those require, that is not installed at the version that
    the manifest pins."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    required = {
        canonicalize_name(distribution.metadata["Name"])
        for distribution in required_distributions(("build",))
    }
    for text in manifest.packages:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        if canonicalize_name(requirement.name) not in required:
            continue
        (pin,) = requirement.specifier
        installed = metadata.version(requirement.name)
        if installed != pin.version:
            raise BuildError(
                f"cannot build executables: {requirement.name} {installed} installed, "
                f"expected {pin.version}"
            )


def test_pyav_codecs() -> None:
    import av
    from av.codec.codec import UnknownCodecError

    for kind, name in REQUIRED_CODECS:
        try:
            av.Codec(name, CODEC_MODES[kind])
        except UnknownCodecError:
            raise BuildError(
                f"cannot build executables: PyAV {metadata.version('av')} has no "
                f"{kind} {name}"
            ) from None


def build_executable(command: str) -> subprocess.CompletedProcess:
    """Runs PyInstaller on the launcher of ``command``. PyInstaller finds the package in
    the repository, whether the install is editable or not, and copies its metadata, from
    which ``--version`` reads the version."""
    environment = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        # Each build keeps its own PyInstaller cache, so that the three builds can run
        # at once.
        "PYINSTALLER_CONFIG_DIR": str(PYINSTALLER_FOLDER / "cache" / command),
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--onefile",
            "--noupx",
            "--log-level",
            "WARN",
            "--name",
            command,
            "--distpath",
            str(PYINSTALLER_FOLDER / "dist"),
            "--workpath",
            str(PYINSTALLER_FOLDER / "work"),
            "--specpath",
            str(PYINSTALLER_FOLDER / "spec"),
            "--paths",
            str(REPOSITORY),
            "--copy-metadata",
            "gyro_doctor",
            *(
                argument
                for module in EXCLUDED_MODULES
                for argument in ("--exclude-module", module)
            ),
            str(LAUNCHERS / COMMANDS[command]),
        ],
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def build_executables() -> dict[str, Path]:
    """Builds the executable of each command, all at once, and returns the path of each,
    keyed by its command."""
    # The builds share these folders, which PyInstaller creates when they do not exist,
    # so two builds that start together would both create them.
    for name in ("dist", "work", "spec"):
        (PYINSTALLER_FOLDER / name).mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = dict(zip(COMMANDS, executor.map(build_executable, COMMANDS)))
    executables = {}
    for command, result in results.items():
        output = (result.stdout + result.stderr).strip()
        if output:
            print(output, file=sys.stderr)
        path = executable_path(command)
        if result.returncode:
            raise BuildError(
                f"cannot build {path.name}: PyInstaller exit status {result.returncode}"
            )
        executables[command] = path
    return executables


def run_options(executable: Path) -> dict[str, subprocess.CompletedProcess]:
    """The run of ``executable`` with ``--help`` and with ``--version``, keyed by the
    option."""
    return {
        option: subprocess.run(
            [str(executable), option], capture_output=True, text=True, check=False
        )
        for option in ("--help", "--version")
    }


def test_executables(executables: dict[str, Path], version: str) -> None:
    """Runs ``--help`` and ``--version`` of every executable at once, and stops on an exit
    status other than 0 or a version line other than the command and ``version``."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        runs = dict(zip(executables, executor.map(run_options, executables.values())))
    for command, results in runs.items():
        name = executables[command].name
        for option, result in results.items():
            if result.returncode:
                print(result.stderr.strip(), file=sys.stderr)
                raise BuildError(
                    f"cannot build {name}: {option} exit status {result.returncode}"
                )
        printed = results["--version"].stdout.strip()
        expected = f"{command} {version}"
        if printed != expected:
            raise BuildError(
                f"cannot build {name}: --version printed {printed}, expected {expected}"
            )
        print(printed)


def macos_minimum() -> str:
    """The highest of the macOS versions in the platform tags of the installed wheels and
    of the macOS that the Python interpreter was built for."""
    target = sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET")
    versions = [tuple(int(part) for part in target.split("."))]
    for distribution in metadata.distributions():
        for line in (distribution.read_text("WHEEL") or "").splitlines():
            if not line.startswith("Tag:"):
                continue
            platform_tag = line.removeprefix("Tag:").strip().split("-")[2]
            versions += [
                (int(major), int(minor))
                for major, minor in re.findall(r"macosx_(\d+)_(\d+)_", platform_tag)
            ]
    return ".".join(str(part) for part in max(versions))


def minimum_system(system: str) -> str:
    if system == "windows-x64":
        return WINDOWS_MINIMUM
    if system == "linux-x64":
        return " ".join(platform.libc_ver())
    return f"macOS {macos_minimum()}"


def required_distributions(extras: tuple[str, ...] = ()) -> list[metadata.Distribution]:
    """The distributions that gyro_doctor requires, with its optional dependency groups
    ``extras``, and every distribution they require, sorted by name. With no group, they
    are the distributions that the executables bundle."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    found = {}
    reached = set()
    gyro_doctor = metadata.distribution("gyro_doctor")
    pending = [(gyro_doctor, extra) for extra in ("", *extras)]
    while pending:
        distribution, extra = pending.pop()
        for text in distribution.requires or []:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({"extra": extra}):
                continue
            name = canonicalize_name(requirement.name)
            if name not in found:
                try:
                    found[name] = metadata.distribution(requirement.name)
                except metadata.PackageNotFoundError:
                    raise BuildError(
                        f"cannot build executables: {requirement.name} not installed"
                    ) from None
            for requested in ("", *sorted(requirement.extras)):
                if (name, requested) not in reached:
                    reached.add((name, requested))
                    pending.append((found[name], requested))
    return [found[name] for name in sorted(found)]


def build() -> dict[str, Path]:
    """Builds the executables of this system and returns the path of each, keyed by its
    command."""
    manifest = read_manifest()
    system = system_name()
    if platform.python_version() != manifest.python:
        raise BuildError(
            f"cannot build executables: Python {platform.python_version()}, "
            f"expected {manifest.python}"
        )
    test_pins(manifest)
    test_pyav_codecs()
    executables = build_executables()
    test_executables(executables, package_version())
    print(f"minimum system: {minimum_system(system)}")
    return executables


def main() -> int:
    # In a workflow log, stdout is a pipe; line buffering keeps its lines in order with
    # those of stderr and of pip.
    sys.stdout.reconfigure(line_buffering=True)
    try:
        executables = build()
    except BuildError as error:
        print(f"build.py: error: {error}", file=sys.stderr)
        return 1
    for path in executables.values():
        print(relative(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
