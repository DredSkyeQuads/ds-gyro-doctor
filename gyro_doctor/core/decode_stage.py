# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""The decode stage, which derives the decode result from the gyro data track, and the
cache writer of the decode result, with the header test, the file stamps, and the read
of a cache file that the cache of every stage uses.

No quaternion is resampled, smoothed, interpolated, or moved in time. A rate sample is
the rotation between the quaternions that open two consecutive measurement groups,
divided by a time and expressed in degrees per second, and it stands at the midpoint of
the interval it spans. Two consecutive quaternions that hold the same value are found by
exact equality of all four components, never by a tolerance.
"""

import contextlib
import zipfile
import zlib
from pathlib import Path

import av
import numpy as np

from gyro_doctor import errors
from gyro_doctor.core import probe, quaternions, telemetry
from gyro_doctor.core.rates import DEG

# Stays at 1 while the product is in development. Once the product is released, a
# developer increments it whenever the derivation of a cached array changes or a stage
# result gains an array, so that the header test raises an error on every result
# written by an earlier derivation.
DERIVATION_VERSION = 1

# What ``np.load`` and ``json.loads`` raise on a cache file that is truncated, empty, or
# overwritten: a zip archive without its directory, a compressed member that does not
# decompress, a file with no data, or bytes that are neither an array nor JSON.
DAMAGED_CACHE_FILE_ERRORS = (ValueError, EOFError, zipfile.BadZipFile, zlib.error)


@contextlib.contextmanager
def cache_file_read(path):
    """Raise ``errors.CannotReadError`` in place of an error of the operating system,
    with the code ``SYSTEM_ERROR``, and in place of each error of
    ``DAMAGED_CACHE_FILE_ERRORS``, with the code ``CACHE_FILE_INVALID``, that the read
    of the cache file at ``path`` inside the block raises.

    The block holds the read of the file and of the members it needs, never a test, so
    that an error a defect raises in a test stays a traceback."""
    with errors.operating_system_error_as(errors.CannotReadError, path):
        try:
            yield
        except DAMAGED_CACHE_FILE_ERRORS as error:
            raise errors.CannotReadError(
                errors.ErrorCode.CACHE_FILE_INVALID, file=path, reason=str(error)
            ) from error


# The period test: T equals the reciprocal of the counted measurement rate within
# 10 nanoseconds.
PERIOD_TEST_TOLERANCE_S = 1e-8

# The handler name of the gyro data track states the grouping pattern, and nothing else
# in the container does. "CAM meta" marks an Osmo Action 6, which stores each
# measurement across three quaternions; "DJI meta" marks an air unit, which stores each
# measurement in one quaternion or in two identical ones.
OSMO_ACTION_6_HANDLER_NAME = "CAM meta"
OSMO_ACTION_6_QUATERNIONS_PER_GROUP = 3


def time_order(quaternion_times_s):
    """The order that sorts by time the quaternions given in file order. The sort is
    stable, so of a time collision the quaternion first in file order stays first."""
    return np.argsort(quaternion_times_s, kind="stable")


def kept_after_time_collisions(times):
    """Per quaternion: True where its time is later than the time before it. The later
    quaternion of a time collision is excluded from the derivation of the rates, since
    an elapsed time of zero cannot divide a rotation."""
    return np.concatenate([[True], np.diff(times) > 0])


def repeats_previous_quaternion(quaternions_wxyz):
    """Per consecutive pair: True where the later quaternion equals the one before it on
    all four components. Exact equality, never a tolerance."""
    return np.all(quaternions_wxyz[1:] == quaternions_wxyz[:-1], axis=1)


def interval_rates(times, quaternions_wxyz):
    """(midpoint times, deg/s rates) between consecutive entries of the given
    sequence, each rotation divided by its own elapsed time and placed at the midpoint
    of the interval it spans."""
    rotations = quaternions.to_rotations(quaternions_wxyz)
    elapsed = np.diff(times)
    rates = (rotations[:-1].inv() * rotations[1:]).as_rotvec() / elapsed[:, None] * DEG
    return 0.5 * (times[:-1] + times[1:]), rates


def consecutive_quaternion_rates(times, quaternions_wxyz):
    """The rate over each interval between two consecutive stored quaternions. Where
    the two quaternions are identical the rotation is zero, so the rate is exactly
    0 deg/s, as the camera stored it."""
    return interval_rates(times, quaternions_wxyz)


def air_unit_group_opening_mask(quaternions_wxyz):
    """Per quaternion of an air unit track: True at the first quaternion of each run of
    identical consecutive quaternions.

    Where the camera stores a measurement in two identical quaternions, the first of the
    pair opens the measurement group, and where it stores each measurement in one
    quaternion, every quaternion opens one."""
    return np.concatenate([[True], ~repeats_previous_quaternion(quaternions_wxyz)])


def osmo_action_6_group_opening_index(times, owning_telemetry_frame):
    """The index of the quaternion that opens each measurement group of an Osmo Action
    6 track, which stores each measurement across three quaternions.

    The quaternions are ordered first by owning telemetry frame and then by time, never
    by time alone. Where the first quaternion of a telemetry frame falls in time inside
    the block of the previous telemetry frame, three consecutive quaternions in time
    order can belong to two telemetry frames and span far less than one measurement,
    which puts a dip and a spike into the rate where no motion occurred.

    The one or two leftover quaternions at the end of the track are left out of the last
    group. A widened final group moves the mean interval between group boundaries away
    from the reciprocal of the counted measurement rate, and the period test raises an
    error on the track.

    Raises an error on a track with fewer quaternions than two whole groups and the
    quaternion that closes the second."""
    order = np.lexsort((times, owning_telemetry_frame))
    group_count = (order.size - 1) // OSMO_ACTION_6_QUATERNIONS_PER_GROUP
    if group_count < 2:
        raise errors.CannotRepairError(
            errors.ErrorCode.ORIENTATIONS_FEWER_THAN_TWO_GROUPS,
            count=int(order.size),
            expected=2 * OSMO_ACTION_6_QUATERNIONS_PER_GROUP + 1,
        )
    last = group_count * OSMO_ACTION_6_QUATERNIONS_PER_GROUP
    return order[np.arange(0, last + 1, OSMO_ACTION_6_QUATERNIONS_PER_GROUP)]


def group_boundary_times(result):
    """The time of the quaternion that opens each measurement group of a decode result.

    Rate sample k spans the interval from ``boundaries[k]`` to ``boundaries[k+1]``.
    """
    kept_times = np.asarray(result["quaternion_times_s"], dtype=np.float64)[
        np.asarray(result["kept_after_time_collisions"])
    ]
    return kept_times[np.asarray(result["group_opening_quaternion_index"])]


def derive_period(boundaries, result):
    """T, the one period the Normalized Rate divides every rotation by, in seconds.

        T = (last group boundary time - first group boundary time) / N

    for N rate samples, so N intervals at spacing T span exactly the elapsed time of the
    video. The median interval and the most common interval both fail to span it.

    The period test asserts T against the reciprocal of the counted measurement rate,
    which spans the rate samples from the first to the last, and raises an error when the
    two differ by more than 10 nanoseconds.
    """
    rate_sample_count = boundaries.size - 1
    period = float((boundaries[-1] - boundaries[0]) / rate_sample_count)
    counted = 1.0 / float(result["measurement_rate_hz"])
    if abs(period - counted) > PERIOD_TEST_TOLERANCE_S:
        raise errors.CannotRepairError(
            errors.ErrorCode.PERIOD_TEST_FAILED,
            period_us=period * 1e6,
            counted_us=counted * 1e6,
            tolerance_ns=PERIOD_TEST_TOLERANCE_S * 1e9,
        )
    return period


def normalized_rate(recorded_rate, boundaries, period):
    """The Normalized Rate from the recorded rate.

    The recorded rate is ``rotation[k] / interval[k]``, so multiplying by
    ``interval[k] / T`` gives ``rotation[k] / T``: every rotation divided by the one
    period T, with every rate sample keeping its own time and its own place.
    """
    return recorded_rate * (np.diff(boundaries) / period)[:, None]


def counted_rate_hz(times):
    """The counted rate of a series: the number of intervals between its first and its
    last sample divided by the time those samples span.

    Counted over the real time span, never ``1 / median(interval)``: where the
    intervals take two values, the median takes the more common one and understates the
    rate."""
    return float(len(times) - 1) / float(times[-1] - times[0])


def video_identity(video):
    """The identity a decode result carries in its header: the file stem, size in
    bytes, and modification time of its video, and the derivation version."""
    stat = Path(video).stat()
    return {
        "video_name": Path(video).stem,
        "video_size_bytes": int(stat.st_size),
        "video_mtime_s": float(stat.st_mtime),
        "derivation_version": int(DERIVATION_VERSION),
    }


def header_test(result, video, path):
    """The header test: raise an error on the cached result at ``path`` when its
    derivation version differs from the current one, and otherwise when any header field
    differs from the video on disk, the error carrying every header field that
    differs."""
    expected = video_identity(video)
    stored = {key: result[key].item() if key in result else None for key in expected}
    if stored["derivation_version"] != expected["derivation_version"]:
        raise errors.CannotReadError(
            errors.ErrorCode.CACHE_VERSION_DIFFERS,
            file=path,
            stored_version=stored["derivation_version"],
            expected_version=expected["derivation_version"],
        )
    differs = [
        key for key, expected_value in expected.items() if stored[key] != expected_value
    ]
    if differs:
        raise errors.CannotReadError(
            errors.ErrorCode.VIDEO_CHANGED_SINCE_RESULT_CACHED,
            file=path,
            video=Path(video).name,
            fields=differs,
        )


def file_stamp(path):
    """The file stamp of the file at ``path``: its name, size in bytes, and modification
    time in nanoseconds.

    Held as separate fields so that an error on a file that disagrees with its file
    stamp can state which field disagrees.
    """
    path = Path(path)
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def file_stamp_test(directory, stamp):
    """The path of the file that ``stamp`` names in ``directory``, after the test that
    the file has the size and modification time the stamp recorded.

    A file that exists under the stamped name can still be a different file, written by
    another run, so the test raises an error on a missing file, and on a file whose size
    or modification time differs, carrying both sides of the difference. The error
    carries no ``file``: the caller adds the cached result whose file stamp it tested.
    """
    path = Path(directory) / stamp["name"]
    if not path.exists():
        raise errors.CannotReadError(
            errors.ErrorCode.STAMPED_FILE_NOT_FOUND, stamped=path
        )
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (stamp["size"], stamp["mtime_ns"]):
        raise errors.CannotReadError(
            errors.ErrorCode.STAMPED_FILE_CHANGED,
            stamped=path,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            stamp_size=stamp["size"],
            stamp_mtime_ns=stamp["mtime_ns"],
        )
    return path


def file_stamps_test(path, directory, stamps, keys):
    """The file stamp test in ``directory`` of each file stamp in ``stamps`` under the
    keys ``keys``, the file stamps of the cached result at ``path``. An error of a file
    stamp test carries ``path`` as its ``file``."""
    try:
        for key in keys:
            file_stamp_test(directory, stamps[key])
    except errors.CannotReadError as error:
        error.values.setdefault("file", path)
        raise


def every_part_present_test(result, path, parts):
    """Raise an error carrying every part of ``parts`` that ``result``, read from the
    cached result at ``path``, lacks. ``result`` is anything that answers ``in`` by part
    name."""
    missing = [name for name in parts if name not in result]
    if missing:
        raise errors.CannotReadError(
            errors.ErrorCode.CACHED_RESULT_PARTS_MISSING,
            file=path,
            count=len(missing),
            parts=missing,
        )


def derive_decode_result(
    quaternion_times_s,
    quaternions_wxyz,
    video_frame_rate,
    identity,
    quaternion_count_per_telemetry_frame,
    owning_telemetry_frame,
    *,
    handler_name,
    group_opening_quaternion_index=None,
):
    """Every part of the decode result, from quaternions already ordered by time.

    ``quaternion_times_s`` and ``quaternions_wxyz`` hold every quaternion and its time
    in time order. ``owning_telemetry_frame`` holds the telemetry frame that owns each
    quaternion, in the same order. A rebuilt track keeps the times, the counts, and the
    owners of the video it was rebuilt from.

    ``group_opening_quaternion_index`` holds the index of the quaternion that opens
    each measurement group. For a rebuilt track the caller passes the measurement groups
    of the video since a repair changes no measurement group. A rebuilt track with no
    gyro motion repeats one orientation at every quaternion. The air unit rule would
    find one measurement group in such a track.

    Without ``group_opening_quaternion_index``, ``handler_name`` selects the grouping
    rule that finds those indexes. The recorded rate, the period T, and the Normalized
    Rate follow from the indexes by one path for every grouping pattern.

    Raises the errors of ``osmo_action_6_group_opening_index`` and
    ``derive_period``."""
    kept = kept_after_time_collisions(quaternion_times_s)
    times, kept_quaternions = quaternion_times_s[kept], quaternions_wxyz[kept]
    consecutive_times, consecutive_rates = consecutive_quaternion_rates(
        times, kept_quaternions
    )
    if group_opening_quaternion_index is not None:
        opening = np.asarray(group_opening_quaternion_index, dtype=int)
    elif handler_name == OSMO_ACTION_6_HANDLER_NAME:
        opening = osmo_action_6_group_opening_index(times, owning_telemetry_frame[kept])
    else:
        opening = np.flatnonzero(air_unit_group_opening_mask(kept_quaternions))
    recorded_times, recorded_rate = interval_rates(
        times[opening], kept_quaternions[opening]
    )
    quaternion_rate = counted_rate_hz(consecutive_times)
    result = dict(
        **identity,
        handler_name=handler_name,
        video_frame_rate=video_frame_rate,
        quaternion_rate_hz=quaternion_rate,
        measurement_rate_hz=counted_rate_hz(recorded_times),
        quaternions_per_video_frame=quaternion_rate / video_frame_rate,
        quaternion_times_s=quaternion_times_s,
        quaternions_wxyz=quaternions_wxyz,
        kept_after_time_collisions=kept,
        consecutive_quaternion_rate_times_s=consecutive_times,
        consecutive_quaternion_rate_deg_s=consecutive_rates,
        recorded_rate_times_s=recorded_times,
        recorded_rate_deg_s=recorded_rate,
        group_opening_quaternion_index=opening,
        quaternion_count_per_telemetry_frame=quaternion_count_per_telemetry_frame,
        owning_telemetry_frame=owning_telemetry_frame,
    )
    boundaries = group_boundary_times(result)
    period = derive_period(boundaries, result)
    result["period_s"] = period
    result["normalized_rate_deg_s"] = normalized_rate(recorded_rate, boundaries, period)
    return result


def decode(video, *, force):
    """The decode stage: the decode result of the video at path ``video``.

    Reads the gyro data track, orders the quaternions by time, and keeps the telemetry
    frame that owns each quaternion. The video frame rate comes from the container
    through PyAV. Raises an error on a container that reports no video frame rate, and
    on a video whose stored camera model name ends in ``"-Dr."`` unless ``force`` is
    set. Every error it raises carries the path of the video as ``file``."""
    video = Path(video)
    try:
        track = telemetry.read_gyro_data_track(video, force=force)
        order = time_order(track.quaternion_times_s)
        counts = track.quaternion_count_per_telemetry_frame
        owning_telemetry_frame = np.repeat(
            np.arange(counts.size, dtype=np.int32), counts
        )
        try:
            video_frame_rate = probe.video_frame_rate(video)
        except av.FFmpegError as error:
            raise errors.CannotReadError(
                errors.ErrorCode.VIDEO_INVALID, file=video, reason=str(error)
            ) from error
        if video_frame_rate is None:
            raise errors.CannotReadError(
                errors.ErrorCode.VIDEO_FRAME_RATE_MISSING, file=video
            )
        return derive_decode_result(
            track.quaternion_times_s[order],
            track.quaternions_wxyz[order],
            video_frame_rate,
            video_identity(video),
            counts,
            owning_telemetry_frame[order],
            handler_name=track.handler_name,
        )
    except errors.GyroDoctorError as error:
        error.values.setdefault("file", video)
        raise


# Every part of the decode result other than the identity in the header, which the
# header test compares field by field.
DECODE_RESULT_PARTS = (
    "handler_name",
    "video_frame_rate",
    "quaternion_rate_hz",
    "measurement_rate_hz",
    "quaternions_per_video_frame",
    "quaternion_times_s",
    "quaternions_wxyz",
    "kept_after_time_collisions",
    "consecutive_quaternion_rate_times_s",
    "consecutive_quaternion_rate_deg_s",
    "recorded_rate_times_s",
    "recorded_rate_deg_s",
    "group_opening_quaternion_index",
    "quaternion_count_per_telemetry_frame",
    "owning_telemetry_frame",
    "period_s",
    "normalized_rate_deg_s",
)


INVENTORY_HEADING = "## Inventory"

INVENTORY_COLUMNS = (
    "file",
    "video",
    "video frame rate",
    "quaternions",
    "measurement groups",
    "identical consecutive quaternions",
    "size",
)

NEW_README_PREAMBLE = """\
# Decode results

One `.npz` per video holding the quaternions of its gyro data track as the camera
stored them, and every series derived from them: the recorded rate, the Normalized
Rate, and the period T. Its header carries the identity of the video and the derivation
version.

The decode stage writes the inventory below each time it writes a decode result. A row
whose `.npz` no longer exists is dropped when the next decode result is written.

"""


def write_npz(path, arrays):
    """Store ``arrays`` at ``path`` as a compressed ``.npz``.

    The arrays go to ``<name>.partial`` beside ``path`` first, which replaces ``path``
    only once it is complete, so a write that is interrupted never leaves a truncated
    result under the name of the result."""
    path = Path(path)
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("wb") as partial_file:
        np.savez_compressed(partial_file, **arrays)
    partial.replace(path)


def decode_result_path(video_cache_directory, video_name):
    """The path of the decode result of the video with file stem ``video_name`` in
    ``video_cache_directory``."""
    return Path(video_cache_directory) / f"{video_name}-decode-result.npz"


def readme_path(video_cache_directory):
    """The path of the ``README.md`` that holds the inventory of the decode results in
    ``video_cache_directory``."""
    return Path(video_cache_directory) / "README.md"


def _spaced(count):
    """A count with a space between thousands, as the inventory table writes them."""
    return f"{int(count):,}".replace(",", " ")


def _inventory_row(path, result):
    """One inventory row for a decode result just written: its key and its markdown
    line.

    Every figure comes from the decode result being written, so the row cannot disagree
    with the file it describes. The share of identical consecutive quaternions counts
    each kept quaternion, in time order, that equals the one before it."""
    quaternion_count = len(result["quaternion_times_s"])
    group_count = len(result["group_opening_quaternion_index"])
    consecutive_times = result["consecutive_quaternion_rate_times_s"]
    span_s = float(consecutive_times[-1] - consecutive_times[0])
    kept = np.asarray(result["quaternions_wxyz"])[
        np.asarray(result["kept_after_time_collisions"])
    ]
    repeated = np.count_nonzero(repeats_previous_quaternion(kept)) / (len(kept) - 1)
    cells = (
        f"`{path.name}`",
        f"`{result['video_name']}`",
        f"{float(result['video_frame_rate']):.4f} fps",
        (
            f"{_spaced(quaternion_count)} over {span_s:.1f} s "
            f"({float(result['quaternion_rate_hz']):.4f} Hz, "
            f"{float(result['quaternions_per_video_frame']):.4f} per video frame)"
        ),
        f"{_spaced(group_count)} ({float(result['measurement_rate_hz']):.4f} Hz)",
        f"{repeated * 100:.4f} %",
        f"{path.stat().st_size / 1e6:.1f} MB",
    )
    return path.name, "| " + " | ".join(cells) + " |"


def _inventory_table(rows, video_cache_directory):
    """The whole table: header, every row whose ``.npz`` is still there in file name
    order, and the totals line as a final row."""
    kept = {
        name: line
        for name, line in rows.items()
        if (video_cache_directory / name).exists()
    }
    ordered = [kept[name] for name in sorted(kept)]
    total_mb = sum((video_cache_directory / name).stat().st_size for name in kept) / 1e6
    blanks = " | ".join("" for _ in INVENTORY_COLUMNS[1:-1])
    decode_results = "decode result" if len(kept) == 1 else "decode results"
    return [
        "| " + " | ".join(INVENTORY_COLUMNS) + " |",
        "|" + "|".join("---" for _ in INVENTORY_COLUMNS) + "|",
        *ordered,
        f"| **{len(kept)} {decode_results}** | {blanks} | **{total_mb:.1f} MB** |",
    ]


def _existing_rows(lines, start, stop):
    """The rows already in the table, keyed by the ``.npz`` filename in the first cell.
    The header and separator lines are skipped; the totals row keys on no file and is
    dropped with the stale ones."""
    rows = {}
    for line in lines[start + 2 : stop]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows[cells[0].strip("`").strip()] = line
    return rows


def update_cache_directory_readme(path, result):
    """Write the row of the decode result at ``path`` into the inventory table of the
    ``README.md`` of its directory, and return the path of the ``README.md``.

    The table is the only part of the file this writes: the prose around it is left
    exactly as it stands. With no README in the directory, one is created from
    ``NEW_README_PREAMBLE``."""
    video_cache_directory = path.parent
    readme = readme_path(video_cache_directory)
    text = (
        readme.read_text(encoding="utf-8") if readme.exists() else NEW_README_PREAMBLE
    )
    lines = text.split("\n")

    heading = next(
        (i for i, line in enumerate(lines) if line.strip() == INVENTORY_HEADING), None
    )
    if heading is None:
        while lines and not lines[-1].strip():
            lines.pop()
        lines += ["", INVENTORY_HEADING, ""]
        heading = len(lines) - 2

    start = next(
        (i for i in range(heading, len(lines)) if lines[i].startswith("|")), None
    )
    if start is None:
        start = heading + 1
        while start < len(lines) and not lines[start].strip():
            start += 1
        stop, rows = start, {}
    else:
        stop = start
        while stop + 1 < len(lines) and lines[stop + 1].startswith("|"):
            stop += 1
        stop += 1
        rows = _existing_rows(lines, start, stop)

    key, row = _inventory_row(path, result)
    rows[key] = row
    lines[start:stop] = _inventory_table(rows, video_cache_directory)
    readme.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return readme


def write_decode_result(result, video_cache_directory):
    """The cache writer of the decode stage: store the decode result in
    ``video_cache_directory`` as ``<video file stem>-decode-result.npz``, record its row
    in the inventory of that directory, and return the path of the decode result and the
    path of the ``README.md`` that holds the inventory.

    An error of the operating system while it writes the decode result or the
    ``README.md`` raises an error carrying the path of that file."""
    path = decode_result_path(video_cache_directory, result["video_name"])
    with errors.operating_system_error_as(errors.CannotWriteError, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_npz(path, result)
    with errors.operating_system_error_as(
        errors.CannotWriteError, readme_path(path.parent)
    ):
        readme = update_cache_directory_readme(path, result)
    return path, readme
