# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""The cache reader of the decode stage and its three tests, and the names of the two
rate series of the decode result.

A rate sample of the recorded rate and the rate sample at the same index of the
Normalized Rate span the same pair of measurement group boundaries, which is what maps
an extent found on the Normalized Rate to exact quaternions.
"""

from pathlib import Path

import numpy as np

from gyro_doctor import errors
from gyro_doctor.core import decode_stage

SERIES_ARRAY = {
    "normalized": "normalized_rate_deg_s",
    "recorded": "recorded_rate_deg_s",
}
SERIES_LABEL = {
    "normalized": "Normalized Rate (rotation / T)",
    "recorded": "recorded rate (rotation / own interval)",
}


def period_test(result):
    """The period test of a decode result: raise an error when T as stored differs from
    T as the group boundaries of the same result derive it by more than
    ``decode_stage.PERIOD_TEST_TOLERANCE_S``, and the error of
    ``decode_stage.derive_period`` when the derived T differs from the reciprocal of the
    counted measurement rate.

    A stored T that its own group boundaries do not derive means the Normalized Rate was
    divided by a period from another derivation."""
    stored = float(result["period_s"])
    derived = decode_stage.derive_period(
        decode_stage.group_boundary_times(result), result
    )
    if abs(stored - derived) > decode_stage.PERIOD_TEST_TOLERANCE_S:
        raise errors.CannotReadError(
            errors.ErrorCode.STORED_PERIOD_DIFFERS,
            stored_us=stored * 1e6,
            derived_us=derived * 1e6,
        )


def rate_series_test(result):
    """The test that the recorded rate and the Normalized Rate have the same shape, with
    one time per rate sample and one group boundary more than rate samples, so the rate
    sample at each index spans the same quaternions in both.

    Raises an error carrying each part whose shape or size disagrees, as the name of the
    part, its shape or size, and the shape or size expected."""
    recorded = np.asarray(result["recorded_rate_deg_s"])
    normalized = np.asarray(result["normalized_rate_deg_s"])
    boundaries = decode_stage.group_boundary_times(result)
    times = np.asarray(result["recorded_rate_times_s"])
    rate_sample_count = recorded.shape[0]
    problems = []
    if normalized.shape != recorded.shape:
        problems.append(("normalized_rate_deg_s", normalized.shape, recorded.shape))
    if boundaries.size != rate_sample_count + 1:
        problems.append(
            ("group_opening_quaternion_index", boundaries.size, rate_sample_count + 1)
        )
    if times.size != rate_sample_count:
        problems.append(("recorded_rate_times_s", times.size, rate_sample_count))
    if problems:
        raise errors.CannotReadError(
            errors.ErrorCode.RATE_SERIES_UNEQUAL, problems=problems
        )


def read_decode_result(path, video):
    """The cache reader of the decode stage: the decode result stored at ``path`` for
    the video at path ``video``, returned only after its three tests pass.

    Every error it raises on purpose is ``errors.CannotReadError`` and carries ``path``
    as ``file``: the error of a file it cannot open or read as a decode result, and the
    error of a test, the period test that ``derive_period`` runs included. The tests run
    after the read, so an error a defect raises in a test stays a traceback."""
    path = Path(path)
    with (
        decode_stage.cache_file_read(path),
        np.load(path, allow_pickle=False) as stored,
    ):
        result = {key: stored[key] for key in stored.files}
    try:
        decode_stage.header_test(result, video, path)
        decode_stage.every_part_present_test(
            result, path, decode_stage.DECODE_RESULT_PARTS
        )
        period_test(result)
        rate_series_test(result)
    except errors.CannotRepairError as error:
        raise errors.CannotReadError(
            error.code, **{**error.values, "file": path}
        ) from error
    except errors.GyroDoctorError as error:
        error.values.setdefault("file", path)
        raise
    return result
