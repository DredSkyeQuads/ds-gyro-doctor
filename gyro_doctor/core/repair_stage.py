# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The repair stage, which attenuates each extent of the detection result with the
ceiling attenuator and produces the repair result and the repair record, with the cache
writer and the cache reader of the repair result.

The ceiling attenuator does not widen an extent, since both fades lie inside it. An axis
outside the repair axes runs at an infinite ceiling, keeps every rate sample, and still
has its extents in the repair record. The repair stage stops the run with an error if
any rate sample outside every extent changed.

The repair stage sets every rate sample of an axis that ``--remove-gyro-motion`` names to
0 degrees per second, whether or not the repair axes or the detection result hold the
axis. None of the tests of the ceiling attenuator runs on the axis since the repair uses
none of the extents that the detector found. The repair stage stops the run with an
error if any rate sample of the axis is not 0.

A Normalized Rate is each rotation divided by the one period T, so it composes back to
orientations by T, and every rotation the repair record reports is a sum of rate samples
times T. Composing it by each rate sample's own interval instead would put the timing
distortion back inside the repaired extents alone, which is motion the camera never
recorded at exactly the rate samples the repair changed.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from gyro_doctor import errors
from gyro_doctor.core import (
    ceiling_attenuator,
    decode_result_reader,
    decode_stage,
    repaired_rates_to_quaternions,
)
from gyro_doctor.core.axes import AXES, axis_column

SERIES = "normalized"

CEILING = "ceiling"

# The parts of the header of the decode result, which the decode result recomputed on
# the repaired track carries unchanged, so that a cached repair result names the video
# and the derivation it belongs to.
HEADER_PARTS = ("video_name", "video_size_bytes", "video_mtime_s", "derivation_version")


def selected_repair_axes(axes, detection_result):
    """The repair axes in the order pitch, yaw, roll. Raises ValueError on an unknown
    axis, a repeated axis, an empty selection, and an axis that the detection result
    does not hold."""
    axes = tuple(axes)
    for axis in axes:
        axis_column(axis)
    if not axes or len(set(axes)) != len(axes):
        raise ValueError(
            f"the repair stage takes one or more distinct axes of {', '.join(AXES)}, "
            f"not {', '.join(axes) or 'none'}"
        )
    processed_axes = [str(stored_axis) for stored_axis in detection_result["axes"]]
    absent = [axis for axis in axes if axis not in processed_axes]
    if absent:
        raise ValueError(
            f"the repair stage repairs only the axes of the detection result, "
            f"{', '.join(processed_axes)}, not {', '.join(absent)}"
        )
    return tuple(axis for axis in AXES if axis in axes)


def excluded_extents(
    identifiers,
    video_frame_ranges,
    extent_identifier,
    extent_first_video_frame,
    extent_last_video_frame,
):
    """The do not repair list resolved against the extents of a detection result:
    ``(excluded, unmatched)``, the sorted identifiers of the extents the list excludes,
    and each identifier of the list that names no extent, in the order of the list.

    ``identifiers`` are the identifiers the list names, such as ``Y1502``.
    ``video_frame_ranges`` are the ranges of the list, each ``(axis letter or None,
    first video frame, last video frame)``, both video frames included; a range without
    an axis letter applies to every axis. ``extent_identifier``,
    ``extent_first_video_frame``, and ``extent_last_video_frame`` give, for each extent,
    its identifier and the video frames of its start and its end.

    The list excludes every extent whose identifier it names, and every extent whose
    video frames a range overlaps on the axis of the range's letter or on every axis. An
    identifier names every extent that carries it, so every extent of an identifier
    that a range finds is excluded."""
    extent_identifier = np.asarray(extent_identifier, str)
    first = np.asarray(extent_first_video_frame, int)
    last = np.asarray(extent_last_video_frame, int)
    axis_letters = extent_identifier.astype("<U1")
    known = set(extent_identifier.tolist())
    overlapped = np.zeros(extent_identifier.size, bool)
    for axis_letter, range_first, range_last in video_frame_ranges:
        on_axis = True if axis_letter is None else axis_letters == axis_letter
        overlapped |= on_axis & (first <= range_last) & (last >= range_first)
    excluded = (known & set(identifiers)) | set(extent_identifier[overlapped].tolist())
    unmatched = tuple(
        identifier for identifier in identifiers if identifier not in known
    )
    return tuple(sorted(excluded)), unmatched


def axis_of_detection_result(detection_result, axis):
    """One axis of the detection result: its extents, which the detection result lists
    in order of their first rate sample, with the peak sample, the classification, and
    the identifier of each, the local median, the clean Robust Standard Deviation
    repeated at every rate sample, and the clean Robust Standard Deviation.

    Every ceiling is a multiple of the clean Robust Standard Deviation of its axis, which
    the detection stage measures on the residual outside every extent after its three
    passes.
    """
    processed_axes = [str(stored_axis) for stored_axis in detection_result["axes"]]
    column = processed_axes.index(axis)
    local_median = detection_result["local_median_deg_s"][:, column]
    clean_robust_standard_deviation = float(
        detection_result["clean_robust_standard_deviation_deg_s"][column]
    )
    on_axis = np.asarray(detection_result["extent_axis"]) == axis
    extents = (
        np.stack(
            [
                np.asarray(detection_result["extent_first_sample"])[on_axis],
                np.asarray(detection_result["extent_last_sample"])[on_axis],
            ],
            axis=1,
        )
        .astype(int)
        .reshape(-1, 2)
    )
    peaks = np.asarray(detection_result["extent_peak_sample"])[on_axis].astype(int)
    classifications = np.array(
        [
            str(classification)
            for classification in np.asarray(detection_result["extent_classification"])[
                on_axis
            ]
        ],
        dtype=object,
    )
    identifiers = [
        str(identifier)
        for identifier in np.asarray(detection_result["extent_identifier"])[on_axis]
    ]
    clean_robust_standard_deviation_at_every_rate_sample = np.full(
        local_median.size, clean_robust_standard_deviation
    )
    return (
        extents,
        peaks,
        classifications,
        identifiers,
        local_median,
        clean_robust_standard_deviation_at_every_rate_sample,
        clean_robust_standard_deviation,
    )


def extent_report(
    rate, repaired, centerline, boundaries, period_s, first, last, gain, extent_entry
):
    """The report of one extent: its start, end, and duration, the rotations of the
    Normalized Rate, the repaired Normalized Rate, and the centerline over it, and
    ``extent_entry``, the entry of the ceiling attenuator for that extent.

    ``first`` and ``last`` are inclusive. Each rotation is a sum of rate samples times
    ``period_s``, the period T, since the Normalized Rate is each rotation divided by T.
    The duration and the start and end times come from the group boundaries. A key that
    ``extent_entry`` shares with the report keeps the report's value, so each key has
    one derivation.
    """
    extent_rate_samples = slice(first, last + 1)
    removed = float(
        np.sum((rate[extent_rate_samples] - repaired[extent_rate_samples]) * period_s)
    )
    recorded = float(np.sum(rate[extent_rate_samples] * period_s))
    report = {
        "first_sample": int(first),
        "last_sample": int(last),
        "rate_sample_count": int(last - first + 1),
        "start_s": float(boundaries[first]),
        "end_s": float(boundaries[last + 1]),
        "duration_ms": float((boundaries[last + 1] - boundaries[first]) * 1e3),
        "gain": float(gain),
        "recorded_rotation_deg": recorded,
        "centerline_rotation_deg": float(
            np.sum(centerline[extent_rate_samples] * period_s)
        ),
        "removed_rotation_deg": removed,
        "kept_rotation_deg": recorded - removed,
    }
    report.update(
        {key: value for key, value in extent_entry.items() if key not in report}
    )
    return report


def changed_only_inside_extents(rate, repaired, extents):
    """The test that the repair changed no rate sample outside every extent, by exact
    comparison, with three counts: the rate samples inside the extents, the rate samples
    changed, and the rate samples changed outside every extent.

    A rate sample changed outside every extent is motion the camera never recorded, so
    the test raises an error and the run stops before the orientations are rebuilt.
    """
    inside_extents = np.zeros(rate.size, dtype=bool)
    for first, last in extents:
        inside_extents[first : last + 1] = True
    changed = rate != repaired
    outside = np.flatnonzero(changed & ~inside_extents)
    if outside.size:
        raise errors.CannotRepairError(
            errors.ErrorCode.RATE_SAMPLES_CHANGED_OUTSIDE_EXTENTS,
            count=int(outside.size),
            rate_sample=int(outside[0]),
        )
    return {
        "rate_samples_inside_extents": int(inside_extents.sum()),
        "rate_samples_changed": int(changed.sum()),
        "rate_samples_changed_outside_extents": int(outside.size),
    }


def kept_every_rate_sample(axis, rate, repaired):
    """The test that the repaired Normalized Rate of an axis outside the repair axes
    equals its Normalized Rate at every rate sample, by exact comparison; raises an error
    carrying the axis otherwise."""
    changed = np.flatnonzero(rate != repaired)
    if changed.size:
        raise errors.CannotRepairError(
            errors.ErrorCode.AXIS_OUTSIDE_REPAIR_AXES_CHANGED,
            count=int(changed.size),
            axis=axis,
            rate_sample=int(changed[0]),
        )


def zero_at_every_rate_sample(axis, repaired):
    """The test that the repaired Normalized Rate of an axis that
    ``--remove-gyro-motion`` names is 0 degrees per second at every rate sample.

    The test compares each rate sample exactly. Where a rate sample is not 0, it raises
    an error that carries the axis, the count of rate samples that are not 0, and the
    first of them."""
    not_zero = np.flatnonzero(repaired != 0.0)
    if not_zero.size:
        raise errors.CannotRepairError(
            errors.ErrorCode.REMOVED_AXIS_NOT_ZERO,
            count=int(not_zero.size),
            axis=axis,
            rate_sample=int(not_zero[0]),
        )


def remove_gyro_motion(rate, period_s, detection_result, axis):
    """The removal of the recorded gyro motion of one axis that ``--remove-gyro-motion``
    names.

    The removal sets every rate sample of the Normalized Rate of the axis to 0 degrees
    per second and tests that every rate sample is 0. It returns the repaired Normalized
    Rate of the axis and the entry of the axis in the repair record.

    The repair treats the whole axis as one extent, repaired toward the ``zero``
    centerline at a gain of 0. The absolute rotation removed is therefore the size of
    the net rotation removed. The repair record keeps the extents that the detector
    found on the axis. It holds no extent of an axis that the detection result does not
    hold.

    Raises the error of ``zero_at_every_rate_sample``."""
    repaired = np.zeros_like(rate)
    zero_at_every_rate_sample(axis, repaired)
    if axis in [str(stored_axis) for stored_axis in detection_result["axes"]]:
        extents, _peaks, _classifications, identifiers, *_rest = (
            axis_of_detection_result(detection_result, axis)
        )
    else:
        extents, identifiers = np.empty((0, 2), dtype=int), []
    removed = float(np.sum((rate - repaired) * period_s))
    return {
        "repaired_rate_deg_s": repaired,
        "record": {
            "detected_extents_in": int(extents.shape[0]),
            "removed_rotation_deg_abs_total": abs(removed),
            "removed_rotation_deg_net": removed,
            "extents": [
                {
                    "identifier": identifier,
                    "first_sample": int(first),
                    "last_sample": int(last),
                }
                for identifier, (first, last) in zip(identifiers, extents, strict=True)
            ],
        },
    }


def ceiling_attenuator_tests(
    rate, repaired, centerline, extents, gains, extent_entries, ceilings
):
    """The three tests of the ceiling attenuator on one axis, and five counts among the
    extents it neither skipped at a video edge nor excluded as the do not repair list
    directs.

    ``ceilings`` holds the ceiling of each extent, in clean Robust Standard Deviations.
    The three tests:

    1. The repaired Normalized Rate equals the Normalized Rate exactly at the first and
       the last rate sample of the extent, which the ceiling attenuator holds at a gain
       of one.
    2. Inside the extent the ceiling attenuator scaled every departure from the
       centerline by a factor between the gain of the extent and one, so it amplified
       and inverted none.
    3. No rate sample in the interior of the extent stands above the ceiling. Inside a
       fade the gain stands above the gain of the extent, and the first and the last
       rate sample are held at a gain of one even with no fade, so the ceiling applies
       to the interior alone. An extent whose fades were shortened is left out, since
       its largest departure can sit inside a fade.

    The first two tests raise an error at the first extent that fails them; the third
    raises an error after every extent, carrying how many failed it.

    The five counts: the extents that passed the first test; the extents that passed the
    second; the extents that failed the third, which is therefore 0; the extents with a
    rate sample above the ceiling anywhere from the first rate sample to the last; and
    the extents whose fades were shortened.
    """
    ceilings = np.asarray(ceilings, float)
    edges_held, ratio_held, over_ceiling, shortened_fades, over_anywhere = (
        0,
        0,
        [],
        0,
        0,
    )
    for index, ((first, last), gain, extent_entry) in enumerate(
        zip(extents, gains, extent_entries, strict=True)
    ):
        if not extent_entry["scaled"]:
            continue
        if repaired[first] != rate[first] or repaired[last] != rate[last]:
            raise errors.CannotRepairError(
                errors.ErrorCode.EXTENT_END_CHANGED, rate_sample=int(first)
            )
        edges_held += 1

        extent_rate_samples = slice(first, last + 1)
        departure_in = rate[extent_rate_samples] - centerline[extent_rate_samples]
        departure_out = repaired[extent_rate_samples] - centerline[extent_rate_samples]
        moved = np.abs(departure_in) > 0
        ratio = departure_out[moved] / departure_in[moved]
        if ratio.size and (ratio.min() < gain - 1e-9 or ratio.max() > 1.0 + 1e-9):
            raise errors.CannotRepairError(
                errors.ErrorCode.DEPARTURE_SCALED_OUTSIDE_GAIN,
                rate_sample=int(first),
                low=float(ratio.min()),
                high=float(ratio.max()),
                gain=float(gain),
            )
        ratio_held += 1

        if extent_entry["fade_shortened"]:
            shortened_fades += 1
        elif (
            extent_entry["peak_in_clean_robust_standard_deviations_after_interior"]
            > ceilings[index] + 1e-6
        ):
            over_ceiling.append(int(first))
        if (
            extent_entry["peak_in_clean_robust_standard_deviations_after"]
            > ceilings[index] + 1e-6
        ):
            over_anywhere += 1

    if over_ceiling:
        raise errors.CannotRepairError(
            errors.ErrorCode.INTERIOR_ABOVE_CEILING,
            count=len(over_ceiling),
            rate_sample=over_ceiling[0],
        )
    return {
        "extents_identical_at_both_boundaries": edges_held,
        "extents_scaled_within_gain_to_one": ratio_held,
        "extents_over_ceiling_in_the_interior": len(over_ceiling),
        "extents_over_ceiling_anywhere_in_the_extent": over_anywhere,
        "extents_with_shortened_fades": shortened_fades,
    }


def ceilings_for(classifications, parameters):
    """The ceiling of each extent, in clean Robust Standard Deviations, by its
    classification.

    Read from the repair parameters the ceiling attenuator applied, so each test measures
    against the ceiling applied.
    """
    return np.array(
        [parameters.ceiling_of(classification) for classification in classifications]
    )


def repair_axis(
    rate,
    boundaries,
    rate_sample_times_s,
    measurement_rate_hz,
    period_s,
    detection_result,
    axis,
    repair_axes,
    parameters,
):
    """The repair of one axis of the detection result: the ceiling attenuator on its
    Normalized Rate, the tests of the repair, and the entry of the axis in the repair
    record.

    Reads only its own axis, so the axes are independent units of work. Raises the
    errors of ``kept_every_rate_sample``, ``changed_only_inside_extents``, and
    ``ceiling_attenuator_tests``.
    """
    repaired_here = axis in repair_axes
    # An axis outside the repair axes runs through the ceiling attenuator rather than
    # around it, so the repair record still holds its extents, each at a gain of one.
    axis_parameters = (
        parameters
        if repaired_here
        else replace(
            parameters,
            repair_oscillation_ceiling=float("inf"),
            repair_burst_ceiling=float("inf"),
            repair_impulse_ceiling=float("inf"),
        )
    )
    (
        extents,
        peaks,
        classifications,
        identifiers,
        local_median,
        clean_robust_standard_deviation_at_every_rate_sample,
        clean_robust_standard_deviation,
    ) = axis_of_detection_result(detection_result, axis)

    started = time.perf_counter()
    result = ceiling_attenuator.attenuate(
        rate,
        measurement_rate_hz,
        extents,
        peaks,
        classifications,
        local_median,
        clean_robust_standard_deviation_at_every_rate_sample,
        rate_sample_times_s=rate_sample_times_s,
        excluded=np.isin(identifiers, parameters.do_not_repair),
        parameters=axis_parameters,
    )
    elapsed_s = time.perf_counter() - started
    if not repaired_here:
        kept_every_rate_sample(axis, rate, result.repaired_rate_deg_s)

    # The ceiling attenuator returns the extents in order of their first rate sample, the
    # order in which the detection result lists them, so each identifier and each
    # ceiling lines up with its extent.
    extent_reports = [
        {
            "identifier": identifier,
            **extent_report(
                rate,
                result.repaired_rate_deg_s,
                result.centerline,
                boundaries,
                period_s,
                int(first),
                int(last),
                gain,
                extent_entry,
            ),
        }
        for identifier, (first, last), gain, extent_entry in zip(
            identifiers,
            result.extents,
            result.gains,
            result.extent_entries,
            strict=True,
        )
    ]
    scaled = [report for report in extent_reports if report["scaled"]]
    kept_reasons = [
        report["reason"] for report in extent_reports if not report["scaled"]
    ]
    gains = np.array([report["gain"] for report in scaled]) if scaled else np.array([])
    durations = (
        np.array([report["duration_ms"] for report in extent_reports])
        if extent_reports
        else np.array([])
    )
    removed = (
        np.array([report["removed_rotation_deg"] for report in scaled])
        if scaled
        else np.array([])
    )
    before = (
        np.array(
            [
                report["peak_in_clean_robust_standard_deviations_before"]
                for report in scaled
            ]
        )
        if scaled
        else np.array([])
    )

    return {
        "repaired_rate_deg_s": result.repaired_rate_deg_s,
        "record": {
            "clean_robust_standard_deviation_deg_s": clean_robust_standard_deviation,
            "repaired": repaired_here,
            "measured_against": (
                "every ceiling is a multiple of the clean Robust Standard Deviation of "
                "the axis from the detection result, and the local-median centerline "
                "is the local median of the detection result"
            ),
            "seconds_to_attenuate": round(elapsed_s, 3),
            "detected_extents_in": int(extents.shape[0]),
            "extents_out": int(result.extents.shape[0]),
            "identity_outside_extents": changed_only_inside_extents(
                rate, result.repaired_rate_deg_s, result.extents
            ),
            "ceiling_attenuator_tests": ceiling_attenuator_tests(
                rate,
                result.repaired_rate_deg_s,
                result.centerline,
                result.extents,
                result.gains,
                result.extent_entries,
                ceilings_for(classifications, axis_parameters),
            ),
            "extent_summary": {
                "scaled_count": len(scaled),
                "skipped_count": kept_reasons.count(ceiling_attenuator.AT_A_VIDEO_EDGE),
                "excluded_count": kept_reasons.count(
                    ceiling_attenuator.ON_THE_DO_NOT_REPAIR_LIST
                ),
                "duration_ms_min": float(durations.min()) if durations.size else None,
                "duration_ms_median": float(np.median(durations))
                if durations.size
                else None,
                "duration_ms_max": float(durations.max()) if durations.size else None,
                "gain_min": float(gains.min()) if gains.size else None,
                "gain_median": float(np.median(gains)) if gains.size else None,
                "gain_max": float(gains.max()) if gains.size else None,
                "passed_through_count": int((gains >= 1.0).sum()) if gains.size else 0,
                "peak_in_clean_robust_standard_deviations_before_min": float(
                    before.min()
                )
                if before.size
                else None,
                "peak_in_clean_robust_standard_deviations_before_median": float(
                    np.median(before)
                )
                if before.size
                else None,
                "peak_in_clean_robust_standard_deviations_before_max": float(
                    before.max()
                )
                if before.size
                else None,
                "removed_rotation_deg_abs_total": float(np.abs(removed).sum())
                if removed.size
                else 0.0,
                "removed_rotation_deg_net": float(removed.sum())
                if removed.size
                else 0.0,
            },
            "attenuator_statistics": result.statistics,
            "extents": extent_reports,
        },
    }


def repair(decode_result, detection_result, axes, parameters):
    """The repair stage: the repair result of the decode result and the detection result
    on the selected repair axes, with the repair parameters given.

    The repair result holds the header of the decode result; the repaired orientations,
    with the time and the time collision status of each; the repaired Normalized Rate
    recomputed from them, with the time of each rate sample; and the repair record.

    The repair stage repairs every axis of the detection result and every axis that
    ``--remove-gyro-motion`` names. One worker per axis repairs those axes in parallel.
    Any other axis keeps its Normalized Rate. Cadence preservation runs only when
    ``--remove-gyro-motion`` names no axis.

    Raises the errors of ``selected_repair_axes``, ``repair_axis``,
    ``remove_gyro_motion``, and the assertions of the rebuild.
    """
    repair_axes = selected_repair_axes(axes, detection_result)
    processed_axes = [str(stored_axis) for stored_axis in detection_result["axes"]]
    removed_axes = tuple(parameters.remove_gyro_motion)
    repaired_axes = [
        axis for axis in AXES if axis in processed_axes or axis in removed_axes
    ]
    rates = np.asarray(decode_result["normalized_rate_deg_s"])
    boundaries = decode_stage.group_boundary_times(decode_result)
    rate_sample_times_s = 0.5 * (boundaries[:-1] + boundaries[1:])
    measurement_rate_hz = float(decode_result["measurement_rate_hz"])
    period_s = float(decode_result["period_s"])

    def repair_one_axis(axis):
        if axis in removed_axes:
            return remove_gyro_motion(
                rates[:, axis_column(axis)], period_s, detection_result, axis
            )
        return repair_axis(
            rates[:, axis_column(axis)],
            boundaries,
            rate_sample_times_s,
            measurement_rate_hz,
            period_s,
            detection_result,
            axis,
            repair_axes,
            parameters,
        )

    with ThreadPoolExecutor(max_workers=len(repaired_axes)) as pool:
        per_axis = list(pool.map(repair_one_axis, repaired_axes))

    repaired = rates.copy()
    for axis, axis_repair in zip(repaired_axes, per_axis, strict=True):
        repaired[:, axis_column(axis)] = axis_repair["repaired_rate_deg_s"]

    assertions = [repaired_rates_to_quaternions.round_trip_identity(decode_result)]
    rebuilt = repaired_rates_to_quaternions.quaternion_track_from_repaired_rates(
        decode_result, repaired, series=SERIES, period_s=period_s
    )
    if not removed_axes:
        assertions.append(
            repaired_rates_to_quaternions.cadence_preservation(decode_result, rebuilt)
        )

    # The handler name comes from the decode result since the rebuilt track has no
    # container to read it from. The measurement groups also come from the decode result
    # since a repair changes no measurement group. The identity comes from the header of
    # the decode result since the stage reads no file.
    recomputed = decode_stage.derive_decode_result(
        np.asarray(decode_result["quaternion_times_s"], dtype=np.float64),
        rebuilt,
        float(decode_result["video_frame_rate"]),
        {name: decode_result[name] for name in HEADER_PARTS},
        decode_result["quaternion_count_per_telemetry_frame"],
        decode_result["owning_telemetry_frame"],
        handler_name=decode_result["handler_name"],
        group_opening_quaternion_index=decode_result["group_opening_quaternion_index"],
    )

    record = {
        "run": {
            "attenuator_method": CEILING,
            "attenuator": "ceiling_attenuator.py",
            "baseline_series": "local_median_deg_s",
            "input_series": decode_result_reader.SERIES_ARRAY[SERIES],
            "input_series_label": decode_result_reader.SERIES_LABEL[SERIES],
            "repaired_rate_series": SERIES,
            "composition_multiplier": repaired_rates_to_quaternions.COMPOSITION_MULTIPLIER[
                SERIES
            ],
            "detected_extents": (
                "every extent of the detection result on each axis of the detection "
                "result"
            ),
            "measurement_rate_source": (
                "measurement_rate_hz of the decode result, counted over the video"
            ),
            "scope": "whole video",
            "repair_parameters": asdict(parameters),
            "repair_axes": list(repair_axes),
        },
        "video": {
            "video_name": str(decode_result["video_name"]),
            "video_size_bytes": int(decode_result["video_size_bytes"]),
            "video_mtime_s": float(decode_result["video_mtime_s"]),
            "derivation_version": int(decode_result["derivation_version"]),
            "video_frame_rate": float(decode_result["video_frame_rate"]),
            "measurement_rate_hz": measurement_rate_hz,
            "period_s": period_s,
            "rate_sample_count": int(rates.shape[0]),
            "axes": processed_axes,
        },
        "assertions": assertions,
        "axes": {
            axis: axis_repair["record"]
            for axis, axis_repair in zip(repaired_axes, per_axis, strict=True)
        },
    }

    return {
        **{part: recomputed[part] for part in HEADER_PARTS},
        "quaternion_times_s": recomputed["quaternion_times_s"],
        "quaternions_wxyz": recomputed["quaternions_wxyz"],
        "kept_after_time_collisions": recomputed["kept_after_time_collisions"],
        "recorded_rate_times_s": recomputed["recorded_rate_times_s"],
        "normalized_rate_deg_s": recomputed["normalized_rate_deg_s"],
        "repair_record": json.dumps(record, default=float),
    }


REPAIR_RESULT_PARTS = (
    *HEADER_PARTS,
    "quaternion_times_s",
    "quaternions_wxyz",
    "kept_after_time_collisions",
    "recorded_rate_times_s",
    "normalized_rate_deg_s",
    "repair_record",
)


def repair_result_path(video_cache_directory, video_name):
    """The path of the repair result of the video with file stem ``video_name`` in
    ``video_cache_directory``."""
    return Path(video_cache_directory) / f"{video_name}-repair-result.npz"


def repair_record_path(video_cache_directory, video_name):
    """The path of the repair record of the video with file stem ``video_name`` in
    ``video_cache_directory``."""
    return Path(video_cache_directory) / f"{video_name}-repair-record.json"


def write_repair_result(result, video_cache_directory, *, file_stamps):
    """The cache writer of the repair stage: store the repair result in
    ``video_cache_directory`` as ``<video file stem>-repair-result.npz`` and the repair
    record as ``<video file stem>-repair-record.json``, creating
    ``video_cache_directory`` with its parents.

    ``file_stamps`` maps ``"decode_result"`` and ``"detection_result"`` to the file
    stamps of the two results the stage read. The repair record inside the stored repair
    result holds them under ``file_stamps``. A file cannot hold its own file stamp, so
    the repair record file holds those two and the file stamp of the repair result under
    ``"repair_result"``. The repair result in memory is left as it is.

    An error of the operating system while it writes a file raises an error carrying the
    path of that file."""
    video_cache_directory = Path(video_cache_directory)
    stem = str(result["video_name"])
    result_path = repair_result_path(video_cache_directory, stem)
    record_path = repair_record_path(video_cache_directory, stem)
    record = json.loads(str(result["repair_record"]))
    record["file_stamps"] = dict(file_stamps)
    with errors.operating_system_error_as(errors.CannotWriteError, result_path):
        video_cache_directory.mkdir(parents=True, exist_ok=True)
        decode_stage.write_npz(
            result_path, {**result, "repair_record": json.dumps(record)}
        )
        record["file_stamps"]["repair_result"] = decode_stage.file_stamp(result_path)
    with errors.operating_system_error_as(errors.CannotWriteError, record_path):
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def read_repair_result(path):
    """The cache reader of the repair stage: the repair result stored at ``path``,
    returned only when every part of the repair result is present and its repair record
    holds the file stamps.

    Every error it raises on purpose is ``errors.CannotReadError`` and carries ``path``
    as ``file``. The tests run after the read of the file and of its repair record, so
    an error a defect raises in a test stays a traceback."""
    path = Path(path)
    with decode_stage.cache_file_read(path):
        with np.load(path, allow_pickle=False) as stored:
            result = {key: stored[key] for key in stored.files}
        record = (
            json.loads(str(result["repair_record"]))
            if "repair_record" in result
            else {}
        )
    decode_stage.every_part_present_test(result, path, REPAIR_RESULT_PARTS)
    decode_stage.every_part_present_test(record, path, ("file_stamps",))
    return result


def same_parameters(record, axes, parameters):
    """Whether the repair record ``record``, as a dictionary, was computed on the repair
    axes ``axes`` with the repair parameters ``parameters``. The parameters go through
    the JSON round trip that the stored record went through, which turns a tuple into a
    list, so both sides compare alike."""
    return record["run"]["repair_axes"] == [
        axis for axis in AXES if axis in axes
    ] and record["run"]["repair_parameters"] == json.loads(
        json.dumps(asdict(parameters), default=float)
    )
