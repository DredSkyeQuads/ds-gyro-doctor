# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The detection stage, which finds the gyro noise events in the Normalized Rate of
each selected axis in three passes and classifies each one, with the run record, the
extent tables, and the cache writer and reader of the detection result.

Each pass measures against the Robust Standard Deviation of the residual over the rate
samples that no earlier pass placed inside an extent, so the order of the passes is part
of the method, and the pass that finds an extent gives it its classification.

The oscillation pass and the burst pass recognize an event by the ratio of the local
Robust Standard Deviation to the Robust Standard Deviation of the pass, and end each side
of an extent where the residual stays within the event end level for the event end time,
since the local Robust Standard Deviation lags the signal by its own noise window at both
ends of an event. The impulse pass recognizes an event by the residual of one rate sample
and bounds its extent by the shape of the residual around that rate sample.

Every time parameter is a time on the track in milliseconds, never a count of rate
samples, since the measurement rate differs between cameras.
"""

import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gyro_doctor import errors
from gyro_doctor.core import decode_stage, robust_local_fit
from gyro_doctor.core.axes import AXES, axis_column

# Rate samples per vectorized block of the local Robust Standard Deviation. The gather
# holds this many times the window width in floats, so at a window of about 200 rate
# samples it bounds the working memory of one axis at about 32 MB.
LOCAL_ROBUST_STANDARD_DEVIATION_BLOCK = 20_000

LARGE_AMPLITUDE_OSCILLATION = "large amplitude oscillation"
MID_AMPLITUDE_BURST = "mid-amplitude burst"
IMPULSE_EVENT = "impulse event"

# The passes in the order they run. Each measures against the Robust Standard Deviation
# left after the passes before it removed their extents from the population.
PASSES = (
    {
        "pass": 1,
        "name": "oscillation pass",
        "classification": LARGE_AMPLITUDE_OSCILLATION,
        "threshold_parameter": "detect_oscillation_threshold",
        "kind": "ratio",
        "measured_against": "full track Robust Standard Deviation",
        "threshold_label": (
            "oscillation threshold, local Robust Standard Deviation over that"
        ),
        "trigger_label": "trigger runs past the oscillation threshold",
    },
    {
        "pass": 2,
        "name": "burst pass",
        "classification": MID_AMPLITUDE_BURST,
        "threshold_parameter": "detect_burst_threshold",
        "kind": "ratio",
        "measured_against": "Robust Standard Deviation of the burst pass",
        "threshold_label": "burst threshold, local Robust Standard Deviation over that",
        "trigger_label": (
            "trigger runs past the burst threshold that begin outside earlier extents"
        ),
    },
    {
        "pass": 3,
        "name": "impulse pass",
        "classification": IMPULSE_EVENT,
        "threshold_parameter": "detect_impulse_threshold",
        "kind": "impulse",
        "measured_against": "Robust Standard Deviation of the impulse pass",
        "threshold_label": "impulse threshold, in Robust Standard Deviations",
        "trigger_label": "peaks accepted by the impulse event criteria",
    },
)

LOCAL_MEDIAN_DESCRIPTION = "robust local linear regression"

# The number of fits of the local median, fixed by the method rather than offered as a
# parameter: robust_local_fit.fit counts the fit by distance as the first.
LOCAL_MEDIAN_FITS = 2

EXTENT_TABLE_HEADER = (
    "identifier,first_sample,last_sample,rate_sample_count,start_s,end_s,duration_ms,"
    "found_by,classification,peak_sample,peak_video_frame,peak_rate_deg_s\n"
)

REPORTED_PERCENTILES = (50, 75, 90, 99, 100)

# The parts of the decode result header that the detection result carries, so that a
# cached detection result names the video and the derivation it belongs to.
HEADER_PARTS = ("video_name", "video_size_bytes", "video_mtime_s", "derivation_version")


@dataclass(frozen=True)
class DetectionParameters:
    """The detection parameters, one field per detection parameter option, named after
    the option and holding its default.

    Each threshold holds one value for all three axes or three values in the order
    pitch, yaw, roll. The thresholds, the event end level, and the shoulder drop are
    multiples of a Robust Standard Deviation; every other parameter is a time in
    milliseconds."""

    detect_oscillation_threshold: tuple[float, ...] = (100.0,)
    detect_burst_threshold: tuple[float, ...] = (10.0,)
    detect_impulse_threshold: tuple[float, ...] = (4.5,)
    local_median_window: float = 100.0
    detect_noise_window: float = 100.0
    detect_event_end_level: float = 2.0
    detect_event_end_time: float = 10.0
    detect_impulse_min_width: float = 3.0
    detect_impulse_max_width: float = 25.0
    detect_impulse_shoulder_drop: float = 1.0
    detect_impulse_shoulder_time: float = 12.0
    detect_impulse_side_end_time: float = 5.0
    detect_impulse_widen: float = 5.0
    detect_impulse_widen_max: float = 50.0


def runs_of(mask):
    """Start and end index of every contiguous True run in ``mask``, inclusive."""
    edges = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1)
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [mask.size - 1]])
    return starts, ends


def widen_impulse_extents(
    first,
    last,
    scaled_residual,
    boundaries,
    event_end_level,
    widening_step_ms,
    maximum_widening_ms,
):
    """Widen each end of each impulse extent outward from its shoulder.

    The first widening step is taken whatever the residual at the shoulder, and each
    further step only while the outermost rate sample still lies outside the event end
    level. An end stops at the maximum widening whether or not the residual has returned
    within the event end level."""
    rate_sample_count = scaled_residual.size
    start = np.empty(first.size, dtype=np.int64)
    end = np.empty(first.size, dtype=np.int64)

    for k in range(first.size):
        for side, edge_time_s, sign in (
            ("start", boundaries[first[k]], -1.0),
            ("end", boundaries[last[k] + 1], +1.0),
        ):
            widened, outermost = 0.0, first[k] if side == "start" else last[k]
            while widened < maximum_widening_ms - 1e-9:
                widened = min(widened + widening_step_ms, maximum_widening_ms)
                target = edge_time_s + sign * widened * 1e-3
                outermost = int(
                    np.clip(
                        np.searchsorted(boundaries, target, "right") - 1,
                        0,
                        rate_sample_count - 1,
                    )
                )
                if abs(scaled_residual[outermost]) <= event_end_level:
                    break
            if side == "start":
                start[k] = outermost
            else:
                end[k] = outermost
    return start, end


def merge_overlapping(start, end):
    """Extents that overlap are one extent: ``[first, last]`` pairs in order."""
    merged = []
    for k in np.argsort(start):
        first, last = int(start[k]), int(end[k])
        if merged and first <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], last)
        else:
            merged.append([first, last])
    return merged


def impulse_event_candidates(scaled_residual, threshold):
    """One peak per run of rate samples beyond the impulse threshold: the rate sample of
    the run that lies farthest from the local median."""
    beyond = np.abs(scaled_residual) > threshold
    starts, ends = runs_of(beyond)
    return np.array(
        [
            int(start + np.argmax(np.abs(scaled_residual[start : end + 1])))
            for start, end in zip(starts, ends)
        ]
    )


def follow_side_to_shoulder(
    residual,
    boundaries,
    peak,
    direction,
    pass_robust_standard_deviation,
    shoulder_drop,
    shoulder_time_ms,
    event_end_level,
    side_end_time_ms,
):
    """Follow one side of the V outward from the peak and return its shoulder.

    The side continues while the residual keeps moving away from the peak, even across
    the local median. It ends at the first point from which the residual turns back by at
    least ``shoulder_drop`` Robust Standard Deviations within ``shoulder_time_ms``. A turn
    that takes longer than the shoulder time is a drift of the residual: the point it
    turned from can no longer end the side, and the side continues from where the
    residual now is. The side also ends once the residual has stayed within the event end
    level for ``side_end_time_ms``. In either case the shoulder is the farthest point the
    side reached since the peak or since the last slow reversal, not the point at which
    the test fired."""
    rate_sample_count = residual.size
    sign = 1.0 if residual[peak] < 0 else -1.0  # the sides of a peak below zero rise
    shoulder_time_s = shoulder_time_ms * 1e-3
    side_end_time_s = side_end_time_ms * 1e-3
    farthest_residual, farthest = residual[peak], peak
    within_since = None
    i = peak
    while True:
        i += direction
        if not (0 <= i < rate_sample_count):
            return farthest
        if abs(residual[i] / pass_robust_standard_deviation) <= event_end_level:
            if within_since is None:
                within_since = i
            elif abs(boundaries[i] - boundaries[within_since]) >= side_end_time_s:
                return farthest
        else:
            within_since = None
        if sign * (residual[i] - farthest_residual) > 0:
            farthest_residual, farthest = residual[i], i
            continue
        retrace = sign * (farthest_residual - residual[i])
        elapsed = abs(boundaries[i] - boundaries[farthest])
        if elapsed <= shoulder_time_s:
            if retrace >= shoulder_drop * pass_robust_standard_deviation:
                return farthest
        else:
            farthest_residual, farthest = residual[i], i


def peak_sample(first, last, scaled_residual):
    """The rate sample of largest departure from the local median inside one extent.

    Not the largest absolute rate: a burst during a fast roll would otherwise report the
    peak of the roll."""
    return first + int(np.argmax(np.abs(scaled_residual[first : last + 1])))


def robust_standard_deviation_outside_extents(residual, claimed):
    """1.4826 times the median of the absolute residual over the rate samples outside
    every extent of the passes run so far."""
    return robust_local_fit.MAD_TO_SIGMA * float(np.median(np.abs(residual[~claimed])))


def local_robust_standard_deviations(residual, times, noise_window_ms):
    """The local Robust Standard Deviation at every rate sample: 1.4826 times the median
    of the absolute residual over every rate sample within the noise window on either
    side of it, including those inside an extent."""
    rate_sample_count = residual.size
    local_robust_standard_deviation = np.empty(rate_sample_count)
    absolute = np.abs(residual)

    for start in range(0, rate_sample_count, LOCAL_ROBUST_STANDARD_DEVIATION_BLOCK):
        stop = min(start + LOCAL_ROBUST_STANDARD_DEVIATION_BLOCK, rate_sample_count)
        block_times = times[start:stop]
        window_start = np.searchsorted(
            times, block_times - noise_window_ms * 1e-3, side="left"
        )
        window_end = np.searchsorted(
            times, block_times + noise_window_ms * 1e-3, side="right"
        )
        widths = window_end - window_start

        for width in np.unique(widths):
            where = np.flatnonzero(widths == width)
            gathered = absolute[window_start[where][:, None] + np.arange(width)]
            local_robust_standard_deviation[start + where] = (
                robust_local_fit.MAD_TO_SIGMA * np.median(gathered, axis=1)
            )
    return local_robust_standard_deviation


def merge_touching(merged):
    """Extents that touch are one extent: one ending at rate sample n and the next
    beginning at n + 1 describe one continuous region, and two rows would hand the repair
    stage two adjacent extents to repair separately."""
    joined = []
    for first, last in merged:
        if joined and first <= joined[-1][1] + 1:
            joined[-1][1] = max(joined[-1][1], last)
        else:
            joined.append([first, last])
    return joined


def extent_rows(
    merged, boundaries, scaled_residual, video_frames, rate, detection_pass
):
    """Every merged extent as one row of the extent table, found by the pass
    ``detection_pass`` of ``PASSES``.

    Rate sample ``i`` spans the interval from ``boundaries[i]`` to ``boundaries[i+1]``,
    so an extent over rate samples ``first`` to ``last`` runs from ``boundaries[first]``
    to ``boundaries[last+1]``."""
    rows = []
    for first, last in merged:
        start_s = float(boundaries[first])
        end_s = float(boundaries[last + 1])
        peak = peak_sample(first, last, scaled_residual)
        rows.append(
            {
                "first_sample": int(first),
                "last_sample": int(last),
                "rate_sample_count": int(last - first + 1),
                "start_s": start_s,
                "end_s": end_s,
                "duration_ms": (end_s - start_s) * 1e3,
                "found_by": detection_pass["name"],
                "classification": detection_pass["classification"],
                "peak_sample": int(peak),
                "peak_video_frame": float(video_frames[peak]),
                "peak_rate_deg_s": float(rate[peak]),
            }
        )
    return rows


def axis_result_test(axis, rows, found_at, fit, full_track_robust_standard_deviation):
    """Raise an error carrying ``axis`` on any violation of what the detection of one
    axis guarantees to the repair stage: extents that never overlap and never touch
    within one classification, each peak inside its own extent, an extent for every
    event a pass counted, a local median fit with weight in every window and a scale for
    every difference, and a full track Robust Standard Deviation that a threshold can be
    a multiple of.

    ``found_at`` holds the rate sample at which each pass counted each event: the first
    rate sample of each trigger run of the oscillation pass and the burst pass that
    begins outside earlier extents, and each accepted peak of the impulse pass."""
    for previous, row in itertools.pairwise(rows):
        extents = {
            "axis": axis,
            "first": previous["first_sample"],
            "last": previous["last_sample"],
            "next_first": row["first_sample"],
            "next_last": row["last_sample"],
        }
        if row["first_sample"] <= previous["last_sample"]:
            raise errors.CannotRepairError(errors.ErrorCode.EXTENTS_OVERLAP, **extents)
        # Extents of one classification never touch, since a merge joins them. Extents
        # of two classifications may: an oscillation that decays into a burst is two
        # adjacent regions that the repair stage treats differently.
        if (
            row["first_sample"] == previous["last_sample"] + 1
            and row["classification"] == previous["classification"]
        ):
            raise errors.CannotRepairError(errors.ErrorCode.EXTENTS_TOUCH, **extents)
    for row in rows:
        if not row["first_sample"] <= row["peak_sample"] <= row["last_sample"]:
            raise errors.CannotRepairError(
                errors.ErrorCode.PEAK_OUTSIDE_EXTENT,
                axis=axis,
                first=row["first_sample"],
                last=row["last_sample"],
                peak=row["peak_sample"],
            )
    for sample in found_at:
        if not any(row["first_sample"] <= sample <= row["last_sample"] for row in rows):
            raise errors.CannotRepairError(
                errors.ErrorCode.EVENT_OUTSIDE_EXTENTS,
                axis=axis,
                rate_sample=int(sample),
            )
    if fit.windows_without_weight:
        raise errors.CannotRepairError(
            errors.ErrorCode.LOCAL_MEDIAN_WINDOWS_WITHOUT_WEIGHT,
            count=int(fit.windows_without_weight),
            axis=axis,
        )
    if fit.zero_scale:
        raise errors.CannotRepairError(
            errors.ErrorCode.LOCAL_MEDIAN_ZERO_SCALE,
            count=int(fit.zero_scale),
            axis=axis,
        )
    if (
        not np.isfinite(full_track_robust_standard_deviation)
        or full_track_robust_standard_deviation <= 0.0
    ):
        raise errors.CannotRepairError(
            errors.ErrorCode.FULL_TRACK_ROBUST_STANDARD_DEVIATION_NOT_ABOVE_ZERO,
            axis=axis,
            value=full_track_robust_standard_deviation,
        )


def threshold_on_axis(values, axis, parameter):
    """The value of a threshold on one axis: one value applies to all three axes, and
    three values apply to pitch, yaw, and roll in that order. Raises ValueError on any
    other count of values."""
    if len(values) == 1:
        return float(values[0])
    if len(values) != len(AXES):
        raise ValueError(
            f"{parameter} takes one value for all three axes or {len(AXES)} in "
            f"{', '.join(AXES)} order, not {len(values)}"
        )
    return float(values[axis_column(axis)])


def runs_within_event_end_level(
    residual, times, pass_robust_standard_deviation, event_end_level, event_end_time_ms
):
    """Every maximal run of rate samples within ``event_end_level`` Robust Standard
    Deviations of the local median that lasts at least ``event_end_time_ms``.

    An oscillation crosses its own local median every half cycle, so the residual is
    briefly within the event end level many times within one event. Only a run that
    stays there for the event end time can end an extent."""
    inside = np.abs(residual) <= event_end_level * pass_robust_standard_deviation
    starts, ends = runs_of(inside)
    if starts.size == 0:
        return starts, ends
    lasted = (times[ends] - times[starts]) * 1e3 >= event_end_time_ms
    return starts[lasted], ends[lasted]


def ratio_pass_extents(
    ratio,
    residual,
    times,
    threshold,
    pass_robust_standard_deviation,
    event_end_level,
    event_end_time_ms,
):
    """``(merged extents, the first rate sample of every trigger run)`` of the
    oscillation pass or the burst pass.

    The ratio only triggers an extent. Both of its ends are read from the residual,
    because the local Robust Standard Deviation is measured over the noise window and
    therefore lags the signal by up to that window at both ends of an event."""
    past = ratio > threshold
    if not past.any():
        return [], np.zeros(0, dtype=np.int64)
    trigger_first, trigger_last = runs_of(past)
    quiet_first, quiet_last = runs_within_event_end_level(
        residual,
        times,
        pass_robust_standard_deviation,
        event_end_level,
        event_end_time_ms,
    )
    start = np.empty(trigger_first.size, dtype=np.int64)
    end = np.empty(trigger_first.size, dtype=np.int64)
    for k in range(trigger_first.size):
        before = quiet_last[quiet_last <= trigger_first[k]]
        start[k] = int(before[-1]) if before.size else 0
        after = quiet_first[quiet_first >= trigger_last[k]]
        end[k] = int(after[0]) if after.size else residual.size - 1
    return merge_touching(merge_overlapping(start, end)), trigger_first


def without_claimed_rate_samples(merged, claimed):
    """``merged``, with every rate sample that an earlier pass claimed cut out of it.

    A rate sample that an earlier pass claimed stays with that pass: an extent that
    reaches into an earlier extent keeps only its free part, and an extent wholly
    claimed disappears, so two passes never emit overlapping rows."""
    kept = []
    for first, last in merged:
        span = np.arange(first, last + 1)
        free = ~claimed[span]
        if not free.any():
            continue
        runs_first, runs_last = runs_of(free)
        for first_free, last_free in zip(runs_first, runs_last, strict=True):
            kept.append([int(span[first_free]), int(span[last_free])])
    kept.sort(key=lambda row: row[0])
    return merge_touching(kept)


def run_impulse_pass(
    residual,
    boundaries,
    video_frames,
    rate,
    pass_robust_standard_deviation,
    claimed,
    threshold,
    parameters,
    detection_pass,
):
    """The impulse pass: find each peak, follow both sides of its V to the shoulders,
    apply the minimum width and the maximum width tests, then widen each accepted extent.
    Returns the rows of the extent table and the accepted peaks.

    The sides are followed on the residual, not on the rate, because the sign of the
    residual is the direction of the departure: against a sloped local median, a rate of
    -10 deg/s about a local median of -50 deg/s is a departure upward."""
    scaled_residual = residual / pass_robust_standard_deviation
    if not (np.abs(scaled_residual) > threshold).any():
        return [], np.zeros(0, dtype=np.int64)
    peaks = impulse_event_candidates(scaled_residual, threshold)
    side_arguments = (
        pass_robust_standard_deviation,
        parameters.detect_impulse_shoulder_drop,
        parameters.detect_impulse_shoulder_time,
        parameters.detect_event_end_level,
        parameters.detect_impulse_side_end_time,
    )
    left = np.array(
        [
            follow_side_to_shoulder(
                residual, boundaries, int(peak), -1, *side_arguments
            )
            for peak in peaks
        ]
    )
    right = np.array(
        [
            follow_side_to_shoulder(
                residual, boundaries, int(peak), +1, *side_arguments
            )
            for peak in peaks
        ]
    )
    # Two peaks whose sides end at the same pair of shoulders belong to one V.
    _, unique = np.unique(np.stack([left, right]), axis=1, return_index=True)
    unique = np.sort(unique)
    peaks, left, right = peaks[unique], left[unique], right[unique]

    shoulder_to_shoulder_ms = (boundaries[right + 1] - boundaries[left]) * 1e3
    # The departure is the run around the peak beyond the event end level, which is a
    # different measure from the width of the V: a drop of one rate sample can have
    # shoulders four rate samples out.
    outside = np.abs(scaled_residual) > parameters.detect_event_end_level
    departure_ms = np.empty(peaks.size)
    for k, peak in enumerate(peaks):
        departure_first = departure_last = int(peak)
        while departure_first > 0 and outside[departure_first - 1]:
            departure_first -= 1
        while departure_last + 1 < residual.size and outside[departure_last + 1]:
            departure_last += 1
        departure_ms[k] = (
            boundaries[departure_last + 1] - boundaries[departure_first]
        ) * 1e3

    accepted = (
        (departure_ms >= parameters.detect_impulse_min_width)
        & (shoulder_to_shoulder_ms <= parameters.detect_impulse_max_width)
        & (~claimed[peaks])
    )
    peaks, left, right = peaks[accepted], left[accepted], right[accepted]
    if peaks.size == 0:
        return [], np.zeros(0, dtype=np.int64)

    start, end = widen_impulse_extents(
        left,
        right,
        scaled_residual,
        boundaries,
        parameters.detect_event_end_level,
        parameters.detect_impulse_widen,
        parameters.detect_impulse_widen_max,
    )
    merged = without_claimed_rate_samples(merge_overlapping(start, end), claimed)
    rows = extent_rows(
        merged, boundaries, scaled_residual, video_frames, rate, detection_pass
    )
    return rows, peaks


def run_ratio_pass(
    residual,
    times,
    boundaries,
    video_frames,
    rate,
    local_robust_standard_deviation,
    pass_robust_standard_deviation,
    claimed,
    threshold,
    parameters,
    detection_pass,
):
    """The oscillation pass or the burst pass: trigger on the ratio of the local Robust
    Standard Deviation to the Robust Standard Deviation of the pass, end each extent on
    the residual, and cut out every rate sample an earlier pass claimed. Returns the
    rows of the extent table and the first rate sample of each trigger run counted.

    A trigger run whose first rate sample an earlier pass claimed is not counted."""
    ratio = local_robust_standard_deviation / pass_robust_standard_deviation
    merged, triggers = ratio_pass_extents(
        ratio,
        residual,
        times,
        threshold,
        pass_robust_standard_deviation,
        parameters.detect_event_end_level,
        parameters.detect_event_end_time,
    )
    if triggers.size:
        triggers = triggers[~claimed[triggers]]
    merged = without_claimed_rate_samples(merged, claimed)
    scaled_residual = residual / pass_robust_standard_deviation
    rows = extent_rows(
        merged, boundaries, scaled_residual, video_frames, rate, detection_pass
    )
    return rows, triggers


def axis_settings(parameters, axis):
    """The parameter values the detection of one axis uses, each threshold taken on that
    axis."""
    settings = asdict(parameters)
    for detection_pass in PASSES:
        name = detection_pass["threshold_parameter"]
        settings[name] = threshold_on_axis(settings[name], axis, name)
    return settings


def percentiles(values):
    """The reported percentiles of ``values``; not a number where there is no value."""
    if values.size == 0:
        return np.full(len(REPORTED_PERCENTILES), np.nan)
    return np.array(
        [
            float(np.percentile(values, percentile))
            for percentile in REPORTED_PERCENTILES
        ]
    )


def detect_axis(decode_result, axis, parameters):
    """The detection of one axis: the local median, the local Robust Standard Deviation,
    the Robust Standard Deviation and the threshold of each pass, the extents that the
    three passes find in order, and the quantities the report of the stage prints.

    Reads only its own axis of the Normalized Rate, so the axes are independent units of
    work. Raises the errors of ``axis_result_test``."""
    settings = axis_settings(parameters, axis)
    rate = decode_result["normalized_rate_deg_s"][:, axis_column(axis)]
    boundaries = decode_stage.group_boundary_times(decode_result)
    rate_sample_times_s = decode_result["recorded_rate_times_s"]
    video_frames = rate_sample_times_s * decode_result["video_frame_rate"]

    fit = robust_local_fit.fit(
        rate, rate_sample_times_s, parameters.local_median_window, LOCAL_MEDIAN_FITS
    )
    residual = rate - fit.local_median
    local_robust_standard_deviation = local_robust_standard_deviations(
        residual, rate_sample_times_s, parameters.detect_noise_window
    )

    claimed = np.zeros(rate.size, dtype=bool)
    passes, rows = [], []
    for detection_pass in PASSES:
        pass_robust_standard_deviation = robust_standard_deviation_outside_extents(
            residual, claimed
        )
        threshold = settings[detection_pass["threshold_parameter"]]
        if detection_pass["kind"] == "impulse":
            pass_rows, found_at = run_impulse_pass(
                residual,
                boundaries,
                video_frames,
                rate,
                pass_robust_standard_deviation,
                claimed,
                threshold,
                parameters,
                detection_pass,
            )
        else:
            pass_rows, found_at = run_ratio_pass(
                residual,
                rate_sample_times_s,
                boundaries,
                video_frames,
                rate,
                local_robust_standard_deviation,
                pass_robust_standard_deviation,
                claimed,
                threshold,
                parameters,
                detection_pass,
            )
        for row in pass_rows:
            claimed[row["first_sample"] : row["last_sample"] + 1] = True
        peak_samples = np.array(
            [row["peak_sample"] for row in pass_rows], dtype=np.int64
        )
        departure_percentiles = percentiles(
            np.abs(residual[peak_samples]) if pass_rows else np.zeros(0)
        )
        passes.append(
            {
                "robust_standard_deviation": pass_robust_standard_deviation,
                "threshold": threshold,
                "extent_count": len(pass_rows),
                "trigger_count": int(found_at.size),
                "found_at": found_at,
                "share": float(
                    sum(row["rate_sample_count"] for row in pass_rows) / rate.size
                ),
                "departure_percentiles": departure_percentiles,
                "departure_percentiles_in_robust_standard_deviations": (
                    departure_percentiles / pass_robust_standard_deviation
                ),
                "duration_percentiles": percentiles(
                    np.array([row["duration_ms"] for row in pass_rows])
                ),
            }
        )
        rows += pass_rows

    rows.sort(key=lambda row: row["first_sample"])
    full_track_robust_standard_deviation = passes[0]["robust_standard_deviation"]
    clean_robust_standard_deviation = robust_standard_deviation_outside_extents(
        residual, claimed
    )
    axis_result_test(
        axis,
        rows,
        np.concatenate(
            [pass_quantities["found_at"] for pass_quantities in passes]
            + [np.zeros(0, dtype=np.int64)]
        ),
        fit,
        full_track_robust_standard_deviation,
    )

    return {
        "axis": axis,
        "settings": settings,
        "rate_sample_times_s": rate_sample_times_s,
        "rate_sample_video_frames": video_frames,
        "local_median_deg_s": fit.local_median,
        "local_robust_standard_deviation_deg_s": local_robust_standard_deviation,
        "full_track_robust_standard_deviation_deg_s": (
            full_track_robust_standard_deviation
        ),
        "clean_robust_standard_deviation_deg_s": clean_robust_standard_deviation,
        "pass_robust_standard_deviation_deg_s": [
            pass_quantities["robust_standard_deviation"] for pass_quantities in passes
        ],
        "pass_threshold": [pass_quantities["threshold"] for pass_quantities in passes],
        "pass_extent_count": [
            pass_quantities["extent_count"] for pass_quantities in passes
        ],
        "extent_count": len(rows),
        "pass_trigger_count": [
            pass_quantities["trigger_count"] for pass_quantities in passes
        ],
        "pass_share_of_rate_samples_inside_extents": [
            pass_quantities["share"] for pass_quantities in passes
        ],
        "share_of_rate_samples_inside_extents": float(claimed.mean()),
        "pass_peak_departure_percentiles_deg_s": [
            pass_quantities["departure_percentiles"] for pass_quantities in passes
        ],
        "pass_peak_departure_percentiles_in_robust_standard_deviations": [
            pass_quantities["departure_percentiles_in_robust_standard_deviations"]
            for pass_quantities in passes
        ],
        "pass_extent_duration_percentiles_ms": [
            pass_quantities["duration_percentiles"] for pass_quantities in passes
        ],
        "rows": rows,
    }


def selected_axes(axes):
    """The selected axes in the order pitch, yaw, roll. Raises ValueError on an unknown
    axis, a repeated axis, and an empty selection."""
    axes = tuple(axes)
    for axis in axes:
        axis_column(axis)
    if not axes or len(set(axes)) != len(axes):
        raise ValueError(
            f"the detection stage takes one or more distinct axes of {', '.join(AXES)}, "
            f"not {', '.join(axes) or 'none'}"
        )
    return tuple(axis for axis in AXES if axis in axes)


def extent_identifier(axis, peak_video_frame):
    """The identifier of an extent: the letter of its axis, ``P``, ``Y``, or ``R``,
    followed by the video frame of its peak, the video frame shown at the time of the
    peak, such as ``Y1502``."""
    return f"{axis[0].upper()}{int(np.floor(peak_video_frame))}"


def detection_result(decode_result, per_axis):
    """The detection result of the detections ``per_axis`` of ``detect_axis``: the
    header of the decode result, every per axis series stacked in columns in the order
    of the part ``axes``, the Robust Standard Deviations and the report quantities of
    each axis and pass, the settings, and every extent of every axis in one table."""

    def stacked(name):
        return np.stack([axis_detection[name] for axis_detection in per_axis], axis=1)

    def by_axis(name, dtype=None):
        return np.array(
            [axis_detection[name] for axis_detection in per_axis], dtype=dtype
        )

    first = per_axis[0]
    extents = [
        (axis_detection["axis"], row)
        for axis_detection in per_axis
        for row in axis_detection["rows"]
    ]

    def column(name, dtype):
        return np.array([row[name] for _axis, row in extents], dtype=dtype)

    return {
        **{name: decode_result[name] for name in HEADER_PARTS},
        "axes": by_axis("axis", str),
        "settings": json.dumps(
            {
                axis_detection["axis"]: axis_detection["settings"]
                for axis_detection in per_axis
            }
        ),
        "rate_sample_times_s": first["rate_sample_times_s"],
        "rate_sample_video_frames": first["rate_sample_video_frames"],
        "local_median_deg_s": stacked("local_median_deg_s"),
        "local_robust_standard_deviation_deg_s": stacked(
            "local_robust_standard_deviation_deg_s"
        ),
        "full_track_robust_standard_deviation_deg_s": by_axis(
            "full_track_robust_standard_deviation_deg_s"
        ),
        "clean_robust_standard_deviation_deg_s": by_axis(
            "clean_robust_standard_deviation_deg_s"
        ),
        "pass_robust_standard_deviation_deg_s": by_axis(
            "pass_robust_standard_deviation_deg_s"
        ),
        "pass_threshold": by_axis("pass_threshold"),
        "pass_extent_count": by_axis("pass_extent_count", np.int64),
        "extent_count": by_axis("extent_count", np.int64),
        "pass_trigger_count": by_axis("pass_trigger_count", np.int64),
        "pass_share_of_rate_samples_inside_extents": by_axis(
            "pass_share_of_rate_samples_inside_extents"
        ),
        "share_of_rate_samples_inside_extents": by_axis(
            "share_of_rate_samples_inside_extents"
        ),
        "reported_percentiles": np.array(REPORTED_PERCENTILES),
        "pass_peak_departure_percentiles_deg_s": by_axis(
            "pass_peak_departure_percentiles_deg_s"
        ),
        "pass_peak_departure_percentiles_in_robust_standard_deviations": by_axis(
            "pass_peak_departure_percentiles_in_robust_standard_deviations"
        ),
        "pass_extent_duration_percentiles_ms": by_axis(
            "pass_extent_duration_percentiles_ms"
        ),
        "extent_identifier": np.array(
            [extent_identifier(axis, row["peak_video_frame"]) for axis, row in extents],
            dtype=str,
        ),
        "extent_axis": np.array([axis for axis, _row in extents], dtype=str),
        "extent_first_sample": column("first_sample", np.int64),
        "extent_last_sample": column("last_sample", np.int64),
        "extent_start_s": column("start_s", np.float64),
        "extent_end_s": column("end_s", np.float64),
        "extent_peak_sample": column("peak_sample", np.int64),
        "extent_peak_video_frame": column("peak_video_frame", np.float64),
        "extent_peak_rate_deg_s": column("peak_rate_deg_s", np.float64),
        "extent_classification": column("classification", str),
        "extent_found_by": column("found_by", str),
    }


def detect(decode_result, axes, parameters):
    """The detection stage: the detection result of the decode result on the selected
    axes, with the detection parameters given.

    The selected axes run in parallel, one worker per axis. Raises the errors of
    ``selected_axes`` and of ``detect_axis``."""
    axes = selected_axes(axes)
    with ThreadPoolExecutor(max_workers=len(axes)) as pool:
        per_axis = list(
            pool.map(lambda axis: detect_axis(decode_result, axis, parameters), axes)
        )
    return detection_result(decode_result, per_axis)


DETECTION_RESULT_PARTS = (
    *HEADER_PARTS,
    "axes",
    "settings",
    "rate_sample_times_s",
    "rate_sample_video_frames",
    "local_median_deg_s",
    "local_robust_standard_deviation_deg_s",
    "full_track_robust_standard_deviation_deg_s",
    "clean_robust_standard_deviation_deg_s",
    "pass_robust_standard_deviation_deg_s",
    "pass_threshold",
    "pass_extent_count",
    "extent_count",
    "pass_trigger_count",
    "pass_share_of_rate_samples_inside_extents",
    "share_of_rate_samples_inside_extents",
    "reported_percentiles",
    "pass_peak_departure_percentiles_deg_s",
    "pass_peak_departure_percentiles_in_robust_standard_deviations",
    "pass_extent_duration_percentiles_ms",
    "extent_identifier",
    "extent_axis",
    "extent_first_sample",
    "extent_last_sample",
    "extent_start_s",
    "extent_end_s",
    "extent_peak_sample",
    "extent_peak_video_frame",
    "extent_peak_rate_deg_s",
    "extent_classification",
    "extent_found_by",
)


def axis_index(result, axis):
    """The column of ``axis`` in every per axis array of a detection result."""
    return [str(stored_axis) for stored_axis in result["axes"]].index(axis)


def extent_table_text(result, axis):
    """The extent table of one axis, one row per extent in order of its first rate
    sample, as CSV text."""
    on_axis = np.flatnonzero(np.asarray(result["extent_axis"]) == axis)
    lines = [EXTENT_TABLE_HEADER]
    for k in on_axis:
        first = int(result["extent_first_sample"][k])
        last = int(result["extent_last_sample"][k])
        start_s = float(result["extent_start_s"][k])
        end_s = float(result["extent_end_s"][k])
        lines.append(
            f"{result['extent_identifier'][k]},{first},{last},{last - first + 1},"
            f"{start_s:.6f},{end_s:.6f},{(end_s - start_s) * 1e3:.4f},"
            f"{result['extent_found_by'][k]},{result['extent_classification'][k]},"
            f"{int(result['extent_peak_sample'][k])},"
            f"{float(result['extent_peak_video_frame'][k]):.3f},"
            f"{float(result['extent_peak_rate_deg_s'][k]):.4f}\n"
        )
    return "".join(lines)


def detection_result_path(video_cache_directory, video_name):
    """The path of the detection result of the video with file stem ``video_name`` in
    ``video_cache_directory``."""
    return Path(video_cache_directory) / f"{video_name}-detection-result.npz"


def run_record_path(video_cache_directory, video_name):
    """The path of the run record of the video with file stem ``video_name`` in
    ``video_cache_directory``."""
    return Path(video_cache_directory) / f"{video_name}-detection-run-record.json"


def extent_table_path(video_cache_directory, video_name, axis):
    """The path of the extent table of ``axis`` of the video with file stem
    ``video_name`` in ``video_cache_directory``."""
    return Path(video_cache_directory) / f"{video_name}-{axis}-extent-table.csv"


def run_record(result, result_path, extent_table_paths, file_stamps):
    """The run record: for each axis the detector processed, the Robust Standard
    Deviation each pass measured against, each threshold, the number of extents of each
    classification, the share of rate samples inside an extent, and the extent table, by
    file name, with the file name of the detection result and the file stamp of the
    decode result the stage read."""
    axes = {}
    for axis in (str(stored_axis) for stored_axis in result["axes"]):
        column = axis_index(result, axis)
        axes[axis] = {
            "passes": [
                {
                    "pass": detection_pass["pass"],
                    "name": detection_pass["name"],
                    "classification": detection_pass["classification"],
                    "robust_standard_deviation_deg_s": float(
                        result["pass_robust_standard_deviation_deg_s"][column][p]
                    ),
                    "threshold": float(result["pass_threshold"][column][p]),
                    "extent_count": int(result["pass_extent_count"][column][p]),
                    "share_of_rate_samples_inside_extents": float(
                        result["pass_share_of_rate_samples_inside_extents"][column][p]
                    ),
                }
                for p, detection_pass in enumerate(PASSES)
            ],
            "share_of_rate_samples_inside_extents": float(
                result["share_of_rate_samples_inside_extents"][column]
            ),
            "extent_table": extent_table_paths[axis].name,
        }
    return {
        "detection_result": result_path.name,
        "file_stamps": file_stamps,
        "axes": axes,
    }


def write_detection_result(result, video_cache_directory, *, file_stamps):
    """The cache writer of the detection stage: store the detection result in
    ``video_cache_directory`` as ``<video file stem>-detection-result.npz``, the run
    record as ``<video file stem>-detection-run-record.json``, and the extent table of
    each axis as ``<video file stem>-<axis>-extent-table.csv``.

    ``file_stamps`` maps ``"decode_result"`` to the file stamp of the decode result the
    stage read. The stored detection result holds it as JSON text under ``file_stamps``,
    and so does the run record; the detection result in memory is left as it is.

    Returns the path of the extent table of each axis, by axis. An error of the operating
    system while it writes a file raises an error carrying the path of that file."""
    video_cache_directory = Path(video_cache_directory)
    stem = str(result["video_name"])
    result_path = detection_result_path(video_cache_directory, stem)
    with errors.operating_system_error_as(errors.CannotWriteError, result_path):
        video_cache_directory.mkdir(parents=True, exist_ok=True)
        decode_stage.write_npz(
            result_path,
            {**result, "file_stamps": json.dumps(file_stamps, sort_keys=True)},
        )
    extent_table_paths = {}
    for axis in (str(stored_axis) for stored_axis in result["axes"]):
        path = extent_table_path(video_cache_directory, stem, axis)
        text = extent_table_text(result, axis)
        with errors.operating_system_error_as(errors.CannotWriteError, path):
            path.write_text(text, encoding="utf-8")
        extent_table_paths[axis] = path
    record_path = run_record_path(video_cache_directory, stem)
    record_text = (
        json.dumps(
            run_record(result, result_path, extent_table_paths, file_stamps), indent=2
        )
        + "\n"
    )
    with errors.operating_system_error_as(errors.CannotWriteError, record_path):
        record_path.write_text(record_text, encoding="utf-8")
    return extent_table_paths


def read_detection_result(path):
    """The cache reader of the detection stage: the detection result stored at ``path``,
    with its ``file_stamps``, returned only when every part of the detection result and
    the file stamps are present.

    Every error it raises on purpose is ``errors.CannotReadError`` and carries ``path``
    as ``file``. The test runs after the read, so an error a defect raises in it stays a
    traceback."""
    path = Path(path)
    with (
        decode_stage.cache_file_read(path),
        np.load(path, allow_pickle=False) as stored,
    ):
        result = {key: stored[key] for key in stored.files}
    decode_stage.every_part_present_test(
        result, path, (*DETECTION_RESULT_PARTS, "file_stamps")
    )
    return result


def same_parameters(stored, axes, parameters):
    """Whether the detection result ``stored`` was computed on ``axes`` with the
    detection parameters ``parameters``.

    ``stored`` needs only the parts ``axes`` and ``settings``, so a caller can pass the
    stored detection result without reading its arrays. Raises the errors of
    ``selected_axes``."""
    axes = selected_axes(axes)
    if [str(stored_axis) for stored_axis in stored["axes"]] != list(axes):
        return False
    return json.loads(str(stored["settings"])) == {
        axis: axis_settings(parameters, axis) for axis in axes
    }
