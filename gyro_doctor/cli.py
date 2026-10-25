# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""``gyro-doctor`` repairs the gyro data track of a DJI O4 video and saves the
repaired video alongside the original.

The ``gyro-doctor`` command, the command line of the product: it parses the options,
resolves the cache, runs the decode stage, the detection stage, the repair stage, and
the write stage in that order, skipping each stage whose cached result is valid, and
prints the line or the report of each stage from the quantities the stage returns, with
no computation of its own. It also holds the grammar of a position, a range, and a
duration in the video.
"""

import argparse
import importlib.metadata
import json
import logging
import math
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np

from gyro_doctor import errors
from gyro_doctor.core import (
    ceiling_attenuator,
    decode_result_reader,
    decode_stage,
    detection_stage,
    repair_stage,
    telemetry,
    write_stage,
)
from gyro_doctor.core.axes import AXES

log = logging.getLogger(__name__)

DECODE_REPORT_HEADING = "decode stage"
DETECTION_REPORT_HEADING = "detection stage"
REPAIR_REPORT_HEADING = "repair stage"
WRITE_REPORT_HEADING = "write stage"
CACHED_RESULT = ": result read from the cache:"

FILES_HEADING = "files"


def main(argv=None):
    """Run ``gyro-doctor`` on the command line arguments ``argv``, which default to
    ``sys.argv[1:]``, and return the exit status: 0 when the run completes, and 1 when
    the run stops with an error.

    The arguments are the video path and every option of ``gyro-doctor``, under its
    name, default, and meaning. The help output groups them under the headings
    "positional arguments", "options", "cache related & charting only run options",
    "detection tuning parameters", "repair tuning parameters", "large amplitude
    oscillation repair parameters", "mid-amplitude burst repair parameters", "impulse
    repair parameters", and "advanced tuning parameters", in that order. ``--help`` leaves out ``--report``
    and the group "advanced tuning parameters"; ``--help-all`` lists every option.

    Where each outcome goes:

    - A mistake in the options that argparse finds, or that a type function raises an
      error for, goes through argparse in its own form, which writes the usage and the
      error line to stderr and raises ``SystemExit(2)`` before any stage runs: more
      than one of ``--decode-only``, ``--detect-only``, and ``--repair-only``, and a
      list, threshold, number, or choice that does not parse, or a number that is not
      finite, except a ceiling of ``inf``. ``--help`` and ``--help-all`` write the help
      to stdout and raise ``SystemExit(0)``. ``--version`` writes the name of the
      command and the version of the package to stdout and raises ``SystemExit(0)``.
    - The test of the options after parsing raises an error that goes through
      ``parser.error``, which writes the usage and the error line to stderr and raises
      ``SystemExit(2)`` before the cache is resolved or any stage runs: a value outside
      its range; ``--no-cache`` with ``--decode-only``, ``--detect-only``, or
      ``--repair-only``; an explicit ``--repair-axes`` that names an axis
      ``--detect-axes`` does not; a video path that is not a file; and an ``--output``
      that resolves to the video, names a directory, or lies in a directory that does
      not exist.
    - An error of the run, ``errors.GyroDoctorError``, is logged through
      ``errors.log_error`` with its error line and any hint line at ERROR level through
      the logger ``gyro_doctor.cli``, and ``main`` returns 1. A ``--do-not-repair`` file
      that cannot be read is such an error, before any stage runs. Without ``--force``
      the error of a video whose stored camera model name ends in ``"-Dr."`` comes
      before the cache is read.
    - When ``--remove-gyro-motion`` names every axis, a warning is logged at WARNING
      level through ``gyro_doctor.cli``. The run then removes the gyro motion of all
      three axes.
    - When ``--repair-axes`` is at its default and ``--detect-axes`` names fewer than
      three axes, a warning is logged at WARNING level through ``gyro_doctor.cli``, and
      the run repairs the detected axes alone.
    - An element of the do not repair list that does not parse is logged as a warning
      at WARNING level through ``gyro_doctor.cli`` before any stage runs. On a run that
      reaches the repair stage, these elements are logged the same way, before the
      repair stage runs or its line prints:

      - an identifier that names no extent;
      - a range of a time and a video frame whose last video frame comes before its
        first;
      - an identifier, or a range after an axis letter, on an axis that
        ``--remove-gyro-motion`` names.

      The run continues without the element.
    - A cached result that fails the header test, a file stamp test, or a test of its
      cache reader, lacks a part, or is damaged, is logged as a warning at WARNING level
      through ``gyro_doctor.cli`` with the error it raised, and its stage runs again and
      overwrites it. A cached result computed with other parameters is not valid, and the
      stage runs again with no warning. Once a stage runs, every later stage runs.
    - By default the run prints nothing on stdout.
    - With ``--verbose`` the run prints one line for each stage that ran or whose
      result it read from the cache, in stage order. A stage that ran prints the line
      that ``decode_report_line``, ``detection_report_line``, ``repair_report_line``, or
      ``write_report_line`` returns, which opens with the heading of the stage. A stage
      whose result the run read from the cache prints its heading, ``CACHED_RESULT``,
      and the path it read.
    - With ``--report``, with or without ``--verbose``, the run prints the report of
      each stage that ran, which opens with the heading of the stage, and the line of
      each stage whose result it read from the cache. With the cache on, the reports
      end with ``FILES_HEADING`` and the path of every file in the cache that the run
      wrote or read, one per line.
    """
    logging.basicConfig(format="gyro-doctor: %(message)s")
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        output_path, repair_axes = _options_test(args)
    except errors.GyroDoctorError as error:
        parser.error(str(error))
    try:
        _run(args, output_path, repair_axes)
    except errors.GyroDoctorError as error:
        errors.log_error(log, error)
        return 1
    return 0


def decode_report(result, path, readme):
    """The report of the decode stage, as a list of lines, formatted from the decode
    result. ``path`` and ``readme`` are the paths of the decode result and of the
    ``README.md`` that holds its inventory row, or None when the cache is off."""
    lines = [DECODE_REPORT_HEADING]
    written = (
        f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB): "
        if path is not None
        else "  "
    )
    lines.append(
        f"{written}"
        f"{len(result['quaternion_times_s'])} quaternions, "
        f"{len(result['recorded_rate_times_s'])} rate samples, "
        f"{result['quaternion_rate_hz']:.2f} Hz, "
        f"{result['quaternions_per_video_frame']:.4f} quaternions per video frame, "
        f"{len(result['quaternion_count_per_telemetry_frame'])} telemetry frames, "
        f"period T {result['period_s'] * 1e6:.4f} us"
    )
    if readme is not None:
        lines.append(f"  recorded its row in {readme}")
    return lines


def decode_report_line(result):
    """The line of the decode stage that ``--verbose`` prints: the numbers of
    quaternions and rate samples, the quaternion rate, and the period T, formatted from
    the decode result."""
    return (
        f"{DECODE_REPORT_HEADING}: "
        f"{len(result['quaternion_times_s'])} quaternions, "
        f"{len(result['recorded_rate_times_s'])} rate samples, "
        f"{result['quaternion_rate_hz']:.2f} Hz, "
        f"period T {result['period_s'] * 1e6:.4f} us"
    )


def write_report(video, output_path, report):
    """The report of the write stage, as a list of lines: the path and size of the
    video, the path of the repaired video, the number of quaternions the write
    overwrote, and the number of telemetry frames it rewrote, from the
    ``WritebackReport`` of the write."""
    return [
        WRITE_REPORT_HEADING,
        f"  input video                 {video}",
        f"  size of the input video     {report.video_size_bytes} bytes",
        f"  repaired video              {output_path}",
        f"  quaternions overwritten     {report.quaternion_messages_overwritten}",
        f"  telemetry frames rewritten  {report.telemetry_frames_rewritten}",
    ]


def write_report_line(output_path, report):
    """The line of the write stage that ``--verbose`` prints: the path of the repaired
    video, ``output_path``, and the number of quaternions the write overwrote, from
    ``report``, the ``WritebackReport`` of the write."""
    return (
        f"{WRITE_REPORT_HEADING}: wrote {output_path}, "
        f"{report.quaternion_messages_overwritten} quaternions overwritten"
    )


def axis_report(result, axis, extent_table_path, video_frame_rate):
    """The report of one axis of the detection stage, as a list of lines, formatted from
    the quantities in the detection result. ``extent_table_path`` is the path of the
    extent table of the axis, or None when the cache is off.

    The report lists each extent of the axis with its identifier, its classification,
    and its start and end as a range of times and as a range of video frames, each in
    the grammar of a range. The detection result holds no video frame rate, so the
    caller passes ``video_frame_rate``, the video frame rate of the decode result."""
    axis_index = detection_stage.axis_index(result, axis)
    settings = json.loads(str(result["settings"]))[axis]
    rate_sample_count = int(np.asarray(result["rate_sample_times_s"]).size)
    lines = [
        f"axis {axis}   Normalized Rate",
        (
            f"local median: {detection_stage.LOCAL_MEDIAN_DESCRIPTION}, local median window +/-"
            f"{settings['local_median_window']:g} ms"
        ),
        "Robust Standard Deviation of a pass: 1.4826 x median(|residual|) over every",
        "       rate sample of the axis that no earlier pass placed inside an extent",
        "",
        f"  {'rate samples':<72} {rate_sample_count:>14}",
        f"  {'noise window, +/- ms':<72} {settings['detect_noise_window']:>14g}",
        (
            f"  {'event end level, in Robust Standard Deviations':<72} "
            f"{settings['detect_event_end_level']:>14g}"
        ),
        f"  {'event end time, ms':<72} {settings['detect_event_end_time']:>14g}",
        "",
    ]

    for p, detection_pass in enumerate(detection_stage.PASSES):
        pass_robust_standard_deviation = float(
            result["pass_robust_standard_deviation_deg_s"][axis_index][p]
        )
        extent_count = int(result["pass_extent_count"][axis_index][p])
        share = float(
            result["pass_share_of_rate_samples_inside_extents"][axis_index][p]
        )
        measured_against = f"  measures against {detection_pass['measured_against']}"
        lines += [
            (
                f"  pass {detection_pass['pass']}, looking for: "
                f"{detection_pass['classification']}"
            ),
            (
                f"  {measured_against + ', deg/s':<72} "
                f"{pass_robust_standard_deviation:>14.4f}"
            ),
            (
                f"  {'  ' + detection_pass['threshold_label']:<72} "
                f"{float(result['pass_threshold'][axis_index][p]):>14g}"
            ),
            (
                f"  {'  ' + detection_pass['trigger_label']:<72} "
                f"{int(result['pass_trigger_count'][axis_index][p]):>14}"
            ),
            (
                f"  {'  extents after merging those that overlap or touch':<72} "
                f"{extent_count:>14}"
            ),
            f"  {'  share of samples inside an extent':<72} {100 * share:>13.4f}%",
        ]
        if extent_count:
            lines.append(
                f"    {'percentile':>10} {'peak departure deg/s':>21} "
                f"{'peak departure in Robust Standard Deviations':>45} "
                f"{'extent duration ms':>20}"
            )
            for percentile, departure, in_robust_standard_deviations, duration in zip(
                result["reported_percentiles"],
                result["pass_peak_departure_percentiles_deg_s"][axis_index][p],
                result["pass_peak_departure_percentiles_in_robust_standard_deviations"][
                    axis_index
                ][p],
                result["pass_extent_duration_percentiles_ms"][axis_index][p],
                strict=True,
            ):
                lines.append(
                    f"    {int(percentile):>10g} {float(departure):>21.4f} "
                    f"{float(in_robust_standard_deviations):>45.4f} "
                    f"{float(duration):>20.4f}"
                )
        else:
            lines.append(
                f"    the {detection_pass['name']} found no extent on this axis, "
                f"so there is no distribution to report"
            )
        lines.append("")

    clean_robust_standard_deviation = float(
        result["clean_robust_standard_deviation_deg_s"][axis_index]
    )
    total_share = float(result["share_of_rate_samples_inside_extents"][axis_index])
    lines += [
        (
            f"  {'clean Robust Standard Deviation, deg/s':<72} "
            f"{clean_robust_standard_deviation:>14.4f}"
        ),
        f"  {'share of samples inside any extent':<72} {100 * total_share:>13.4f}%",
        "",
    ]
    on_axis = np.flatnonzero(np.asarray(result["extent_axis"]) == axis)
    if on_axis.size:
        lines.append(
            f"    {'identifier':<10} {'classification':<27} {'start to end time':<23} "
            f"start to end video frame"
        )
        for k in on_axis:
            start_s, end_s = (
                float(result[key][k]) for key in ("extent_start_s", "extent_end_s")
            )
            times = f"{format_time(start_s)}-{format_time(end_s)}"
            video_frames = (
                f"f{reported_video_frame(start_s, video_frame_rate)}-"
                f"f{reported_video_frame(end_s, video_frame_rate)}"
            )
            lines.append(
                f"    {result['extent_identifier'][k]!s:<10} "
                f"{result['extent_classification'][k]!s:<27} {times:<23} {video_frames}"
            )
        lines.append("")
    if extent_table_path is not None:
        lines += [
            (
                f"  extent table  {extent_table_path}  "
                f"({int(result['extent_count'][axis_index])} rows, "
                f"{rate_sample_count} samples)"
            ),
            "",
        ]
    return lines


def detection_report_line(result):
    """The line of the detection stage that ``--verbose`` prints, formatted from the
    detection result: the axes, then the number of extents and the number of each
    classification, each label once with one value per axis, separated by slashes in
    the order of the axes."""
    axes = [str(axis) for axis in result["axes"]]
    axis_indexes = [detection_stage.axis_index(result, axis) for axis in axes]
    extent_counts = "/".join(
        str(int(result["extent_count"][axis_index])) for axis_index in axis_indexes
    )
    classifications = ", ".join(
        f"{detection_pass['classification']} "
        + "/".join(
            str(int(result["pass_extent_count"][axis_index][p]))
            for axis_index in axis_indexes
        )
        for p, detection_pass in enumerate(detection_stage.PASSES)
    )
    return (
        f"{DETECTION_REPORT_HEADING}: {'/'.join(axes)} extents {extent_counts}, "
        f"{classifications}"
    )


REPORTED_REASONS = {
    ceiling_attenuator.AT_A_VIDEO_EDGE: "skipped at a video edge",
    ceiling_attenuator.ON_THE_DO_NOT_REPAIR_LIST: "excluded",
}


REMOVED_GYRO_MOTION = "gyro motion removed by --remove-gyro-motion"


def removed_axes_of(record):
    """The axes of the repair record ``record`` whose gyro motion the repair removed, in
    the order of the record."""
    removed = record["run"]["repair_parameters"]["remove_gyro_motion"]
    return [axis for axis in record["axes"] if axis in removed]


def repair_report(record):
    """The report of the repair stage, as a list of lines, formatted from the repair
    record. For each axis it counts the extents the repair scaled, skipped at a video
    edge, and excluded as the do not repair list directs, and lists each extent with its
    identifier and the gain the repair applied to it, or ``REPORTED_REASONS`` of the
    reason the repair kept it at its recorded values. For an axis that
    ``--remove-gyro-motion`` names, it states ``REMOVED_GYRO_MOTION`` and the absolute
    and the net rotation removed."""
    video = record["video"]
    lines = [
        REPAIR_REPORT_HEADING,
        f"video               {video['video_name']}",
        f"video frame rate    {video['video_frame_rate']:.4f} fps",
        f"measurement rate    {video['measurement_rate_hz']:.4f} Hz  (counted)",
    ]
    removed_axes = removed_axes_of(record)
    for axis, block in record["axes"].items():
        if axis in removed_axes:
            lines += [
                "",
                f"axis {axis}",
                f"  {REMOVED_GYRO_MOTION}",
                (
                    f"  {'rotation removed, absolute':<55} "
                    f"{block['removed_rotation_deg_abs_total']:.4f} deg"
                ),
                (
                    f"  {'rotation removed, net':<55} "
                    f"{block['removed_rotation_deg_net']:.4f} deg"
                ),
            ]
            continue
        summary = block["extent_summary"]
        identity_outside_extents = block["identity_outside_extents"]
        tests = block["ceiling_attenuator_tests"]
        lines += [
            "",
            f"axis {axis}",
            (
                f"  {'detected extents in':<55} {block['detected_extents_in']} "
                f"({summary['scaled_count']} scaled, "
                f"{summary['skipped_count']} skipped, "
                f"{summary['excluded_count']} excluded)"
            ),
            (
                f"  {'duration ms min/median/max':<55} "
                f"{summary['duration_ms_min']:.2f} / "
                f"{summary['duration_ms_median']:.2f} / "
                f"{summary['duration_ms_max']:.2f}"
                if summary["duration_ms_min"] is not None
                else f"  {'duration ms min/median/max':<55} none"
            ),
            (
                f"  {'gain min/median/max':<55} "
                f"{summary['gain_min']:.4f} / {summary['gain_median']:.4f} / "
                f"{summary['gain_max']:.4f}"
                if summary["gain_min"] is not None
                else f"  {'gain min/median/max':<55} none"
            ),
            (
                f"  {'extents left at a gain of one':<55} "
                f"{summary['passed_through_count']}"
            ),
            (
                f"  {'rotation removed, absolute':<55} "
                f"{summary['removed_rotation_deg_abs_total']:.4f} deg"
            ),
            (
                f"  {'rotation removed, net':<55} "
                f"{summary['removed_rotation_deg_net']:.4f} deg"
            ),
            (
                f"  {'samples inside extents':<55} "
                f"{identity_outside_extents['rate_samples_inside_extents']}"
            ),
            (
                f"  {'samples changed':<55} "
                f"{identity_outside_extents['rate_samples_changed']}"
            ),
            (
                f"  {'peak in clean Robust Standard Deviations min/median/max':<55} "
                + " / ".join(
                    f"{summary[key]:.2f}"
                    for key in (
                        "peak_in_clean_robust_standard_deviations_before_min",
                        "peak_in_clean_robust_standard_deviations_before_median",
                        "peak_in_clean_robust_standard_deviations_before_max",
                    )
                )
                if summary["peak_in_clean_robust_standard_deviations_before_min"]
                is not None
                else (
                    f"  {'peak in clean Robust Standard Deviations min/median/max':<55}"
                    " none"
                )
            ),
            (
                f"  {'over ceiling anywhere in the extent':<55} "
                f"{tests['extents_over_ceiling_anywhere_in_the_extent']}"
            ),
            (
                f"  {'extents with shortened fades':<55} "
                f"{tests['extents_with_shortened_fades']}"
            ),
        ]
        if block["extents"]:
            lines.append(f"    {'identifier':<10} gain")
            lines += [
                f"    {extent['identifier']:<10} "
                + (
                    f"{extent['gain']:.4f}"
                    if extent["scaled"]
                    else REPORTED_REASONS[extent["reason"]]
                )
                for extent in block["extents"]
            ]
    return lines


def repair_report_line(record):
    """The line of the repair stage that ``--verbose`` prints, formatted from the repair
    record: the axes, then the number of extents the repair attenuated below a gain of
    one, the number it left at a gain of one, the number it skipped at a video edge, and
    the number it excluded as the do not repair list directs, each label once with one
    value per axis, separated by slashes in the order of the axes. The line then names
    the axes whose gyro motion the repair removed. The counts leave those axes out."""
    removed_axes = removed_axes_of(record)
    counted_axes = [axis for axis in record["axes"] if axis not in removed_axes]
    parts = []
    if counted_axes:
        summaries = [record["axes"][axis]["extent_summary"] for axis in counted_axes]
        attenuated = "/".join(
            str(summary["scaled_count"] - summary["passed_through_count"])
            for summary in summaries
        )
        at_a_gain_of_one = "/".join(
            str(summary["passed_through_count"]) for summary in summaries
        )
        skipped = "/".join(str(summary["skipped_count"]) for summary in summaries)
        excluded = "/".join(str(summary["excluded_count"]) for summary in summaries)
        parts.append(
            f"{'/'.join(counted_axes)} attenuated {attenuated}, "
            f"at a gain of one {at_a_gain_of_one}, skipped at a video edge {skipped}, "
            f"excluded {excluded}"
        )
    if removed_axes:
        parts.append(f"gyro motion removed from {'/'.join(removed_axes)}")
    return f"{REPAIR_REPORT_HEADING}: {'; '.join(parts)}"


_DECODE, _DETECTION, _REPAIR, _WRITE = range(4)


def _finite(text):
    """A number given on the command line, which raises ``argparse.ArgumentTypeError``
    unless it is a finite float."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            errors.message(errors.ErrorCode.VALUE_NOT_A_NUMBER, text=text)
        ) from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(
            errors.message(errors.ErrorCode.VALUE_NOT_FINITE, text=text)
        )
    return value


def _ceiling(text):
    """A ceiling given on the command line: a finite float, or ``inf``, which keeps
    every rate sample of an extent at its recorded value."""
    if text == "inf":
        return math.inf
    return _finite(text)


def _threshold(text):
    """A threshold: one value for all three axes, or three values for pitch, yaw, and
    roll in that order, as one comma separated token. Any other count of values, or a
    value that is not a finite number, raises ``argparse.ArgumentTypeError``."""
    values = tuple(_finite(part) for part in text.split(","))
    if len(values) not in (1, len(AXES)):
        raise argparse.ArgumentTypeError(
            errors.message(
                errors.ErrorCode.THRESHOLD_VALUE_COUNT_INVALID,
                text=text,
                count=len(values),
            )
        )
    return values


def _axes(text):
    """A list of axes as one comma separated token, each element one of pitch, yaw, and
    roll and none repeated, returned in the order pitch, yaw, roll. An unknown or a
    repeated axis raises ``argparse.ArgumentTypeError``."""
    axes = text.split(",")
    unknown = [axis for axis in axes if axis not in AXES]
    if unknown:
        raise argparse.ArgumentTypeError(
            errors.message(
                errors.ErrorCode.AXIS_UNKNOWN, text=text, axes=",".join(unknown)
            )
        )
    repeated = [axis for axis in AXES if axes.count(axis) > 1]
    if repeated:
        raise argparse.ArgumentTypeError(
            errors.message(
                errors.ErrorCode.AXIS_REPEATED, text=text, axes=",".join(repeated)
            )
        )
    return tuple(axis for axis in AXES if axis in axes)


@dataclass(frozen=True)
class Position:
    """A position in the video: a time in seconds, counted from video frame 0 at time
    0, or a video frame, counted from 0. Exactly one of the two is set. ``text`` is the
    position as the user typed it, which an error names, and takes no part in a
    comparison."""

    seconds: float | None = None
    video_frame: int | None = None
    text: str | None = field(default=None, compare=False)

    def to_video_frame(self, video_frame_rate):
        """The video frame of this position. The video frame of a time is the video
        frame shown at that time: the time times ``video_frame_rate``, with no offset,
        rounded down to a whole video frame."""
        if self.video_frame is not None:
            return self.video_frame
        return math.floor(self.seconds * video_frame_rate)


@dataclass(frozen=True)
class Range:
    """A range in the video, from ``first`` through ``last``, both included."""

    first: Position
    last: Position


@dataclass(frozen=True)
class Duration:
    """A duration in the video: a time in seconds, or a count of video frames. Exactly
    one of the two is set. ``text`` is the duration as the user typed it, which an error
    names, and takes no part in a comparison."""

    seconds: float | None = None
    video_frames: int | None = None
    text: str | None = field(default=None, compare=False)


def _seconds(text):
    """The seconds of a time, or None when ``text`` is not a time. A time is seconds
    with the unit ``s``; minutes and seconds; or hours, minutes, and seconds. In a time
    with a colon, every field after the first is two digits below 60, and only the last
    field may carry a decimal fraction. Every field is ASCII digits, with no sign and no
    exponent."""
    if match := re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)s", text):
        return float(match[1])
    if match := re.fullmatch(r"([0-9]+):([0-5][0-9](?:\.[0-9]+)?)", text):
        return 60 * int(match[1]) + float(match[2])
    if match := re.fullmatch(r"([0-9]+):([0-5][0-9]):([0-5][0-9](?:\.[0-9]+)?)", text):
        return 3600 * int(match[1]) + 60 * int(match[2]) + float(match[3])
    return None


def format_time(seconds):
    """A time, rounded to the nearest millisecond: minutes and seconds, ``1:15.500``,
    or, at or past one hour, hours, minutes, and seconds, ``1:02:15.500``.

    A time before 0 is written as ``0:00.000``. The grammar has no minus sign, and the
    first telemetry frame can place quaternions before time 0, so an extent at the start
    of a video can start before it."""
    hours, milliseconds = divmod(max(round(seconds * 1000), 0), 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    if hours:
        return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"
    return f"{minutes}:{whole_seconds:02d}.{milliseconds:03d}"


class GrammarError(argparse.ArgumentTypeError):
    """A mistake in a position, a range, a duration, or an identifier, which argparse
    prints as a mistake in the option. It carries its code, so the do not repair list,
    which reads its elements outside argparse, names the mistake."""

    def __init__(self, code, **values):
        super().__init__(errors.message(code, **values))
        self.code = code


def parse_position(text):
    """A position: seconds with the unit ``s`` (``75s``, ``75.5s``); minutes and seconds
    (``1:15``, ``1:15.5``); hours, minutes, and seconds (``1:02:15``); or ``f`` and a
    video frame (``f4500``). In a time with a colon, every field after the first is
    below 60. Any other text is a mistake in the option and raises ``GrammarError``."""
    if match := re.fullmatch(r"f([0-9]+)", text):
        return Position(video_frame=int(match[1]), text=text)
    seconds = _seconds(text)
    if seconds is None:
        raise GrammarError(errors.ErrorCode.VALUE_NOT_A_POSITION, text=text)
    return Position(seconds=seconds, text=text)


def parse_range(text):
    """A range: two positions with a hyphen between them, ``1:15-1:17`` or
    ``f4500-f4620``, from the first position through the second, both included. A range
    whose second position comes before its first is a mistake in the option, as is any
    text other than two positions and one hyphen; each raises ``GrammarError``.

    The order of two times, or of two video frames, is tested here. The order of a time
    and a video frame depends on the video frame rate, so a range that holds one of each
    is returned untested, and the caller, which knows the video frame rate, tests its
    order."""
    parts = text.split("-")
    if len(parts) != 2 or not all(parts):
        raise GrammarError(errors.ErrorCode.VALUE_NOT_A_RANGE, text=text)
    first, last = (parse_position(part) for part in parts)
    both_times = first.seconds is not None and last.seconds is not None
    both_video_frames = first.video_frame is not None and last.video_frame is not None
    if (both_times and last.seconds < first.seconds) or (
        both_video_frames and last.video_frame < first.video_frame
    ):
        raise GrammarError(errors.ErrorCode.RANGE_REVERSED, text=text)
    return Range(first, last)


def parse_duration(text):
    """A duration: a time, as a position states it, or a count of video frames followed
    by ``f`` (``15s``, ``360f``). Any other text is a mistake in the option and raises
    ``GrammarError``."""
    if match := re.fullmatch(r"([0-9]+)f", text):
        return Duration(video_frames=int(match[1]), text=text)
    seconds = _seconds(text)
    if seconds is None:
        raise GrammarError(errors.ErrorCode.VALUE_NOT_A_DURATION, text=text)
    return Duration(seconds=seconds, text=text)


def reported_video_frame(seconds, video_frame_rate):
    """The video frame that the report of the detection stage prints for the time
    ``seconds``: the video frame of the time that ``format_time`` prints, so that the
    printed time, given to the product, names the printed video frame."""
    return parse_position(format_time(seconds)).to_video_frame(video_frame_rate)


@dataclass(frozen=True)
class DoNotRepairElement:
    """An element of the do not repair list that parses: an identifier, such as
    ``Y1502``, or a range, with ``axis_letter``, the letter ``P``, ``Y``, or ``R`` of
    the axis the range applies to, or None for every axis. ``text`` is the element as
    the user typed it. ``file`` and ``line`` are the ``--do-not-repair`` file and the
    number of the line that holds the element, both None for an element of
    ``--do-not-repair-list``."""

    text: str
    file: Path | None = None
    line: int | None = None
    identifier: str | None = None
    axis_letter: str | None = None
    range: Range | None = None


# The codes of each warning of the do not repair list: the code for an element of
# --do-not-repair-list, and the code for a line of the --do-not-repair file.
_INVALID_CODES = (
    errors.ErrorCode.DO_NOT_REPAIR_ELEMENT_INVALID,
    errors.ErrorCode.DO_NOT_REPAIR_LINE_INVALID,
)
_NAMES_NO_EXTENT_CODES = (
    errors.ErrorCode.DO_NOT_REPAIR_ELEMENT_NAMES_NO_EXTENT,
    errors.ErrorCode.DO_NOT_REPAIR_LINE_NAMES_NO_EXTENT,
)
_ON_REMOVED_AXIS_CODES = (
    errors.ErrorCode.DO_NOT_REPAIR_ELEMENT_ON_REMOVED_AXIS,
    errors.ErrorCode.DO_NOT_REPAIR_LINE_ON_REMOVED_AXIS,
)

# The axis of each axis letter of the do not repair list.
_AXIS_OF_LETTER = {axis[0].upper(): axis for axis in AXES}


def _do_not_repair_warning(codes, text, file, line, **values):
    """The warning of the element ``text`` of the do not repair list, as ``(code,
    values)``: the first code of ``codes`` for an element of ``--do-not-repair-list``,
    where ``file`` is None, and the second for the element on line ``line`` of the
    ``--do-not-repair`` file ``file``. ``values`` are further values of the warning."""
    element_code, line_code = codes
    if file is None:
        return element_code, {"element": text, **values}
    return line_code, {"file": file, "line": line, "element": text, **values}


def _do_not_repair_element(text, file, line):
    """The element of the do not repair list that ``text`` states. Without a hyphen,
    ``text`` is an identifier: an axis letter, ``P``, ``Y``, or ``R``, followed by a
    video frame number. With a hyphen, it is a range in the grammar of ``parse_range``,
    after an optional axis letter. Text that does not parse raises ``GrammarError``."""
    if "-" not in text:
        if re.fullmatch(r"[PYR][0-9]+", text) is None:
            raise GrammarError(errors.ErrorCode.VALUE_NOT_AN_IDENTIFIER, text=text)
        return DoNotRepairElement(text, file, line, identifier=text)
    axis_letter = text[0] if text[0] in "PYR" else None
    try:
        element_range = parse_range(text[1:] if axis_letter else text)
    except GrammarError as error:
        # A mistake in the whole range names the element as the user typed it, with its
        # axis letter; a mistake in one position names that position.
        if axis_letter and error.code is not errors.ErrorCode.VALUE_NOT_A_POSITION:
            raise GrammarError(error.code, text=text) from None
        raise
    return DoNotRepairElement(
        text, file, line, axis_letter=axis_letter, range=element_range
    )


def _do_not_repair_file_text(path):
    """The text of the ``--do-not-repair`` file ``path``, read as UTF-8 with or without
    a byte order mark. A file that cannot be read raises ``errors.CannotReadError``."""
    if not path.exists():
        raise errors.CannotReadError(
            errors.ErrorCode.DO_NOT_REPAIR_FILE_NOT_FOUND, file=path
        )
    if path.is_dir():
        raise errors.CannotReadError(
            errors.ErrorCode.DO_NOT_REPAIR_FILE_IS_DIRECTORY, file=path
        )
    with errors.operating_system_error_as(errors.CannotReadError, path):
        data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise errors.CannotReadError(
            errors.ErrorCode.DO_NOT_REPAIR_FILE_NOT_UTF8, file=path
        ) from None


def read_do_not_repair_list(listed, path):
    """The do not repair list, as ``(elements, warnings)``: the elements of ``listed``,
    the token of ``--do-not-repair-list``, followed by those of ``path``, the
    ``--do-not-repair`` file, either of which is None when not given; and the warning of
    each element that does not parse, which ``elements`` leaves out, each warning as
    ``(code, values)``.

    ``listed`` holds its elements between commas. The file holds one element on each
    line, with the whitespace around it stripped, and a blank line or a line that begins
    with ``#`` is ignored. An empty element is ignored. A file that cannot be read
    raises ``errors.CannotReadError``."""
    given = (
        [(text, None, None) for text in listed.split(",")] if listed is not None else []
    )
    if path is not None:
        given += [
            (text.strip(), path, number)
            for number, text in enumerate(
                _do_not_repair_file_text(path).splitlines(), start=1
            )
            if not text.strip().startswith("#")
        ]
    elements, warnings = [], []
    for text, file, line in given:
        if not text:
            continue
        try:
            elements.append(_do_not_repair_element(text, file, line))
        except GrammarError as error:
            warnings.append(
                _do_not_repair_warning(
                    _INVALID_CODES,
                    text,
                    file,
                    line,
                    cause=error.code,
                    reason=str(error),
                )
            )
    return elements, warnings


def _axis_letter(element):
    """The axis letter of an element of the do not repair list.

    The axis letter of an identifier is its first letter. The axis letter of a range is
    the letter before the range. A range on every axis has no axis letter. The function
    returns None for it."""
    if element.identifier is not None:
        return element.identifier[0]
    return element.axis_letter


def resolve_do_not_repair_list(
    elements,
    extent_identifier,
    extent_start_s,
    extent_end_s,
    video_frame_rate,
    removed_axes=(),
):
    """The do not repair list ``elements`` resolved against the extents of a detection
    result, as ``(excluded, warnings)``: the sorted identifiers of the extents the list
    excludes, and the warning of each element the resolution leaves out, as ``(code,
    values)``, in the order of the list.

    ``extent_identifier``, ``extent_start_s``, and ``extent_end_s`` give the identifier,
    the start time, and the end time of each extent. Each range of the list converts to
    video frames at ``video_frame_rate``, and the start and the end of each extent
    convert to the video frames that the report of the detection stage prints, so a
    range read off the report names the extents the report shows in it. A range of a
    time and a video frame whose last video frame comes before its first, and an
    identifier that names no extent, is a warning.

    ``removed_axes`` are the axes that ``--remove-gyro-motion`` names. An identifier, or
    a range after an axis letter, on one of those axes is a warning. The resolution
    leaves such an element out."""
    removed_letters = {axis[0].upper() for axis in removed_axes}
    resolved = [_axis_letter(element) not in removed_letters for element in elements]
    video_frame_ranges = [
        None
        if element.range is None or not is_resolved
        else (
            element.axis_letter,
            element.range.first.to_video_frame(video_frame_rate),
            element.range.last.to_video_frame(video_frame_rate),
        )
        for element, is_resolved in zip(elements, resolved, strict=True)
    ]
    excluded, unmatched = repair_stage.excluded_extents(
        [
            element.identifier
            for element, is_resolved in zip(elements, resolved, strict=True)
            if is_resolved and element.identifier is not None
        ],
        [
            video_frame_range
            for video_frame_range in video_frame_ranges
            if video_frame_range and video_frame_range[1] <= video_frame_range[2]
        ],
        extent_identifier,
        [
            reported_video_frame(float(start_s), video_frame_rate)
            for start_s in extent_start_s
        ],
        [
            reported_video_frame(float(end_s), video_frame_rate)
            for end_s in extent_end_s
        ],
    )
    warnings = []
    for element, is_resolved, video_frame_range in zip(
        elements, resolved, video_frame_ranges, strict=True
    ):
        place = (element.text, element.file, element.line)
        if not is_resolved:
            warnings.append(
                _do_not_repair_warning(
                    _ON_REMOVED_AXIS_CODES,
                    *place,
                    axis=_AXIS_OF_LETTER[_axis_letter(element)],
                )
            )
        elif element.identifier is not None and element.identifier in unmatched:
            warnings.append(_do_not_repair_warning(_NAMES_NO_EXTENT_CODES, *place))
        elif video_frame_range and video_frame_range[2] < video_frame_range[1]:
            code = errors.ErrorCode.RANGE_REVERSED
            warnings.append(
                _do_not_repair_warning(
                    _INVALID_CODES,
                    *place,
                    cause=code,
                    reason=errors.message(code, text=element.text),
                )
            )
    return excluded, warnings


def _field_of(option):
    """The field of the detection parameters or the repair parameters that holds
    ``option``, which is also the attribute argparse gives it."""
    return option.removeprefix("--").replace("-", "_")


def _shown(value):
    """A default as the help text shows it, and the value of an ``invalid --OPTION
    VALUE`` error as parsed."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, tuple):
        return ",".join(f"{one:g}" for one in value)
    return f"{value:g}"


class _HelpAll(argparse.Action):
    """``--help-all``: print the help with every option, the advanced options included,
    and exit, as ``--help`` prints the help without them."""

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(
            option_strings,
            dest=argparse.SUPPRESS,
            default=argparse.SUPPRESS,
            nargs=0,
            **kwargs,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        _parser(advanced=True).print_help()
        parser.exit()


def _parser(advanced=False):
    """The parser of the command line, with the options in their groups. Unless
    ``advanced`` is true, as it is for ``--help-all``, the help leaves out ``--report``
    and the group "advanced tuning parameters"."""
    detection_parameters = detection_stage.DetectionParameters()
    repair_parameters = ceiling_attenuator.RepairParameters()
    all_axes = ",".join(AXES)

    def advanced_help(text):
        return text if advanced else argparse.SUPPRESS

    parser = argparse.ArgumentParser(
        prog="gyro-doctor",
        usage="%(prog)s [options] VIDEO",
        description=(
            "gyro-doctor repairs the gyro data track of a DJI O4 video and saves the "
            "repaired video alongside the original."
        ),
    )

    parser.add_argument("video", type=Path, metavar="VIDEO", help="The video to repair")
    parser.add_argument(
        "--help-all",
        action=_HelpAll,
        help="Show help message including advanced options",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {importlib.metadata.version('gyro_doctor')}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        help=(
            "The path of the repaired video (default: adds the suffix '-repaired' to the "
            "video name before the file extension)"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line summary for each stage",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force repair of a video already marked as repaired",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help=advanced_help("Print the log detail of every stage"),
    )

    cache_options = parser.add_argument_group(
        "cache related & charting only run options"
    )
    cache_options.add_argument(
        "--no-cache",
        action="store_true",
        help="Disables caching. Charts and renders will be unavailable.",
    )
    stop_options = cache_options.add_mutually_exclusive_group()
    stop_options.add_argument(
        "--decode-only",
        action="store_true",
        help="Stop the run after the decode stage",
    )
    stop_options.add_argument(
        "--detect-only",
        action="store_true",
        help="Stop the run after the detection stage",
    )
    stop_options.add_argument(
        "--repair-only",
        action="store_true",
        help="Stop the run after the repair stage",
    )
    cache_options.add_argument(
        "--cache-dir",
        type=Path,
        metavar="PATH",
        help=(
            "The location of the cache (default: .gyro-doctor-cache under the current "
            "working directory)"
        ),
    )

    detection_tuning_parameters = parser.add_argument_group(
        "detection tuning parameters"
    )
    detection_tuning_parameters.add_argument(
        "--detect-axes",
        type=_axes,
        default=AXES,
        metavar="LIST",
        help=f"The detect axes to process (default: {all_axes})",
    )
    detection_tuning_parameters.add_argument(
        "--detect-oscillation-threshold",
        type=_threshold,
        default=detection_parameters.detect_oscillation_threshold,
        metavar="VALUE or LIST",
        help=(
            "Noise threshold for a local segment of the video. When the noise of the "
            "gyro signal reaches this many times the Robust Standard Deviation of the "
            "axis, Gyro Doctor detects a large amplitude oscillation. Set one value for "
            "all three axes, or "
            f"three values in the order {all_axes} "
            f"(default: {_shown(detection_parameters.detect_oscillation_threshold)})"
        ),
    )
    detection_tuning_parameters.add_argument(
        "--detect-burst-threshold",
        type=_threshold,
        default=detection_parameters.detect_burst_threshold,
        metavar="VALUE or LIST",
        help=(
            "Noise threshold for a local segment of the video. When the noise of the "
            "gyro signal reaches this many times the Robust Standard Deviation of the "
            "axis, Gyro Doctor detects a mid-amplitude burst. Set one value for all "
            "three axes, or set each "
            f"axis individually in the order {all_axes} "
            f"(default: {_shown(detection_parameters.detect_burst_threshold)})"
        ),
    )
    detection_tuning_parameters.add_argument(
        "--detect-impulse-threshold",
        type=_threshold,
        default=detection_parameters.detect_impulse_threshold,
        metavar="VALUE or LIST",
        help=(
            "Noise threshold for a single sample of the gyro signal. When a sample "
            "differs from the local median by this many times the Robust Standard "
            "Deviation of the axis, Gyro Doctor detects an impulse candidate, which the "
            "advanced tuning parameters of --help-all qualify as an impulse event. Set "
            "one value for "
            f"all three axes, or set each axis individually in the order {all_axes} "
            f"(default: {_shown(detection_parameters.detect_impulse_threshold)})"
        ),
    )

    repair_tuning_parameters = parser.add_argument_group("repair tuning parameters")
    repair_tuning_parameters.add_argument(
        "--repair-axes",
        type=_axes,
        metavar="LIST",
        help=f"The repair axes to process (default: {all_axes})",
    )
    repair_tuning_parameters.add_argument(
        "--remove-gyro-motion",
        type=_axes,
        default=repair_parameters.remove_gyro_motion,
        metavar="LIST",
        help=(
            "Removes all the gyro motion of the axes in LIST from the gyro data track "
            "of the repaired video. Comma separated, such as yaw,pitch,roll "
            "(default: none)"
        ),
    )
    repair_tuning_parameters.add_argument(
        "--do-not-repair-list",
        metavar="LIST",
        help=(
            "Do not repair the named extents, time ranges, and frame ranges in LIST. "
            "Comma separated, such as Y1502,0:00-0:05,Rf4500-f4620. Frame numbers begin "
            "with f. P, R, or Y indicates the axis of a range, and a range without one "
            "applies to all axes (default: none)"
        ),
    )
    # The field do_not_repair of the repair parameters holds the identifiers of the
    # extents the list excludes, not the path of the file, so the option stores the path
    # under a name of its own.
    repair_tuning_parameters.add_argument(
        "--do-not-repair",
        type=Path,
        dest="do_not_repair_file",
        metavar="FILE",
        help=(
            "A text file for the do not repair list, one item or range on each line. A "
            "blank line or a line that begins with # is ignored (default: none)"
        ),
    )

    for title, ceiling_option, centerline_option, classification, event in (
        (
            "large amplitude oscillation repair parameters",
            "--repair-oscillation-ceiling",
            "--repair-oscillation-centerline",
            "a large amplitude oscillation",
            "the oscillation",
        ),
        (
            "mid-amplitude burst repair parameters",
            "--repair-burst-ceiling",
            "--repair-burst-centerline",
            "a mid-amplitude burst",
            "the burst",
        ),
        (
            "impulse repair parameters",
            "--repair-impulse-ceiling",
            "--repair-impulse-centerline",
            "an impulse event",
            "the impulse event",
        ),
    ):
        classification_parameters = parser.add_argument_group(title)
        ceiling = getattr(repair_parameters, _field_of(ceiling_option))
        classification_parameters.add_argument(
            ceiling_option,
            type=_ceiling,
            default=ceiling,
            metavar="VALUE",
            help=(
                f"Ceiling for the repair of {classification}. The repair reduces {event} "
                "until the gyro signal peak is VALUE times the noise of the axis "
                "excluding all detected events. A value of 0 collapses the motion to the "
                f"centerline, and inf leaves {event} as recorded "
                f"(default: {_shown(ceiling)})"
            ),
        )
        centerline = getattr(repair_parameters, _field_of(centerline_option))
        classification_parameters.add_argument(
            centerline_option,
            choices=ceiling_attenuator.CENTERLINES,
            default=centerline,
            help=(
                f"The centerline toward which {classification} is attenuated "
                f"(default: {centerline})"
            ),
        )

    advanced_tuning_parameters = parser.add_argument_group("advanced tuning parameters")
    for option, meaning in (
        (
            "--local-median-window",
            (
                "The time in milliseconds on either side of the window's center used "
                "to draw a smoothed line that follows the overall trend of the gyro "
                "signal. The smoothed local median is similar to a rolling average. "
                "Detection and repair both use the smoothed local median, and in both "
                "stages a shorter window leads to less repair. In the detection stage, "
                "a shorter window lets the smoothed local median follow quicker "
                "motion, so less of that motion is measured as noise. Less of it then "
                "reaches the oscillation and burst thresholds, and detection finds "
                "fewer large amplitude oscillations and mid-amplitude bursts. In the "
                "repair stage, a gyro noise event repaired with the local-median "
                "centerline is reduced toward the smoothed local median. A shorter "
                "window causes the repair to follow more of the original gyro signal, "
                "attenuating the gyro noise event less while keeping more of what may "
                "be real motion inside it."
            ),
        ),
        (
            "--detect-noise-window",
            (
                "The time in milliseconds on either side of the window's center over "
                "which detection measures the local noise of the gyro signal. The "
                "oscillation and burst thresholds use the local noise for detection. A "
                "shorter window leads to more repair since it allows a single noise "
                "event to occupy more of the window, increasing the noise event's "
                "influence on the measurement."
            ),
        ),
        (
            "--detect-event-end-level",
            (
                "Defines the signal level at the edges of a detected noise event "
                "indicating where the noise event ends, given as a multiple of the "
                "noise distance from the smoothed local median. Detection identifies "
                "gyro noise events, then uses this parameter to determine where each "
                "gyro noise event starts and ends. A higher value leads to shorter "
                "events and less repair, since shorter events leave more of the "
                "original gyro signal untouched."
            ),
        ),
        (
            "--detect-event-end-time",
            (
                "Defines how long in milliseconds that the gyro signal must stay within "
                "--detect-event-end-level for a large amplitude oscillation or a "
                "mid-amplitude burst to end. This prevents brief crossings within "
                "--detect-event-end-level from ending an event prematurely. A longer "
                "value leads to more repair, since the longer time requirement waits "
                "for the noise to settle longer before the event is considered "
                "complete."
            ),
        ),
        (
            "--detect-impulse-min-width",
            (
                "The shortest time in milliseconds that a peak must remain higher than "
                "the --detect-event-end-level to count as an impulse event. A longer "
                "value leads to less repair since a longer value rejects brief peaks."
            ),
        ),
        (
            "--detect-impulse-max-width",
            (
                "The longest time in milliseconds of an impulse event before widening. "
                "Any impulse event whose edges cannot be determined within this window "
                "of time is rejected from the impulse candidates, since impulse events "
                "by definition are very short events. A longer value leads to more "
                "repair, since a longer value accepts wider peaks that trail off more "
                "gradually instead of returning quickly."
            ),
        ),
        (
            "--detect-impulse-shoulder-drop",
            (
                "The turn back size as a multiple of the noise that indicates the "
                "shoulder at the end of each side of an impulse event has been reached. "
                "A higher value finds potentially fewer but wider and therefore more "
                "complete impulse events, preventing premature indication an impulse "
                "event is complete which could lead to fragmentation of that event."
            ),
        ),
        (
            "--detect-impulse-shoulder-time",
            (
                "The time in milliseconds within which the gyro signal must turn back by "
                "--detect-impulse-shoulder-drop to indicate the end of a side of an "
                "impulse event. A shorter value rejects slow gently sloped returns, so a "
                "wide hill shaped peak in the gyro signal is not mistaken for an impulse "
                "event. A longer value accepts slower returns, causing more hill shaped "
                "peaks to count as impulse events and be repaired."
            ),
        ),
        (
            "--detect-impulse-side-end-time",
            (
                "The time in milliseconds that the gyro signal must stay within "
                "--detect-event-end-level to end a side of an impulse event. A longer "
                "value finds fewer but wider impulse events by rejecting brief returns "
                "to normal as incomplete, so each side of the event can expand farther "
                "from the peak. Since peaks wider than --detect-impulse-max-width are "
                "rejected, making this value larger may result in fewer impulse events."
            ),
        ),
        (
            "--detect-impulse-widen",
            (
                "The iterative step size in milliseconds by which each impulse event is "
                "widened past its shoulders, once always and then until the gyro signal "
                "returns within --detect-event-end-level, up to "
                "--detect-impulse-widen-max. A longer value widens every impulse event "
                "more."
            ),
        ),
        (
            "--detect-impulse-widen-max",
            (
                "The longest time in milliseconds that each end of an impulse event can "
                "be widened past its shoulder. Widening stops at this limit even where "
                "the gyro signal is still beyond --detect-event-end-level. A longer "
                "value lets an impulse event grow wider, leading to more repair."
            ),
        ),
    ):
        default = getattr(detection_parameters, _field_of(option))
        advanced_tuning_parameters.add_argument(
            option,
            type=_finite,
            default=default,
            metavar="VALUE",
            help=advanced_help(f"{meaning} (default: {_shown(default)})"),
        )
    advanced_tuning_parameters.add_argument(
        "--repair-fade",
        type=_finite,
        default=repair_parameters.repair_fade,
        metavar="VALUE",
        help=advanced_help(
            "The time in milliseconds over which the repair blends into the original "
            "gyro signal at each end of a gyro noise event. A longer fade makes the "
            "repair edges more gradual and leads to less repair inside the edges of "
            "each event. A fade is shortened to no more than half of an event since it "
            "fades both sides. "
            f"(default: {_shown(repair_parameters.repair_fade)})"
        ),
    )
    advanced_tuning_parameters.add_argument(
        "--repair-fade-shape",
        choices=ceiling_attenuator.FADE_SHAPES,
        default=repair_parameters.repair_fade_shape,
        help=advanced_help(
            "The choice of the shape of the fade at each end of a repair. A "
            "raised-cosine fade eases in and out along a smooth curve with no sudden "
            "change. A linear fade changes at a constant rate, with a corner at each "
            "edge of the fade. "
            f"(default: {repair_parameters.repair_fade_shape})"
        ),
    )
    advanced_tuning_parameters.add_argument(
        "--repair-edges",
        action="store_true",
        default=repair_parameters.repair_edges,
        help=advanced_help(
            "Repair a gyro noise event at the beginning or end of the video. By "
            "default such an event is left as recorded, since it begins or ends on "
            "gyro noise, leaving the centerline nothing clean to connect to and the "
            "repair nothing clean to blend into. Repairing such an event can add false "
            "motion. "
            f"(default: {_shown(repair_parameters.repair_edges)})"
        ),
    )
    return parser


# The range of every option that takes a number: the code of the error that a value
# outside the range raises, the options, and the test that a value lies inside the
# range. A ceiling of inf lies inside its range.
RANGES = (
    (
        errors.ErrorCode.VALUE_NOT_ABOVE_ZERO,
        (
            "--detect-oscillation-threshold",
            "--detect-burst-threshold",
            "--detect-impulse-threshold",
            "--detect-event-end-level",
            "--detect-impulse-shoulder-drop",
        ),
        lambda value: value > 0,
    ),
    (
        errors.ErrorCode.TIME_NOT_ABOVE_ZERO,
        ("--local-median-window", "--detect-noise-window", "--detect-impulse-widen"),
        lambda value: value > 0,
    ),
    (
        errors.ErrorCode.TIME_BELOW_ZERO,
        (
            "--detect-event-end-time",
            "--detect-impulse-min-width",
            "--detect-impulse-max-width",
            "--detect-impulse-shoulder-time",
            "--detect-impulse-side-end-time",
            "--detect-impulse-widen-max",
            "--repair-fade",
        ),
        lambda value: value >= 0,
    ),
    (
        errors.ErrorCode.CEILING_BELOW_ZERO,
        (
            "--repair-oscillation-ceiling",
            "--repair-burst-ceiling",
            "--repair-impulse-ceiling",
        ),
        lambda value: value >= 0,
    ),
)

STOP_OPTION_STAGES = {
    "--decode-only": "decode",
    "--detect-only": "detection",
    "--repair-only": "repair",
}


def _range_test(args):
    """Raise ``errors.InvalidOptionValueError`` for the first option of ``RANGES`` whose
    value lies outside its range. A threshold lies outside its range when any of its
    values does, and the error names its values as one token."""
    for code, options, in_range in RANGES:
        for option in options:
            value = getattr(args, _field_of(option))
            values = value if isinstance(value, tuple) else (value,)
            if not all(in_range(one) for one in values):
                raise errors.InvalidOptionValueError(
                    code, option=option, value=_shown(value)
                )


def _options_test(args):
    """The output path and the repair axes of the run, after the test of every value and
    every combination of options that argparse does not test, which raises an error of
    ``errors`` for the first that fails. When ``--remove-gyro-motion`` names every axis,
    logs a warning. When ``--repair-axes`` is at its default and ``--detect-axes`` names
    fewer axes, logs a warning, and the repair axes are the detected axes."""
    _range_test(args)
    stop_options = [
        option for option in STOP_OPTION_STAGES if getattr(args, _field_of(option))
    ]
    if args.no_cache and stop_options:
        raise errors.CannotCombineOptionsError(
            errors.ErrorCode.STOP_OPTION_WITHOUT_CACHE,
            option="--no-cache",
            other_option=stop_options[0],
            stage=STOP_OPTION_STAGES[stop_options[0]],
        )
    if args.repair_axes is not None:
        absent = [axis for axis in args.repair_axes if axis not in args.detect_axes]
        if absent:
            raise errors.CannotCombineOptionsError(
                errors.ErrorCode.REPAIR_AXIS_NOT_DETECTED,
                option=f"--repair-axes {','.join(args.repair_axes)}",
                other_option=f"--detect-axes {','.join(args.detect_axes)}",
                absent=",".join(absent),
            )
    video = args.video
    if not video.is_file():
        raise errors.CannotReadError(errors.ErrorCode.VIDEO_NOT_A_FILE, file=video)
    if args.output is None:
        output_path = video.with_name(f"{video.stem}-repaired{video.suffix}")
    else:
        output_path = args.output
    if output_path.exists() and output_path.samefile(video):
        raise errors.CannotWriteError(
            errors.ErrorCode.OUTPUT_IS_VIDEO, file=output_path, video=video
        )
    if output_path.is_dir():
        raise errors.CannotWriteError(
            errors.ErrorCode.OUTPUT_IS_DIRECTORY, file=output_path
        )
    if not output_path.parent.is_dir():
        raise errors.CannotWriteError(
            errors.ErrorCode.OUTPUT_DIRECTORY_NOT_FOUND,
            file=output_path,
            directory=output_path.parent,
        )
    if len(args.remove_gyro_motion) == len(AXES):
        errors.log_warning(log, errors.ErrorCode.GYRO_MOTION_REMOVED_FROM_EVERY_AXIS)
    if args.repair_axes is not None:
        return output_path, args.repair_axes
    if len(args.detect_axes) < len(AXES):
        errors.log_warning(
            log,
            errors.ErrorCode.REPAIR_AXES_NARROWED_TO_DETECTED,
            absent=",".join(axis for axis in AXES if axis not in args.detect_axes),
            detect_axes=",".join(args.detect_axes),
        )
    return output_path, args.detect_axes


def _parameters(parameters_class, args):
    """The detection parameters or the repair parameters of the run, one field per
    option. ``do_not_repair`` is no option: the run fills it once it resolves the do not
    repair list against the detection result."""
    return parameters_class(
        **{
            parameter.name: getattr(args, parameter.name)
            for parameter in fields(parameters_class)
            if parameter.name != "do_not_repair"
        }
    )


def _last_stage(args):
    """The stage the run ends at."""
    if args.decode_only:
        return _DECODE
    if args.detect_only:
        return _DETECTION
    if args.repair_only:
        return _REPAIR
    return _WRITE


def stored_parts(path, parts, json_parts=()):
    """The names of every member of the cached result at ``path``, and each member of
    ``parts`` and of ``json_parts`` that it holds, read without the other arrays:
    ``(present, members)``. Each member of ``json_parts`` is JSON text and is returned
    parsed. The read is inside ``decode_stage.cache_file_read``, which raises
    ``errors.CannotReadError`` for a file that cannot be opened or is damaged."""
    with (
        decode_stage.cache_file_read(path),
        np.load(path, allow_pickle=False) as stored,
    ):
        present = set(stored.files)
        members = {name: stored[name] for name in parts if name in present}
        members |= {
            name: json.loads(str(stored[name]))
            for name in json_parts
            if name in present
        }
    return present, members


def _decode_result_valid(path, video):
    """Whether the cached decode result at ``path`` is valid for ``video``, tested
    without reading its arrays. A decode result that fails a test is not valid, with a
    warning."""
    if not path.exists():
        return False
    try:
        present, header = stored_parts(path, detection_stage.HEADER_PARTS)
        decode_stage.header_test(header, video, path)
        decode_stage.every_part_present_test(
            present, path, decode_stage.DECODE_RESULT_PARTS
        )
    except errors.CannotReadError as error:
        errors.log_cached_result_warning(log, error, "decode stage")
        return False
    return True


def _detection_result_valid(path, video, video_cache_directory, axes, parameters):
    """Whether the cached detection result at ``path`` is valid for ``video``, the
    detected axes, and the detection parameters, tested without reading its arrays. A
    detection result that fails a test is not valid, with a warning; one computed with
    other parameters is not valid, with no warning."""
    if not path.exists():
        return False
    try:
        present, members = stored_parts(
            path, (*detection_stage.HEADER_PARTS, "axes", "settings"), ("file_stamps",)
        )
        decode_stage.header_test(members, video, path)
        decode_stage.every_part_present_test(
            present, path, (*detection_stage.DETECTION_RESULT_PARTS, "file_stamps")
        )
        if not detection_stage.same_parameters(members, axes, parameters):
            return False
        decode_stage.file_stamps_test(
            path, video_cache_directory, members["file_stamps"], ("decode_result",)
        )
    except errors.CannotReadError as error:
        errors.log_cached_result_warning(log, error, "detection stage")
        return False
    return True


def _repair_result_valid(path, video, video_cache_directory, axes, parameters):
    """Whether the cached repair result at ``path`` is valid for ``video``, the repair
    axes, and the repair parameters, tested without reading its arrays. A repair result
    that fails a test is not valid, with a warning; one computed with other parameters
    is not valid, with no warning."""
    if not path.exists():
        return False
    try:
        present, members = stored_parts(
            path, repair_stage.HEADER_PARTS, ("repair_record",)
        )
        decode_stage.header_test(members, video, path)
        decode_stage.every_part_present_test(
            present, path, repair_stage.REPAIR_RESULT_PARTS
        )
        record = members["repair_record"]
        decode_stage.every_part_present_test(record, path, ("file_stamps",))
        if not repair_stage.same_parameters(record, axes, parameters):
            return False
        decode_stage.file_stamps_test(
            path,
            video_cache_directory,
            record["file_stamps"],
            ("decode_result", "detection_result"),
        )
    except errors.CannotReadError as error:
        errors.log_cached_result_warning(log, error, "repair stage")
        return False
    return True


def _resolve_do_not_repair_list_from_cache(
    elements, detection_path, decode_path, removed_axes
):
    """``resolve_do_not_repair_list`` of ``elements`` against the extents of the cached
    detection result at ``detection_path``, at the video frame rate of the cached decode
    result at ``decode_path``.

    The function reads each file without its other arrays. ``removed_axes`` are the axes
    that ``--remove-gyro-motion`` names. A file that cannot be opened or is damaged
    raises ``errors.CannotReadError``."""
    _, extents = stored_parts(
        detection_path, ("extent_identifier", "extent_start_s", "extent_end_s")
    )
    _, decoded = stored_parts(decode_path, ("video_frame_rate",))
    return resolve_do_not_repair_list(
        elements,
        extents["extent_identifier"],
        extents["extent_start_s"],
        extents["extent_end_s"],
        float(decoded["video_frame_rate"]),
        removed_axes,
    )


def _print(lines):
    print("\n".join(lines))


def _run(args, output_path, repair_axes):
    """Run the stages from the first stage whose cached result is not valid through the
    stage the run ends at. Print nothing on stdout by default. With ``--verbose`` print
    one line for each stage that ran or whose result the run read from the cache. With
    ``--report`` print the report of each stage that ran, the line of each stage whose
    result the run read from the cache, and, with the cache on, the path of every file
    in the cache that the run wrote or read.

    The run reads the do not repair list before any stage runs. A run that reaches the
    repair stage resolves the list against the cached detection result before the cache
    test of the repair result, and against the detection result the run computes when
    the detection stage runs. The identifiers of the extents the list excludes fill
    ``do_not_repair`` of the repair parameters, and the warnings of the last resolution
    are logged once, before the repair stage runs or its line prints."""
    video, detect_axes = args.video, args.detect_axes
    print_cached_lines = args.verbose or args.report
    detection_parameters = _parameters(detection_stage.DetectionParameters, args)
    repair_parameters = _parameters(ceiling_attenuator.RepairParameters, args)
    last_stage = _last_stage(args)
    do_not_repair_list, list_warnings = read_do_not_repair_list(
        args.do_not_repair_list, args.do_not_repair_file
    )
    for code, values in list_warnings:
        errors.log_warning(log, code, **values)
    if not args.force:
        telemetry.already_repaired_test(video)

    cache_on = not args.no_cache
    video_cache_directory = (
        args.cache_dir or Path.cwd() / ".gyro-doctor-cache"
    ) / video.stem
    decode_path = decode_stage.decode_result_path(video_cache_directory, video.stem)
    readme_path = decode_stage.readme_path(video_cache_directory)
    detection_path = detection_stage.detection_result_path(
        video_cache_directory, video.stem
    )
    run_record_path = detection_stage.run_record_path(video_cache_directory, video.stem)
    extent_table_paths = [
        detection_stage.extent_table_path(video_cache_directory, video.stem, axis)
        for axis in detect_axes
    ]
    repair_path = repair_stage.repair_result_path(video_cache_directory, video.stem)
    repair_record_path = repair_stage.repair_record_path(
        video_cache_directory, video.stem
    )
    write_record_path = write_stage.write_record_path(video_cache_directory, video.stem)

    resolution_warnings = []
    run_from = _DECODE
    if cache_on and _decode_result_valid(decode_path, video):
        run_from = _DETECTION
        if last_stage >= _DETECTION and _detection_result_valid(
            detection_path,
            video,
            video_cache_directory,
            detect_axes,
            detection_parameters,
        ):
            run_from = _REPAIR
        if run_from == _REPAIR and last_stage >= _REPAIR:
            try:
                excluded, resolution_warnings = _resolve_do_not_repair_list_from_cache(
                    do_not_repair_list,
                    detection_path,
                    decode_path,
                    repair_parameters.remove_gyro_motion,
                )
            except errors.CannotReadError as error:
                errors.log_cached_result_warning(log, error, "detection stage")
                run_from = _DETECTION
            else:
                repair_parameters = replace(repair_parameters, do_not_repair=excluded)
                if _repair_result_valid(
                    repair_path,
                    video,
                    video_cache_directory,
                    repair_axes,
                    repair_parameters,
                ):
                    run_from = _WRITE

    # The reads go from the last stage back to the first: an error of a cache reader
    # runs that stage again, and that stage then takes the results before it.
    decode_result = detection_result = repair_result = None
    if run_from == _WRITE and last_stage == _WRITE:
        try:
            repair_result = repair_stage.read_repair_result(repair_path)
        except errors.CannotReadError as error:
            errors.log_cached_result_warning(log, error, "repair stage")
            run_from = _REPAIR
    if run_from == _REPAIR and last_stage >= _REPAIR:
        try:
            detection_result = detection_stage.read_detection_result(detection_path)
        except errors.CannotReadError as error:
            errors.log_cached_result_warning(log, error, "detection stage")
            run_from = _DETECTION
    if run_from in (_DETECTION, _REPAIR) and last_stage >= run_from:
        try:
            decode_result = decode_result_reader.read_decode_result(decode_path, video)
        except errors.CannotReadError as error:
            errors.log_cached_result_warning(log, error, "decode stage")
            run_from = _DECODE

    if run_from == _DECODE:
        decode_result = decode_stage.decode(video, force=args.force)
        written = (
            decode_stage.write_decode_result(decode_result, video_cache_directory)
            if cache_on
            else (None, None)
        )
        if args.report:
            _print(decode_report(decode_result, *written))
        elif args.verbose:
            print(decode_report_line(decode_result))
    elif print_cached_lines:
        print(f"{DECODE_REPORT_HEADING}{CACHED_RESULT} {decode_path}")

    if last_stage >= _DETECTION and run_from <= _DETECTION:
        try:
            detection_result = detection_stage.detect(
                decode_result, detect_axes, detection_parameters
            )
        except errors.GyroDoctorError as error:
            error.values.setdefault("file", video)
            raise
        if cache_on:
            extent_table_path_by_axis = detection_stage.write_detection_result(
                detection_result,
                video_cache_directory,
                file_stamps={"decode_result": decode_stage.file_stamp(decode_path)},
            )
        if args.report:
            lines = [DETECTION_REPORT_HEADING]
            for axis in (str(axis) for axis in detection_result["axes"]):
                lines += axis_report(
                    detection_result,
                    axis,
                    extent_table_path_by_axis[axis] if cache_on else None,
                    float(decode_result["video_frame_rate"]),
                )
            _print(lines)
        elif args.verbose:
            print(detection_report_line(detection_result))
    elif last_stage >= _DETECTION and print_cached_lines:
        print(f"{DETECTION_REPORT_HEADING}{CACHED_RESULT} {detection_path}")

    if last_stage >= _REPAIR:
        if run_from <= _DETECTION:
            excluded, resolution_warnings = resolve_do_not_repair_list(
                do_not_repair_list,
                detection_result["extent_identifier"],
                detection_result["extent_start_s"],
                detection_result["extent_end_s"],
                float(decode_result["video_frame_rate"]),
                repair_parameters.remove_gyro_motion,
            )
            repair_parameters = replace(repair_parameters, do_not_repair=excluded)
        # A run that resolved the list against the cached detection result and then ran
        # the detection stage after all resolved the list again above; only the warnings
        # of the last resolution are logged.
        for code, values in resolution_warnings:
            errors.log_warning(log, code, **values)

    if last_stage >= _REPAIR and run_from <= _REPAIR:
        try:
            repair_result = repair_stage.repair(
                decode_result, detection_result, repair_axes, repair_parameters
            )
        except errors.GyroDoctorError as error:
            error.values.setdefault("file", video)
            raise
        if cache_on:
            repair_stage.write_repair_result(
                repair_result,
                video_cache_directory,
                file_stamps={
                    "decode_result": decode_stage.file_stamp(decode_path),
                    "detection_result": decode_stage.file_stamp(detection_path),
                },
            )
        record = json.loads(str(repair_result["repair_record"]))
        if args.report:
            _print(repair_report(record))
        elif args.verbose:
            print(repair_report_line(record))
    elif last_stage >= _REPAIR and print_cached_lines:
        print(f"{REPAIR_REPORT_HEADING}{CACHED_RESULT} {repair_path}")

    if last_stage == _WRITE:
        report = write_stage.write(video, repair_result, output_path, force=args.force)
        if cache_on:
            write_stage.store_write_record(
                video_cache_directory, video.stem, output_path, repair_path
            )
        if args.report:
            _print(write_report(video, output_path, report))
        elif args.verbose:
            print(write_report_line(output_path, report))

    if cache_on and args.report:
        files = [decode_path, readme_path]
        if last_stage >= _DETECTION:
            files += [detection_path, run_record_path, *extent_table_paths]
        if last_stage >= _REPAIR:
            files += [repair_path, repair_record_path]
        if last_stage == _WRITE:
            files.append(write_record_path)
        _print([FILES_HEADING, *(str(path) for path in files if path.exists())])
