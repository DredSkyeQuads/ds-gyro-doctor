# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""``gyro-doctor-render`` renders the stabilized video before and after the repair
side by side using Gyroflow.

The ``gyro-doctor-render`` command, in its two modes: ``original-vs-repair``, the
render of the original against the repaired video, and ``raw-vs-stab``, the render of the
video as shot. Either mode composes two columns side by side under a title bar, each a
video pane over a three panel scrolling rate strip, from the Gyroflow renders and the
unstabilized recode that the command keeps under ``<cache>/<video stem>/render/`` and
from the results that ``gyro-doctor`` stored for the video in the cache; the command
runs no stage.
"""

import argparse
import contextlib
import functools
import importlib.metadata
import json
import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import av
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from gyro_doctor import errors
from gyro_doctor.cli import parse_duration, parse_position
from gyro_doctor.core import decode_stage, detection_stage, repair_stage, write_stage
from gyro_doctor.core.axes import axis_column
from gyro_doctor.core.probe import keyframe_indices
from gyro_doctor.tools import chart
from gyro_doctor.tools.chart import (
    READ_ERROR_CODES,
    TRANSIENT_OUTLIER_THRESHOLD,
    _largest_absolute_rate,
    _largest_absolute_rate_over_three_axes,
    _peaks_beyond_rate_axis,
    _rate_axis_scale,
    _read_decode_result,
    _read_detection_result,
    detection_result_tested,
    repair_result_tested,
)

log = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# The Gyroflow dials and the presets
# -------------------------------------------------------------------------

LENS_CORRECTION_AMOUNT = 0.25
DEFAULT_FOV = 0.9
DEFAULT_SMOOTHNESS = 0.5
# Gyroflow's adaptive zoom window: 0 turns adaptive zoom off, a value below -0.9 is a
# static zoom, and a value above 0 a dynamic zoom over that many seconds.
DEFAULT_ZOOM_WINDOW = 0.0
# The depth of the solver of the crop that Gyroflow's "Max zoom" caps.
MAX_ZOOM_ITERATIONS = 5

STANDARD = {
    "smoothness": DEFAULT_SMOOTHNESS,
    "fov": DEFAULT_FOV,
    "lens": LENS_CORRECTION_AMOUNT,
    "zoom_window": DEFAULT_ZOOM_WINDOW,
    "max_zoom": None,
}
ZOOMING = {
    "smoothness": 0.15,
    "fov": 1.0,
    "lens": 0.20,
    "zoom_window": 2.0,
    "max_zoom": 130.0,
}
PRESETS = {"standard": STANDARD, "zooming": ZOOMING}
PRESET_NAMES = {"standard": "STANDARD", "zooming": "Zooming"}


def resolve_preset(preset):
    """The dials of the preset named ``preset``."""
    return PRESETS[preset]


def provenance_line(preset):
    """The line of the Gyroflow dials of a preset, ``name: key=value ...``, ASCII and
    single spaced, with units: lens and maxzoom in percent, and zoomspeed, Gyroflow's
    "Zooming speed", in seconds. ``maxzoom`` is printed only when the preset carries it, so
    the line states what the Gyroflow render used and nothing it did not."""
    dials = resolve_preset(preset)
    parts = [
        f"SM={dials['smoothness']:g}",
        f"FOV={dials['fov']:g}",
        f"lens={dials['lens'] * 100:g}%",
        f"zoomspeed={dials['zoom_window']}s",
    ]
    if dials["max_zoom"] is not None:
        parts.append(f"maxzoom={dials['max_zoom']:g}%")
    return f"{PRESET_NAMES[preset]}: {' '.join(parts)}"


# -------------------------------------------------------------------------
# Gyroflow
# -------------------------------------------------------------------------

# Gyroflow's integration method that reads a video's own quaternions rather than deriving
# an orientation from a rate track.
NATIVE_INTEGRATION_METHOD = 0


def resolve_gyroflow():
    """The Gyroflow executable that ``GYRO_DOCTOR_GYROFLOW`` names, with no search of
    ``PATH``. Raises ``errors.CannotWriteError`` when the variable is unset or names no
    file, to which the caller adds the path of the composite as ``file``."""
    value = os.environ.get("GYRO_DOCTOR_GYROFLOW")
    if not value:
        raise errors.CannotWriteError(errors.ErrorCode.GYROFLOW_NOT_SET)
    executable = Path(value)
    if not executable.is_file():
        raise errors.CannotWriteError(errors.ErrorCode.GYROFLOW_NOT_A_FILE, path=value)
    return executable


def _run_gyroflow(executable, arguments):
    """Run Gyroflow, a program of the graphical subsystem, with no window, and return what
    it wrote to stdout and stderr. Its exit status is not tested: whether it wrote its
    render is, by the caller."""
    process = subprocess.run(
        [str(executable), *arguments],
        cwd=str(Path(executable).parent),
        capture_output=True,
        text=True,
        timeout=2400,
        check=False,
    )
    return process.stdout + process.stderr


def render_gyroflow_project(executable, project, gyroflow_render):
    """Render the Gyroflow project at ``project`` through the Gyroflow command line into
    ``gyroflow_render``.

    Both paths are made absolute, since Gyroflow runs in the directory of its executable,
    where a relative project path names no file and Gyroflow renders nothing. A Gyroflow
    run that writes no render logs the end of Gyroflow's output at INFO level and raises
    ``errors.CannotWriteError``."""
    project = Path(project).resolve()
    gyroflow_render = Path(gyroflow_render).resolve()
    output = _run_gyroflow(executable, [str(project), "-f", "--stdout-progress"])
    if not gyroflow_render.exists():
        log.info("%s", output[-2000:])
        raise errors.CannotWriteError(
            errors.ErrorCode.GYROFLOW_WROTE_NO_RENDER,
            file=gyroflow_render,
            project=project,
        )


def file_url(path):
    """Gyroflow's ``file:///`` URL of a local path, with forward slashes."""
    return "file:///" + Path(path).resolve().as_posix().lstrip("/")


def _path_from_file_url(url):
    """The path that a ``file:///`` URL of a Gyroflow project names."""
    return Path(url2pathname(urlparse(str(url)).path)).resolve()


def apply_stabilization_dials(
    stabilization,
    *,
    smoothness,
    fov,
    lens_correction,
    adaptive_zoom_window,
    max_zoom,
    max_zoom_iterations,
):
    """Set the dials of a Gyroflow render in the ``stabilization`` block of a Gyroflow
    project, and leave every other field as it is: the lens correction amount, the field of
    view, the adaptive zoom window, the smoothness of the smoothing method, and, when
    ``max_zoom`` is not None, Gyroflow's "Max zoom" with the depth of its solver."""
    stabilization["lens_correction_amount"] = lens_correction
    stabilization["fov"] = fov
    stabilization["adaptive_zoom_window"] = adaptive_zoom_window
    for parameter in stabilization["smoothing_params"]:
        parameter["value"] = smoothness
    if max_zoom is not None:
        stabilization["max_zoom"] = max_zoom
        stabilization["max_zoom_iterations"] = max_zoom_iterations
    return stabilization


def container_signature(video_path):
    """The size and the modification time in nanoseconds of the video at ``video_path``,
    in hexadecimal, which the file name of its Gyroflow render carries.

    Without it a video rewritten in place would be answered by the Gyroflow render of its
    earlier content, and a repaired video has the size of the original by construction, so
    its video frame count cannot tell the two apart. It signs the input video, whose
    modification time Gyroflow does not back date."""
    stat = Path(video_path).stat()
    return f"{stat.st_size:x}-{stat.st_mtime_ns:x}"


def _trim_text(trim_ranges_ms):
    """The trim of a Gyroflow project as one token, each range its first and last
    millisecond with a hyphen between, such as ``0-15000``."""
    return ",".join(f"{start:g}-{end:g}" for start, end in trim_ranges_ms or ())


def verify_gyroflow_render(
    gyroflow_render, video_path, *, video_frame_count, trim_ranges_ms, stabilization
):
    """Raise ``errors.CannotReadError`` unless the Gyroflow render ``gyroflow_render`` is
    the render of ``video_path`` asked for: the Gyroflow project beside it names
    ``video_path`` as its video file and as its gyro source, integrates with
    ``NATIVE_INTEGRATION_METHOD``, carries no inline motion data, and holds
    ``trim_ranges_ms`` and ``stabilization``, the smoothing method and the dials; and the
    Gyroflow render holds at least ``video_frame_count`` video frames. An error of the
    project has the project as its ``file``, and an error of the Gyroflow render the
    Gyroflow render. To the error of other dials the caller adds the name of the preset
    as ``preset``.

    Gyroflow renders a few video frames past the end of its trim, a number that varies
    with Gyroflow, and the composite leaves them out; only a Gyroflow render with fewer
    than ``video_frame_count`` video frames cannot make the composite. It reads back the
    project Gyroflow was handed, not what the caller meant to write, and counts the video
    frames of the Gyroflow render, never its modification time, which Gyroflow back dates.
    The container signature in the file name keys the input video."""
    gyroflow_render = Path(gyroflow_render)
    project_path = gyroflow_render.with_suffix(".gyroflow")
    resolved_video_path = Path(video_path).resolve()
    if not project_path.exists():
        raise errors.CannotReadError(
            errors.ErrorCode.GYROFLOW_PROJECT_NOT_FOUND, file=project_path
        )
    try:
        with errors.operating_system_error_as(errors.CannotReadError, project_path):
            project = json.loads(project_path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise errors.CannotReadError(
            errors.ErrorCode.GYROFLOW_PROJECT_INVALID, file=project_path
        ) from error

    def same(path):
        return os.path.normcase(str(path)) == os.path.normcase(str(resolved_video_path))

    gyro_source = project.get("gyro_source") or {}
    if not project.get("videofile") or not same(
        _path_from_file_url(project["videofile"])
    ):
        raise errors.CannotReadError(
            errors.ErrorCode.GYROFLOW_PROJECT_VIDEO_DIFFERS,
            file=project_path,
            named=project.get("videofile"),
            video=resolved_video_path,
        )
    if not gyro_source.get("filepath") or not same(
        _path_from_file_url(gyro_source["filepath"])
    ):
        raise errors.CannotReadError(
            errors.ErrorCode.GYROFLOW_PROJECT_GYRO_SOURCE_DIFFERS,
            file=project_path,
            named=gyro_source.get("filepath"),
            video=resolved_video_path,
        )
    if gyro_source.get("integration_method") != NATIVE_INTEGRATION_METHOD:
        raise errors.CannotReadError(
            errors.ErrorCode.GYROFLOW_PROJECT_INTEGRATION_METHOD_DIFFERS,
            file=project_path,
            method=gyro_source.get("integration_method"),
            expected=NATIVE_INTEGRATION_METHOD,
        )
    if (
        gyro_source.get("raw_imu") is not None
        or gyro_source.get("quaternions") is not None
    ):
        raise errors.CannotReadError(
            errors.ErrorCode.GYROFLOW_PROJECT_CARRIES_INLINE_MOTION, file=project_path
        )
    if project.get("trim_ranges_ms") != trim_ranges_ms:
        raise errors.CannotReadError(
            errors.ErrorCode.GYROFLOW_PROJECT_TRIM_DIFFERS,
            file=project_path,
            trim=_trim_text(project.get("trim_ranges_ms")),
            expected=_trim_text(trim_ranges_ms),
        )
    if project.get("stabilization") != stabilization:
        raise errors.CannotReadError(
            errors.ErrorCode.GYROFLOW_PROJECT_STABILIZATION_SETTINGS_DIFFER,
            file=project_path,
            settings=project.get("stabilization"),
            expected=stabilization,
        )
    try:
        video_frames_rendered = _video_stream_info(gyroflow_render)["video_frame_count"]
    except av.error.FFmpegError as error:
        raise errors.CannotReadError(
            errors.ErrorCode.VIDEO_INVALID,
            file=gyroflow_render,
            reason=str(error),
        ) from error
    if video_frames_rendered < video_frame_count:
        raise errors.CannotReadError(
            errors.ErrorCode.GYROFLOW_RENDER_TOO_FEW_VIDEO_FRAMES,
            file=gyroflow_render,
            count=video_frames_rendered,
            expected=video_frame_count,
        )


def _video_stream_info(path):
    """The width, the height, the video frame count, and the video frame rate of the video
    stream of the video at ``path``, read through PyAV."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return {
            "width": stream.width,
            "height": stream.height,
            "video_frame_count": stream.frames,
            "video_frame_rate": stream.average_rate,
        }


def _gyroflow_render_path(render_directory, source, tag, first, last, preset):
    """The path that ``render_native_leg`` takes for the Gyroflow render of video frames
    ``first`` to ``last`` of the video ``source`` with ``preset``, before it folds in the
    container signature. Every Gyroflow render takes its path from here, so a Gyroflow
    render of the original video made in one mode is found and reused in the other."""
    return render_directory / f"{source.stem[:24]}_{tag}_f{first}-f{last}_{preset}.mp4"


def render_native_leg(video_path, gyroflow_render_path, *, first, last, dials):
    """The Gyroflow render of video frames ``first`` to ``last`` of the video at
    ``video_path``, stabilized from the video's own gyro data track with ``dials``, and
    whether it was rendered now: ``(path, rendered)``.

    The container signature of the video is folded into the file name, so the path is
    not ``gyroflow_render_path`` as given. A Gyroflow render that already stands under
    that name is reused only when it passes ``verify_gyroflow_render`` for video frames
    ``first`` to ``last`` and ``dials``, and is otherwise rendered again, a project that
    does not parse included; a Gyroflow render just made that fails it raises the
    ``errors.CannotReadError`` of the test. A video that PyAV cannot read raises
    ``errors.CannotReadError`` with the code ``VIDEO_INVALID``, and an error of the
    operating system while the Gyroflow project is written raises
    ``errors.CannotWriteError`` of the project.

    The render writes the Gyroflow project itself, beside the Gyroflow render: the video as
    the video file and as the gyro source; the geometry and timing of the video, since
    Gyroflow reads a project without probing its video and converts the trim through its
    duration; the smoothing method and the dials; the trim; and the output. It carries no
    ``calibration_data``, so Gyroflow loads the gyro data track, the lens profile, and the
    rolling shutter readout time from the video."""
    gyroflow_render_path = Path(gyroflow_render_path).resolve()
    try:
        gyroflow_render_path = gyroflow_render_path.with_name(
            f"{gyroflow_render_path.stem}_{container_signature(video_path)}"
            f"{gyroflow_render_path.suffix}"
        )
        stream_info = _video_stream_info(video_path)
    except av.error.FFmpegError as error:
        raise errors.CannotReadError(
            errors.ErrorCode.VIDEO_INVALID, file=video_path, reason=str(error)
        ) from error
    video_frame_rate = float(stream_info["video_frame_rate"])
    # Gyroflow compares the trim with the time of each video frame rounded to a whole
    # microsecond, so a trim that starts at the exact time of the first video frame can fall
    # just after it and lose it, as video frame 10 of a video at 60000/1001 video frames per
    # second is lost; a start 1 microsecond earlier cannot.
    trim_ranges_ms = [
        [
            max(first / video_frame_rate * 1000.0 - 0.001, 0.0),
            (last + 1) / video_frame_rate * 1000.0,
        ]
    ]
    stabilization = apply_stabilization_dials(
        {"method": "Default", "smoothing_params": [{"name": "smoothness"}]},
        smoothness=dials["smoothness"],
        fov=dials["fov"],
        lens_correction=dials["lens"],
        adaptive_zoom_window=dials["zoom_window"],
        max_zoom=dials["max_zoom"],
        max_zoom_iterations=MAX_ZOOM_ITERATIONS,
    )
    video_frame_count = last - first + 1
    if gyroflow_render_path.exists():
        try:
            verify_gyroflow_render(
                gyroflow_render_path,
                video_path,
                video_frame_count=video_frame_count,
                trim_ranges_ms=trim_ranges_ms,
                stabilization=stabilization,
            )
        except errors.CannotReadError:
            # Removed before Gyroflow renders it again, so that a Gyroflow run that
            # writes nothing is never answered by the old file.
            gyroflow_render_path.unlink()
        else:
            return gyroflow_render_path, False

    executable = resolve_gyroflow()
    project = {
        "videofile": file_url(video_path),
        "video_info": {
            "width": stream_info["width"],
            "height": stream_info["height"],
            "num_frames": stream_info["video_frame_count"],
            "fps": video_frame_rate,
            "duration_ms": stream_info["video_frame_count"] / video_frame_rate * 1000.0,
        },
        "stabilization": stabilization,
        "gyro_source": {
            "filepath": file_url(video_path),
            "integration_method": NATIVE_INTEGRATION_METHOD,
        },
        "trim_ranges_ms": trim_ranges_ms,
        "output": {
            "codec": "H.265/HEVC",
            "output_folder": file_url(gyroflow_render_path.parent) + "/",
            "output_filename": gyroflow_render_path.name,
            "output_width": 1280,
            "output_height": 720,
            "use_gpu": True,
        },
    }
    project_path = gyroflow_render_path.with_suffix(".gyroflow")
    with errors.operating_system_error_as(errors.CannotWriteError, project_path):
        gyroflow_render_path.parent.mkdir(parents=True, exist_ok=True)
        project_path.write_text(json.dumps(project, indent=2), encoding="utf-8")
    render_gyroflow_project(executable, project_path, gyroflow_render_path)
    verify_gyroflow_render(
        gyroflow_render_path,
        video_path,
        video_frame_count=video_frame_count,
        trim_ranges_ms=trim_ranges_ms,
        stabilization=stabilization,
    )
    return gyroflow_render_path, True


def crop_to_16_9(width, height):
    """The box ``(left, top, crop_width, crop_height)`` that crops a picture of ``width`` by
    ``height`` to 16:9 at its center, along whichever side is too long. A picture wider than
    16:9 keeps its height and its width becomes ``round(height * 16 / 9)``; a picture
    narrower than 16:9 keeps its width and its height becomes ``round(width * 9 / 16)``; a
    16:9 picture is not cropped. The cropped side is centered at ``(side - crop) // 2``."""
    if width * 9 > height * 16:
        crop_width = round(height * 16 / 9)
        return (width - crop_width) // 2, 0, crop_width, height
    crop_height = round(width * 9 / 16)
    return 0, (height - crop_height) // 2, width, crop_height


def render_raw_leg(video_path, recode_path, *, first, last, video_frame_rate):
    """The unstabilized recode of video frames ``first`` to ``last`` of the video at
    ``video_path``, written to ``recode_path`` again on every run, and its path.

    ``video_frame_rate`` is the video frame rate of the decode result. The recode seeks
    to the last keyframe at or before ``first`` that ``keyframe_indices`` finds, decodes
    forward counting video frames, crops each video frame with ``crop_to_16_9``, scales
    it to 1280 by 720, and encodes it through PyAV with ``libvpx-vp9``. A video with no
    keyframe at or before ``first`` and a seek that lands on another video frame raise
    ``errors.CannotReadError`` of the video; a first video frame written whose timestamp
    is not ``first``, a recode that holds another count of video frames, and an error of
    the operating system where the recode is opened for writing raise
    ``errors.CannotWriteError`` of the recode."""
    recode_path = Path(recode_path).resolve()
    keyframes = [
        index
        for index in keyframe_indices(video_path, video_frame_rate)
        if index <= first
    ]
    if not keyframes:
        raise errors.CannotReadError(
            errors.ErrorCode.KEYFRAME_MISSING_BEFORE_VIDEO_FRAME,
            file=video_path,
            video_frame=first,
        )
    keyframe = keyframes[-1]
    count = last - first + 1
    with errors.operating_system_error_as(errors.CannotWriteError, recode_path):
        recode_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with contextlib.ExitStack() as stack:
        source = stack.enter_context(av.open(str(video_path)))
        with errors.operating_system_error_as(errors.CannotWriteError, recode_path):
            target = stack.enter_context(av.open(str(recode_path), "w"))
        stream = source.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        left, top, crop_width, crop_height = crop_to_16_9(stream.width, stream.height)
        source.seek(
            round(keyframe / video_frame_rate / float(time_base)),
            backward=True,
            any_frame=False,
            stream=stream,
        )
        encoder = target.add_stream(
            "libvpx-vp9", rate=stream.average_rate, options=VP9_ENCODER_OPTIONS
        )
        encoder.width, encoder.height, encoder.pix_fmt = 1280, 720, "yuv420p"
        video_frame = None
        for decoded_video_frame in source.decode(stream):
            timestamp_video_frame = round(
                float(decoded_video_frame.pts * time_base) * video_frame_rate
            )
            if video_frame is None:
                if timestamp_video_frame != keyframe:
                    raise errors.CannotReadError(
                        errors.ErrorCode.SEEK_MISSED_KEYFRAME,
                        file=video_path,
                        keyframe=keyframe,
                        landed=timestamp_video_frame,
                    )
                video_frame = keyframe
            else:
                video_frame += 1
            if video_frame < first:
                continue
            if video_frame > last:
                break
            if written == 0 and timestamp_video_frame != first:
                raise errors.CannotWriteError(
                    errors.ErrorCode.RECODE_STARTS_AT_OTHER_VIDEO_FRAME,
                    file=recode_path,
                    first=timestamp_video_frame,
                    expected=first,
                )
            picture = decoded_video_frame.to_ndarray(format="rgb24")[
                top : top + crop_height, left : left + crop_width
            ]
            scaled = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(picture), format="rgb24"
            ).reformat(width=1280, height=720, format="yuv420p", interpolation="AREA")
            scaled.pts = written
            for packet in encoder.encode(scaled):
                target.mux(packet)
            written += 1
        for packet in encoder.encode():
            target.mux(packet)
    if written != count:
        raise errors.CannotWriteError(
            errors.ErrorCode.RECODE_VIDEO_FRAME_COUNT_DIFFERS,
            file=recode_path,
            count=written,
            expected=count,
        )
    return recode_path


# -------------------------------------------------------------------------
# The series, the difference, and the marks
# -------------------------------------------------------------------------


def frame_grid_average(times, values, video_frame_rate, video_frame_count):
    """The rate samples ``values`` at ``times`` averaged into ``video_frame_count``
    video frames at ``video_frame_rate``: video frame i covers
    [i / video_frame_rate, (i + 1) / video_frame_rate), and its value is the mean of
    every finite rate sample whose time lies in it, or not a number where none does. A
    rate sample is never picked out per video frame, which would alias."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    video_frame = np.floor(times * video_frame_rate).astype(np.int64)
    keep = (video_frame >= 0) & (video_frame < video_frame_count) & np.isfinite(values)
    video_frame, values = video_frame[keep], values[keep]
    sums = np.bincount(video_frame, weights=values, minlength=video_frame_count)
    counts = np.bincount(video_frame, minlength=video_frame_count)
    averaged = np.full(video_frame_count, np.nan)
    covered = counts > 0
    averaged[covered] = sums[covered] / counts[covered]
    return averaged


def normalized_rate_per_video_frame(result, video_frame_rate, video_frame_count):
    """The Normalized Rate of a decode result, or the repaired Normalized Rate of a
    repair result, averaged per video frame: one row per video frame of the video, one
    column per axis."""
    times = result["recorded_rate_times_s"]
    rates = np.asarray(result["normalized_rate_deg_s"])
    return np.stack(
        [
            frame_grid_average(
                times, rates[:, column], video_frame_rate, video_frame_count
            )
            for column in range(rates.shape[1])
        ],
        axis=1,
    )


# Two series averaged per video frame count as equal where they differ by no more than
# this many degrees per second. Outside every extent the original and the repaired
# Normalized Rate differ only by the rounding of the rebuild, far below it.
_DIFFERENCE_TOLERANCE_DEG_S = 0.5


def difference_spans(reference, other):
    """Per axis column, the spans ``(first, last)`` of video frames where ``reference``
    and ``other`` differ by more than ``_DIFFERENCE_TOLERANCE_DEG_S``, for all three
    axes, so a change on an axis the repair left alone shows. A video frame where either
    series is not a number counts as equal."""
    reference, other = np.asarray(reference), np.asarray(other)
    spans_by_axis = {}
    for column in range(reference.shape[1]):
        differ = (
            np.abs(reference[:, column] - other[:, column])
            > _DIFFERENCE_TOLERANCE_DEG_S
        )
        spans, start, count = [], 0, len(differ)
        while start < count:
            if differ[start]:
                end = start
                while end + 1 < count and differ[end + 1]:
                    end += 1
                spans.append((start, end))
                start = end + 1
            else:
                start += 1
        spans_by_axis[column] = spans
    return spans_by_axis


# The color of the marks of each classification string, in the order of the legend. They
# skip the orange and the turquoise of the two series, the white of the marker of the video
# frame on screen, and the purple of a difference no mark covers; green is not used.
DECORATOR_COLORS = {
    detection_stage.LARGE_AMPLITUDE_OSCILLATION: "#CC79A7",
    detection_stage.MID_AMPLITUDE_BURST: "#F0E442",
    detection_stage.IMPULSE_EVENT: "#1f4fd6",
}


def parse_color_override(text):
    """``--decorator-color``: ``"<classification string>=#rrggbb"`` as
    ``(classification string, "#rrggbb")``. The classification string may hold spaces;
    only the first ``=`` separates. Any other text raises
    ``argparse.ArgumentTypeError``."""
    label, separator, value = text.partition("=")
    value = value.strip()
    if not separator or not re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
        raise argparse.ArgumentTypeError(
            errors.message(
                errors.ErrorCode.VALUE_NOT_A_CLASSIFICATION_AND_COLOR, text=text
            )
        )
    return label.strip(), value


def read_decorators(detection, colors):
    """The marks of the extents of the detection result ``detection``, as axis decorators
    in video frames of the video: one group per classification string, each with its
    label, its color, its spans ``(axis column, first video frame, last video frame)``, and
    its markers ``(axis column, video frame of the peak, identifier)``, the identifier the
    detection result stores for the extent.

    ``colors`` adds to or overrides ``DECORATOR_COLORS``. The groups come in the declared
    order of ``DECORATOR_COLORS``, then any classification string of ``colors`` alone, in
    the order given. A classification string with no declared color raises
    ``errors.CannotReadError`` naming it, to which the caller adds the path of the
    detection result as ``file``, never a color that falls back, since a color that falls
    back is how two groups come to look alike.

    The video frame of a rate sample is the floor of its ``rate_sample_video_frames``,
    the video frame whose average holds it."""
    declared = dict(DECORATOR_COLORS)
    declared.update(colors)
    labels = [str(label) for label in detection["extent_classification"]]
    unknown = sorted({label for label in labels if label not in declared})
    if unknown:
        raise errors.CannotReadError(
            errors.ErrorCode.CLASSIFICATION_WITHOUT_COLOR,
            classifications=",".join(unknown),
        )
    video_frame = np.floor(
        np.asarray(detection["rate_sample_video_frames"], dtype=np.float64)
    )
    first = video_frame[np.asarray(detection["extent_first_sample"], dtype=np.int64)]
    last = video_frame[np.asarray(detection["extent_last_sample"], dtype=np.int64)]
    peak = np.floor(np.asarray(detection["extent_peak_video_frame"], dtype=np.float64))
    identifiers = np.asarray(detection["extent_identifier"]).astype(str)
    spans, markers = {}, {}
    for index, label in enumerate(labels):
        column = axis_column(str(detection["extent_axis"][index]))
        spans.setdefault(label, []).append(
            (column, int(first[index]), int(last[index]))
        )
        markers.setdefault(label, []).append(
            (column, int(peak[index]), str(identifiers[index]))
        )
    return [
        {
            "label": label,
            "color": ImageColor.getrgb(declared[label]),
            "spans": spans[label],
            "markers": markers[label],
        }
        for label in declared
        if label in spans
    ]


def _mask_to_spans(series, spans_by_axis, pad):
    """A copy of ``series`` that is not a number everywhere except within ``pad`` video
    frames of each span of its axis, so the series shows only where it differs from the
    reference and rejoins it visibly at each end."""
    masked = np.full_like(series, np.nan)
    for column in range(series.shape[1]):
        for first, last in spans_by_axis.get(column, []):
            low, high = max(0, first - pad), min(len(series) - 1, last + pad)
            masked[low : high + 1, column] = series[low : high + 1, column]
    return masked


def localize_decorators(decorators, low, high):
    """Axis decorators in video frames of the video, localized to video frames ``low`` to
    ``high - 1``: spans ``(first, last)`` and markers ``(video frame, identifier)`` counted
    from video frame ``low``, by axis column, keeping only what falls inside those video
    frames. A group with nothing inside them is dropped, so the legend lists only what is
    drawn."""
    localized = []
    for group in decorators:
        spans, markers = {}, {}
        for column, first, last in group["spans"]:
            if last >= low and first < high:
                spans.setdefault(column, []).append(
                    (max(0, first - low), min(high - low - 1, last - low))
                )
        for column, video_frame, identifier in group["markers"]:
            if low <= video_frame < high:
                markers.setdefault(column, []).append((video_frame - low, identifier))
        if spans or markers:
            localized.append(
                {
                    "label": group["label"],
                    "color": group["color"],
                    "spans": spans,
                    "markers": markers,
                }
            )
    return localized


def build_leg(mp4, series, *, title, provenance, reference, decorators, window):
    """One column of the render, from the video ``mp4`` and ``series``, an ordered list
    of ``(label, array, color)`` whose arrays hold one row per video frame of the video.

    ``window`` = ``(low, high)`` slices every series to video frames ``low`` to
    ``high - 1`` of the video, and localizes ``decorators`` to them, so video frame 0 of
    the column is video frame ``low`` of the video. The trace labeled ``reference`` is
    drawn last, on top, the visible trace everywhere; every other trace is not a number
    except within one video frame of where it differs from the reference, and the spans of
    that difference are the computed difference of the column. The legend lists the series
    in the reverse of the drawing order, so that the reference, drawn over every other
    trace, comes first, and lists each group of marks with something in ``window``;
    ``prepare_column`` adds ``changed, not flagged`` where the fills draw a span of the
    difference that no mark claims."""
    low, high = window
    sliced = [(label, array[low:high], color) for label, array, color in series]
    reference_series = next(array for label, array, _ in sliced if label == reference)
    difference_spans_by_axis, ordered = {}, []
    for label, array, color in sliced:
        if label == reference:
            continue
        difference = difference_spans(reference_series, array)
        for column, spans in difference.items():
            difference_spans_by_axis.setdefault(column, []).extend(spans)
        ordered.append((label, _mask_to_spans(array, difference, 1), color))
    ordered += [entry for entry in sliced if entry[0] == reference]
    groups = localize_decorators(decorators, low, high)
    return {
        "mp4": str(mp4),
        "series": ordered,
        "title": title,
        "provenance": provenance,
        "difference_spans": difference_spans_by_axis,
        "decorators": groups,
        "series_legend": [
            (color, label, "line") for label, _, color in reversed(ordered)
        ],
        "marker_legend": [(group["color"], group["label"], "fill") for group in groups],
    }


def verbose_lines_of_column(column, first, last, video_frame_rate, preset):
    """The lines ``--verbose`` prints of what the column ``column`` draws, read from the
    column itself: video frames ``first`` to ``last``; per axis, the spans of the computed
    difference and their video frames; per group of marks, its spans and markers in those
    video frames and how many of their video frames the difference covers, since the rest
    draw faint. A column that draws one series computes no difference, so its lines state
    the spans and markers of each group alone."""
    names = ("pitch", "yaw", "roll")
    difference = column["difference_spans"]
    compared = len(column["series"]) > 1
    lines = [
        (
            f"video frames f{first} to f{last} "
            f"({first / video_frame_rate:.3f}-{(last + 1) / video_frame_rate:.3f} s, "
            f"{last - first + 1} video frames), preset {preset}"
        )
    ]
    for column_index, name in enumerate(names if compared else ()):
        spans = difference.get(column_index, [])
        video_frames = sum(end - start + 1 for start, end in spans)
        lines.append(
            f"    {name:5s}  {len(spans):4d} difference spans  {video_frames:6d} "
            f"video frames"
        )
    for group in column["decorators"]:
        span_list = [span for spans in group["spans"].values() for span in spans]
        merged = {
            column_index: _span_merge(spans)
            for column_index, spans in group["spans"].items()
        }
        total = sum(
            end - start + 1 for spans in merged.values() for start, end in spans
        )
        covered = sum(
            end - start + 1
            for column_index, spans in merged.items()
            for start, end in _span_intersect(spans, difference.get(column_index, []))
        )
        share = f"{100 * covered / total:.0f}%" if total else "n/a"
        marker_count = sum(len(markers) for markers in group["markers"].values())
        line = f"    {group['label']:30s} {len(span_list):4d} spans  {marker_count:4d} markers"
        if compared:
            line += f"  {covered}/{total} marked video frames changed ({share})"
        lines.append(line)
    return lines


# -------------------------------------------------------------------------
# The drawing
# -------------------------------------------------------------------------

COLUMN_WIDTH = 900
PANEL_HEIGHT = 80
PANEL_TOP = 6  # the gap between the video pane and the first panel
TIME_AXIS_HEIGHT = 20  # the scrolling seconds axis under the panels
LEGEND_HEIGHT = 100  # one fixed height for the worst case
LEGEND_ROW_Y = (46, 66, 86)  # the baselines of the item rows under the provenance line
GUTTER_WIDTH = 10
STRIP_BACKGROUND = 20
# The identifiers stand in rows below the panel name and the upper scale value, and each
# identifier keeps one row for the whole composite.
IDENTIFIER_PX = 13
IDENTIFIER_PAD = 2
IDENTIFIER_FIRST_ROW_TOP = 22
# The distance in pixels from the bottom of one row to the top of the next.
IDENTIFIER_ROW_GAP = 2
# The least distance in pixels from the right of one box to the left of the next box in
# one row of the identifiers.
IDENTIFIER_CLEARANCE_PX = 2
IDENTIFIER_ROW_COUNT = 3
# The callout of a peak beyond the rate axis starts this many pixels right of its peak,
# and no two callouts of one axis stand closer than the spacing.
CALLOUT_PX = 13
CALLOUT_PAD = 2
CALLOUT_OFFSET_PX = 3
CALLOUT_SPACING_PX = 76
AXIS_PANEL_NAMES = [("PITCH", 0), ("YAW", 1), ("ROLL", 2)]

# The series take colors chosen for the dark strip.
A_SERIES_COLOR = (255, 165, 40)
B_SERIES_COLOR = (40, 220, 205)
CHANGED_NOT_FLAGGED_COLOR = (106, 61, 154)
VIDEO_FRAME_ON_SCREEN_MARKER_COLOR = (255, 255, 255)
# Each label puts a colon after its letter, as the column titles do, since Pillow's
# default face draws the space after a capital letter almost invisibly.
ORIGINAL_SERIES_LABEL = "A: Normalized Rate (Original)"
REPAIRED_SERIES_LABEL = "B: Normalized Rate (Repaired)"
# Both columns of raw-vs-stab draw the one original Normalized Rate; A and B name the
# video of each column.
RAW_VS_STAB_A_SERIES_LABEL = "A: Normalized Rate (No stabilization)"
RAW_VS_STAB_B_SERIES_LABEL = "B: Normalized Rate (Stabilized)"

# A span the computed difference covers is filled at SPAN_FILL_ALPHA; a span a mark flags
# that the difference does not cover is filled at DECORATOR_UNCHANGED_ALPHA, so a fade or a
# stretch the repair left alone reads faint instead of claiming the whole extent changed.
SPAN_FILL_ALPHA = 0.28
DECORATOR_UNCHANGED_ALPHA = 0.10

# The column title is Roboto Medium, light on the black of the title bar, located through
# matplotlib's font manager rather than a path of one platform. It carries a whole file
# name, so it shrinks to fit and is never cut short: the end of the name tells the original
# from the repaired video. Medium explicitly, since the default weight resolves to Roboto
# Light.
TITLE_FAMILY = "Roboto"
TITLE_WEIGHT = 500
COLUMN_TITLE_MAX_PX = 22
COLUMN_TITLE_COLOR = (215, 215, 215)

# The title bar across the top of the composite, above both columns, holds the title and,
# under it, each column title centered over its column, in the face of the column title.
# It is 146 pixels high so that, above the columns of a 1280 by 720 Gyroflow render, 872
# pixels high, the composite is 1810 by 1018 pixels, 16:9; 1018 is the even height nearest
# 16:9, since libvpx-vp9 needs even dimensions. The title is larger than the column title,
# and the longer title, RAW_VS_STAB_TITLE, leaves at least 40 pixels clear on each side.
TITLE_BAR_HEIGHT = 146
TITLE_BAR_COLOR = (0, 0, 0)
TITLE_BAR_PX = 40
TITLE_BAR_TEXT_COLOR = (245, 197, 24)
TITLE_BAR_TEXT_CENTER_Y = 50
COLUMN_TITLE_CENTER_Y = 118
ORIGINAL_VS_REPAIR_TITLE = (
    "Gyro Doctor: DJI Gyro Noise Visualization - Original vs. Repaired"
)
RAW_VS_STAB_TITLE = (
    "Gyro Doctor: DJI Gyro Noise Visualization - Raw vs. Stabilized (not repaired)"
)

SCROLL_WINDOW_S = 3.0
HEADROOM = 1.25
# A peak is called out where it passes the rate axis by this factor.
CALLOUT_FACTOR = 1.15


@functools.lru_cache(maxsize=64)
def _title_font(size_px):
    """The Roboto face at ``size_px`` pixels, or the face matplotlib finds closest when
    Roboto is not installed. Cached by size, since the column titles try one size after
    another as they shrink to fit."""
    from matplotlib import font_manager

    path = font_manager.findfont(
        font_manager.FontProperties(family=TITLE_FAMILY, weight=TITLE_WEIGHT),
        fallback_to_default=True,
    )
    return ImageFont.truetype(path, size_px)


@functools.lru_cache(maxsize=16)
def _default_font(size_px):
    """Pillow's own default face at ``size_px`` pixels, for every label but the
    title."""
    return ImageFont.load_default(size_px)


def _blend_rectangle(image, box, color, alpha):
    """Blend ``color`` at ``alpha`` over the rectangle ``box`` = ``(left, top, right,
    bottom)`` of ``image``, clipped to the image."""
    left, top = max(box[0], 0), max(box[1], 0)
    right, bottom = min(box[2], image.width), min(box[3], image.height)
    region = np.asarray(image.crop((left, top, right, bottom)), dtype=np.float64)
    blended = alpha * np.asarray(color, dtype=np.float64) + (1 - alpha) * region
    image.paste(Image.fromarray(blended.astype(np.uint8)), (left, top))


def _column_title_px(column_titles):
    """The one size at which every title of ``column_titles`` is drawn, so that no
    column title is drawn smaller than another: the largest whole size from
    ``COLUMN_TITLE_MAX_PX`` down to 10 pixels at which the widest of them, as drawn,
    fits the column width less 16 pixels. Where the widest is wider than that at 10
    pixels, every title is drawn whole at 10 pixels, never cut short."""
    size_px = COLUMN_TITLE_MAX_PX
    while size_px > 10 and any(
        _title_font(size_px).getbbox(title)[2] > COLUMN_WIDTH - 16
        for title in column_titles
    ):
        size_px -= 1
    return size_px


def _label_box(image, text, top_left, *, size_px=15, pad=5):
    """Dark text on a light grey box, half transparent, with a border, legible over any
    footage. ``top_left`` is the top left corner of the box."""
    font = _default_font(size_px)
    left, top, right, bottom = font.getbbox(text)
    x, y = top_left
    box_right, box_bottom = x + (right - left) + 2 * pad, y + (bottom - top) + 2 * pad
    _blend_rectangle(image, (x, y, box_right, box_bottom), (175, 175, 175), 0.5)
    draw = ImageDraw.Draw(image)
    draw.rectangle((x, y, box_right, box_bottom), outline=(90, 90, 90))
    draw.text((x + pad - left, y + pad - top), text, font=font, fill=(25, 25, 25))


def _label_box_size(text, *, size_px, pad):
    """The width and height of the box that ``_label_box`` draws for ``text``."""
    left, top, right, bottom = _default_font(size_px).getbbox(text)
    return right - left + 2 * pad, bottom - top + 2 * pad


def identifier_box_size(identifier):
    """The width and height of the box of ``identifier``. Every identifier is capital
    letters and digits, so every box has the height of ``P0``."""
    width, _height = _label_box_size(
        identifier, size_px=IDENTIFIER_PX, pad=IDENTIFIER_PAD
    )
    return width, _label_box_size("P0", size_px=IDENTIFIER_PX, pad=IDENTIFIER_PAD)[1]


def callout_text(peak):
    """The callout of a peak beyond the rate axis: its direction and its value."""
    return f"{'^' if peak > 0 else 'v'} {abs(peak):.1f}"


def callout_top(peak):
    """The top of the box of the callout of ``peak``, counted from the top of its panel:
    at the top of the panel for an upward peak, at the bottom for a downward one."""
    return 4 if peak > 0 else PANEL_HEIGHT - 20


def identifier_row_band(row):
    """The top and the bottom of row ``row`` of the identifiers, counted from the top of
    its panel."""
    height = identifier_box_size("P0")[1]
    top = IDENTIFIER_FIRST_ROW_TOP + row * (height + IDENTIFIER_ROW_GAP)
    return top, top + height


def kept_callouts(peaks_beyond_rate_axis, video_frames_per_pixel):
    """The callouts of a composite, ``{axis column: [(video frame, signed peak), ...]}``
    in order along the video, decided once from every peak of
    ``peaks_beyond_rate_axis``: on each axis the largest peaks first, leaving out a peak
    within ``CALLOUT_SPACING_PX`` of one already kept, so that a callout never appears
    or disappears as the strip scrolls."""
    spacing = CALLOUT_SPACING_PX * video_frames_per_pixel
    kept = {}
    for axis_index, peaks in peaks_beyond_rate_axis.items():
        chosen = []
        for video_frame, peak in sorted(
            peaks, key=lambda video_frame_and_peak: -abs(video_frame_and_peak[1])
        ):
            if all(abs(video_frame - other) >= spacing for other, _peak in chosen):
                chosen.append((video_frame, peak))
        kept[axis_index] = sorted(chosen)
    return kept


def identifier_rows(decorators, callouts, video_frames_per_pixel):
    """The row of each identifier in its panel, ``{(axis column, identifier): row}``,
    decided once from every extent of ``decorators`` and every callout of ``callouts``,
    so that an identifier keeps its row as the strip scrolls and covers no callout and no
    other identifier wherever a row is free at its place.

    Every label rides with one point of the series, so its place along the video, in
    video frames, is the same on every video frame of the composite: an identifier ends
    one pixel left of its marker, and a callout starts ``CALLOUT_OFFSET_PX`` right of its
    peak. A callout reserves its place in every row its box crosses. In order along the
    video, each identifier takes the first row where it covers nothing already reserved,
    or else, with every row taken there, the row where it covers the fewest."""
    gap = IDENTIFIER_CLEARANCE_PX * video_frames_per_pixel
    taken = {}
    for axis_index, axis_callouts in callouts.items():
        for video_frame, peak in axis_callouts:
            width, height = _label_box_size(
                callout_text(peak), size_px=CALLOUT_PX, pad=CALLOUT_PAD
            )
            start = video_frame + CALLOUT_OFFSET_PX * video_frames_per_pixel
            place = (start, start + width * video_frames_per_pixel)
            box_top = callout_top(peak)
            for row in range(IDENTIFIER_ROW_COUNT):
                row_top, row_bottom = identifier_row_band(row)
                if row_top <= box_top + height and box_top <= row_bottom:
                    taken.setdefault(
                        axis_index, [[] for _row in range(IDENTIFIER_ROW_COUNT)]
                    )[row].append(place)
    places = {}
    for group in decorators:
        for axis_index, markers in group["markers"].items():
            for video_frame, identifier in markers:
                end = video_frame - video_frames_per_pixel
                places[axis_index, identifier] = (
                    end - identifier_box_size(identifier)[0] * video_frames_per_pixel,
                    end,
                )
    rows = {}
    for (axis_index, identifier), (start, end) in sorted(
        places.items(), key=lambda item: item[1]
    ):
        reserved = taken.setdefault(
            axis_index, [[] for _row in range(IDENTIFIER_ROW_COUNT)]
        )
        covered = [
            sum(
                start < other_end + gap and other_start < end + gap
                for other_start, other_end in reserved[row]
            )
            for row in range(IDENTIFIER_ROW_COUNT)
        ]
        row = next(
            (candidate for candidate, count in enumerate(covered) if count == 0),
            int(np.argmin(covered)),
        )
        reserved[row].append((start, end))
        rows[axis_index, identifier] = row
    return rows


def _legend_entry(draw, x, y, color, term, kind):
    """Draw one swatch and ``term`` with its baseline at ``y``, and return the x just past
    the term. ``kind`` is ``fill``, a solid rectangle, or ``line``."""
    if kind == "fill":
        draw.rectangle((x, y - 9, x + 20, y - 1), fill=color)
    else:
        draw.line((x, y - 5, x + 20, y - 5), fill=color, width=2)
    text_x = x + 24
    font = _default_font(11)
    draw.text((text_x, y), term, font=font, fill=(215, 215, 215), anchor="ls")
    return text_x + draw.textlength(term, font=font)


def _legend_band(width, provenance, series_items, marker_items):
    """The legend band of one column, one fixed height on a dark ground: the provenance
    line on top; on the left the series, one per row; and, set off to the right of them by
    a divider, the marks, in columns filled down the rows. Only what the column draws is
    listed."""
    band = Image.new("RGB", (width, LEGEND_HEIGHT), (32, 32, 32))
    draw = ImageDraw.Draw(band)
    draw.line((0, 0, width, 0), fill=(80, 80, 80))
    draw.text(
        (10, 20), provenance, font=_default_font(12), fill=(205, 205, 205), anchor="ls"
    )

    series_right = 10
    for row, (color, term, kind) in enumerate(series_items):
        end = _legend_entry(draw, 10, LEGEND_ROW_Y[row], color, term, kind)
        series_right = max(series_right, end)

    if marker_items:
        marker_x = series_right + 24
        draw.line(
            (marker_x - 12, 40, marker_x - 12, LEGEND_HEIGHT - 6), fill=(70, 70, 70)
        )
        rows, column_x, column_width = len(LEGEND_ROW_Y), marker_x, 0
        for index, (color, term, kind) in enumerate(marker_items):
            if index and index % rows == 0:
                column_x += column_width + 28
                column_width = 0
            end = _legend_entry(
                draw, column_x, LEGEND_ROW_Y[index % rows], color, term, kind
            )
            column_width = max(column_width, end - column_x)
    return band


def _polyline_split(draw, xs, ys, color):
    """Draw a polyline through ``(xs, ys)``, broken where a value is not a number. The y
    values are never clipped: Pillow clips the drawing to the image, so a value beyond the
    rate axis leaves the panel rather than lying flat along its edge."""
    points = []
    for x, y in zip(xs, ys, strict=True):
        if np.isfinite(y):
            points.append((x, round(float(y))))
            continue
        if len(points) > 1:
            draw.line(points, fill=color, width=1)
        points = []
    if len(points) > 1:
        draw.line(points, fill=color, width=1)


def _span_intersect(spans, other_spans):
    """The spans of video frames, both ends included, present in both lists."""
    intersection = []
    for span_first, span_last in spans:
        for other_first, other_last in other_spans:
            low, high = max(span_first, other_first), min(span_last, other_last)
            if low <= high:
                intersection.append((low, high))
    return intersection


def _span_merge(spans):
    """The spans of video frames ``spans``, both ends included, sorted, with spans that
    overlap or touch joined into one, so no video frame lies in two spans."""
    merged = []
    for first, last in sorted(spans):
        if merged and first <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def _span_subtract(spans, cut):
    """``spans`` with every span of ``cut`` removed, so the result and ``cut`` partition
    the original and no video frame is covered twice."""
    remaining = []
    for span_first, span_last in spans:
        uncut_spans = [(span_first, span_last)]
        for cut_first, cut_last in cut:
            kept = []
            for uncut_first, uncut_last in uncut_spans:
                if cut_last < uncut_first or cut_first > uncut_last:
                    kept.append((uncut_first, uncut_last))
                    continue
                if uncut_first < cut_first:
                    kept.append((uncut_first, cut_first - 1))
                if cut_last < uncut_last:
                    kept.append((cut_last + 1, uncut_last))
            uncut_spans = kept
        remaining.extend(uncut_spans)
    return remaining


def _fill_plan(decorators, difference):
    """The fills of one column, ``[(axis column, spans, color, alpha), ...]`` in drawing
    order, and per axis column the spans of the difference that no mark claims, which the
    plan fills in the color of ``changed, not flagged``: ``(plan, changed_not_flagged)``.
    Per axis, three regions from the marks and the computed difference:

      marked and changed: the color of the group, at SPAN_FILL_ALPHA;
      marked, not changed: the color of the group, at DECORATOR_UNCHANGED_ALPHA, faint;
      changed, not marked: CHANGED_NOT_FLAGGED_COLOR, a difference no mark accounts for.

    The regions partition each axis, so no video frame is filled twice and blended into a
    color that means nothing; where two classifications claim one video frame, the group
    declared first keeps it. The plan depends on the column alone, never on the video frame
    drawn, so ``prepare_column`` builds it once."""
    plan, changed_not_flagged = [], {}
    for column in range(3):
        changed_spans = difference.get(column, [])
        claimed, taken = [], []
        for group in decorators:
            spans = _span_subtract(group["spans"].get(column, []), taken)
            if not spans:
                continue
            changed = _span_intersect(spans, changed_spans)
            plan.append(
                (
                    column,
                    _span_subtract(spans, changed_spans),
                    group["color"],
                    DECORATOR_UNCHANGED_ALPHA,
                )
            )
            plan.append((column, changed, group["color"], SPAN_FILL_ALPHA))
            claimed.extend(changed)
            taken.extend(spans)
        changed_not_flagged[column] = _span_subtract(changed_spans, claimed)
        plan.append(
            (
                column,
                changed_not_flagged[column],
                CHANGED_NOT_FLAGGED_COLOR,
                SPAN_FILL_ALPHA,
            )
        )
    return plan, changed_not_flagged


def prepare_column(column, video_frames_each_side, peaks_beyond_rate_axis):
    """A column with the parts of it that no video frame changes: its fills, the
    callouts it keeps of the peaks of ``peaks_beyond_rate_axis`` and the row of each
    identifier clear of them, for a strip that shows ``2 * video_frames_each_side``
    video frames, and its rendered legend band. Built once before the pass over the
    video frames, never per video frame.

    The marker legend lists ``changed, not flagged`` after the groups of marks only where
    the fills draw a span of the difference that no mark claims, so a difference that the
    marks claim in full lists no entry for a color the column never draws."""
    video_frames_per_pixel = 2 * video_frames_each_side / COLUMN_WIDTH
    callouts = kept_callouts(peaks_beyond_rate_axis, video_frames_per_pixel)
    fills, changed_not_flagged = _fill_plan(
        column["decorators"], column["difference_spans"]
    )
    marker_legend = list(column["marker_legend"])
    if any(changed_not_flagged.values()):
        marker_legend.append(
            (CHANGED_NOT_FLAGGED_COLOR, "changed, not flagged", "fill")
        )
    return {
        **column,
        "fills": fills,
        "callouts": callouts,
        "identifier_rows": identifier_rows(
            column["decorators"], callouts, video_frames_per_pixel
        ),
        "legend_band": _legend_band(
            COLUMN_WIDTH,
            column["provenance"],
            column["series_legend"],
            marker_legend,
        ),
    }


def draw_column(
    pane,
    column,
    rate_axis_half_range,
    video_frame_on_screen,
    video_frames_each_side,
    video_frame_rate,
    first_video_frame,
):
    """One column at video frame ``video_frame_on_screen`` of the video of the column: the
    video pane ``pane``, already scaled to the column width, with the video frame counter;
    the three panel scrolling strip, with its seconds axis, its rate axes, the identifier
    of each extent beside its marker, and the marker of the video frame on screen; and the
    legend band.

    ``column`` is a column of ``prepare_column``. ``rate_axis_half_range`` is one half
    range of the rate axis for the three panels, or three, one per panel. Nothing is
    clipped: a trace beyond ``rate_axis_half_range`` leaves the panel, and a kept
    callout, where there is one, labels its peak. ``first_video_frame``, the video frame
    of the video at which every array starts, serves the labels alone, so the counter
    and the seconds axis read video frames and times of the video, while every array
    stays indexed from 0."""
    rate_axis_half_ranges = (
        list(rate_axis_half_range)
        if np.ndim(rate_axis_half_range)
        else [rate_axis_half_range] * 3
    )
    counter_video_frame = video_frame_on_screen + first_video_frame
    _label_box(
        pane,
        f"f{counter_video_frame}  t={counter_video_frame / video_frame_rate:.2f}s",
        (8, pane.height - 30),
    )

    strip_height = PANEL_TOP + 3 * PANEL_HEIGHT + TIME_AXIS_HEIGHT
    strip = np.full((strip_height, COLUMN_WIDTH, 3), STRIP_BACKGROUND, np.uint8)
    low, high, span = (
        video_frame_on_screen - video_frames_each_side,
        video_frame_on_screen + video_frames_each_side,
        2 * video_frames_each_side,
    )

    def x_of(video_frame):
        return round((video_frame - low) / span * COLUMN_WIDTH)

    # A grid line at each whole second of the video in view; the fills blend over them.
    seconds = range(
        int(np.ceil((low + first_video_frame) / video_frame_rate)),
        int(np.floor((high + first_video_frame) / video_frame_rate)) + 1,
    )
    second_x = [
        (second, x_of(second * video_frame_rate - first_video_frame))
        for second in seconds
    ]
    for _second, x in second_x:
        if 0 <= x < COLUMN_WIDTH:
            strip[PANEL_TOP : PANEL_TOP + 3 * PANEL_HEIGHT, x] = 44

    for axis_index, spans, color, alpha in column["fills"]:
        top = PANEL_TOP + axis_index * PANEL_HEIGHT
        for first, last in spans:
            if not (low <= last and first <= high):
                continue
            left, right = max(x_of(first), 0), min(x_of(last), COLUMN_WIDTH)
            if right - left < 4:  # keeps a narrow region legible
                left, right = left - 2, left + 3
            region = strip[
                top : top + PANEL_HEIGHT, max(left, 0) : min(right, COLUMN_WIDTH)
            ]
            if region.size:
                region[:] = (alpha * np.asarray(color) + (1 - alpha) * region).astype(
                    np.uint8
                )

    image = Image.fromarray(strip)
    draw = ImageDraw.Draw(image)
    axis_y = strip_height - 6
    label_font, scale_font = _default_font(12), _default_font(11)
    draw.text(
        (4, axis_y), "time (s)", font=label_font, fill=(150, 150, 150), anchor="ls"
    )
    for second, x in second_x:
        if x > 64:
            draw.text(
                (x + 2, axis_y),
                f"{second}",
                font=label_font,
                fill=(150, 150, 150),
                anchor="ls",
            )

    for group in column["decorators"]:
        for axis_index, markers in group["markers"].items():
            top = PANEL_TOP + axis_index * PANEL_HEIGHT
            for video_frame, _identifier in markers:
                if low <= video_frame <= high:
                    x = x_of(video_frame)
                    draw.line((x, top, x, top + PANEL_HEIGHT), fill=group["color"])

    # The series in their order: the first underneath, the reference last, on top. The
    # rows of the identifiers keep them clear of the callouts; the callouts are drawn after
    # the identifiers for the one case where an identifier finds no free row.
    callouts = []
    for name, axis_index in AXIS_PANEL_NAMES:
        top = PANEL_TOP + axis_index * PANEL_HEIGHT
        middle = top + PANEL_HEIGHT // 2
        draw.line((0, middle, COLUMN_WIDTH, middle), fill=(70, 70, 70))
        for _label, series, color in column["series"]:
            first, last = max(low, 0), min(high + 1, len(series))
            values = series[first:last, axis_index]
            xs = [x_of(video_frame) for video_frame in range(first, last)]
            ys = middle - values / rate_axis_half_ranges[axis_index] * (
                PANEL_HEIGHT // 2 - 6
            )
            _polyline_split(draw, xs, ys, color)
        draw.text(
            (6, top + 14),
            f"{name} (deg/s)",
            font=label_font,
            fill=(205, 205, 205),
            anchor="ls",
        )
        draw.text(
            (COLUMN_WIDTH - 42, top + 12),
            f"+{rate_axis_half_ranges[axis_index]:.0f}",
            font=scale_font,
            fill=(140, 140, 140),
            anchor="ls",
        )
        draw.text(
            (COLUMN_WIDTH - 42, top + PANEL_HEIGHT - 5),
            f"-{rate_axis_half_ranges[axis_index]:.0f}",
            font=scale_font,
            fill=(140, 140, 140),
            anchor="ls",
        )
        # Each callout rides with its peak and scrolls in and out cut by the column edge.
        for video_frame, peak in column["callouts"].get(axis_index, ()):
            text = callout_text(peak)
            left = x_of(video_frame) + CALLOUT_OFFSET_PX
            width = _label_box_size(text, size_px=CALLOUT_PX, pad=CALLOUT_PAD)[0]
            if left < COLUMN_WIDTH and left + width > 0:
                callouts.append((text, (left, top + callout_top(peak))))

    # Each identifier ends one pixel left of its marker, in its row of the composite, and
    # scrolls in and out cut by the column edge.
    for group in column["decorators"]:
        for axis_index, markers in group["markers"].items():
            top = PANEL_TOP + axis_index * PANEL_HEIGHT
            for video_frame, identifier in markers:
                width = identifier_box_size(identifier)[0]
                left = x_of(video_frame) - 1 - width
                if left < COLUMN_WIDTH and left + width > 0:
                    row_top, _row_bottom = identifier_row_band(
                        column["identifier_rows"][axis_index, identifier]
                    )
                    _label_box(
                        image,
                        identifier,
                        (left, top + row_top),
                        size_px=IDENTIFIER_PX,
                        pad=IDENTIFIER_PAD,
                    )
    for text, position in callouts:
        _label_box(image, text, position, size_px=CALLOUT_PX, pad=CALLOUT_PAD)

    center = COLUMN_WIDTH // 2
    draw.line(
        (center, PANEL_TOP, center, PANEL_TOP + 3 * PANEL_HEIGHT),
        fill=VIDEO_FRAME_ON_SCREEN_MARKER_COLOR,
    )

    composed = Image.new(
        "RGB", (COLUMN_WIDTH, pane.height + strip_height + LEGEND_HEIGHT)
    )
    composed.paste(pane, (0, 0))
    composed.paste(image, (0, pane.height))
    composed.paste(column["legend_band"], (0, pane.height + strip_height))
    return composed


def _title_bar(width, title, column_titles):
    """The title bar of the composite, ``width`` pixels wide and ``TITLE_BAR_HEIGHT``
    high, in the face of the column title: ``title`` centered across it, and under it each
    title of ``column_titles``, in the order of the columns, centered over its column, all
    at the one size of ``_column_title_px``."""
    bar = Image.new("RGB", (width, TITLE_BAR_HEIGHT), TITLE_BAR_COLOR)
    draw = ImageDraw.Draw(bar)
    draw.text(
        (width // 2, TITLE_BAR_TEXT_CENTER_Y),
        title,
        font=_title_font(TITLE_BAR_PX),
        fill=TITLE_BAR_TEXT_COLOR,
        anchor="mm",
    )
    column_title_font = _title_font(_column_title_px(column_titles))
    for index, column_title in enumerate(column_titles):
        draw.text(
            (
                index * (COLUMN_WIDTH + GUTTER_WIDTH) + COLUMN_WIDTH // 2,
                COLUMN_TITLE_CENTER_Y,
            ),
            column_title,
            font=column_title_font,
            fill=COLUMN_TITLE_COLOR,
            anchor="mm",
        )
    return bar


def _compose(columns, title_bar):
    """One canvas of ``title_bar`` above the two columns, A, a gutter, and B."""
    width = sum(column.width for column in columns) + GUTTER_WIDTH * (len(columns) - 1)
    canvas = Image.new(
        "RGB", (width, title_bar.height + columns[0].height), (40, 40, 40)
    )
    canvas.paste(title_bar, (0, 0))
    x = 0
    for column in columns:
        canvas.paste(column, (x, title_bar.height))
        x += column.width + GUTTER_WIDTH
    return canvas


# The libvpx-vp9 encoder at a constant quality, with the rows of each video frame encoded
# in parallel. A constant quality needs a target bit rate of 0.
VP9_ENCODER_OPTIONS = {
    "crf": "30",
    "b": "0",
    "deadline": "good",
    "cpu-used": "4",
    "row-mt": "1",
}


def write_composite(
    column_videos,
    columns,
    *,
    title,
    rate_axis_half_range,
    peaks_beyond_rate_axis,
    output,
    video_frame_rate,
    video_frames_each_side,
    first_video_frame,
    video_frame_count,
):
    """The composite, ``video_frame_count`` video frames long, under the title bar of
    ``title``: one pass over the video frames of ``column_videos``, the video of each
    column, decoded together from their first video frame and counted, never sought, so
    video frame k of the composite is video frame k of the video of each column and row k
    of every series. The pass stops after ``video_frame_count`` video frames, which leaves
    out the video frames Gyroflow renders past the end of its trim. Each composite video
    frame is encoded through PyAV with ``libvpx-vp9``. An error of the operating system
    where the composite is opened for writing raises ``errors.CannotWriteError`` of the
    composite. Returns the number of video frames the composite holds."""
    prepared = [
        prepare_column(column, video_frames_each_side, column_peaks_beyond_rate_axis)
        for column, column_peaks_beyond_rate_axis in zip(
            columns, peaks_beyond_rate_axis, strict=True
        )
    ]
    width = len(columns) * COLUMN_WIDTH + GUTTER_WIDTH * (len(columns) - 1)
    title_bar = _title_bar(width, title, [column["title"] for column in columns])
    with contextlib.ExitStack() as stack:
        sources = [stack.enter_context(av.open(str(path))) for path in column_videos]
        streams = [source.streams.video[0] for source in sources]
        for stream in streams:
            stream.thread_type = "AUTO"
        pane_height = round(streams[0].height * COLUMN_WIDTH / streams[0].width)
        height = (
            TITLE_BAR_HEIGHT
            + pane_height
            + PANEL_TOP
            + 3 * PANEL_HEIGHT
            + TIME_AXIS_HEIGHT
            + LEGEND_HEIGHT
        )
        with errors.operating_system_error_as(errors.CannotWriteError, output):
            target = stack.enter_context(av.open(str(output), "w"))
        encoder = target.add_stream(
            "libvpx-vp9",
            rate=streams[0].average_rate,
            options=VP9_ENCODER_OPTIONS,
        )
        encoder.width, encoder.height, encoder.pix_fmt = width, height, "yuv420p"
        written = 0
        decoded = zip(
            *(source.decode(stream) for source, stream in zip(sources, streams))
        )
        for video_frame, decoded_video_frames in enumerate(decoded):
            if video_frame >= video_frame_count:
                break
            panes = [
                decoded_video_frame.to_image(
                    width=COLUMN_WIDTH, height=pane_height, interpolation="AREA"
                )
                for decoded_video_frame in decoded_video_frames
            ]
            canvas = _compose(
                [
                    draw_column(
                        pane,
                        column,
                        rate_axis_half_range,
                        video_frame,
                        video_frames_each_side,
                        video_frame_rate,
                        first_video_frame,
                    )
                    for pane, column in zip(panes, prepared, strict=True)
                ],
                title_bar,
            )
            composite_video_frame = av.VideoFrame.from_image(canvas)
            composite_video_frame.pts = video_frame
            for packet in encoder.encode(composite_video_frame):
                target.mux(packet)
            written += 1
        for packet in encoder.encode():
            target.mux(packet)
    return written


def render_compare(
    columns, *, title, output, video_frame_rate, first_video_frame, rate_axis_scale
):
    """Write the composite of the two columns under the title bar of ``title`` to
    ``output`` and return the lines ``--verbose`` prints of it: the rate axis and the video
    frames written.

    The rate axis holds on every video frame: by default plus and minus ``HEADROOM`` times
    the largest absolute rate of every series drawn in the composite, over the three axes,
    that is not a transient outlier; with ``rate_axis_scale`` ``peraxis`` each panel its own;
    with values, the values given. A peak is called out where a series passes the axis by
    ``CALLOUT_FACTOR``."""
    every_series = [series for column in columns for _, series, _ in column["series"]]
    video_frame_count = len(every_series[0])
    last = video_frame_count - 1
    if rate_axis_scale is None:
        rate_axis_half_range = HEADROOM * _largest_absolute_rate_over_three_axes(
            every_series, 0, last, TRANSIENT_OUTLIER_THRESHOLD
        )
        described = (
            f"plus and minus {rate_axis_half_range:.1f} deg/s on the three panels"
        )
    elif rate_axis_scale == "peraxis":
        rate_axis_half_range = [
            HEADROOM
            * max(
                1e-6,
                _largest_absolute_rate(
                    every_series, column, 0, last, TRANSIENT_OUTLIER_THRESHOLD
                ),
            )
            for column in range(3)
        ]
        described = "plus and minus " + ", ".join(
            f"{value:.1f}" for value in rate_axis_half_range
        )
        described += " deg/s on pitch, yaw, and roll"
    else:
        rate_axis_half_range = list(rate_axis_scale)
        described = "plus and minus " + ", ".join(
            f"{value:g}" for value in rate_axis_half_range
        )
        described += " deg/s on pitch, yaw, and roll"
    rate_axis_half_ranges = (
        rate_axis_half_range
        if np.ndim(rate_axis_half_range)
        else [rate_axis_half_range] * 3
    )

    peaks_beyond_rate_axis = []
    for column in columns:
        peaks_by_axis = {}
        for _name, axis_index in AXIS_PANEL_NAMES:
            peaks = _peaks_beyond_rate_axis(
                [series[:, axis_index] for _, series, _ in column["series"]],
                rate_axis_half_ranges[axis_index],
            )
            peaks_by_axis[axis_index] = [
                (video_frame, peak)
                for video_frame, peak in peaks
                if abs(peak) > CALLOUT_FACTOR * rate_axis_half_ranges[axis_index]
            ]
        peaks_beyond_rate_axis.append(peaks_by_axis)

    written = write_composite(
        [column["mp4"] for column in columns],
        columns,
        title=title,
        rate_axis_half_range=rate_axis_half_range,
        peaks_beyond_rate_axis=peaks_beyond_rate_axis,
        output=output,
        video_frame_rate=video_frame_rate,
        video_frames_each_side=round(SCROLL_WINDOW_S * video_frame_rate / 2),
        first_video_frame=first_video_frame,
        video_frame_count=video_frame_count,
    )
    return [
        f"rate axis: {described}",
        f"wrote {output}, {written} video frames",
    ]


# -------------------------------------------------------------------------
# The cache
# -------------------------------------------------------------------------


def _read_repair_result(path, video, video_cache_directory, command):
    """The repair result at ``path``, tested as ``gyro-doctor`` tests it, comparing no
    parameters. A missing repair result, a file that cannot be opened or is damaged, and
    a repair result that fails a test raise ``errors.CannotReadError`` with ``command``,
    the ``gyro-doctor`` command its hint names."""
    if not path.exists():
        raise errors.CannotReadError(
            errors.ErrorCode.CACHED_RESULT_NOT_FOUND, file=path, command=command
        )
    try:
        return repair_result_tested(path, video, video_cache_directory)
    except errors.CannotReadError as error:
        error.values.setdefault("command", command)
        raise


def _read_write_record(video_cache_directory, video, command):
    """The repaired video that the write record in ``video_cache_directory`` locates,
    after the test of both of its file stamps against the files on disk. A missing write
    record, a file that cannot be opened or is damaged, and a repaired video or a repair
    result that disagrees with its file stamp raise ``errors.CannotReadError`` of the
    write record with ``command``, the ``gyro-doctor`` command its hint names."""
    path = write_stage.write_record_path(video_cache_directory, video.stem)
    if not path.exists():
        raise errors.CannotReadError(
            errors.ErrorCode.CACHED_RESULT_NOT_FOUND, file=path, command=command
        )
    try:
        with decode_stage.cache_file_read(path):
            record = write_stage.read_write_record(path)
            repaired_video = Path(record["repaired_video"])
            stamps = {
                key: record["file_stamps"][key]
                for key in ("repair_result", "repaired_video")
            }
    except KeyError as error:
        raise errors.CannotReadError(
            errors.ErrorCode.CACHE_FILE_INVALID,
            file=path,
            reason=str(error),
            command=command,
        ) from error
    except errors.CannotReadError as error:
        error.values.setdefault("command", command)
        raise
    try:
        decode_stage.file_stamps_test(
            path, video_cache_directory, stamps, ("repair_result",)
        )
        decode_stage.file_stamps_test(
            path, repaired_video.parent, stamps, ("repaired_video",)
        )
    except errors.CannotReadError as error:
        error.values.setdefault("command", command)
        raise
    return repaired_video


# -------------------------------------------------------------------------
# The command
# -------------------------------------------------------------------------


def _cache_location(args, *options):
    """The directory of the cache that holds the results of the video of the parsed options
    ``args``, and the ``gyro-doctor`` command, with ``options`` before the video, that
    fills it: ``(video_cache_directory, command)``."""
    video = args.video
    video_cache_directory = (
        args.cache_dir or Path.cwd() / ".gyro-doctor-cache"
    ) / video.stem
    cache_option = [] if args.cache_dir is None else [f"--cache-dir {args.cache_dir}"]
    command = " ".join(["gyro-doctor", *options, *cache_option, str(video)])
    return video_cache_directory, command


def _default_mode(args):
    """The mode of the render when ``--mode`` is not given: ``original-vs-repair`` when the
    cache holds a repair result of the video that passes its tests, otherwise
    ``raw-vs-stab`` when it holds a detection result that passes its tests. When it holds
    neither, neither mode is ready, and ``errors.CannotReadError`` is raised with the two
    ``gyro-doctor`` commands that make a mode ready."""
    video = args.video
    video_cache_directory, command = _cache_location(args)
    repair = chart._read_repair_result(
        repair_stage.repair_result_path(video_cache_directory, video.stem),
        video,
        video_cache_directory,
        command,
    )
    if repair is not None:
        return "original-vs-repair"
    detection = _read_detection_result(
        detection_stage.detection_result_path(video_cache_directory, video.stem),
        video,
        video_cache_directory,
        command,
    )
    if detection is not None:
        return "raw-vs-stab"
    _, detect_command = _cache_location(args, "--detect-only")
    raise errors.CannotReadError(
        errors.ErrorCode.NO_MODE_READY,
        file=video_cache_directory,
        detect_command=detect_command,
        repair_command=command,
    )


def _detection_decorators(args, detection_path, video, video_cache_directory, command):
    """The decorators of the detection result at ``detection_path``; none, with a warning
    that names the error, when the detection result is missing or fails a test. A file
    that cannot be opened or is damaged raises its ``errors.CannotReadError``."""
    try:
        detection = detection_result_tested(
            detection_path, video, video_cache_directory, command
        )
    except errors.CannotReadError as error:
        if error.code in READ_ERROR_CODES:
            raise
        errors.log_detection_result_unused_warning(log, error)
        return []
    try:
        return read_decorators(detection, dict(args.decorator_color or ()))
    except errors.CannotReadError as error:
        error.values.setdefault("file", detection_path)
        raise


def original_vs_repair(args):
    """Make the render of the original against the repaired video that the parsed options
    ``args`` ask for, from the cache of the video, and return the lines ``--verbose``
    prints."""
    video = args.video
    video_cache_directory, command = _cache_location(args)
    decode = _read_decode_result(
        decode_stage.decode_result_path(video_cache_directory, video.stem),
        video,
        command,
    )
    repair = _read_repair_result(
        repair_stage.repair_result_path(video_cache_directory, video.stem),
        video,
        video_cache_directory,
        command,
    )
    repaired_video = _read_write_record(video_cache_directory, video, command)
    detection_path = detection_stage.detection_result_path(
        video_cache_directory, video.stem
    )
    decorators = _detection_decorators(
        args, detection_path, video, video_cache_directory, command
    )

    first, last = args.first_video_frame, args.last_video_frame
    count = last - first + 1
    output = args.output or Path.cwd() / (
        f"{video.stem}-original-vs-repair-f{first + count // 2}-{count}f.MP4"
    )
    render_directory = video_cache_directory / "render"
    dials = resolve_preset(args.preset)
    lines = []
    gyroflow_renders = []
    for tag, source in (("original", video), ("repaired", repaired_video)):
        try:
            path, rendered = render_native_leg(
                source,
                _gyroflow_render_path(
                    render_directory, source, tag, first, last, args.preset
                ),
                first=first,
                last=last,
                dials=dials,
            )
        except errors.GyroDoctorError as error:
            error.values.setdefault("file", output)
            error.values.setdefault("preset", args.preset)
            raise
        gyroflow_renders.append(path)
        lines.append(f"{tag} video: {'rendered' if rendered else 'reused'} {path}")

    video_frame_rate = float(decode["video_frame_rate"])
    video_frame_count = args.video_frame_count
    original = normalized_rate_per_video_frame(
        decode, video_frame_rate, video_frame_count
    )
    repaired = normalized_rate_per_video_frame(
        repair, video_frame_rate, video_frame_count
    )
    provenance = provenance_line(args.preset)
    window = (first, last + 1)
    column_a = build_leg(
        gyroflow_renders[0],
        [(ORIGINAL_SERIES_LABEL, original, A_SERIES_COLOR)],
        title=f"A: {video.name}",
        provenance=provenance,
        reference=ORIGINAL_SERIES_LABEL,
        decorators=[],
        window=window,
    )
    column_b = build_leg(
        gyroflow_renders[1],
        [
            (REPAIRED_SERIES_LABEL, repaired, B_SERIES_COLOR),
            (ORIGINAL_SERIES_LABEL, original, A_SERIES_COLOR),
        ],
        title=f"B: {repaired_video.name}",
        provenance=provenance,
        reference=ORIGINAL_SERIES_LABEL,
        decorators=decorators,
        window=window,
    )
    lines = (
        verbose_lines_of_column(column_b, first, last, video_frame_rate, args.preset)
        + lines
        + render_compare(
            [column_a, column_b],
            title=ORIGINAL_VS_REPAIR_TITLE,
            output=output,
            video_frame_rate=video_frame_rate,
            first_video_frame=first,
            rate_axis_scale=args.rate_axis_scale,
        )
    )
    return lines


def raw_vs_stab(args):
    """Make the render of the video as shot that the parsed options ``args`` ask for, from
    the cache of the video, and return the lines ``--verbose`` prints: in column A the
    unstabilized recode of ``render_raw_leg``, in column B the Gyroflow render of the
    original video of ``render_native_leg``, each over the original Normalized Rate and the
    marks. The recode and the Gyroflow render run in parallel."""
    video = args.video
    video_cache_directory, detect_command = _cache_location(args, "--detect-only")
    decode = _read_decode_result(
        decode_stage.decode_result_path(video_cache_directory, video.stem),
        video,
        detect_command,
    )
    detection_path = detection_stage.detection_result_path(
        video_cache_directory, video.stem
    )
    decorators = _detection_decorators(
        args, detection_path, video, video_cache_directory, detect_command
    )

    first, last = args.first_video_frame, args.last_video_frame
    count = last - first + 1
    output = args.output or Path.cwd() / (
        f"{video.stem}-raw-vs-stab-f{first + count // 2}-{count}f.MP4"
    )
    render_directory = video_cache_directory / "render"
    video_frame_rate = float(decode["video_frame_rate"])
    with ThreadPoolExecutor(max_workers=2) as executor:
        gyroflow_render_future = executor.submit(
            render_native_leg,
            video,
            _gyroflow_render_path(
                render_directory, video, "original", first, last, args.preset
            ),
            first=first,
            last=last,
            dials=resolve_preset(args.preset),
        )
        recode_future = executor.submit(
            render_raw_leg,
            video,
            render_directory / f"{video.stem[:24]}_unstabilized_f{first}-f{last}.mp4",
            first=first,
            last=last,
            video_frame_rate=video_frame_rate,
        )
        try:
            gyroflow_render, rendered = gyroflow_render_future.result()
        except errors.GyroDoctorError as error:
            error.values.setdefault("file", output)
            error.values.setdefault("preset", args.preset)
            raise
        recode_path = recode_future.result()

    original = normalized_rate_per_video_frame(
        decode, video_frame_rate, args.video_frame_count
    )
    window = (first, last + 1)
    column_a = build_leg(
        recode_path,
        [(RAW_VS_STAB_A_SERIES_LABEL, original, A_SERIES_COLOR)],
        title=f"A: {video.name}, cropped, no stabilization",
        provenance="no stabilization, no lens correction, center crop to 16:9",
        reference=RAW_VS_STAB_A_SERIES_LABEL,
        decorators=decorators,
        window=window,
    )
    column_b = build_leg(
        gyroflow_render,
        [(RAW_VS_STAB_B_SERIES_LABEL, original, A_SERIES_COLOR)],
        title=f"B: {video.name}, stabilized",
        provenance=provenance_line(args.preset),
        reference=RAW_VS_STAB_B_SERIES_LABEL,
        decorators=decorators,
        window=window,
    )
    return (
        verbose_lines_of_column(column_b, first, last, video_frame_rate, args.preset)
        + [
            f"original video: {'rendered' if rendered else 'reused'} {gyroflow_render}",
            f"unstabilized recode: wrote {recode_path}",
        ]
        + render_compare(
            [column_a, column_b],
            title=RAW_VS_STAB_TITLE,
            output=output,
            video_frame_rate=video_frame_rate,
            first_video_frame=first,
            rate_axis_scale=args.rate_axis_scale,
        )
    )


def _resolve_video_span(args):
    """The first and the last video frame that ``--start`` and ``--duration`` select, and
    the video frame count of the video, from the video frame count and the video frame rate
    read through PyAV. A video that PyAV cannot read raises ``errors.CannotReadError``,
    and a ``--start`` or a ``--duration`` that reaches past the last video frame, or
    selects no video frame, raises ``errors.InvalidOptionValueError``."""
    try:
        stream_info = _video_stream_info(args.video)
    except av.error.FFmpegError as error:
        raise errors.CannotReadError(
            errors.ErrorCode.VIDEO_INVALID, file=args.video, reason=str(error)
        ) from error
    video_frame_rate = float(stream_info["video_frame_rate"])
    count_of_video = stream_info["video_frame_count"]
    last_of_video = count_of_video - 1
    first = 0 if args.start is None else args.start.to_video_frame(video_frame_rate)
    if first > last_of_video:
        raise errors.InvalidOptionValueError(
            errors.ErrorCode.START_PAST_LAST_VIDEO_FRAME,
            option="--start",
            value=args.start.text,
            last=last_of_video,
        )
    if args.duration is None:
        count = count_of_video - first
    elif args.duration.video_frames is not None:
        count = args.duration.video_frames
    else:
        count = round(args.duration.seconds * video_frame_rate)
    if count < 1:
        raise errors.InvalidOptionValueError(
            errors.ErrorCode.DURATION_WITHOUT_VIDEO_FRAMES,
            option="--duration",
            value=args.duration.text,
        )
    if first + count - 1 > last_of_video:
        raise errors.InvalidOptionValueError(
            errors.ErrorCode.DURATION_PAST_LAST_VIDEO_FRAME,
            option="--duration",
            value=args.duration.text,
            end=first + count - 1,
            last=last_of_video,
        )
    return first, first + count - 1, count_of_video


def parse_args(argv=None):
    """The options of ``gyro-doctor-render`` parsed from ``argv``, which defaults to
    ``sys.argv[1:]``, with ``--start`` and ``--duration`` resolved to
    ``first_video_frame`` and ``last_video_frame`` and the ``video_frame_count`` of the
    video. A mistake in the options, and the error of a video path that is not a video, of
    a ``--start`` or a ``--duration`` that reaches past the last video frame, and of an
    ``--output`` that is the video or lies in a directory that does not exist, go through
    argparse, which prints the usage and the error line to stderr and exits with status
    2, before any cache is read. ``--help`` and ``--version`` go through argparse, which
    prints the help, or the name of the command and the version of the package, to stdout
    and exits with status 0."""
    parser = argparse.ArgumentParser(
        prog="gyro-doctor-render",
        usage="%(prog)s [options] VIDEO",
        description=(
            "gyro-doctor-render renders the stabilized video before and after the "
            "repair side by side using Gyroflow."
        ),
    )
    parser.add_argument("video", type=Path, metavar="VIDEO", help="The original video")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {importlib.metadata.version('gyro_doctor')}",
    )
    parser.add_argument(
        "--mode",
        choices=("original-vs-repair", "raw-vs-stab"),
        metavar="MODE",
        help=(
            "The render to make: original-vs-repair, the original video against the "
            "repaired video, or raw-vs-stab, the video as shot against the video "
            "stabilized by Gyroflow (default: original-vs-repair when the cache holds a "
            "repair result of the video, otherwise raw-vs-stab when it holds a detection "
            "result)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        help=(
            "The complete path of the composite (default: the video stem, the mode, the "
            "center video frame of the video frames that --start and --duration select, "
            "and their count, with the extension .MP4, in the current directory, such as "
            "DJI_0002-original-vs-repair-f4950-900f.MP4)"
        ),
    )
    parser.add_argument(
        "--start",
        type=parse_position,
        metavar="POSITION",
        help=(
            "The first video frame to render: a time such as 75s, 1:15, or 1:02:15, "
            "or a video frame such as f4500 (default: the first video frame)"
        ),
    )
    parser.add_argument(
        "--duration",
        type=parse_duration,
        metavar="DURATION",
        help=(
            "The duration to render from --start: a time such as 15s, or a count of "
            "video frames "
            "such as 900f (default: the rest of the video)"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        metavar="PATH",
        help=(
            "The cache, as gyro-doctor takes it (default: .gyro-doctor-cache under the "
            "current working directory)"
        ),
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="zooming",
        metavar="NAME",
        help=(
            "The Gyroflow dials of every Gyroflow render, standard or zooming "
            "(default: zooming)"
        ),
    )
    parser.add_argument(
        "--rate-axis-scale",
        type=_rate_axis_scale,
        metavar="VALUE or LIST",
        help=(
            "The rate axis of the three panels, in degrees per second: one value sets "
            "plus and minus that value on all three panels; three values, as one comma "
            "separated token, set pitch, yaw, and roll; peraxis gives each panel its own "
            "rate axis at 1.25 times the largest absolute rate on its axis that is not a "
            "transient outlier (default: one rate axis at 1.25 times the largest "
            "absolute rate of the series drawn in the composite, over the three axes, that "
            "is not a transient outlier)"
        ),
    )
    parser.add_argument(
        "--decorator-color",
        type=parse_color_override,
        action="append",
        metavar="CLASS=#RRGGBB",
        help=(
            'The color of the marks of one classification, such as "impulse '
            'event=#1f4fd6". Repeatable (default: the color of each classification)'
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print what the render drew and the Gyroflow renders it made or reused",
    )
    args = parser.parse_args(argv)
    try:
        _options_test(args)
        args.first_video_frame, args.last_video_frame, args.video_frame_count = (
            _resolve_video_span(args)
        )
    except errors.GyroDoctorError as error:
        parser.error(str(error))
    return args


def _options_test(args):
    """Raise an error of ``errors`` for a video path that is not a file, and for an
    ``--output`` that is the video or lies in a directory that does not exist."""
    if not args.video.is_file():
        raise errors.CannotReadError(errors.ErrorCode.VIDEO_NOT_A_FILE, file=args.video)
    if args.output is not None:
        if args.output.exists() and args.output.samefile(args.video):
            raise errors.CannotWriteError(
                errors.ErrorCode.OUTPUT_IS_VIDEO, file=args.output, video=args.video
            )
        if not args.output.parent.is_dir():
            raise errors.CannotWriteError(
                errors.ErrorCode.OUTPUT_DIRECTORY_NOT_FOUND,
                file=args.output,
                directory=args.output.parent,
            )


def main(argv=None):
    """Run ``gyro-doctor-render`` on the command line arguments ``argv``, which default to
    ``sys.argv[1:]``, and return the exit status: 0 when the composite is written, and 1 on
    an error of the cache, of Gyroflow, of the recode, or of the colors of the marks, and
    when neither mode is ready, ``errors.GyroDoctorError``, logged through
    ``errors.log_error`` with its error line and any hint line at ERROR level through the
    logger ``gyro_doctor.tools.render``.

    A mistake in the options exits with status 2 through argparse, and ``--help`` and
    ``--version`` print to stdout and exit with status 0 through argparse. Otherwise
    nothing is printed on stdout unless ``--verbose`` is given, which prints what the
    render drew and the Gyroflow renders it made or reused, and logs at INFO level the end
    of the output of a Gyroflow run that wrote no render. Without ``--mode``,
    ``--verbose`` also logs at INFO level each cached result that the render does not use,
    with the error it raised, through the logger ``gyro_doctor.tools.chart`` of
    ``chart._read_repair_result`` and ``chart._read_detection_result``."""
    logging.basicConfig(format="gyro-doctor-render: %(message)s")
    args = parse_args(argv)
    level = logging.INFO if args.verbose else logging.NOTSET
    log.setLevel(level)
    chart.log.setLevel(level)
    try:
        mode = args.mode or _default_mode(args)
        if mode == "raw-vs-stab":
            lines = raw_vs_stab(args)
        else:
            lines = original_vs_repair(args)
    except errors.GyroDoctorError as error:
        errors.log_error(log, error)
        return 1
    if args.verbose:
        print("\n".join(lines))
    return 0
