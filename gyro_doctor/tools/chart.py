# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""``gyro-doctor-chart`` charts the rotational rate of the camera on each axis before
and after the repair.

The ``gyro-doctor-chart`` command, the Normalized Rate Quaternion Gyro Chart: it
draws the Normalized Rate of the three axes of a video as three panels over a window of
the video, from the results that ``gyro-doctor`` stored for the video in
``<cache>/<video stem>/``, with the detected extents and the repaired Normalized Rate
where the cache holds them, and runs no stage.
"""

import argparse
import colorsys
import importlib.metadata
import json
import logging
import math
from pathlib import Path
from typing import NamedTuple

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression

from gyro_doctor import errors
from gyro_doctor.cli import parse_position, stored_parts
from gyro_doctor.core import (
    decode_result_reader,
    decode_stage,
    detection_stage,
    repair_stage,
    robust_local_fit,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Annotation
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator

log = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Colors, axes, and the view scales
# -------------------------------------------------------------------------

PALETTE = ["#e8820e", "#009E73"]
AXIS_COLUMNS = [("pitch", 0), ("yaw", 1), ("roll", 2)]
AXIS_ORDER = ("pitch", "yaw", "roll")
# Also the color of the impulse threshold: the B series is a dense field of markers and
# the threshold a thin pair of lines, so a reader tells the two apart.
B_SERIES_COLOR = PALETTE[1]
VIDEO_FRAME_BOUNDARY_COLOR = "#9a9a9a"
BOX_EDGE_COLOR = "#444444"

# The curve through the rate samples of a series, drawn under every marker and over the
# local median, in the series color with this share of white blended into it.
RATE_CURVE_BLEND_TOWARD_WHITE = 0.5
RATE_CURVE_WIDTH_PX = 1.5
RATE_CURVE_ZORDER = 2.5
# The most pixels of axes width that one straight segment of the drawn curve spans.
RATE_CURVE_SEGMENT_PX = 1.0

# The boundary lines of an extent are a dark, saturated version of the extent's own fill:
# the hue is kept, and the saturation and the value are replaced.
EXTENT_BOUNDARY_LINE_SATURATION = 0.72
EXTENT_BOUNDARY_LINE_VALUE = 0.52
EXTENT_BOUNDARY_LINE_WIDTH = 1.1

# One opaque fill per classification. Each is light enough that a marker drawn over it
# stays legible.
EXTENT_FILL_PALETTE = ["#FBDDDD", "#E7DDF4", "#FBF0D5"]
EXTENT_CLASSIFICATION_FILLS = {
    detection_stage.LARGE_AMPLITUDE_OSCILLATION: EXTENT_FILL_PALETTE[0],
    detection_stage.MID_AMPLITUDE_BURST: EXTENT_FILL_PALETTE[1],
    detection_stage.IMPULSE_EVENT: EXTENT_FILL_PALETTE[2],
}

# The line of each threshold, from the Okabe-Ito set: the oscillation threshold, the burst
# threshold, and the impulse threshold, keyed by the number of the pass that applies it.
PASS_THRESHOLD_COLORS = {1: "#0072B2", 2: "#E69F00", 3: "#009E73"}
PASS_THRESHOLD_LINE_WIDTH = 0.6
# The legend sample is drawn heavier than the line, so its color can be read.
PASS_THRESHOLD_LEGEND_LINE_WIDTH = 1.8
# The label of each threshold line at the right edge of its panel: its box pad as a share
# of its font size, its gap from its line, and its inset from the right edge.
THRESHOLD_LABEL_FONTSIZE = 7.0
THRESHOLD_LABEL_BOX_PAD = 0.1
THRESHOLD_LABEL_GAP_POINTS = 2.0
THRESHOLD_LABEL_INSET_POINTS = 3.0

# The 99th percentile over the median of the absolute value of a Laplace distribution,
# ln(100) / ln(2): where the peaks of a stretch that just meets the test of the burst pass
# reach.
LAPLACE_TAIL_FACTOR = 6.643856189774724

# The local median: very light purple, dashed, and faint, since it is context.
LOCAL_MEDIAN_COLOR = "#b9a7e0"
LOCAL_MEDIAN_LINE_ALPHA = 0.55
LOCAL_MEDIAN_LABEL = "local median"

RIGHT_AXIS_RATIO_COLOR = "#7a0f52"
RIGHT_AXIS_RATIO_LABEL = "Local / Full Track RSD"
RIGHT_AXIS_RATIO_LINE_WIDTH = 1.0
# The repeated right hand scale is written this many points to the right of the repeated
# left hand scale: the values clear of the left hand values, and the name clear of both.
RIGHT_AXIS_RATIO_NAME_OFFSET_POINTS = 124.0
RIGHT_AXIS_RATIO_VALUE_OFFSET_POINTS = 54.0
# The repeated name of the ratio axis, in a box padded by this share of its font size.
RIGHT_AXIS_RATIO_NAME_FONTSIZE = 7.5
RIGHT_AXIS_RATIO_NAME_BOX_PAD = 0.15
# The real tick values and name of the ratio axis end this far inside the canvas edge.
RIGHT_AXIS_RATIO_EDGE_CLEARANCE_POINTS = 4.0

# A page is one screen view; a canvas is a whole number of pages. The window is a number
# of video frames, and the ultra-wide view is 15 seconds. px_per_sample is the floor of
# pixels of axes width per rate sample, and the diameter of a marker.
PAGE_WIDTH_PX = 3000.0
VIEW_SCALES = {
    "normal": {"pages": 1, "window_video_frames": 24, "px_per_sample": 2.5},
    "wide": {"pages": 5, "window_video_frames": 120, "px_per_sample": 2.5},
    "ultra-wide": {"pages": 21, "window_seconds": 15.0, "px_per_sample": 2.0},
}
DEFAULT_VIEW = "normal"
DEFAULT_CANVAS_HEIGHT_PX = 1200
DPI = 130

# The default rate axis: HEADROOM times the largest absolute rate that is not a transient
# outlier, an absolute rate above its median plus TRANSIENT_OUTLIER_THRESHOLD times its
# Robust Standard Deviation.
HEADROOM = 1.3
TRANSIENT_OUTLIER_THRESHOLD = 8.0

# -------------------------------------------------------------------------
# The header, the legend, and the boxes
# -------------------------------------------------------------------------

HEADER_TITLE_BAND_INCHES = 0.44
HEADER_SUBTITLE_BAND_INCHES = 0.26
HEADER_SUBTITLE_FONTSIZE = 9.5
HEADER_TITLE_FONTSIZE = 15.0
HEADER_LINE_INCHES = 0.17
HEADER_PADDING_INCHES = 0.12
# A header taller than this share of the figure would leave the panels smaller than it.
HEADER_MAX_FIGURE_FRACTION = 0.45
HEADER_SUBTITLE_LINES_RESERVED = 2
# The least number of lines the header holds for the legend, or for the data box and the
# note below it. A chart that needs more reserves what it needs.
HEADER_BOX_LINES_RESERVED = 4
HEADER_BLOCK_GAP_PX = 40.0
HEADER_BLOCK_INSET_PX = 18.0

# Three labeled major ticks per page are the aim and two the least: with one, a scroll
# position can show none.
TICK_LABELS_PER_PAGE_TARGET = 3.0
TICK_LABELS_PER_PAGE_REQUIRED = 2.0
MINOR_TICKS_PER_MAJOR = 5

# Two legend columns, each a group: on the left the series, the local median, the ratio
# series, and the estimated peak height; on the right, every entry ranked from
# LEGEND_RANK_EXTENTS, the extents and then the threshold lines.
LEGEND_COLUMNS = 2
LEGEND_RANK_EXTENTS = 100
LEGEND_RANK_THRESHOLDS = 200

REPEATED_PANEL_NAME_Y = 0.97
REPEATED_PANEL_NAME_CLEARANCE_POINTS = 6.0
VIDEO_FRAME_AXIS_NAME_CLEARANCE_POINTS = 8.0
# The identifier of an extent stands one row below the repeated panel name, which
# would otherwise cover it, in a box padded by this share of its font size.
EXTENT_IDENTIFIER_BELOW_PANEL_NAME_POINTS = 16.0
EXTENT_IDENTIFIER_FONTSIZE = 9.0
EXTENT_IDENTIFIER_BOX_PAD = 0.2
BOX_STYLE = "round,pad=0.5,rounding_size=0.25"
BOX_PAD_FONT_FRACTION = 0.5
DATA_BOX_FONTSIZE = 8.5
# Written below the data box, right aligned with it, where the chart shows an RSD.
RSD_NOTE = "RSD = Robust Standard Deviation"
DATA_BOX_NOTE_GAP_POINTS = 2.0

# A label above a panel sits this many inches above the panel's top edge, taking the
# levels in turn so that two labels near each other do not overlap. The width of a label
# is estimated from the number of its characters.
LABEL_LEVELS_INCHES = (0.03, 0.17)
LABEL_WIDTH_PER_CHARACTER_POINTS = 5.6
LABEL_BOX_PADDING_POINTS = 8.0
# The top panel carries the video frame axis above it, so its labels start above it.
VIDEO_FRAME_AXIS_BAND_INCHES = 0.20

# -------------------------------------------------------------------------
# The titles and the series
# -------------------------------------------------------------------------

TITLE = "Normalized Rate Quaternion Gyro Chart"
SERIES_LABEL = "Normalized Rate"
AB_TITLE = f"{TITLE}: A=Original, B=Repaired"
A_SERIES_LABEL = "A Normalized Rate"
B_SERIES_LABEL = "B Normalized Rate"
DRAWN_SERIES = "normalized"
AS_RECORDED_STAGE = "as recorded"
AS_REPAIRED_STAGE = "as repaired"


class SeriesRecord(NamedTuple):
    """What one drawn series is: the series of the decode result, the video and the
    derivation version from the header of the result that holds it, and the stage that
    produced it, ``as recorded`` or ``as repaired``."""

    series: str
    video_name: str
    derivation_version: int
    stage: str


class ShadedSpanSet(NamedTuple):
    """The extents of one classification, keyed by axis as ``(start s, end s,
    identifier)``, with the file they were read from and the series their detection was
    computed on."""

    label: str
    fill: str
    spans: dict
    source: str
    series: str

    @property
    def chart_form(self):
        return (self.label, self.fill, self.spans)


class LocalMedianLevel(NamedTuple):
    """One level drawn as a pair of lines at ``multiple`` times the Robust Standard
    Deviation of its ``LocalMedianLevels`` on either side of the local median, with its
    label on the panel and its legend entry. A dashed level is an estimate, and a solid
    level is a rule the detector applies."""

    multiple: float
    label: str
    color: str
    legend_rank: int
    legend_label: str
    linestyle: str = "-"


class LocalMedianLevels(NamedTuple):
    """The local median of one axis and the levels drawn about it, sliced to the window.
    ``robust_standard_deviation_deg_s`` is the one Robust Standard Deviation the levels
    are multiples of, repeated at every rate sample."""

    axis: str
    times: np.ndarray
    local_median: np.ndarray
    robust_standard_deviation_deg_s: np.ndarray
    levels: tuple


class RatioThreshold(NamedTuple):
    """A horizontal line on the right hand axis at the height where one pass fires. The
    label states the rule the pass applies."""

    height: float
    label: str
    color: str
    legend_rank: int
    legend_label: str


class LocalRobustStandardDeviationRatio(NamedTuple):
    """The local Robust Standard Deviation of one axis divided by its full track Robust
    Standard Deviation, sliced to the window."""

    axis: str
    times: np.ndarray
    ratio: np.ndarray


# -------------------------------------------------------------------------
# The rate axis and the peaks beyond it
# -------------------------------------------------------------------------


def _largest_absolute_rate(arrays, column, first_row, last_row, outlier_threshold):
    """The largest absolute value in rows ``first_row`` to ``last_row`` of ``column`` of
    every array that is not a transient outlier, an absolute value above the median plus
    ``outlier_threshold`` times the Robust Standard Deviation of the absolute values; 0
    where the rows hold no finite value."""
    values = [np.asarray(array)[first_row : last_row + 1, column] for array in arrays]
    values = np.concatenate(values) if values else np.empty(0)
    values = np.abs(values[np.isfinite(values)])
    if not len(values):
        return 0.0
    median = np.median(values)
    robust_standard_deviation = robust_local_fit.MAD_TO_SIGMA * np.median(
        np.abs(values - median)
    )
    kept = values[values <= median + outlier_threshold * robust_standard_deviation]
    return float(kept.max())


def _largest_absolute_rate_over_three_axes(
    arrays, first_row, last_row, outlier_threshold
):
    """The half range of one rate axis shared by the three axes, before the headroom: the
    largest over the three axes of ``_largest_absolute_rate``, and at least 1e-6."""
    return max(
        1e-6,
        max(
            _largest_absolute_rate(
                arrays, column, first_row, last_row, outlier_threshold
            )
            for _name, column in AXIS_COLUMNS
        ),
    )


def _peaks_beyond_rate_axis(axis_series, rate_axis_half_range):
    """The contiguous runs of video frames where the largest absolute value over the
    series of one axis exceeds ``rate_axis_half_range``, one ``(index, signed peak)``
    per run at its peak. A value that is not a number is never the largest, so a series
    drawn only where it differs from another leaves the peaks of the other found."""
    stack = np.vstack(axis_series)
    magnitude = np.where(np.isfinite(stack), np.abs(stack), -np.inf)
    loudest = np.argmax(magnitude, axis=0)
    peak = stack[loudest, np.arange(stack.shape[1])]
    over = np.isfinite(peak) & (np.abs(peak) > rate_axis_half_range)
    peaks, start, count = [], 0, len(over)
    while start < count:
        if over[start]:
            end = start
            while end < count and over[end]:
                end += 1
            peak_index = start + int(np.argmax(np.abs(peak[start:end])))
            peaks.append((peak_index, float(peak[peak_index])))
            start = end
        else:
            start += 1
    return peaks


def _peaks_beyond_rate_axis_at_rate_sample_times(times, values, rate_axis_half_range):
    """The contiguous runs, in time order over every series given for one axis, where
    the absolute value exceeds ``rate_axis_half_range``, one ``(time, signed peak)`` per
    run at its peak rate sample. The series carry their own rate sample times, so the
    runs are found after merging them into one time order."""
    merged_times = np.concatenate(times)
    merged_values = np.concatenate(values)
    order = np.argsort(merged_times, kind="stable")
    merged_times, merged_values = merged_times[order], merged_values[order]
    over = np.isfinite(merged_values) & (np.abs(merged_values) > rate_axis_half_range)
    peaks, start, count = [], 0, len(over)
    while start < count:
        if over[start]:
            end = start
            while end < count and over[end]:
                end += 1
            peak_index = start + int(np.argmax(np.abs(merged_values[start:end])))
            peaks.append(
                (float(merged_times[peak_index]), float(merged_values[peak_index]))
            )
            start = end
        else:
            start += 1
    return peaks


# -------------------------------------------------------------------------
# The window, the extents, and the layout helpers
# -------------------------------------------------------------------------


def view_settings(view, video_frame_rate):
    """``(window in video frames, canvas width in px, px per rate sample floor)`` for a
    view on a video at ``video_frame_rate`` video frames per second."""
    scale = VIEW_SCALES[view]
    window_video_frames = scale.get("window_video_frames")
    if window_video_frames is None:
        window_video_frames = round(scale["window_seconds"] * video_frame_rate)
    return (
        window_video_frames,
        round(scale["pages"] * PAGE_WIDTH_PX),
        scale["px_per_sample"],
    )


def clamped_center(requested_video_frame, window_video_frames, video_frame_count):
    """``(center video frame drawn, (first video frame, last video frame))`` of the
    window.

    The window is centered on ``requested_video_frame``, with half the window before it.
    Where the start or the end of the video falls in the window, the window begins at
    video frame 0 or ends at the last video frame and keeps its length. A video with
    fewer video frames than the window gives the whole video as the window. The center
    drawn is the first video frame of the window plus half its length."""
    window_video_frames = min(window_video_frames, video_frame_count)
    first = min(
        max(requested_video_frame - window_video_frames // 2, 0),
        video_frame_count - window_video_frames,
    )
    return first + window_video_frames // 2, (first, first + window_video_frames - 1)


def window_lines(requested_video_frame, drawn_center):
    """The data box lines stating the requested center and the center drawn, where the
    window was slid to fit the video; none where it was not."""
    if drawn_center == requested_video_frame:
        return ()
    return (
        ("center frame requested", f"f{requested_video_frame}"),
        ("center frame drawn", f"f{drawn_center}, window slid to fit the video"),
    )


def _window_indices(times, window_start_s, window_end_s):
    """The first and the last index of ``times`` inside the window [``window_start_s``,
    ``window_end_s``]."""
    first = int(np.searchsorted(times, window_start_s, side="left"))
    last = int(np.searchsorted(times, window_end_s, side="right")) - 1
    return first, last


def _extent_boundary_line_color(fill):
    """The color of the boundary lines of an extent: the hue of its fill, at
    ``EXTENT_BOUNDARY_LINE_SATURATION`` and ``EXTENT_BOUNDARY_LINE_VALUE``."""
    hue, _saturation, _value = colorsys.rgb_to_hsv(*mcolors.to_rgb(fill))
    return colorsys.hsv_to_rgb(
        hue, EXTENT_BOUNDARY_LINE_SATURATION, EXTENT_BOUNDARY_LINE_VALUE
    )


def _resolve_time_span_sets(named_time_span_sets):
    """``[(label, fill, boundary line color, {axis name: [(start s, end s, identifier),
    ...]}), ...]`` for the extents. An extent is drawn at its exact times, never widened
    to video frame boundaries."""
    resolved = []
    for label, fill, time_spans_by_axis in named_time_span_sets:
        extents_by_axis = {
            name: list(time_spans_by_axis[name]) for name, _column in AXIS_COLUMNS
        }
        resolved.append(
            (label, fill, _extent_boundary_line_color(fill), extents_by_axis)
        )
    return resolved


def _video_frame_range_text(first_video_frame, last_video_frame):
    """A range of video frames as the title writes it: ``f####`` for one video frame,
    ``f####-f####`` for more."""
    if first_video_frame == last_video_frame:
        return f"f{first_video_frame}"
    return f"f{first_video_frame}-f{last_video_frame}"


def _lay_out_and_measure(figure, panel, dpi, panel_rectangle):
    """Lay the panels out inside ``panel_rectangle`` without resizing the figure, and
    return ``(the realized axes width in px, the px the margins take)``."""
    figure.tight_layout(rect=panel_rectangle)
    return _axes_width_and_margins_px(figure, panel, dpi)


def _axes_width_and_margins_px(figure, panel, dpi):
    """``(the realized width in px of panel, the px the margins take)``."""
    width_px = float(figure.get_size_inches()[0]) * dpi
    axes_width_px = panel.get_position().width * width_px
    return axes_width_px, width_px - axes_width_px


def _calculate_panel_right_edge(figure, right_hand_axes):
    """The right edge of the panels, as a share of the figure width, that keeps the tick
    values and the name of the right hand axes ``right_hand_axes`` on the canvas with
    ``RIGHT_AXIS_RATIO_EDGE_CLEARANCE_POINTS`` to spare, measured from their text. The
    edge moves in by as far as they reach past the canvas, and never out."""
    renderer = figure.canvas.get_renderer()
    reach_px = max(
        right_hand_axis.get_tightbbox(renderer).x1
        for right_hand_axis in right_hand_axes.values()
    )
    overflow_px = (
        reach_px
        + RIGHT_AXIS_RATIO_EDGE_CLEARANCE_POINTS * DPI / 72.0
        - figure.bbox.width
    )
    return figure.subplotpars.right - max(overflow_px, 0.0) / figure.bbox.width


def _data_box_text(lines):
    """The data box as monospace text in two columns, each padded to one width, so the
    labels read down the left and the values down the right."""
    label_width = max(len(label) for label, _value in lines)
    value_width = max(len(value) for _label, value in lines)
    return "\n".join(
        f"{label:<{label_width}}   {value:>{value_width}}" for label, value in lines
    )


def _largest_nice_step(maximum_step):
    """The largest step of the 1, 2, 2.5, and 5 ladder of each decade that does not
    exceed ``maximum_step``."""
    decade = 10.0 ** math.floor(math.log10(maximum_step))
    candidates = [
        multiple * decade * scale
        for scale in (0.01, 0.1, 1.0, 10.0)
        for multiple in (1.0, 2.0, 2.5, 5.0)
    ]
    return max(step for step in candidates if step <= maximum_step * (1.0 + 1e-12))


def _tick_positions(low, high, step):
    """Every multiple of ``step`` inside [``low``, ``high``]."""
    return np.arange(math.ceil(low / step), math.floor(high / step) + 1.0) * step


def _apply_x_tick_intervals(
    panel,
    video_frame_axis,
    axes_width_px,
    window_start_s,
    window_end_s,
    video_frame_rate,
):
    """Install the labeled major ticks and the minor ticks of the time axis and the video
    frame axis, with the major step chosen from the realized axes width so that a page
    holds ``TICK_LABELS_PER_PAGE_TARGET`` labels, and return the steps and the labels per
    page realized."""
    span_s = window_end_s - window_start_s
    span_video_frames = span_s * video_frame_rate
    widest_page_share_per_label = PAGE_WIDTH_PX / TICK_LABELS_PER_PAGE_TARGET

    seconds_step = _largest_nice_step(
        span_s * widest_page_share_per_label / axes_width_px
    )
    seconds_major = _tick_positions(window_start_s, window_end_s, seconds_step)
    decimals = max(0, math.ceil(-math.log10(seconds_step)))
    panel.xaxis.set_major_locator(FixedLocator(seconds_major))
    panel.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:.{decimals}f}")
    )
    panel.xaxis.set_minor_locator(MultipleLocator(seconds_step / MINOR_TICKS_PER_MAJOR))

    video_frame_step = max(
        1.0,
        float(
            round(
                _largest_nice_step(
                    span_video_frames * widest_page_share_per_label / axes_width_px
                )
            )
        ),
    )
    video_frame_major = _tick_positions(
        window_start_s * video_frame_rate,
        window_end_s * video_frame_rate,
        video_frame_step,
    )
    video_frame_axis.set_ticks(
        video_frame_major,
        labels=[f"{round(position)}" for position in video_frame_major],
    )
    video_frame_axis.xaxis.set_minor_locator(
        MultipleLocator(max(1.0, video_frame_step / MINOR_TICKS_PER_MAJOR))
    )

    px_per_seconds_label = axes_width_px * seconds_step / span_s
    px_per_video_frame_label = axes_width_px * video_frame_step / span_video_frames
    return {
        "seconds_axis_tick_step_s": float(seconds_step),
        "seconds_axis_tick_labels_per_page": float(
            PAGE_WIDTH_PX / px_per_seconds_label
        ),
        "video_frame_axis_tick_step_frames": float(video_frame_step),
        "video_frame_axis_tick_labels_per_page": float(
            PAGE_WIDTH_PX / px_per_video_frame_label
        ),
    }


def _reserve_video_frame_axis_name(panel, widest_tick_label):
    """``frame (n)``, the name of the video frame axis above ``panel``, written in the
    left margin where ``widest_tick_label`` centered on the panel's left edge would push
    it, so that the layout leaves room for it wherever the first tick falls.
    ``_place_video_frame_axis_name`` moves it once the ticks are set."""
    widths, _heights = _text_sizes_points(
        [widest_tick_label], plt.rcParams["xtick.labelsize"]
    )
    return panel.annotate(
        "frame (n)",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-(VIDEO_FRAME_AXIS_NAME_CLEARANCE_POINTS + 0.5 * widths[0]), 0.0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=10,
        annotation_clip=False,
    )


def _place_video_frame_axis_name(figure, panel, video_frame_axis, name):
    """Move ``name``, the name of the video frame axis above ``panel``, onto the line of
    its tick labels, ending ``VIDEO_FRAME_AXIS_NAME_CLEARANCE_POINTS`` left of the panel
    and of the leftmost tick label. Called after the ticks are set."""
    renderer = figure.canvas.get_renderer()
    # A secondary axis takes its place and its limits from its parent only when drawn.
    video_frame_axis.apply_aspect(
        video_frame_axis.get_axes_locator()(video_frame_axis, renderer)
    )
    leftmost = min(
        (
            label.get_window_extent(renderer)
            for label in video_frame_axis.get_xticklabels()
        ),
        key=lambda extent: extent.x0,
    )
    panel_extent = panel.get_window_extent(renderer)
    right_px = (
        min(leftmost.x0, panel_extent.x0)
        - VIDEO_FRAME_AXIS_NAME_CLEARANCE_POINTS * DPI / 72.0
    )
    name.xyann = (
        (right_px - panel_extent.x0) * 72.0 / DPI,
        (0.5 * (leftmost.y0 + leftmost.y1) - panel_extent.y1) * 72.0 / DPI,
    )


def _header_block_positions(canvas_width_px, block_width_px):
    """``(the left edge in px of every header block, the drawn width of a block, the
    pitch between blocks)``.

    A canvas no wider than a page takes one block over its full width. A wider canvas is
    read by scrolling, and a view of one page shows a whole block only where the pitch
    plus the width of a block fits in a page, so the pitch is a page less a block."""
    if canvas_width_px <= PAGE_WIDTH_PX:
        return [0.0], float(canvas_width_px), float(canvas_width_px)
    pitch = PAGE_WIDTH_PX - block_width_px
    if pitch <= 0.0:
        raise ValueError(
            f"one header block is {block_width_px:.0f} px wide, which leaves no room to "
            f"repeat it inside a {PAGE_WIDTH_PX:.0f} px page"
        )
    lefts, left = [], 0.0
    while left + block_width_px <= canvas_width_px:
        lefts.append(left)
        left += pitch
    return lefts, block_width_px, pitch


def _data_x_at_px(panel, canvas_width_px, x_px, window_start_s, window_end_s):
    """The time under canvas pixel ``x_px``, held inside the window."""
    position = panel.get_position()
    inside = (x_px / canvas_width_px - position.x0) / position.width
    return window_start_s + min(max(inside, 0.0), 1.0) * (window_end_s - window_start_s)


def label_center_on_canvas(center, half_width, canvas_left, canvas_right):
    """The center of a label that reaches ``half_width`` either side of it: ``center``
    where the label lies whole between ``canvas_left`` and ``canvas_right``, and otherwise
    moved inward from the end it would cross, so that the label stays whole."""
    return min(max(center, canvas_left + half_width), canvas_right - half_width)


def _draw_labels_above_panels(
    panels, pending, figure_width_inches, figure_height_inches
):
    """Write each queued label in the clear space above its panel, at a height in inches
    off the panel's realized top edge.

    ``pending`` is ``{panel index: [(time, text, color), ...]}``. Each label is centered
    on its time, or moved inward by ``label_center_on_canvas`` where it would cross an
    edge of the canvas. The labels are sorted by time and each takes the lowest level
    where it overlaps no label already there, or else the level whose last label ends
    soonest."""
    for panel_index, labels in pending.items():
        panel = panels[panel_index]
        position = panel.get_position()
        panel_inches = position.height * figure_height_inches
        base = VIDEO_FRAME_AXIS_BAND_INCHES if panel_index == 0 else 0.0
        x_left, x_right = panel.get_xlim()
        axes_points = position.width * figure_width_inches * 72.0
        time_per_figure_width = (x_right - x_left) / position.width
        canvas_left = x_left - position.x0 * time_per_figure_width
        canvas_right = x_right + (1.0 - position.x1) * time_per_figure_width
        occupied = [[] for _level in LABEL_LEVELS_INCHES]
        for time, text, color in sorted(labels):
            half_width = (
                0.5
                * (
                    len(text) * LABEL_WIDTH_PER_CHARACTER_POINTS
                    + LABEL_BOX_PADDING_POINTS
                )
                * (x_right - x_left)
                / axes_points
            )
            x_data = label_center_on_canvas(time, half_width, canvas_left, canvas_right)
            level = next(
                (
                    i
                    for i, taken in enumerate(occupied)
                    if all(
                        x_data - half_width > end or x_data + half_width < start
                        for start, end in taken
                    )
                ),
                min(
                    range(len(occupied)),
                    key=lambda candidate_level: (
                        occupied[candidate_level][-1][1]
                        if occupied[candidate_level]
                        else -np.inf
                    ),
                ),
            )
            occupied[level].append((x_data - half_width, x_data + half_width))
            height_inches = base + LABEL_LEVELS_INCHES[level]
            panel.annotate(
                text,
                xy=(x_data, 1.0 + height_inches / panel_inches),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": BOX_EDGE_COLOR,
                    "lw": 0.8,
                },
                annotation_clip=False,
                zorder=7,
            )


def _legend_handles_in_groups(handles):
    """The legend handles, ordered and padded so that each column holds one group.

    Sorted by ``legend_rank``, stable, then split at ``LEGEND_RANK_EXTENTS`` into the left
    column and the right column, and both padded with blank entries to the same length,
    so that the column major fill puts each group in its own column."""
    ordered = sorted(handles, key=lambda handle: getattr(handle, "legend_rank", 0))
    left_column = [
        handle
        for handle in ordered
        if getattr(handle, "legend_rank", 0) < LEGEND_RANK_EXTENTS
    ]
    right_column = [
        handle
        for handle in ordered
        if getattr(handle, "legend_rank", 0) >= LEGEND_RANK_EXTENTS
    ]
    if not right_column:
        return left_column
    rows = max(len(left_column), len(right_column))
    left_column += [
        Line2D([], [], ls="none", marker="", label="")
        for _ in range(rows - len(left_column))
    ]
    right_column += [
        Line2D([], [], ls="none", marker="", label="")
        for _ in range(rows - len(right_column))
    ]
    return left_column + right_column


def _data_box_lines(
    *, quaternion_rate_hz, measurement_rate_hz, video_frame_rate, extra_lines
):
    """The data box as ``(label, value)`` pairs, one value per line: the video frame rate,
    and the quaternion rate and the measurement rate in hertz to one decimal place,
    followed by ``extra_lines``."""
    return [
        ("video frame rate", f"{video_frame_rate:.2f} fps"),
        ("quaternion rate", f"{quaternion_rate_hz:.1f} Hz"),
        ("measurement rate", f"{measurement_rate_hz:.1f} Hz"),
        *extra_lines,
    ]


def _raised_box_lines(reserved, block_width_px, component_widths_px):
    """How many of the reserved lines the legend and the data box, with the note below it,
    take by moving up beside the title and the subtitles rather than down into the
    panels.

    Only the lines beyond ``HEADER_BOX_LINES_RESERVED`` move, as high as the top of the
    title band, and only where one header block is wide enough to hold the legend, the
    widest centered line, and the data box side by side with a gap between each."""
    extra = reserved - HEADER_BOX_LINES_RESERVED
    if extra <= 0:
        return 0
    clear_px = (
        block_width_px
        - 2 * HEADER_BLOCK_INSET_PX
        - component_widths_px["legend"]
        - component_widths_px["data_box"]
        - component_widths_px["centered"]
        - 2 * HEADER_BLOCK_GAP_PX
    )
    if clear_px < 0.0:
        return 0
    room_inches = (
        HEADER_TITLE_BAND_INCHES
        + HEADER_SUBTITLE_LINES_RESERVED * HEADER_SUBTITLE_BAND_INCHES
    )
    return min(extra, int(room_inches / HEADER_LINE_INCHES))


def _panel_rectangle_and_header_heights(
    figure_height_inches, box_lines_reserved, raised_box_lines
):
    """``(the rectangle of the panels, the height of the first subtitle line, the height
    of the top of the legend and the data box)``, each a fraction of the figure.

    The header is one fixed height for the worst case it reserves, so that charts with
    the same reservation have the same panel geometry. The raised lines take no height
    from the panels."""
    box_lines = box_lines_reserved - raised_box_lines
    header_inches = (
        HEADER_TITLE_BAND_INCHES
        + HEADER_SUBTITLE_LINES_RESERVED * HEADER_SUBTITLE_BAND_INCHES
        + HEADER_PADDING_INCHES
        + box_lines * HEADER_LINE_INCHES
    )
    if header_inches > HEADER_MAX_FIGURE_FRACTION * figure_height_inches:
        raise ValueError(
            f"the header band needs {header_inches:.2f} inches to hold {box_lines} lines "
            f"of legend or data box, which is more than "
            f"{HEADER_MAX_FIGURE_FRACTION:.0%} of the {figure_height_inches:.2f} inch "
            f"figure"
        )
    subtitle_y = 1.0 - HEADER_TITLE_BAND_INCHES / figure_height_inches
    subtitles_bottom_y = (
        subtitle_y
        - HEADER_SUBTITLE_LINES_RESERVED
        * HEADER_SUBTITLE_BAND_INCHES
        / figure_height_inches
    )
    return (
        (0.0, 0.0, 1.0, 1.0 - header_inches / figure_height_inches),
        subtitle_y,
        subtitles_bottom_y
        + raised_box_lines * HEADER_LINE_INCHES / figure_height_inches,
    )


def _header_block_width_px(handles, title, subtitle_lines, data_box_text, dpi):
    """``(the width in px of one header block, the widths it was taken over)``: the
    widest of the title, a subtitle line, and the legend and the data box side by side.

    The handles are grouped in place before the legend is measured, so the legend is
    measured in the arrangement it is drawn in. The text is measured on a small figure at
    the same dpi, since a text extent depends on the font and the dpi alone."""
    handles[:] = _legend_handles_in_groups(handles)
    small_figure = plt.figure(figsize=(1.0, 1.0), dpi=dpi)
    title_artist = small_figure.text(0.0, 0.0, title, fontsize=HEADER_TITLE_FONTSIZE)
    subtitle_artists = [
        small_figure.text(0.0, 0.0, line, fontsize=HEADER_SUBTITLE_FONTSIZE)
        for line in subtitle_lines
    ]
    box_artist = small_figure.text(
        0.0,
        0.0,
        data_box_text,
        fontsize=DATA_BOX_FONTSIZE,
        linespacing=1.35,
        family="DejaVu Sans Mono",
        bbox={"boxstyle": BOX_STYLE, "fc": "white", "ec": BOX_EDGE_COLOR, "lw": 0.8},
    )
    legend_artist = small_figure.legend(
        handles=handles,
        loc="upper left",
        fontsize=9,
        frameon=True,
        fancybox=True,
        ncol=LEGEND_COLUMNS,
        borderpad=BOX_PAD_FONT_FRACTION,
    )
    legend_artist.get_frame().set_boxstyle(BOX_STYLE)
    small_figure.canvas.draw()
    renderer = small_figure.canvas.get_renderer()
    components = {
        "centered": float(
            max(
                (
                    title_artist.get_window_extent(renderer).width,
                    *(
                        artist.get_window_extent(renderer).width
                        for artist in subtitle_artists
                    ),
                )
            )
        ),
        "legend": float(legend_artist.get_window_extent(renderer).width),
        "data_box": float(box_artist.get_window_extent(renderer).width),
    }
    plt.close(small_figure)
    return (
        max(
            components["centered"],
            components["legend"] + HEADER_BLOCK_GAP_PX + components["data_box"],
        ),
        components,
    )


def _repeat_y_scale(panels, x_data, rate_axis_half_ranges):
    """Write each panel's name and its tick values at ``x_data``, since on a canvas read
    by scrolling the real ones leave the view with the first page.
    ``rate_axis_half_ranges`` holds the half range of each panel."""
    for (name, column), panel in zip(AXIS_COLUMNS, panels, strict=True):
        panel.annotate(
            f"{name} (deg/s)",
            (x_data, REPEATED_PANEL_NAME_Y),
            xycoords=panel.get_xaxis_transform(),
            xytext=(-REPEATED_PANEL_NAME_CLEARANCE_POINTS, 0),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=8.5,
            color="#222222",
            bbox={
                "boxstyle": "square,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.85,
            },
            zorder=8,
            annotation_clip=False,
        )
        for value in panel.get_yticks():
            if abs(value) > rate_axis_half_ranges[column]:
                continue
            panel.text(
                x_data,
                value,
                f"{value:g}",
                ha="left",
                va="center",
                fontsize=7.5,
                color="#666666",
                bbox={
                    "boxstyle": "square,pad=0.1",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.85,
                },
                zorder=8,
            )


# -------------------------------------------------------------------------
# The local median, the threshold lines, and the right hand axis
# -------------------------------------------------------------------------


def draw_local_median_levels(panel, local_median_levels):
    """Draw one axis's local median, dashed, and each of its levels as a pair of lines
    about the local median in the level's own color and line style."""
    for level in sorted(
        local_median_levels.levels, key=lambda level: level.multiple, reverse=True
    ):
        for sign in (1.0, -1.0):
            panel.plot(
                local_median_levels.times,
                local_median_levels.local_median
                + sign
                * level.multiple
                * local_median_levels.robust_standard_deviation_deg_s,
                color=level.color,
                lw=PASS_THRESHOLD_LINE_WIDTH,
                ls=level.linestyle,
                zorder=4,
            )
    panel.plot(
        local_median_levels.times,
        local_median_levels.local_median,
        color=LOCAL_MEDIAN_COLOR,
        lw=0.8,
        ls="--",
        alpha=LOCAL_MEDIAN_LINE_ALPHA,
        zorder=2,
    )


def local_median_legend_handles(local_median_levels):
    """The legend entries of the local median and of each level drawn about it, one
    entry per level whatever the number of panels it is drawn on."""
    if not local_median_levels:
        return []
    local_median_entry = Line2D(
        [0],
        [0],
        color=LOCAL_MEDIAN_COLOR,
        lw=1.2,
        ls="--",
        label=LOCAL_MEDIAN_LABEL,
    )
    local_median_entry.legend_rank = 10
    handles, seen = [local_median_entry], set()
    for local_median_levels_of_axis in local_median_levels:
        for level in local_median_levels_of_axis.levels:
            if level.legend_label in seen:
                continue
            seen.add(level.legend_label)
            entry = Line2D(
                [0],
                [0],
                color=level.color,
                ls=level.linestyle,
                lw=PASS_THRESHOLD_LEGEND_LINE_WIDTH,
                label=level.legend_label,
            )
            entry.legend_rank = level.legend_rank
            handles.append(entry)
    return handles


def right_hand_ratio_legend_handles(ratios, thresholds):
    """The legend entry of the ratio series and one entry per threshold, whatever the
    number of panels it is drawn on."""
    if not ratios:
        return []
    ratio_series_entry = Line2D(
        [0], [0], color=RIGHT_AXIS_RATIO_COLOR, lw=1.6, label=RIGHT_AXIS_RATIO_LABEL
    )
    ratio_series_entry.legend_rank = 20
    handles, seen = [ratio_series_entry], set()
    for lines in thresholds.values():
        for line in lines:
            if line.legend_label in seen:
                continue
            seen.add(line.legend_label)
            entry = Line2D(
                [0],
                [0],
                color=line.color,
                lw=PASS_THRESHOLD_LEGEND_LINE_WIDTH,
                label=line.legend_label,
            )
            entry.legend_rank = line.legend_rank
            handles.append(entry)
    return handles


def attach_right_hand_ratio_axes(panels, ratios, thresholds, y_max):
    """Give the panel of each ratio series a right hand axis that carries the series and
    the threshold lines of its axis, from 0 to ``y_max``, and return the right hand axes
    keyed by axis name.

    Called after the panels are laid out: a right hand axis present during the layout
    would take height from the panels."""
    right_hand_axes = {}
    for ratio_series in ratios:
        panel = panels[AXIS_ORDER.index(ratio_series.axis)]
        right_hand_axis = panel.twinx()
        right_hand_axis.set_in_layout(False)
        right_hand_axis.plot(
            ratio_series.times,
            ratio_series.ratio,
            color=RIGHT_AXIS_RATIO_COLOR,
            lw=RIGHT_AXIS_RATIO_LINE_WIDTH,
            zorder=6,
        )
        for line in thresholds[ratio_series.axis]:
            right_hand_axis.axhline(
                line.height,
                color=line.color,
                lw=PASS_THRESHOLD_LINE_WIDTH,
                zorder=7,
            )
        right_hand_axis.set_xlim(*panel.get_xlim())
        right_hand_axis.set_ylim(0.0, y_max)
        right_hand_axis.set_ylabel(RIGHT_AXIS_RATIO_LABEL, color=RIGHT_AXIS_RATIO_COLOR)
        right_hand_axis.tick_params(
            axis="y", colors=RIGHT_AXIS_RATIO_COLOR, labelsize=7.5
        )
        right_hand_axes[ratio_series.axis] = right_hand_axis
    return right_hand_axes


def ratio_axis_name_bottom(
    values_under_name, name_height, y_max, panel_height, clearance
):
    """The bottom of the box of the name of the ratio axis, in points above the panel's
    bottom edge, for the ratio series ``values_under_name`` under the whole width of the
    name, on a right hand axis from 0 to ``y_max`` over a panel ``panel_height`` points
    high: ``THRESHOLD_LABEL_GAP_POINTS`` above the highest of them, or, where the name,
    ``name_height`` high, has no room there, as far below the lowest. Either way the name
    stays ``clearance`` points inside the panel's border lines."""
    heights = np.asarray(values_under_name, dtype=np.float64) / y_max * panel_height
    bottom = heights.max() + THRESHOLD_LABEL_GAP_POINTS
    if bottom + name_height > panel_height - clearance:
        bottom = heights.min() - THRESHOLD_LABEL_GAP_POINTS - name_height
    return min(max(bottom, clearance), panel_height - clearance - name_height)


def write_right_hand_ratio_scale(right_hand_axes, ratios, x_data, y_max):
    """Write the name and the tick values of the right hand axis inside each panel at
    ``x_data``, wherever the left hand scale is repeated and once at the left edge of the
    window, since the real right hand tick values sit at the far end of the canvas. The
    name stands just above or below the ratio series under it, placed by
    ``ratio_axis_name_bottom``."""
    widths, text_heights = _text_sizes_points(
        [RIGHT_AXIS_RATIO_LABEL], RIGHT_AXIS_RATIO_NAME_FONTSIZE
    )
    pad_points = RIGHT_AXIS_RATIO_NAME_BOX_PAD * RIGHT_AXIS_RATIO_NAME_FONTSIZE
    for ratio_series in ratios:
        right_hand_axis = right_hand_axes[ratio_series.axis]
        figure_width_inches, figure_height_inches = (
            right_hand_axis.figure.get_size_inches()
        )
        position = right_hand_axis.get_position()
        x_left, x_right = right_hand_axis.get_xlim()
        seconds_per_point = (x_right - x_left) / (
            position.width * figure_width_inches * 72.0
        )
        name_start = (
            x_data
            + (RIGHT_AXIS_RATIO_NAME_OFFSET_POINTS - pad_points) * seconds_per_point
        )
        name_end = name_start + (widths[0] + 2.0 * pad_points) * seconds_per_point
        # The drawn line runs on to the rate sample past each end of the name.
        first = max(
            int(np.searchsorted(ratio_series.times, name_start, side="right")) - 1, 0
        )
        last = int(np.searchsorted(ratio_series.times, name_end, side="left")) + 1
        panel_height = position.height * figure_height_inches * 72.0
        bottom = ratio_axis_name_bottom(
            ratio_series.ratio[first:last],
            text_heights[0] + 2.0 * pad_points,
            y_max,
            panel_height,
            right_hand_axis.spines["top"].get_linewidth() + THRESHOLD_LABEL_GAP_POINTS,
        )
        right_hand_axis.annotate(
            RIGHT_AXIS_RATIO_LABEL,
            (x_data, bottom / panel_height),
            xycoords=("data", "axes fraction"),
            # The box reaches its pad below the text, so the text rises by the pad.
            xytext=(RIGHT_AXIS_RATIO_NAME_OFFSET_POINTS, pad_points),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=RIGHT_AXIS_RATIO_NAME_FONTSIZE,
            color=RIGHT_AXIS_RATIO_COLOR,
            bbox={
                "boxstyle": f"square,pad={RIGHT_AXIS_RATIO_NAME_BOX_PAD}",
                "fc": "white",
                "ec": "none",
                "alpha": 0.85,
            },
            zorder=8,
            annotation_clip=False,
        )
        for value in right_hand_axis.get_yticks():
            if value < 0.0 or value > y_max:
                continue
            right_hand_axis.annotate(
                f"{value:g}",
                (x_data, value),
                xytext=(RIGHT_AXIS_RATIO_VALUE_OFFSET_POINTS, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.5,
                color=RIGHT_AXIS_RATIO_COLOR,
                bbox={
                    "boxstyle": "square,pad=0.1",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.85,
                },
                zorder=8,
                annotation_clip=False,
            )


def threshold_label_bottoms(lines, heights, panel_height, border_width):
    """The bottom of each threshold label at the right edge of a panel, in points above
    the panel's bottom edge, for the lines at ``lines`` points above that edge, each label
    ``heights`` points high, in a panel ``panel_height`` points high whose border lines
    are ``border_width`` points wide.

    Every label stays inside the panel, clear of its top and bottom border lines by the
    border width and ``THRESHOLD_LABEL_GAP_POINTS``, so that no box covers a border line.
    Each label starts ``THRESHOLD_LABEL_GAP_POINTS`` above its line, or as far below it
    where it would not clear the top border line. Labels that would overlap are then moved
    apart by the least distance that clears them, the least sum of the squares of the
    moves, keeping the order of their starts and staying inside that range.

    In that order, each bottom less the heights of the labels below it turns no overlap
    into a sequence that never falls, and the range into one range for every label, so
    the placement is the isotonic regression of the starts so reduced, held to that
    range."""
    lines = np.asarray(lines, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    clearance = border_width + THRESHOLD_LABEL_GAP_POINTS
    above = lines + THRESHOLD_LABEL_GAP_POINTS
    starts = np.where(
        above + heights <= panel_height - clearance,
        above,
        lines - THRESHOLD_LABEL_GAP_POINTS - heights,
    )
    order = np.argsort(starts, kind="stable")
    heights_below = np.cumsum(heights[order]) - heights[order]
    reduced = np.clip(
        isotonic_regression(starts[order] - heights_below).x,
        clearance,
        panel_height - clearance - heights.sum(),
    )
    bottoms = np.empty_like(starts)
    bottoms[order] = reduced + heights_below
    return bottoms


def _text_sizes_points(texts, fontsize, fontweight="normal"):
    """``(widths, heights)`` in points of each of ``texts`` at ``fontsize`` and
    ``fontweight``, measured on a small figure at ``DPI``, since a text extent depends on
    the font and the dpi alone."""
    small_figure = plt.figure(figsize=(1.0, 1.0), dpi=DPI)
    renderer = small_figure.canvas.get_renderer()
    extents = [
        small_figure.text(
            0.0, 0.0, text, fontsize=fontsize, fontweight=fontweight
        ).get_window_extent(renderer)
        for text in texts
    ]
    plt.close(small_figure)
    return (
        np.array([extent.width for extent in extents]) * 72.0 / DPI,
        np.array([extent.height for extent in extents]) * 72.0 / DPI,
    )


def draw_threshold_labels(
    panels,
    right_hand_axes,
    local_median_levels,
    thresholds,
    rate_axis_half_ranges,
    ratio_y_max,
    figure_width_inches,
    figure_height_inches,
):
    """Write the label of every threshold line at the right edge of its panel: the
    levels of ``local_median_levels`` on the rate axis, at the upper line of each pair,
    and the lines of ``thresholds`` on the right hand axis, placed together by
    ``threshold_label_bottoms``. ``rate_axis_half_ranges`` holds the half range of each
    panel and ``ratio_y_max`` the top of the right hand axis. Return the box of each
    label, keyed by axis name, as ``(left, right, bottom, top)`` in points from the
    panel's lower left corner.

    Called after the panels are laid out, since the labels are placed in points of the
    realized panel height, and after the right hand axes are attached, so that the labels
    are drawn over them."""
    local_median_levels_by_axis = {
        local_median_levels_of_axis.axis: local_median_levels_of_axis
        for local_median_levels_of_axis in local_median_levels
    }
    boxes = {name: [] for name, _column in AXIS_COLUMNS}
    for panel, (name, column) in zip(panels, AXIS_COLUMNS, strict=True):
        labels = []
        if name in local_median_levels_by_axis:
            local_median_levels_of_axis = local_median_levels_by_axis[name]
            labels += [
                (
                    (
                        local_median_levels_of_axis.local_median[-1]
                        + level.multiple
                        * local_median_levels_of_axis.robust_standard_deviation_deg_s[
                            -1
                        ]
                        + rate_axis_half_ranges[column]
                    )
                    / (2.0 * rate_axis_half_ranges[column]),
                    level.label,
                    level.color,
                )
                for level in local_median_levels_of_axis.levels
            ]
        labels += [
            (line.height / ratio_y_max, line.label, line.color)
            for line in thresholds.get(name, ())
        ]
        if not labels:
            continue
        shares, texts, colors = zip(*labels, strict=True)
        panel_points = panel.get_position().height * figure_height_inches * 72.0
        widths, text_heights = _text_sizes_points(texts, THRESHOLD_LABEL_FONTSIZE)
        pad_points = THRESHOLD_LABEL_BOX_PAD * THRESHOLD_LABEL_FONTSIZE
        heights = text_heights + 2.0 * pad_points
        bottoms = threshold_label_bottoms(
            np.array(shares) * panel_points,
            heights,
            panel_points,
            panel.spines["top"].get_linewidth(),
        )
        right = (
            panel.get_position().width * figure_width_inches * 72.0
            - THRESHOLD_LABEL_INSET_POINTS
            + pad_points
        )
        boxes[name] = [
            (right - width - 2.0 * pad_points, right, bottom, bottom + height)
            for width, bottom, height in zip(widths, bottoms, heights, strict=True)
        ]
        for bottom, text, color in zip(bottoms, texts, colors, strict=True):
            right_hand_axes.get(name, panel).annotate(
                text,
                xy=(1.0, bottom / panel_points),
                xycoords="axes fraction",
                # The box reaches its pad below the text, so the text rises by the pad.
                xytext=(-THRESHOLD_LABEL_INSET_POINTS, pad_points),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=THRESHOLD_LABEL_FONTSIZE,
                color=color,
                bbox={
                    "boxstyle": f"square,pad={THRESHOLD_LABEL_BOX_PAD}",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.85,
                },
                zorder=8,
                annotation_clip=False,
            )
    return boxes


def extent_identifier_top(top, height, left, right, label_boxes):
    """The top of the box of an extent identifier, ``height`` high and from ``left`` to
    ``right``: ``top`` where it overlaps no box of ``label_boxes``, each ``(left, right,
    bottom, top)``, and otherwise moved down, keeping its horizontal place, to
    ``THRESHOLD_LABEL_GAP_POINTS`` below each label box it would overlap in turn, from the
    highest down, until it is clear of every one."""
    beside = sorted(
        (box for box in label_boxes if box[0] < right and box[1] > left),
        key=lambda box: box[3],
        reverse=True,
    )
    for _left, _right, bottom, box_top in beside:
        if top - height < box_top and top > bottom:
            top = bottom - THRESHOLD_LABEL_GAP_POINTS
    return top


def _lower_extent_identifiers(
    panels, identifiers, label_boxes, figure_width_inches, figure_height_inches
):
    """Move each extent identifier of ``identifiers``, keyed by axis name, down inside its
    extent below every threshold label box of ``label_boxes`` it would overlap, placed by
    ``extent_identifier_top``."""
    pad_points = EXTENT_IDENTIFIER_BOX_PAD * EXTENT_IDENTIFIER_FONTSIZE
    for panel, (name, _column) in zip(panels, AXIS_COLUMNS, strict=True):
        if not identifiers[name] or not label_boxes[name]:
            continue
        position = panel.get_position()
        panel_width = position.width * figure_width_inches * 72.0
        panel_height = position.height * figure_height_inches * 72.0
        x_left, x_right = panel.get_xlim()
        widths, text_heights = _text_sizes_points(
            [annotation.get_text() for annotation in identifiers[name]],
            EXTENT_IDENTIFIER_FONTSIZE,
            fontweight="bold",
        )
        top = (
            REPEATED_PANEL_NAME_Y * panel_height
            - EXTENT_IDENTIFIER_BELOW_PANEL_NAME_POINTS
            + pad_points
        )
        for annotation, width, text_height in zip(
            identifiers[name], widths, text_heights, strict=True
        ):
            center = (annotation.xy[0] - x_left) / (x_right - x_left) * panel_width
            half_width = 0.5 * width + pad_points
            lowered = extent_identifier_top(
                top,
                text_height + 2.0 * pad_points,
                center - half_width,
                center + half_width,
                label_boxes[name],
            )
            annotation.xyann = (0.0, annotation.xyann[1] - (top - lowered))


# -------------------------------------------------------------------------
# The drawing
# -------------------------------------------------------------------------


def rate_sample_curve(times, values, px_per_second):
    """``(times, values)`` of the curve drawn through the rate samples ``values`` at
    ``times``, at ``px_per_second`` pixels of axes width per second: one
    ``PchipInterpolator`` evaluated at each rate sample time and, between them, at points
    no more than ``RATE_CURVE_SEGMENT_PX`` pixels apart.

    ``PchipInterpolator`` is a piecewise cubic Hermite interpolation that preserves
    monotonicity: it passes through every rate sample and never overshoots between two
    neighbors, so it has no peak or trough where no rate sample is. A natural cubic spline
    overshoots and would draw motion that was not recorded."""
    times = np.asarray(times, dtype=np.float64)
    intervals = np.diff(times)
    steps = np.ceil(intervals * px_per_second / RATE_CURVE_SEGMENT_PX).astype(int)
    interval = np.repeat(np.arange(intervals.size), steps)
    step = np.arange(interval.size) - np.repeat(np.cumsum(steps) - steps, steps)
    evaluated = np.append(
        times[interval] + intervals[interval] * step / steps[interval], times[-1]
    )
    return evaluated, PchipInterpolator(times, values)(evaluated)


def _rate_curve_color(color):
    """The color of the curve through a series: the series color with
    ``RATE_CURVE_BLEND_TOWARD_WHITE`` of white blended into it."""
    red_green_blue = np.asarray(mcolors.to_rgb(color))
    return tuple(
        red_green_blue + RATE_CURVE_BLEND_TOWARD_WHITE * (1.0 - red_green_blue)
    )


def series_legend_handles(series):
    """The legend entry of each series of ``series``, given in the order drawn, listed in
    the reverse of that order, so that A, drawn over B, comes first. A series with no
    finite value in the window reads ``(n/a)``."""
    handles = []
    for label, _times, rates, color in reversed(series):
        not_available = "" if len(rates) and np.isfinite(rates).any() else " (n/a)"
        handles.append(
            Line2D(
                [0],
                [0],
                color="none",
                marker="o",
                markersize=5,
                markerfacecolor=color,
                markeredgecolor="none",
                label=label + not_available,
            )
        )
    return handles


def draw_normalized_rate_quaternion_gyro_chart(
    output_path,
    *,
    series,
    window_start_s,
    window_end_s,
    video_frame_rate,
    subtitle_lines,
    quaternion_rate_hz,
    measurement_rate_hz,
    title,
    px_per_sample,
    canvas_width_px,
    canvas_height_px,
    rate_axis_scale,
    named_time_span_sets,
    peak_callout_series,
    extra_data_box_lines,
    local_median_levels,
    right_hand_ratio,
    right_hand_ratio_thresholds,
    right_hand_ratio_y_max,
    data_box_note,
):
    """Draw the Normalized Rate Quaternion Gyro Chart of the window [``window_start_s``,
    ``window_end_s``] seconds to ``output_path`` and return the lines that ``--verbose``
    prints about it.

    ``series`` is a list of ``(label, rate sample times in seconds, array (N, 3) in
    degrees per second, color)``, drawn in that order, each rate sample one marker at
    its own time, never resampled, over the curve of ``rate_sample_curve`` through the
    markers of its series. A series with no finite value in the window reads ``(n/a)``
    in the legend. ``video_frame_rate`` drives only the video frame axis, the video
    frame boundaries, and the title's window.

    ``rate_axis_scale`` is None for one rate axis at ``HEADROOM`` times
    ``_largest_absolute_rate_over_three_axes`` over every series in the window,
    ``"peraxis"`` for each panel at ``HEADROOM`` times ``_largest_absolute_rate`` of its
    own axis, or three half ranges for pitch, yaw, and roll. A rate sample beyond the
    rate axis leaves the panel, and the largest peak beyond it is called out above the
    panel, from the series named in ``peak_callout_series``.

    Everything laid over the series is passed in whole and computed nowhere here: the
    extents of ``named_time_span_sets``, drawn on the panel of their own axis in one
    opaque fill per classification with a boundary line at each end, and the identifier
    of each inside the extent one row below the panel name, over the middle of its part
    inside the window, in the color of its boundary lines; the local median and the
    levels of each ``LocalMedianLevels``; the ratio series of ``right_hand_ratio`` with
    ``right_hand_ratio_thresholds``, on a right hand axis from 0 to
    ``right_hand_ratio_y_max``; the data box lines of ``extra_data_box_lines``; and
    ``data_box_note``, where it is given, below the data box.

    The canvas is ``canvas_width_px`` by ``canvas_height_px`` at ``DPI``, whatever the
    window holds. Two invariants raise ``ValueError`` before anything is written: at least
    ``px_per_sample`` pixels of axes width per rate sample of the densest series, measured
    after layout, and at least ``TICK_LABELS_PER_PAGE_REQUIRED`` labeled major ticks per
    page."""
    resolved = []
    for label, times, rates, color in series:
        times = np.asarray(times, dtype=float)
        first, last = _window_indices(times, window_start_s, window_end_s)
        resolved.append(
            (label, times[first : last + 1], np.asarray(rates)[first : last + 1], color)
        )
    rate_sample_count = max(len(times) for _label, times, _rates, _color in resolved)
    callout_from = [
        series_entry
        for series_entry in resolved
        if series_entry[0] in peak_callout_series
    ]
    windowed = [rates for _label, _times, rates, _color in resolved]
    longest = max(len(rates) for rates in windowed)
    if rate_axis_scale is None:
        rate_axis_scale = (
            HEADROOM
            * _largest_absolute_rate_over_three_axes(
                windowed, 0, longest, TRANSIENT_OUTLIER_THRESHOLD
            ),
        ) * len(AXIS_COLUMNS)
    elif rate_axis_scale == "peraxis":
        rate_axis_scale = tuple(
            HEADROOM
            * max(
                1e-6,
                _largest_absolute_rate(
                    windowed, column, 0, longest, TRANSIENT_OUTLIER_THRESHOLD
                ),
            )
            for _name, column in AXIS_COLUMNS
        )

    video_frame_boundaries = (
        np.arange(
            int(np.ceil(window_start_s * video_frame_rate)),
            int(np.floor(window_end_s * video_frame_rate)) + 1,
        )
        / video_frame_rate
    )
    span_sets = _resolve_time_span_sets(named_time_span_sets)
    # 72 points per inch: a marker this wide is px_per_sample pixels, never wider than
    # the spacing of two rate samples of the densest series.
    marker_points = px_per_sample * 72.0 / DPI

    figure_width_inches = canvas_width_px / DPI
    figure_height_inches = canvas_height_px / DPI
    figure, panels = plt.subplots(
        3, 1, sharex=True, figsize=(figure_width_inches, figure_height_inches)
    )
    for local_median_levels_of_axis in local_median_levels:
        draw_local_median_levels(
            panels[AXIS_ORDER.index(local_median_levels_of_axis.axis)],
            local_median_levels_of_axis,
        )
    callouts_drawn = 0
    pending_labels = {panel_index: [] for panel_index, _axis in enumerate(AXIS_COLUMNS)}
    identifiers = {name: [] for name, _column in AXIS_COLUMNS}
    for panel_index, (name, column) in enumerate(AXIS_COLUMNS):
        panel = panels[panel_index]
        for _label, fill, _boundary_line_color, extents_by_axis in span_sets:
            for start_s, end_s, _identifier in extents_by_axis[name]:
                panel.axvspan(start_s, end_s, color=fill, lw=0, zorder=0)
        for _label, _fill, boundary_line_color, extents_by_axis in span_sets:
            for start_s, end_s, identifier in extents_by_axis[name]:
                panel.axvline(
                    start_s,
                    color=boundary_line_color,
                    lw=EXTENT_BOUNDARY_LINE_WIDTH,
                    zorder=1,
                )
                panel.axvline(
                    end_s,
                    color=boundary_line_color,
                    lw=EXTENT_BOUNDARY_LINE_WIDTH,
                    zorder=1,
                )
                # Inside the extent, so it never covers the video frame axis above
                # the panels or the top edge of the panel.
                identifiers[name].append(
                    panel.annotate(
                        identifier,
                        xy=(
                            0.5
                            * (max(start_s, window_start_s) + min(end_s, window_end_s)),
                            REPEATED_PANEL_NAME_Y,
                        ),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, -EXTENT_IDENTIFIER_BELOW_PANEL_NAME_POINTS),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=EXTENT_IDENTIFIER_FONTSIZE,
                        fontweight="bold",
                        color=mcolors.to_hex(boundary_line_color),
                        bbox={
                            "boxstyle": f"round,pad={EXTENT_IDENTIFIER_BOX_PAD}",
                            "fc": "white",
                            "ec": BOX_EDGE_COLOR,
                            "lw": 0.8,
                        },
                        zorder=7,
                    )
                )
        for boundary in video_frame_boundaries:
            panel.axvline(
                boundary, color=VIDEO_FRAME_BOUNDARY_COLOR, lw=0.5, alpha=0.55, zorder=0
            )
        for order, (_label, times, rates, color) in enumerate(resolved):
            panel.plot(
                times,
                rates[:, column],
                linestyle="none",
                marker="o",
                markersize=marker_points,
                markerfacecolor=color,
                markeredgecolor="none",
                zorder=3 + order,
            )
        panel.set_ylim(-rate_axis_scale[column], rate_axis_scale[column])
        panel.set_ylabel(f"{name} (deg/s)")
        panel.grid(True, axis="y", alpha=0.25)
        peaks = _peaks_beyond_rate_axis_at_rate_sample_times(
            [times for _label, times, _rates, _color in callout_from],
            [rates[:, column] for _label, _times, rates, _color in callout_from],
            rate_axis_scale[column],
        )
        callouts_drawn += len(peaks)
        if peaks:
            # One label per panel, at the largest peak, with the count of the runs that
            # left the panel.
            peak_time, peak = max(
                peaks, key=lambda time_and_peak: abs(time_and_peak[1])
            )
            panel.axvline(
                peak_time, color=BOX_EDGE_COLOR, ls="--", lw=0.9, alpha=0.6, zorder=6
            )
            arrow = "^" if peak > 0 else "v"
            of_many = (
                f", largest of {len(peaks)} runs beyond the rate axis"
                if len(peaks) > 1
                else ""
            )
            pending_labels[panel_index].append(
                (peak_time, f"{arrow} {abs(peak):.1f} deg/s{of_many}", "#222222")
            )

    panels[2].set_xlabel("time (s)")
    panels[2].set_xlim(window_start_s, window_end_s)
    video_frame_axis = panels[0].secondary_xaxis(
        "top",
        functions=(
            lambda seconds: seconds * video_frame_rate,
            lambda video_frame: video_frame / video_frame_rate,
        ),
    )
    video_frame_axis.set_xlabel("")
    # The video frame at the right end of the axis has the most digits.
    video_frame_axis_name = _reserve_video_frame_axis_name(
        panels[0], f"{round(window_end_s * video_frame_rate)}"
    )

    # The video frame boundaries, the curves through the rate samples, and the locator of
    # a peak beyond the rate axis take no legend entry.
    handles = series_legend_handles(resolved)
    for index, (label, fill, boundary_line_color, extents_by_axis) in enumerate(
        span_sets
    ):
        if any(extents_by_axis[name] for name, _column in AXIS_COLUMNS):
            patch = Patch(
                facecolor=fill,
                edgecolor=boundary_line_color,
                lw=EXTENT_BOUNDARY_LINE_WIDTH,
                label=label,
            )
            patch.legend_rank = LEGEND_RANK_EXTENTS + index
            handles.append(patch)
    # Every handle is appended before the header block is measured, so the legend drawn
    # is the legend measured.
    handles += local_median_legend_handles(local_median_levels)
    handles += right_hand_ratio_legend_handles(
        right_hand_ratio, right_hand_ratio_thresholds
    )

    box_lines = _data_box_lines(
        quaternion_rate_hz=quaternion_rate_hz,
        measurement_rate_hz=measurement_rate_hz,
        video_frame_rate=video_frame_rate,
        extra_lines=extra_data_box_lines,
    )
    window_video_frame_range = _video_frame_range_text(
        round(window_start_s * video_frame_rate),
        round(window_end_s * video_frame_rate) - 1,
    )

    # The header block, which holds the legend, the data box, the title, and the
    # subtitles, repeats across a canvas wider than a page. It is measured before the
    # header rectangle is solved, since its width decides whether the boxes can move up
    # beside the title.
    data_box = _data_box_text(box_lines)
    natural_width, block_component_widths = _header_block_width_px(
        handles, title, subtitle_lines, data_box, DPI
    )
    note_lines = 0 if data_box_note is None else 1
    needed_box_lines = max(
        len(box_lines) + note_lines, math.ceil(len(handles) / LEGEND_COLUMNS)
    )
    reserved_box_lines = max(HEADER_BOX_LINES_RESERVED, needed_box_lines)
    lefts, block_width, pitch = _header_block_positions(
        canvas_width_px, natural_width + 2 * HEADER_BLOCK_INSET_PX
    )
    raised_box_lines = _raised_box_lines(
        reserved_box_lines, block_width, block_component_widths
    )
    panel_rectangle, subtitle_y, boxes_top_y = _panel_rectangle_and_header_heights(
        figure_height_inches, reserved_box_lines, raised_box_lines
    )

    # The panels are laid out before the header is placed, since the tick steps are
    # chosen from the axes width the layout realizes.
    required_px = rate_sample_count * px_per_sample
    realized_px, margin_px = _lay_out_and_measure(
        figure, panels[2], DPI, panel_rectangle
    )
    # After the layout, so that the right hand axes take no height from the panels.
    right_hand_axes = attach_right_hand_ratio_axes(
        panels, right_hand_ratio, right_hand_ratio_thresholds, right_hand_ratio_y_max
    )
    if right_hand_axes:
        figure.subplots_adjust(
            right=_calculate_panel_right_edge(figure, right_hand_axes)
        )
        realized_px, margin_px = _axes_width_and_margins_px(figure, panels[2], DPI)
    if realized_px < required_px:
        plt.close(figure)
        raise ValueError(
            f"chart legibility invariant not met for {output_path}: the "
            f"{rate_sample_count} rate samples in the window need {required_px:.1f} px "
            f"of axes width at {px_per_sample} px per rate sample, and the "
            f"{canvas_width_px:.0f} px canvas realizes {realized_px:.1f} px "
            f"({realized_px / rate_sample_count:.3f} px per rate sample); a canvas "
            f"{required_px + margin_px:.0f} px wide would hold them"
        )

    # After the layout, since the density of each curve follows the realized axes width.
    px_per_second = realized_px / (window_end_s - window_start_s)
    curve_width_points = RATE_CURVE_WIDTH_PX * 72.0 / DPI
    for panel, (_name, column) in zip(panels, AXIS_COLUMNS, strict=True):
        for _label, times, rates, color in resolved:
            panel.plot(
                *rate_sample_curve(times, rates[:, column], px_per_second),
                color=_rate_curve_color(color),
                lw=curve_width_points,
                antialiased=True,
                zorder=RATE_CURVE_ZORDER,
            )

    _draw_labels_above_panels(
        panels, pending_labels, figure_width_inches, figure_height_inches
    )
    if right_hand_ratio:
        write_right_hand_ratio_scale(
            right_hand_axes,
            right_hand_ratio,
            float(min(ratio_series.times[0] for ratio_series in right_hand_ratio)),
            right_hand_ratio_y_max,
        )
    threshold_label_boxes = draw_threshold_labels(
        panels,
        right_hand_axes,
        local_median_levels,
        right_hand_ratio_thresholds,
        rate_axis_scale,
        right_hand_ratio_y_max,
        figure_width_inches,
        figure_height_inches,
    )
    _lower_extent_identifiers(
        panels,
        identifiers,
        threshold_label_boxes,
        figure_width_inches,
        figure_height_inches,
    )
    ticks = _apply_x_tick_intervals(
        panels[2],
        video_frame_axis,
        realized_px,
        window_start_s,
        window_end_s,
        video_frame_rate,
    )
    labels_per_page = min(
        ticks["seconds_axis_tick_labels_per_page"],
        ticks["video_frame_axis_tick_labels_per_page"],
    )
    if labels_per_page < TICK_LABELS_PER_PAGE_REQUIRED:
        plt.close(figure)
        raise ValueError(
            f"chart tick invariant not met for {output_path}: the tick steps "
            f"{ticks['seconds_axis_tick_step_s']:g} s and "
            f"{ticks['video_frame_axis_tick_step_frames']:g} video frames put "
            f"{labels_per_page:.2f} labeled major ticks in a {PAGE_WIDTH_PX:.0f} px "
            f"page, below the required {TICK_LABELS_PER_PAGE_REQUIRED:.0f}"
        )
    _place_video_frame_axis_name(
        figure, panels[0], video_frame_axis, video_frame_axis_name
    )

    for index, left_px in enumerate(lefts):
        left_px, right_px = (
            left_px + HEADER_BLOCK_INSET_PX,
            left_px + block_width - HEADER_BLOCK_INSET_PX,
        )
        if index:
            # The first block sits over the figure margin, where the real scales are.
            x_data = _data_x_at_px(
                panels[2], canvas_width_px, left_px, window_start_s, window_end_s
            )
            _repeat_y_scale(panels, x_data, rate_axis_scale)
            write_right_hand_ratio_scale(
                right_hand_axes, right_hand_ratio, x_data, right_hand_ratio_y_max
            )
        legend = figure.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(left_px / canvas_width_px, boxes_top_y),
            fontsize=9,
            frameon=True,
            fancybox=True,
            ncol=LEGEND_COLUMNS,
            edgecolor=BOX_EDGE_COLOR,
            facecolor="white",
            framealpha=1.0,
            borderpad=BOX_PAD_FONT_FRACTION,
        )
        legend.get_frame().set_boxstyle(BOX_STYLE)
        # The data box text drops by its own pad, so the frames of the two boxes line up.
        pad_points = BOX_PAD_FONT_FRACTION * DATA_BOX_FONTSIZE
        pad_fraction = (pad_points / 72.0) / (canvas_height_px / DPI)
        box_text = figure.text(
            right_px / canvas_width_px,
            boxes_top_y - pad_fraction,
            data_box,
            ha="right",
            va="top",
            fontsize=DATA_BOX_FONTSIZE,
            linespacing=1.35,
            family="DejaVu Sans Mono",
            bbox={
                "boxstyle": BOX_STYLE,
                "fc": "white",
                "ec": BOX_EDGE_COLOR,
                "lw": 0.8,
            },
        )
        if data_box_note is not None:
            # From the lower right corner of the text, out by the pad to the corner of
            # the frame.
            figure.add_artist(
                Annotation(
                    data_box_note,
                    xy=(1.0, 0.0),
                    xycoords=box_text,
                    xytext=(pad_points, -(pad_points + DATA_BOX_NOTE_GAP_POINTS)),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    fontsize=DATA_BOX_FONTSIZE,
                    family="DejaVu Sans Mono",
                )
            )
        center = 0.5 * (left_px + right_px) / canvas_width_px
        figure.text(
            center,
            0.988,
            f"{title}    {window_video_frame_range}",
            ha="center",
            va="top",
            fontsize=HEADER_TITLE_FONTSIZE,
        )
        for line_index, line in enumerate(subtitle_lines):
            figure.text(
                center,
                subtitle_y
                - line_index * HEADER_SUBTITLE_BAND_INCHES / (canvas_height_px / DPI),
                line,
                ha="center",
                va="top",
                fontsize=HEADER_SUBTITLE_FONTSIZE,
                color="#444444",
            )

    panel_share = sum(panel.get_position().height for panel in panels)
    panel_positions = [panel.get_position() for panel in panels]
    gaps = (panel_positions[0].y0 - panel_positions[1].y1) + (
        panel_positions[1].y0 - panel_positions[2].y1
    )
    if len(set(rate_axis_scale)) == 1:
        rate_axis = f"rate axis +/-{rate_axis_scale[0]:.1f} deg/s"
    else:
        rate_axis = "rate axis " + ", ".join(
            f"{name} +/-{rate_axis_scale[column]:.1f}" for name, column in AXIS_COLUMNS
        )
        rate_axis += " deg/s"
    summary = [
        (
            f"  canvas height {canvas_height_px:.0f} px: header band "
            f"{(1.0 - panel_positions[0].y1) * canvas_height_px:.0f}, panels "
            f"{panel_share * canvas_height_px:.0f}, gaps between panels "
            f"{gaps * canvas_height_px:.0f}, time axis and its labels "
            f"{panel_positions[2].y0 * canvas_height_px:.0f}"
        ),
        (
            f"  header box reserved {reserved_box_lines} line(s) for "
            f"{needed_box_lines} needed, {raised_box_lines} of them taken beside the "
            f"title rather than out of the panels"
        ),
        (
            f"{Path(output_path).name}: panels are {panel_share:.0%} of the canvas "
            f"height, {realized_px / rate_sample_count:.3f} px per rate sample "
            f"({rate_sample_count} rate samples, axes {realized_px:.0f} px wide, "
            f"canvas "
            f"{canvas_width_px:.0f} x {canvas_height_px:.0f} px at dpi {DPI}), "
            f"{canvas_width_px / PAGE_WIDTH_PX:.2f} pages of {PAGE_WIDTH_PX:.0f} px, "
            f"{len(lefts)} header block(s) {block_width:.0f} px wide every "
            f"{pitch:.0f} px, major ticks every "
            f"{ticks['seconds_axis_tick_step_s']:g} s / "
            f"{ticks['video_frame_axis_tick_step_frames']:g} video frames "
            f"({labels_per_page:.2f} labels per page), {rate_axis}, "
            f"{callouts_drawn} run(s) beyond the rate axis"
        ),
    ]
    with errors.operating_system_error_as(errors.CannotWriteError, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=DPI)
    plt.close(figure)
    return summary


# -------------------------------------------------------------------------
# What the chart draws, from the results in the cache
# -------------------------------------------------------------------------


def series_record(arrays, series, stage):
    """The record of one drawn series, with the video and the derivation version read
    from the header of the result that holds its arrays."""
    return SeriesRecord(
        series=series,
        video_name=str(arrays["video_name"].item()),
        derivation_version=int(arrays["derivation_version"].item()),
        stage=stage,
    )


def normalized_rates(arrays):
    """``(rate sample times in seconds, the Normalized Rate in degrees per second)`` as
    the result ``arrays`` stores them, read and never recomputed, so the chart draws the
    array the detector read."""
    return (
        np.asarray(arrays["recorded_rate_times_s"], dtype=np.float64),
        np.asarray(arrays["normalized_rate_deg_s"], dtype=np.float64),
    )


def classified_extent_span_sets(detection, source, window_start_s, window_end_s):
    """The extents of the detection result that overlap the window [``window_start_s``,
    ``window_end_s``], one ``ShadedSpanSet`` per classification the detection result
    holds, keyed by axis, each with the identifier the detection result stores for it.

    A classification with no fill raises ``errors.CannotReadError``, to which the caller
    adds the path of the detection result as ``file``, since an extent left out of the
    chart reads as an extent never found."""
    axis_of_extent = np.asarray(detection["extent_axis"]).astype(str)
    classification_of_extent = np.asarray(detection["extent_classification"]).astype(
        str
    )
    start_s = np.asarray(detection["extent_start_s"], dtype=np.float64)
    end_s = np.asarray(detection["extent_end_s"], dtype=np.float64)
    present = set(classification_of_extent.tolist())
    unknown = sorted(present - set(EXTENT_CLASSIFICATION_FILLS))
    if unknown:
        raise errors.CannotReadError(
            errors.ErrorCode.CLASSIFICATION_UNKNOWN,
            classifications=",".join(unknown),
        )
    identifiers = np.asarray(detection["extent_identifier"]).astype(str)
    in_window = (end_s > window_start_s) & (start_s < window_end_s)
    span_sets = []
    for classification, fill in EXTENT_CLASSIFICATION_FILLS.items():
        if classification not in present:
            continue
        spans = {}
        for axis in AXIS_ORDER:
            drawn = (
                in_window
                & (axis_of_extent == axis)
                & (classification_of_extent == classification)
            )
            spans[axis] = list(
                zip(
                    start_s[drawn].tolist(),
                    end_s[drawn].tolist(),
                    identifiers[drawn].tolist(),
                    strict=True,
                )
            )
        span_sets.append(
            ShadedSpanSet(classification, fill, spans, source, DRAWN_SERIES)
        )
    return span_sets


def ratio_axis_drawn(show_noise_ratio, span_sets):
    """Whether the chart draws the ratio axis, its series and the oscillation threshold
    and the burst threshold, on all three panels: with ``--show-noise-ratio``, or where
    ``span_sets``, which hold only the extents that overlap the window, hold a large
    amplitude oscillation or a mid-amplitude burst on any axis, the two classifications
    whose passes test the ratio."""
    return show_noise_ratio or any(
        span_set.label
        in (
            detection_stage.LARGE_AMPLITUDE_OSCILLATION,
            detection_stage.MID_AMPLITUDE_BURST,
        )
        and any(span_set.spans.values())
        for span_set in span_sets
    )


def detection_threshold_levels(detection, axis):
    """``(the levels drawn on the rate axis of the panel of axis, the Robust Standard
    Deviation of the impulse pass)``: the estimated peak height at the burst threshold,
    dashed, and the impulse threshold, both as multiples of that Robust Standard
    Deviation.

    The burst pass fires where the local Robust Standard Deviation, 1.4826 times the
    median of the absolute residual, exceeds the burst threshold times the Robust Standard
    Deviation of the burst pass. Divided by 1.4826, that product is the level that more
    than half the rate samples lie outside when the test is just met, and times
    ``LAPLACE_TAIL_FACTOR`` it is the level the peaks reach. It is derived from the
    threshold, never from what was detected, so it is drawn where nothing fired."""
    column = detection_stage.axis_index(detection, axis)
    pass_robust_standard_deviation = detection["pass_robust_standard_deviation_deg_s"][
        column
    ]
    pass_threshold = detection["pass_threshold"][column]
    impulse_pass_robust_standard_deviation = float(pass_robust_standard_deviation[2])
    impulse_threshold = float(pass_threshold[2])
    half_outside_deg_s = (
        float(pass_threshold[1])
        * float(pass_robust_standard_deviation[1])
        / robust_local_fit.MAD_TO_SIGMA
    )
    levels = [
        LocalMedianLevel(
            multiple=(
                LAPLACE_TAIL_FACTOR
                * half_outside_deg_s
                / impulse_pass_robust_standard_deviation
            ),
            label="estimated peak height at the burst threshold",
            color=PASS_THRESHOLD_COLORS[2],
            legend_rank=33,
            legend_label="estimated peak height at the burst threshold",
            linestyle="--",
        ),
        LocalMedianLevel(
            multiple=impulse_threshold,
            label=f"impulse threshold: {impulse_threshold:g}x RSD of the impulse pass",
            color=PASS_THRESHOLD_COLORS[3],
            legend_rank=LEGEND_RANK_THRESHOLDS + 3,
            legend_label="impulse threshold",
        ),
    ]
    return levels, impulse_pass_robust_standard_deviation


def ratio_thresholds(detection, axis):
    """The lines on the right hand axis of the panel of ``axis``: the oscillation
    threshold and the burst threshold, each at its threshold times the Robust Standard
    Deviation of its pass divided by the full track Robust Standard Deviation, the divisor
    of the drawn ratio. The label states the rule."""
    column = detection_stage.axis_index(detection, axis)
    full_track_robust_standard_deviation = float(
        detection["full_track_robust_standard_deviation_deg_s"][column]
    )
    lines = []
    for pass_index, (number, legend_label, rule) in enumerate(
        (
            (1, "oscillation threshold", "full track RSD"),
            (2, "burst threshold", "RSD of the burst pass"),
        )
    ):
        threshold = float(detection["pass_threshold"][column][pass_index])
        pass_robust_standard_deviation = float(
            detection["pass_robust_standard_deviation_deg_s"][column][pass_index]
        )
        lines.append(
            RatioThreshold(
                height=(
                    threshold
                    * pass_robust_standard_deviation
                    / full_track_robust_standard_deviation
                ),
                label=f"{legend_label}: {threshold:g}x {rule}",
                color=PASS_THRESHOLD_COLORS[number],
                legend_rank=LEGEND_RANK_THRESHOLDS + number,
                legend_label=legend_label,
            )
        )
    return lines


def local_median_levels_from_detection_result(detection, window_start_s, window_end_s):
    """The local median and the levels of each axis the detection result holds, sliced to
    the window [``window_start_s``, ``window_end_s``]."""
    times = np.asarray(detection["rate_sample_times_s"], dtype=np.float64)
    window = np.flatnonzero((times >= window_start_s) & (times <= window_end_s))
    local_median_levels = []
    for axis in (str(axis) for axis in detection["axes"]):
        column = detection_stage.axis_index(detection, axis)
        levels, impulse_pass_robust_standard_deviation = detection_threshold_levels(
            detection, axis
        )
        local_median_levels.append(
            LocalMedianLevels(
                axis=axis,
                times=times[window],
                local_median=np.asarray(detection["local_median_deg_s"])[
                    window, column
                ],
                robust_standard_deviation_deg_s=np.full(
                    window.size, impulse_pass_robust_standard_deviation
                ),
                levels=tuple(levels),
            )
        )
    return local_median_levels


def right_hand_ratios_from_detection_result(detection, window_start_s, window_end_s):
    """The local Robust Standard Deviation over the full track Robust Standard Deviation
    of each axis the detection result holds, sliced to the window [``window_start_s``,
    ``window_end_s``]."""
    times = np.asarray(detection["rate_sample_times_s"], dtype=np.float64)
    window = np.flatnonzero((times >= window_start_s) & (times <= window_end_s))
    ratios = []
    for axis in (str(axis) for axis in detection["axes"]):
        column = detection_stage.axis_index(detection, axis)
        local_robust_standard_deviation = np.asarray(
            detection["local_robust_standard_deviation_deg_s"]
        )[window, column]
        full_track_robust_standard_deviation = float(
            detection["full_track_robust_standard_deviation_deg_s"][column]
        )
        ratios.append(
            LocalRobustStandardDeviationRatio(
                axis=axis,
                times=times[window],
                ratio=(
                    local_robust_standard_deviation
                    / full_track_robust_standard_deviation
                ),
            )
        )
    return ratios


def local_median_description(detection):
    """The subtitle line that states what the local median is and its window. One local
    median serves all three passes, so the line names no pass."""
    settings = json.loads(str(detection["settings"]))
    window_ms = float(settings[str(detection["axes"][0])]["local_median_window"])
    return (
        f"local median: {detection_stage.LOCAL_MEDIAN_DESCRIPTION}, local median window "
        f"+/-{window_ms:g} ms"
    )


def robust_standard_deviation_lines(detection):
    """The data box lines of the four Robust Standard Deviations of the detection result,
    one line each with the three axes, and ``n/a`` for an axis it does not hold."""
    held = [str(axis) for axis in detection["axes"]]
    pass_robust_standard_deviation = np.asarray(
        detection["pass_robust_standard_deviation_deg_s"]
    )
    lines = []
    for label, values in (
        (
            "Large Oscillation Pass RSD",
            detection["full_track_robust_standard_deviation_deg_s"],
        ),
        ("Mid Burst Pass RSD", pass_robust_standard_deviation[:, 1]),
        ("Impulse Pass RSD", pass_robust_standard_deviation[:, 2]),
        (
            "Clean Robust Standard Deviation",
            detection["clean_robust_standard_deviation_deg_s"],
        ),
    ):
        lines.append(
            (
                label,
                " | ".join(
                    f"{axis} {float(values[held.index(axis)]):.3f}"
                    if axis in held
                    else f"{axis} n/a"
                    for axis in AXIS_ORDER
                ),
            )
        )
    return tuple(lines)


def data_box_note(detection):
    """The note below the data box: ``RSD_NOTE`` where a detection result is drawn, since
    only the data box lines and the labels drawn from a detection result name an RSD;
    None without one."""
    return None if detection is None else RSD_NOTE


def verbose_lines(
    video,
    video_frame_rate,
    first_video_frame,
    last_video_frame,
    window_start_s,
    window_end_s,
    series,
    span_sets,
    records,
):
    """The lines ``--verbose`` prints about the video, the window, each series drawn, and
    each set of extents."""
    lines = [f"=== {video.name} ==="]
    for record in records:
        lines.append(
            f"  {record.stage:<36} {decode_result_reader.SERIES_LABEL[record.series]}, "
            f"{record.video_name}, derivation version {record.derivation_version}"
        )
    lines.append(f"  {'video frame rate':<36} {video_frame_rate:.4f} fps")
    lines.append(
        f"  {'window':<36} video frames {first_video_frame} to {last_video_frame} "
        f"({last_video_frame - first_video_frame + 1} video frames), "
        f"{window_start_s:.4f} s to {window_end_s:.4f} s"
    )
    for label, times, _rates, _color in series:
        rate_samples_drawn = int(
            ((times >= window_start_s) & (times <= window_end_s)).sum()
        )
        rate = decode_stage.counted_rate_hz(times)
        lines.append(
            f"  {label:<36} {rate_samples_drawn:5d} rate samples drawn, {rate:.1f} Hz, "
            f"{rate / video_frame_rate:.2f} per video frame"
        )
    for span_set in span_sets:
        lines.append(f"  extents '{span_set.label}'")
        lines.append(f"    {'source':<34} {span_set.source}")
        lines.append(
            f"    {'computed on':<34} {decode_result_reader.SERIES_LABEL[span_set.series]}"
        )
        shaded_extents = [
            (axis, start, end, identifier)
            for axis in AXIS_ORDER
            for start, end, identifier in span_set.spans[axis]
        ]
        for axis, start, end, identifier in shaded_extents:
            lines.append(
                f"    {'shaded':<34} {identifier} {axis} {start:.4f} s to {end:.4f} s"
            )
        if not shaded_extents:
            lines.append(f"    {'shaded':<34} none in this window")
    return lines


def titled(name, prefix):
    """The chart's name with the words of ``--title-prefix`` before it."""
    prefix = (prefix or "").strip()
    return f"{prefix} {name}" if prefix else name


# -------------------------------------------------------------------------
# The cache
# -------------------------------------------------------------------------


# The codes of a cached result that cannot be opened or is damaged, which is an error of
# the command, where a cached result that fails a test counts as absent.
READ_ERROR_CODES = (errors.ErrorCode.SYSTEM_ERROR, errors.ErrorCode.CACHE_FILE_INVALID)


def _read_decode_result(path, video, command):
    """The decode result at ``path``, read and tested by the cache reader of the decode
    stage. A missing decode result, one that fails a test, and a file that cannot be
    opened or is damaged raise ``errors.CannotReadError`` with ``command``, the
    ``gyro-doctor`` command its hint names."""
    if not path.exists():
        raise errors.CannotReadError(
            errors.ErrorCode.CACHED_RESULT_NOT_FOUND, file=path, command=command
        )
    try:
        return decode_result_reader.read_decode_result(path, video)
    except errors.CannotReadError as error:
        error.values.setdefault("command", command)
        raise


def detection_result_tested(path, video, video_cache_directory, command):
    """The detection result at ``path``, tested as ``gyro-doctor`` tests it, comparing no
    parameters. A missing detection result, one that fails a test, and a file that
    cannot be opened or is damaged raise ``errors.CannotReadError`` with ``command``, the
    ``gyro-doctor`` command its hint names."""
    if not path.exists():
        raise errors.CannotReadError(
            errors.ErrorCode.CACHED_RESULT_NOT_FOUND, file=path, command=command
        )
    try:
        present, members = stored_parts(
            path, detection_stage.HEADER_PARTS, ("file_stamps",)
        )
        decode_stage.header_test(members, video, path)
        decode_stage.every_part_present_test(
            present, path, (*detection_stage.DETECTION_RESULT_PARTS, "file_stamps")
        )
        decode_stage.file_stamps_test(
            path, video_cache_directory, members["file_stamps"], ("decode_result",)
        )
        return detection_stage.read_detection_result(path)
    except errors.CannotReadError as error:
        error.values.setdefault("command", command)
        raise


def _read_detection_result(path, video, video_cache_directory, command):
    """The detection result at ``path`` of ``detection_result_tested``; None when there is
    none or it fails a test, with the error logged at INFO level. A file that cannot be
    opened or is damaged raises its ``errors.CannotReadError``."""
    if not path.exists():
        return None
    try:
        return detection_result_tested(path, video, video_cache_directory, command)
    except errors.CannotReadError as error:
        if error.code in READ_ERROR_CODES:
            raise
        log.info("%s; the detection result is not used", error)
        return None


def _read_repair_result(path, video, video_cache_directory, command):
    """The repair result at ``path``, tested as ``gyro-doctor`` tests it, comparing no
    parameters; None when there is none or it fails a test, with the error logged at
    INFO level. Of the repair record it reads the file stamps alone, for the test. A file
    that cannot be opened or is damaged raises ``errors.CannotReadError`` with
    ``command``, the ``gyro-doctor`` command its hint names."""
    if not path.exists():
        return None
    try:
        return repair_result_tested(path, video, video_cache_directory)
    except errors.CannotReadError as error:
        if error.code in READ_ERROR_CODES:
            error.values.setdefault("command", command)
            raise
        log.info("%s; the repair result is not used", error)
        return None


def repair_result_tested(path, video, video_cache_directory):
    """The repair result at ``path``, after the header test, the test that every part is
    present, and the file stamp test of each file stamp of its repair record, comparing
    no parameters. Every error it raises is ``errors.CannotReadError``."""
    present, members = stored_parts(path, repair_stage.HEADER_PARTS, ("repair_record",))
    decode_stage.header_test(members, video, path)
    decode_stage.every_part_present_test(
        present, path, repair_stage.REPAIR_RESULT_PARTS
    )
    record = members["repair_record"]
    decode_stage.every_part_present_test(record, path, ("file_stamps",))
    decode_stage.file_stamps_test(
        path,
        video_cache_directory,
        record["file_stamps"],
        ("decode_result", "detection_result"),
    )
    return repair_stage.read_repair_result(path)


# -------------------------------------------------------------------------
# The command
# -------------------------------------------------------------------------


def draw_chart_from_cache(args):
    """Draw the chart that the parsed options ``args`` ask for, from the cache of the
    video, and return the lines ``--verbose`` prints."""
    video = args.video
    video_cache_directory = (
        args.cache_dir or Path.cwd() / ".gyro-doctor-cache"
    ) / video.stem
    cache_option = "" if args.cache_dir is None else f" --cache-dir {args.cache_dir}"
    decode_path = decode_stage.decode_result_path(video_cache_directory, video.stem)
    decode = _read_decode_result(
        decode_path, video, f"gyro-doctor --decode-only{cache_option} {video}"
    )
    detection_path = detection_stage.detection_result_path(
        video_cache_directory, video.stem
    )
    detect_command = f"gyro-doctor --detect-only{cache_option} {video}"
    detection = _read_detection_result(
        detection_path, video, video_cache_directory, detect_command
    )
    if args.show_noise_ratio and detection is None:
        raise errors.CannotReadError(
            errors.ErrorCode.SHOW_NOISE_RATIO_WITHOUT_DETECTION_RESULT,
            file=detection_path,
            command=detect_command,
        )
    repair = None
    repair_path = repair_stage.repair_result_path(video_cache_directory, video.stem)
    repair_command = f"gyro-doctor --repair-only{cache_option} {video}"
    if args.series is None:
        repair = _read_repair_result(
            repair_path, video, video_cache_directory, repair_command
        )

    video_frame_rate = float(decode["video_frame_rate"])
    # One telemetry frame for each video frame.
    video_frame_count = len(decode["quaternion_count_per_telemetry_frame"])
    window_video_frames, canvas_width_px, px_per_sample = view_settings(
        args.view, video_frame_rate
    )
    requested_video_frame = args.center.to_video_frame(video_frame_rate)
    drawn_center, (first_video_frame, last_video_frame) = clamped_center(
        requested_video_frame, window_video_frames, video_frame_count
    )
    window_start_s, window_end_s = (
        first_video_frame / video_frame_rate,
        (last_video_frame + 1) / video_frame_rate,
    )
    output = args.output
    if output is None:
        output = Path.cwd() / (
            f"{video.stem}-chart-f{drawn_center}-"
            f"{last_video_frame - first_video_frame + 1}f.png"
        )

    records = [series_record(decode, DRAWN_SERIES, AS_RECORDED_STAGE)]
    times, rates = normalized_rates(decode)
    period = float(decode["period_s"])
    drawn = [(SERIES_LABEL, times, rates, PALETTE[0])]
    if repair is not None:
        records.append(series_record(repair, DRAWN_SERIES, AS_REPAIRED_STAGE))
        repaired_times, repaired_rates = normalized_rates(repair)
        # The repaired series first and the original over it, so the repaired series
        # shows only where the two differ.
        drawn = [
            (
                B_SERIES_LABEL,
                repaired_times,
                repaired_rates,
                B_SERIES_COLOR,
            ),
            (A_SERIES_LABEL, times, rates, PALETTE[0]),
        ]

    span_sets, local_median_levels, detection_lines, subtitle_lines = (
        [],
        [],
        (),
        [video.stem],
    )
    if detection is not None:
        try:
            span_sets = classified_extent_span_sets(
                detection, detection_path.name, window_start_s, window_end_s
            )
        except errors.CannotReadError as error:
            error.values.setdefault("file", detection_path)
            raise
        local_median_levels = local_median_levels_from_detection_result(
            detection, window_start_s, window_end_s
        )
        detection_lines = robust_standard_deviation_lines(detection)
        subtitle_lines.append(local_median_description(detection))

    lines = verbose_lines(
        video,
        video_frame_rate,
        first_video_frame,
        last_video_frame,
        window_start_s,
        window_end_s,
        drawn,
        span_sets,
        records,
    )
    lines.append(f"  {'period T':<36} {period * 1e6:.4f} us")

    ratios, ratio_limit, ratio_lines = [], None, {}
    if ratio_axis_drawn(args.show_noise_ratio, span_sets):
        ratios = right_hand_ratios_from_detection_result(
            detection, window_start_s, window_end_s
        )
        ratio_lines = {
            axis: ratio_thresholds(detection, axis)
            for axis in (str(axis) for axis in detection["axes"])
        }
        # The right hand axis reaches the highest threshold line over the three panels.
        ratio_limit = max(
            line.height for axis_lines in ratio_lines.values() for line in axis_lines
        )
        lines.append(
            f"  {'ratio axis':<36} 0 to {ratio_limit:g}, shared by the three panels"
        )
        for axis, axis_lines in ratio_lines.items():
            for line in axis_lines:
                lines.append(f"  {axis:<6}{line.label:<70} at height {line.height:.3f}")

    lines += draw_normalized_rate_quaternion_gyro_chart(
        output,
        series=drawn,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        video_frame_rate=video_frame_rate,
        subtitle_lines=subtitle_lines,
        quaternion_rate_hz=float(decode["quaternion_rate_hz"]),
        measurement_rate_hz=float(decode["measurement_rate_hz"]),
        title=titled(AB_TITLE if repair is not None else TITLE, args.title_prefix),
        px_per_sample=px_per_sample,
        canvas_width_px=canvas_width_px,
        canvas_height_px=DEFAULT_CANVAS_HEIGHT_PX,
        rate_axis_scale=args.rate_axis_scale,
        named_time_span_sets=[span_set.chart_form for span_set in span_sets],
        # A peak beyond the rate axis is called out on the original series.
        peak_callout_series=[A_SERIES_LABEL if repair is not None else SERIES_LABEL],
        extra_data_box_lines=(
            (("period T", f"{period * 1e6:.4f} us"),)
            + window_lines(requested_video_frame, drawn_center)
            + detection_lines
        ),
        local_median_levels=local_median_levels,
        right_hand_ratio=ratios,
        right_hand_ratio_thresholds=ratio_lines,
        right_hand_ratio_y_max=ratio_limit,
        data_box_note=data_box_note(detection),
    )
    lines.append(f"  wrote {output}")
    return lines


def _rate_axis_scale(text):
    """``--rate-axis-scale``: ``peraxis``, or the half range of each panel in degrees per
    second, from one value for the three panels or three values for pitch, yaw, and roll
    in that order as one comma separated token, each a finite number greater than 0. Any
    other text raises ``argparse.ArgumentTypeError``."""
    if text == "peraxis":
        return text
    try:
        values = tuple(float(part) for part in text.split(","))
    except ValueError:
        values = ()
    if len(values) not in (1, len(AXIS_ORDER)) or not all(
        math.isfinite(value) and value > 0 for value in values
    ):
        raise argparse.ArgumentTypeError(
            errors.message(errors.ErrorCode.VALUE_NOT_A_RATE_AXIS_SCALE, text=text)
        )
    return values * len(AXIS_ORDER) if len(values) == 1 else values


def parse_args(argv=None):
    """The options of ``gyro-doctor-chart`` parsed from ``argv``, which defaults to
    ``sys.argv[1:]``. A mistake in the options, and the error of a video path that is
    not a file or of an ``--output`` whose extension is not ``.png`` in any letter case,
    go through argparse, which prints the usage and the error line to stderr and exits
    with status 2, before any cache is read. ``--help`` and ``--version`` go through
    argparse, which prints the help, or the name of the command and the version of the
    package, to stdout and exits with status 0."""
    parser = argparse.ArgumentParser(
        prog="gyro-doctor-chart",
        usage="%(prog)s [options] VIDEO",
        description=(
            "gyro-doctor-chart charts the rotational rate of the camera on each axis "
            "before and after the repair."
        ),
    )
    parser.add_argument(
        "video",
        type=Path,
        metavar="VIDEO",
        help="The video whose cached results the chart draws",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {importlib.metadata.version('gyro_doctor')}",
    )
    parser.add_argument(
        "--center",
        type=parse_position,
        required=True,
        metavar="POSITION",
        help=(
            "The middle of the drawn window: a time such as 75s, 1:15, or 1:02:15, or a "
            "video frame such as f4500. The window slides to fit at the ends of the "
            "video"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        help=(
            "The complete path of the PNG (default: the video stem, chart, the center "
            "video frame, and the duration of the window in video frames, in the "
            "current directory, such as DJI_0002-chart-f4500-24f.png)"
        ),
    )
    parser.add_argument(
        "--series",
        choices=("original",),
        help=(
            "original draws the original Normalized Rate alone (default: every series "
            "the cache holds)"
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
        "--view",
        choices=tuple(VIEW_SCALES),
        default=DEFAULT_VIEW,
        metavar="NAME",
        help=(
            "The duration of the window: normal, 24 video frames; wide, 120 video "
            f"frames; ultra-wide, 15 seconds (default: {DEFAULT_VIEW})"
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
            "rate axis at 1.3 times the largest absolute rate on its axis that is not a "
            "transient outlier (default: one rate axis at 1.3 times the largest "
            "absolute rate in the window, over the three axes and every series drawn, "
            "that is not a transient outlier)"
        ),
    )
    parser.add_argument(
        "--show-noise-ratio",
        action="store_true",
        help=(
            "Draw the local Robust Standard Deviation divided by the full track Robust "
            "Standard Deviation using a right hand axis for each panel, with lines at "
            "the oscillation threshold and the burst threshold, in every window "
            "(default: only in a window that contains a large amplitude oscillation or "
            "a mid-amplitude burst). Requires a detection result in the cache"
        ),
    )
    parser.add_argument(
        "--title-prefix", metavar="WORDS", help="Words placed before the title"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print what the chart drew"
    )
    args = parser.parse_args(argv)
    if not args.video.is_file():
        parser.error(
            str(
                errors.CannotReadError(
                    errors.ErrorCode.VIDEO_NOT_A_FILE, file=args.video
                )
            )
        )
    if args.output is not None and args.output.suffix.lower() != ".png":
        parser.error(
            str(
                errors.CannotWriteError(
                    errors.ErrorCode.OUTPUT_NOT_PNG, file=args.output
                )
            )
        )
    return args


def main(argv=None):
    """Run ``gyro-doctor-chart`` on the command line arguments ``argv``, which default to
    ``sys.argv[1:]``, and return the exit status: 0 when the chart is written, and 1 on
    an error of the cache or of what the chart draws, ``errors.GyroDoctorError``, logged
    through ``errors.log_error`` with its error line and any hint line at ERROR level
    through the logger ``gyro_doctor.tools.chart``.

    A mistake in the options exits with status 2 through argparse, and ``--help`` and
    ``--version`` print to stdout and exit with status 0 through argparse. Otherwise
    nothing is printed on stdout unless ``--verbose`` is given, which prints what the
    chart drew and logs at INFO level each cached result that was not used, with the error
    it raised."""
    logging.basicConfig(format="gyro-doctor-chart: %(message)s")
    args = parse_args(argv)
    log.setLevel(logging.INFO if args.verbose else logging.NOTSET)
    try:
        lines = draw_chart_from_cache(args)
    except errors.GyroDoctorError as error:
        errors.log_error(log, error)
        return 1
    if args.verbose:
        print("\n".join(lines))
    return 0
