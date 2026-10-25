# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The robust local linear regression that fits the local median of one axis: at every
rate sample, one weighted straight line over the rate samples within the local median
window on either side, evaluated at the rate sample's own time, with tricube weights by
distance and, in each further fit, Tukey's bisquare function of the difference from the
previous fit.

The window is bounded in time, not in rate samples, since the intervals between rate
samples are not equal, and the fit is evaluated at every rate sample with no grid and no
interpolation. Where the window runs off the end of the track it is one sided; a
straight line evaluated at the rate sample's own time stays unbiased there, where a
local mean would report the center of whatever rate samples it had.

The bisquare function measures each difference against ``BISQUARE_TUNING_CONSTANT``
times a local scale, 1.4826 times the median of the absolute differences over the same
window. A scale taken over the whole track would be inflated by the gyro noise events,
and the rate samples inside an event would never be downweighted.
"""

from dataclasses import dataclass

import numpy as np

# The factor that turns a median absolute deviation into a Robust Standard Deviation.
MAD_TO_SIGMA = 1.4826

# The customary tuning constant of Tukey's bisquare function, which gives 95 percent
# efficiency on Gaussian noise; not a tuned value.
BISQUARE_TUNING_CONSTANT = 4.685

# The most cells one gathered block holds. A width group of windows is split into blocks
# of at most CELL_BUDGET // width rows, so a wider window gives fewer rows per block,
# not a larger block.
CELL_BUDGET = 4_000_000


@dataclass(frozen=True)
class Fit:
    """One robust local regression of one axis.

    ``local_median`` is the fitted value at each rate sample's own time.
    ``windows_without_weight`` counts the windows of the last fit that held no weight
    and fell back to the rate sample's own value, and ``zero_scale`` the rate samples of
    the last reweighting whose own difference scale was zero, so that their own
    difference could not be judged and they kept full weight."""

    local_median: np.ndarray
    windows_without_weight: int
    zero_scale: int


def _tricube(scaled_offset):
    """Cleveland's tricube weight of ``scaled_offset``, an offset in time divided by the
    local median window; zero where its absolute value reaches 1."""
    absolute = np.abs(scaled_offset)
    weight = np.zeros_like(absolute)
    inside = absolute < 1.0
    weight[inside] = (1.0 - absolute[inside] ** 3) ** 3
    return weight


def _bisquare(scaled_difference):
    """Tukey's bisquare weight of ``scaled_difference``, a difference divided by
    ``BISQUARE_TUNING_CONSTANT`` times the local scale; zero where its absolute value
    reaches 1."""
    absolute = np.abs(scaled_difference)
    weight = np.zeros_like(absolute)
    inside = absolute < 1.0
    weight[inside] = (1.0 - absolute[inside] ** 2) ** 2
    return weight


def _window_groups(times, local_median_window_s):
    """Every rate sample's window, grouped by how many rate samples it holds.

    Windows bounded in time differ by a rate sample or two and cannot be one strided
    view, so each width is gathered and reduced at once, and each width's rows are
    chunked so a gathered block stays inside ``CELL_BUDGET`` however wide the window is.

    Yields ``(rows, window_start, width)``: the rate sample indices in this chunk, where
    each one's window starts, and how many rate samples it holds.
    """
    window_start = np.searchsorted(times, times - local_median_window_s, side="left")
    window_end = np.searchsorted(times, times + local_median_window_s, side="right")
    widths = window_end - window_start
    for width in np.unique(widths):
        width = int(width)
        rows_of_width = np.flatnonzero(widths == width)
        rows_per_block = max(1, CELL_BUDGET // max(width, 1))
        for start in range(0, rows_of_width.size, rows_per_block):
            rows = rows_of_width[start : start + rows_per_block]
            yield rows, window_start[rows], width


def _weighted_local_fit(rate, times, local_median_window_s, robustness_weights, groups):
    """One weighted local linear fit: the fitted value at every rate sample's own time,
    and how many windows held no weight and fell back to the rate sample's own value.

    The weighted normal equations are solved in closed form on offsets from the rate
    sample's own time, so the fitted value is the intercept and no matrix is built.
    """
    fitted = np.empty(rate.size)
    windows_without_weight = 0
    for rows, window_start, width in groups:
        window_indices = window_start[:, None] + np.arange(width)
        center_times = times[rows][:, None]
        offsets = times[window_indices] - center_times
        weights = (
            _tricube(offsets / local_median_window_s)
            * robustness_weights[window_indices]
        )
        window_rate = rate[window_indices]

        weight_sum = weights.sum(axis=1)
        weighted_rate_sum = (weights * window_rate).sum(axis=1)
        usable = weight_sum > 0
        # A window with no weight at all, where every neighbor was downweighted to zero,
        # keeps the rate sample's own value and is counted.
        value = np.where(
            usable, weighted_rate_sum / np.where(usable, weight_sum, 1.0), rate[rows]
        )
        windows_without_weight += int((~usable).sum())

        if width > 1:
            weighted_offset_sum = (weights * offsets).sum(axis=1)
            weighted_squared_offset_sum = (weights * offsets * offsets).sum(axis=1)
            weighted_offset_rate_sum = (weights * offsets * window_rate).sum(axis=1)
            determinant = (
                weight_sum * weighted_squared_offset_sum
                - weighted_offset_sum * weighted_offset_sum
            )
            # A window whose weighted offsets carry no spread cannot support a slope;
            # it keeps the weighted mean computed above.
            has_spread = usable & (determinant > 0)
            linear = (
                weighted_squared_offset_sum * weighted_rate_sum
                - weighted_offset_sum * weighted_offset_rate_sum
            ) / np.where(has_spread, determinant, 1.0)
            value = np.where(has_spread, linear, value)

        fitted[rows] = value
    return fitted, windows_without_weight


def _local_difference_scale(difference, groups):
    """1.4826 times the median of the absolute difference over the local median window
    of each rate sample."""
    scale = np.empty(difference.size)
    absolute = np.abs(difference)
    for rows, window_start, width in groups:
        window_indices = window_start[:, None] + np.arange(width)
        scale[rows] = MAD_TO_SIGMA * np.median(absolute[window_indices], axis=1)
    return scale


def fit(rate, times, local_median_window_ms, fits):
    """Fit ``rate`` sampled at ``times`` over ``local_median_window_ms`` on either side
    of each rate sample, and return a :class:`Fit`.

    ``fits`` is the number of fits. The first weights each rate sample by its distance
    alone; each further fit reweights every rate sample by Tukey's bisquare function of
    its difference from the previous fit, measured against ``BISQUARE_TUNING_CONSTANT``
    times the local difference scale, and fits again. Two fits are one fit by distance
    and one reweighted fit.
    """
    rate = np.asarray(rate, dtype=float)
    times = np.asarray(times, dtype=float)
    local_median_window_s = float(local_median_window_ms) * 1e-3
    groups = list(_window_groups(times, local_median_window_s))

    robustness_weights = np.ones(rate.size)
    zero_scale = 0
    local_median, windows_without_weight = _weighted_local_fit(
        rate, times, local_median_window_s, robustness_weights, groups
    )

    for _ in range(fits - 1):
        difference = rate - local_median
        scale = _local_difference_scale(difference, groups)
        # A rate sample whose scale is zero has nothing to judge its difference against,
        # so it keeps full weight.
        measurable = scale > 0
        zero_scale = int((~measurable).sum())
        scaled_difference = np.where(
            measurable,
            difference / np.where(measurable, BISQUARE_TUNING_CONSTANT * scale, 1.0),
            0.0,
        )
        robustness_weights = _bisquare(scaled_difference)
        local_median, windows_without_weight = _weighted_local_fit(
            rate, times, local_median_window_s, robustness_weights, groups
        )

    return Fit(
        local_median=local_median,
        windows_without_weight=windows_without_weight,
        zero_scale=zero_scale,
    )
