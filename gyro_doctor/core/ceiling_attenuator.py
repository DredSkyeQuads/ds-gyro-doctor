# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""The ceiling attenuator of the repair stage, which reduces each gyro noise event that
the detector found until the largest departure in the interior of its extent is at or
below the ceiling of its classification, and the repair parameters it takes.

The clean Robust Standard Deviation is taken outside every extent, so a gyro noise event
does not enlarge the clean Robust Standard Deviation against which its own ceiling is
measured. One gain applies across the interior of an extent, so a gain above 0 keeps the
shape, sign, and timing of the gyro noise event and reduces only its amplitude. The
first and last rate samples of an extent come back identical to their recorded values,
and no rate sample outside an extent is read or changed.
"""

from dataclasses import dataclass

import numpy as np

RAISED_COSINE = "raised-cosine"
LINEAR = "linear"
FADE_SHAPES = (RAISED_COSINE, LINEAR)

LOCAL_MEDIAN = "local-median"
CHORD = "chord"
ZERO = "zero"
CENTERLINES = (LOCAL_MEDIAN, CHORD, ZERO)

# The three classification strings of the detection result, verbatim: the repair stage
# keys the centerline and the ceiling of an extent on them.
LARGE_AMPLITUDE_OSCILLATION = "large amplitude oscillation"
MID_AMPLITUDE_BURST = "mid-amplitude burst"
IMPULSE_EVENT = "impulse event"
CLASSIFICATIONS = (LARGE_AMPLITUDE_OSCILLATION, MID_AMPLITUDE_BURST, IMPULSE_EVENT)

# The reasons the ceiling attenuator keeps an extent at its recorded values.
AT_A_VIDEO_EDGE = "at a video edge"
ON_THE_DO_NOT_REPAIR_LIST = "on the do not repair list"


@dataclass(frozen=True)
class RepairParameters:
    """The repair parameters: one field per repair parameter option other than
    ``--do-not-repair-list`` and ``--do-not-repair``, named after the option and holding
    its default, and ``do_not_repair``.

    Each ceiling is a multiple of the clean Robust Standard Deviation of the axis.
    ``repair_fade`` is a time in milliseconds.

    ``remove_gyro_motion`` holds the axes whose recorded gyro motion the repair stage
    removes. The axes stand in the order pitch, yaw, roll. The ceiling attenuator never
    runs on those axes.

    ``do_not_repair`` holds the sorted identifiers of the extents that the do not
    repair list excludes. The run finds those extents by resolving the list against the
    detection result. A cached repair result therefore stays valid for any list that
    excludes the same extents."""

    repair_oscillation_centerline: str = CHORD
    repair_burst_centerline: str = LOCAL_MEDIAN
    repair_impulse_centerline: str = LOCAL_MEDIAN
    repair_oscillation_ceiling: float = 0.0
    repair_burst_ceiling: float = 2.0
    repair_impulse_ceiling: float = 0.5
    repair_fade: float = 4.0
    repair_fade_shape: str = RAISED_COSINE
    repair_edges: bool = False
    remove_gyro_motion: tuple = ()
    do_not_repair: tuple = ()

    def centerline_of(self, classification) -> str:
        """The centerline toward which an extent of this classification is
        attenuated."""
        return {
            LARGE_AMPLITUDE_OSCILLATION: self.repair_oscillation_centerline,
            MID_AMPLITUDE_BURST: self.repair_burst_centerline,
            IMPULSE_EVENT: self.repair_impulse_centerline,
        }[str(classification)]

    def ceiling_of(self, classification) -> float:
        """The ceiling of an extent of this classification, in clean Robust Standard
        Deviations."""
        return float(
            {
                LARGE_AMPLITUDE_OSCILLATION: self.repair_oscillation_ceiling,
                MID_AMPLITUDE_BURST: self.repair_burst_ceiling,
                IMPULSE_EVENT: self.repair_impulse_ceiling,
            }[str(classification)]
        )


@dataclass
class CeilingAttenuationResult:
    repaired_rate_deg_s: np.ndarray
    # The centerline toward which each extent was attenuated: the local median handed
    # in, and inside every extent neither skipped at a video edge nor excluded by the do
    # not repair list, the centerline of the classification of that extent.
    centerline: np.ndarray
    gains: np.ndarray  # one per extent, in the order of extents
    # (k, 2) the inclusive first and last rate samples of the extents, sorted by first
    # rate sample.
    extents: np.ndarray
    extent_entries: list  # one dict per extent, in the order of extents
    statistics: dict


def departure_gain(peak_departure, clean_robust_standard_deviation, ceiling):
    """The gain that places `peak_departure` at `ceiling` clean Robust Standard
    Deviations, and `peak_departure` in clean Robust Standard Deviations.

    An extent already at or below the ceiling takes a gain of one. A clean Robust
    Standard Deviation that is not finite or not positive cannot express a ceiling, so
    that extent also takes a gain of one rather than an arbitrary attenuation, and its
    departure in clean Robust Standard Deviations is infinite, or 0 where
    `peak_departure` is 0.
    """
    if (
        not np.isfinite(clean_robust_standard_deviation)
        or clean_robust_standard_deviation <= 0
    ):
        return 1.0, np.inf if peak_departure > 0 else 0.0
    departure_in_clean_robust_standard_deviations = (
        peak_departure / clean_robust_standard_deviation
    )
    if departure_in_clean_robust_standard_deviations <= ceiling:
        return 1.0, departure_in_clean_robust_standard_deviations
    return float(ceiling / departure_in_clean_robust_standard_deviations), float(
        departure_in_clean_robust_standard_deviations
    )


def extent_centerline(choice, rate, local_median, times, start, end):
    """The centerline ``choice`` across the extent ``rate[start:end]``.

    ``local-median`` is the local median that the detector fitted. ``zero`` is a
    constant rate of 0 degrees per second. ``chord`` is the straight line in time from
    the rate at the first rate sample of the extent to the rate at its last rate sample:
    in time because the intervals between rate samples are not uniform, and through
    those two rate samples so that the chord rejoins the signal at both ends of the
    extent. Raises ValueError on a choice that is none of ``CENTERLINES``.
    """
    if choice == LOCAL_MEDIAN:
        return local_median[start:end]
    if choice == ZERO:
        return np.zeros(end - start)
    if choice != CHORD:
        raise ValueError(f"centerline must be one of {CENTERLINES}, not {choice!r}")
    span_times = times[start:end]
    first_value, last_value = float(rate[start]), float(rate[end - 1])
    width = float(span_times[-1] - span_times[0])
    if width <= 0:
        return np.full(end - start, 0.5 * (first_value + last_value))
    return (
        first_value + (last_value - first_value) * (span_times - span_times[0]) / width
    )


def extent_gain(rate, centerline, clean_robust_standard_deviation, start, end, ceiling):
    """The gain that places the largest departure of the extent ``rate[start:end]`` from
    ``centerline`` at ``ceiling`` clean Robust Standard Deviations.

    The largest departure anywhere in the extent, not the detector's ``peak_sample``.
    That rate sample is the rate sample of the largest departure from the local median,
    so it is the rate sample of the largest departure from the centerline only while the
    centerline is the local median. Under any other centerline a gain taken at that rate
    sample can leave another rate sample of the interior above the ceiling.

    ``clean_robust_standard_deviation`` holds the clean Robust Standard Deviation of the
    axis at every rate sample. Returns the gain, the largest departure in clean Robust
    Standard Deviations, and the largest departure and the clean Robust Standard
    Deviation at its rate sample.
    """
    departure = np.abs(rate[start:end] - centerline)
    clean_robust_standard_deviation_inside_extent = clean_robust_standard_deviation[
        start:end
    ]
    usable = np.isfinite(clean_robust_standard_deviation_inside_extent) & (
        clean_robust_standard_deviation_inside_extent > 0
    )
    if not usable.any():
        largest = float(departure.max())
        clean_robust_standard_deviation_at_largest_departure = float("nan")
    else:
        ratio = np.where(
            usable,
            departure
            / np.where(usable, clean_robust_standard_deviation_inside_extent, 1.0),
            -np.inf,
        )
        largest_departure_index = int(np.argmax(ratio))
        largest = float(departure[largest_departure_index])
        clean_robust_standard_deviation_at_largest_departure = float(
            clean_robust_standard_deviation_inside_extent[largest_departure_index]
        )
    gain, departure_in_clean_robust_standard_deviations = departure_gain(
        largest, clean_robust_standard_deviation_at_largest_departure, ceiling
    )
    return (
        gain,
        departure_in_clean_robust_standard_deviations,
        largest,
        clean_robust_standard_deviation_at_largest_departure,
    )


def gain_at_each_rate_sample(rate_sample_count, gain, fade_samples, fade_shape):
    """The gain at each of the ``rate_sample_count`` rate samples of an extent: ``gain``
    across the interior, a ramp between ``gain`` and one across each fade of
    ``fade_samples`` rate samples, and one at the first and last rate samples. Raises
    ValueError on a fade shape that is none of ``FADE_SHAPES``.

    The first and last rate samples are set to one explicitly, rather than left to the
    arithmetic of the ramp, so that both come back identical to their recorded values.
    """
    if fade_shape not in FADE_SHAPES:
        raise ValueError(f"fade_shape must be one of {FADE_SHAPES}, not {fade_shape!r}")
    rate_sample_gains = np.full(rate_sample_count, float(gain))
    if fade_samples <= 0:
        rate_sample_gains[0] = rate_sample_gains[-1] = 1.0
        return rate_sample_gains

    position = (np.arange(fade_samples) + 1.0) / (fade_samples + 1.0)
    if fade_shape == RAISED_COSINE:
        # A half cosine, so the slope of the gain is zero at both ends of each fade.
        ramp = 0.5 * (1.0 + np.cos(np.pi * position))
    else:
        ramp = 1.0 - position
    entry = gain + (1.0 - gain) * ramp

    rate_sample_gains[:fade_samples] = entry
    rate_sample_gains[rate_sample_count - fade_samples :] = entry[::-1]
    rate_sample_gains[0] = rate_sample_gains[-1] = 1.0
    return rate_sample_gains


def attenuate(
    normalized_rate_deg_s,
    measurement_rate_hz,
    detected_extents,
    peak_samples,
    classifications,
    local_median_deg_s,
    clean_robust_standard_deviation_deg_s,
    rate_sample_times_s,
    excluded,
    parameters: RepairParameters,
) -> CeilingAttenuationResult:
    """Attenuate each extent of one axis until the largest departure in its interior
    from its centerline is at or below the ceiling of its classification.

    Parameters
    ----------
    normalized_rate_deg_s : (n,) the Normalized Rate of one axis.
    measurement_rate_hz : the counted measurement rate, through which the fade length
        ``parameters.repair_fade`` becomes a whole number of rate samples.
    detected_extents : (k, 2) integer array of the inclusive first and last rate samples
        of each extent, which do not overlap.
    peak_samples : (k,) the peak sample of each extent as the detector found it, which
        the extent entries of the result report.
    classifications : (k,) the classification of each extent. It selects the centerline
        and the ceiling, so an extent whose classification is none of
        ``CLASSIFICATIONS`` raises ValueError naming the classification rather than
        being attenuated with the centerline and the ceiling of another classification.
    local_median_deg_s : (n,) the local median.
    clean_robust_standard_deviation_deg_s : (n,) the clean Robust Standard Deviation of
        the axis at every rate sample.
    rate_sample_times_s : (n,) the time of each rate sample in seconds. The chord
        centerline is a straight line in time, and drawn over the rate sample index it
        would be a different line, because the intervals between rate samples are not
        uniform.
    excluded : (k,) boolean, True for each extent of ``detected_extents``, in its order,
        that the do not repair list excludes.
    parameters : the repair parameters.

    The result holds the extents sorted by first rate sample. No rate sample outside an
    extent changes. Raises ValueError on a local median whose length differs from the
    length of the Normalized Rate.
    """
    rate = np.asarray(normalized_rate_deg_s, float).ravel()
    times = np.asarray(rate_sample_times_s, float).ravel()
    local_median = np.asarray(local_median_deg_s, float).ravel()
    clean_robust_standard_deviation = np.asarray(
        clean_robust_standard_deviation_deg_s, float
    ).ravel()
    rate_sample_count = rate.size
    # The local median comes from the detection result and the Normalized Rate from the
    # decode result, so a detection result of another video raises an error here.
    if local_median.size != rate_sample_count:
        raise ValueError(
            f"the local median holds {local_median.size} rate samples against a "
            f"Normalized Rate of {rate_sample_count}"
        )

    extents = np.asarray(detected_extents, int).reshape(-1, 2)
    peaks = np.asarray(peak_samples, int).ravel()
    classifications = np.asarray(classifications, dtype=object).ravel()
    unknown = sorted(
        {str(classification) for classification in classifications}
        - set(CLASSIFICATIONS)
    )
    if unknown:
        raise ValueError(
            f"extents carry classifications this attenuator has no treatment for: "
            f"{', '.join(unknown)}; the known ones are {', '.join(CLASSIFICATIONS)}"
        )

    excluded = np.asarray(excluded, bool).ravel()

    order = np.argsort(extents[:, 0])
    extents, peaks, classifications, excluded = (
        extents[order],
        peaks[order],
        classifications[order],
        excluded[order],
    )

    fade_samples_nominal = round(parameters.repair_fade * 1e-3 * measurement_rate_hz)
    repaired = rate.copy()
    effective_centerline = local_median.copy()
    gains, extent_entries = [], []

    for (first_sample, last_sample), peak_sample, classification, is_excluded in zip(
        extents, peaks, classifications, excluded, strict=True
    ):
        start, end = int(first_sample), int(last_sample) + 1
        span = end - start

        # The do not repair list is tested before the video edge, so an extent both at a
        # video edge and on the list counts once, as excluded.
        touches_edge = (start == 0) or (end == rate_sample_count)
        if is_excluded:
            reason = ON_THE_DO_NOT_REPAIR_LIST
        elif touches_edge and not parameters.repair_edges:
            reason = AT_A_VIDEO_EDGE
        else:
            reason = None
        if reason is not None:
            gains.append(1.0)
            extent_entries.append(
                {
                    "first_sample": start,
                    "last_sample": end - 1,
                    "scaled": False,
                    "reason": reason,
                    "gain": 1.0,
                }
            )
            continue

        # A ceiling holds of the repaired extent only when the gain is measured against
        # the centerline the extent is attenuated toward, so the gain is taken against
        # the centerline of the classification rather than against the local median.
        baseline = extent_centerline(
            parameters.centerline_of(classification),
            rate,
            local_median,
            times,
            start,
            end,
        )
        (
            gain,
            departure_in_clean_robust_standard_deviations,
            peak_departure,
            clean_robust_standard_deviation_at_largest_departure,
        ) = extent_gain(
            rate,
            baseline,
            clean_robust_standard_deviation,
            start,
            end,
            parameters.ceiling_of(classification),
        )
        deviation = rate[start:end] - baseline
        effective_centerline[start:end] = baseline

        # Both fades lie inside the extent. Where they are shortened to fit, the largest
        # departure can sit inside a fade and stand above the ceiling after the repair,
        # which is why fade_shortened is recorded.
        fade_samples = min(fade_samples_nominal, span // 2)
        fade_shortened = fade_samples < fade_samples_nominal

        rate_sample_gains = gain_at_each_rate_sample(
            span, gain, fade_samples, parameters.repair_fade_shape
        )
        # Algebraically `baseline + rate_sample_gains * deviation`, but written as a
        # reduction of the input so that a gain of one returns every rate sample of the
        # input identical. Adding a deviation back onto a centerline does not:
        # `baseline + (rate - baseline)` can differ from `rate` by a rounding step,
        # which would change the first and last rate samples of the extent and every
        # extent passed through at a gain of one.
        repaired[start:end] = rate[start:end] - (1.0 - rate_sample_gains) * deviation

        attenuated_deviation = repaired[start:end] - baseline
        input_energy = float(np.sum(deviation**2))
        output_energy = float(np.sum(attenuated_deviation**2))

        clean_robust_standard_deviation_inside_extent = clean_robust_standard_deviation[
            start:end
        ]
        usable_scale = clean_robust_standard_deviation_inside_extent > 0
        in_clean_robust_standard_deviations_after = np.zeros(span)
        in_clean_robust_standard_deviations_after[usable_scale] = (
            np.abs(attenuated_deviation[usable_scale])
            / clean_robust_standard_deviation_inside_extent[usable_scale]
        )
        peak_after_in_clean_robust_standard_deviations = (
            float(in_clean_robust_standard_deviations_after.max())
            if usable_scale.any()
            else float("inf")
        )

        # The ceiling applies to the interior alone. Inside a fade the gain stands above
        # the gain of the extent, which is what blends the attenuation into the signal,
        # so a departure near either end keeps more of its height. The interior figure
        # is the one the ceiling is asserted on; the figure over the whole extent is
        # reported, because how far a fade lets a departure through is a property of
        # the fade length worth measuring rather than a failure.
        #
        # At least one rate sample at each end is excluded whatever the fade length: the
        # first and last rate samples are always held at a gain of one so that they come
        # back identical to their recorded values, which means they keep their whole
        # departure and the ceiling does not apply to them. With a fade they lie inside
        # it and the distinction does not show; at a fade length of 0 they are the only
        # rate samples that can stand above the ceiling.
        excluded_at_each_end = max(fade_samples, 1)
        interior = slice(excluded_at_each_end, span - excluded_at_each_end)
        interior_after_in_clean_robust_standard_deviations = (
            float(in_clean_robust_standard_deviations_after[interior].max())
            if in_clean_robust_standard_deviations_after[interior].size
            and usable_scale.any()
            else 0.0
        )

        gains.append(gain)
        extent_entries.append(
            {
                "first_sample": start,
                "last_sample": end - 1,
                "rate_sample_count": span,
                "scaled": True,
                "peak_sample": int(peak_sample),
                "peak_departure_deg_s": peak_departure,
                "clean_robust_standard_deviation_deg_s": (
                    clean_robust_standard_deviation_at_largest_departure
                ),
                "peak_in_clean_robust_standard_deviations_before": (
                    departure_in_clean_robust_standard_deviations
                ),
                "peak_in_clean_robust_standard_deviations_after": (
                    peak_after_in_clean_robust_standard_deviations
                ),
                "peak_in_clean_robust_standard_deviations_after_interior": (
                    interior_after_in_clean_robust_standard_deviations
                ),
                "gain": gain,
                "fade_samples": int(fade_samples),
                "fade_used_ms": fade_samples / measurement_rate_hz * 1e3,
                "fade_shortened": bool(fade_shortened),
                "energy_ratio": output_energy / input_energy
                if input_energy > 0
                else 1.0,
            }
        )

    gains = np.asarray(gains, float)
    scaled_entries = [entry for entry in extent_entries if entry["scaled"]]
    kept_reasons = [entry["reason"] for entry in extent_entries if not entry["scaled"]]
    statistics = {
        "extent_count": int(extents.shape[0]),
        "attenuated_count": int(
            sum(1 for entry in scaled_entries if entry["gain"] < 1.0)
        ),
        "passed_through_count": int(
            sum(1 for entry in scaled_entries if entry["gain"] >= 1.0)
        ),
        "skipped_count": kept_reasons.count(AT_A_VIDEO_EDGE),
        "excluded_count": kept_reasons.count(ON_THE_DO_NOT_REPAIR_LIST),
        "shortened_fade_count": int(
            sum(1 for entry in scaled_entries if entry["fade_shortened"])
        ),
        "repair_oscillation_centerline": parameters.repair_oscillation_centerline,
        "repair_burst_centerline": parameters.repair_burst_centerline,
        "repair_impulse_centerline": parameters.repair_impulse_centerline,
        "repair_oscillation_ceiling": parameters.repair_oscillation_ceiling,
        "repair_burst_ceiling": parameters.repair_burst_ceiling,
        "repair_impulse_ceiling": parameters.repair_impulse_ceiling,
        "gain_min": float(gains.min()) if gains.size else None,
        "gain_median": float(np.median(gains)) if gains.size else None,
        "gain_max": float(gains.max()) if gains.size else None,
        "peak_in_clean_robust_standard_deviations_before_median": float(
            np.median(
                [
                    entry["peak_in_clean_robust_standard_deviations_before"]
                    for entry in scaled_entries
                ]
            )
        )
        if scaled_entries
        else None,
        "energy_ratio_median": float(
            np.median([entry["energy_ratio"] for entry in scaled_entries])
        )
        if scaled_entries
        else None,
        "share_of_rate_samples_inside_scaled_extents": float(
            sum(entry["rate_sample_count"] for entry in scaled_entries)
            / rate_sample_count
        ),
    }
    return CeilingAttenuationResult(
        repaired_rate_deg_s=repaired,
        centerline=effective_centerline,
        gains=gains,
        extents=extents.copy(),
        extent_entries=extent_entries,
        statistics=statistics,
    )
