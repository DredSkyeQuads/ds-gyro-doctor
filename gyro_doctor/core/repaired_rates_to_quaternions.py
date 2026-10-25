# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""The rebuild of orientations from repaired rates, in the one function that composes a
repaired rate series back into the quaternions of the gyro data track, and its two
assertions, round trip identity and cadence preservation.

Each rate series has its own multiplier. A rate sample of the Normalized Rate is a
rotation divided by the period T, so its multiplier is T. A rate sample of the recorded
rate is a rotation divided by its own interval between group boundaries, so its
multiplier is that interval. A rate sample multiplied by the multiplier of the other
series gives a rotation the camera did not make, which is fabricated motion. The caller
therefore names the series, and the rebuild raises an error on a Normalized Rate without
T, a T other than the T derived from the group boundaries of the video, and a recorded
rate with a T.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from gyro_doctor import errors
from gyro_doctor.core import decode_stage
from gyro_doctor.core.rates import DEG

# The text that describes the multiplier of each rate series. The multiplier turns a
# rate sample of that series back into a rotation. Naming the series selects the
# multiplier, so the two are never chosen apart.
COMPOSITION_MULTIPLIER = {
    "normalized": "the period T",
    "recorded": "each sample's own elapsed interval",
}

# The largest difference, in degrees per second, that round trip identity allows between
# a rate of the rebuilt orientations and the rate the decode result holds.
ROUND_TRIP_IDENTITY_TOLERANCE_DEG_S = 1e-6


def _hamilton_product(left, right):
    """The Hamilton product of two quaternions in w, x, y, z order, which composes as
    scipy's ``R_a * R_b`` does."""
    w0, x0, y0, z0 = left
    w1, x1, y1, z1 = right
    return (
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    )


def _composed_orientations(first_orientation_wxyz, rotation_deg):
    """The orientation at each group boundary, composed forward from
    ``first_orientation_wxyz`` by the rotation of each rate sample. The last row is the
    orientation at the last group boundary, which closes the interval the last rate
    sample spans.

    ``rotation_deg`` is (N, 3), in degrees, with the columns pitch, yaw, and roll, which
    are the x, y, and z of the rotation vector. Each rotation composes on the right of
    the orientation before it, since the derivation of the rates takes the rotation of a
    rate sample as the inverse of one orientation times the next.
    """
    rotation_xyzw = Rotation.from_rotvec(rotation_deg / DEG).as_quat()
    rotation_wxyz = np.column_stack(
        [
            rotation_xyzw[:, 3],
            rotation_xyzw[:, 0],
            rotation_xyzw[:, 1],
            rotation_xyzw[:, 2],
        ]
    )
    orientations = np.empty((rotation_deg.shape[0] + 1, 4), dtype=np.float64)
    orientations[0] = first_orientation_wxyz
    orientation = tuple(float(component) for component in first_orientation_wxyz)
    for i in range(rotation_wxyz.shape[0]):
        orientation = _hamilton_product(orientation, rotation_wxyz[i])
        orientations[i + 1] = orientation
    return orientations


def _wxyz_to_rotations(quaternions_wxyz):
    quaternions_wxyz = np.asarray(quaternions_wxyz, dtype=np.float64)
    return Rotation.from_quat(
        np.column_stack(
            [
                quaternions_wxyz[:, 1],
                quaternions_wxyz[:, 2],
                quaternions_wxyz[:, 3],
                quaternions_wxyz[:, 0],
            ]
        )
    )


def _rotations_to_wxyz(rotations):
    quaternions_xyzw = rotations.as_quat()
    return np.column_stack(
        [
            quaternions_xyzw[:, 3],
            quaternions_xyzw[:, 0],
            quaternions_xyzw[:, 1],
            quaternions_xyzw[:, 2],
        ]
    )


def spreads_one_measurement_across_quaternions(decode_result):
    """Whether the camera stored each measurement across several quaternions, read from
    the handler name in the header of the decode result.

    An Osmo Action 6 writes ``CAM meta`` and stores each measurement across three
    quaternions; an air unit writes ``DJI meta`` and stores each measurement in one
    quaternion or in two identical ones."""
    return bool(
        decode_result["handler_name"] == decode_stage.OSMO_ACTION_6_HANDLER_NAME
    )


def redistribute_across_group_quaternions(
    kept_quaternions,
    group_opening_quaternion_index,
    rebuilt_orientations_at_group_boundaries,
):
    """The kept quaternions of the rebuilt track of a camera that stores each measurement
    across several quaternions: the repaired rotation of each measurement group placed
    across the quaternions of that group in the shares measured from them.

    ``kept_quaternions`` holds the kept quaternions of the recorded track,
    ``group_opening_quaternion_index`` the index of the quaternion at each group
    boundary, the last of which closes the last group, and
    ``rebuilt_orientations_at_group_boundaries`` the rebuilt orientation at each group
    boundary.

    The repair changes one value per measurement group, so the rebuild decides what the
    quaternions between two group boundaries hold. One orientation across the whole group
    would replace the gradual rotation the camera recorded within the group with a hold
    and a jump, which is motion the camera never recorded and moves an axis the repair
    left unchanged.

    Each quaternion keeps its recorded rotation from the orientation that opens its
    group. On top of that rotation goes the change the repair made to the rotation of the
    group, scaled by the share of the recorded rotation of the group reached at that
    quaternion. The group keeps the shape the camera recorded, and the orientation that
    opens each group is its rebuilt orientation. A group the repair left unchanged takes
    no change of its own, so its quaternions keep their recorded rotations relative to
    one another and take only the carry forward of earlier repairs.
    """
    quaternion_count = kept_quaternions.shape[0]
    group_count = group_opening_quaternion_index.size - 1
    quaternion_index = np.arange(quaternion_count)
    group = np.clip(
        np.searchsorted(group_opening_quaternion_index, quaternion_index, side="right")
        - 1,
        0,
        group_count - 1,
    )

    kept_rotations = _wxyz_to_rotations(kept_quaternions)
    recorded_openings = _wxyz_to_rotations(
        kept_quaternions[group_opening_quaternion_index[:-1]]
    )
    recorded_closings = _wxyz_to_rotations(
        kept_quaternions[group_opening_quaternion_index[1:]]
    )
    rebuilt_openings = _wxyz_to_rotations(rebuilt_orientations_at_group_boundaries[:-1])
    rebuilt_closings = _wxyz_to_rotations(rebuilt_orientations_at_group_boundaries[1:])

    recorded_rotation_within_group = recorded_openings[group].inv() * kept_rotations
    recorded_group_rotation = recorded_openings.inv() * recorded_closings
    repair_change = (
        recorded_group_rotation.inv() * (rebuilt_openings.inv() * rebuilt_closings)
    ).as_rotvec()

    recorded_group_rotation_vector = recorded_group_rotation.as_rotvec()[group]
    denominator = np.einsum(
        "ij,ij->i", recorded_group_rotation_vector, recorded_group_rotation_vector
    )
    progressed = np.einsum(
        "ij,ij->i",
        recorded_rotation_within_group.as_rotvec(),
        recorded_group_rotation_vector,
    )
    fraction = np.where(
        denominator > 0.0,
        progressed / np.where(denominator > 0.0, denominator, 1.0),
        0.0,
    )

    scaled_repair_change = Rotation.from_rotvec(
        fraction[:, None] * repair_change[group]
    )
    rebuilt_quaternions = _rotations_to_wxyz(
        rebuilt_openings[group] * recorded_rotation_within_group * scaled_repair_change
    )

    # A quaternion after the last group boundary belongs to no group, so it keeps its
    # recorded rotation relative to that boundary.
    tail = quaternion_index[group_opening_quaternion_index[-1] :]
    recorded_at_last_boundary = _wxyz_to_rotations(
        np.repeat(
            kept_quaternions[group_opening_quaternion_index[-1]][None],
            tail.size,
            axis=0,
        )
    )
    rebuilt_at_last_boundary = _wxyz_to_rotations(
        np.repeat(rebuilt_orientations_at_group_boundaries[-1][None], tail.size, axis=0)
    )
    rebuilt_quaternions[tail] = _rotations_to_wxyz(
        rebuilt_at_last_boundary
        * (recorded_at_last_boundary.inv() * kept_rotations[tail])
    )
    return rebuilt_quaternions


def _composition_seconds(decode_result, boundaries, series, period_s):
    """The multiplier of each rate sample, which turns it back into a rotation.

    A rate sample of the recorded rate is multiplied by its own interval between group
    boundaries. A rate sample of the Normalized Rate is multiplied by T, which the caller
    states and which must equal the T derived from the group boundaries of the video.
    Any other multiplier rebuilds rotations the camera did not make, so each other
    pairing of series and T raises an error.
    """
    if series not in COMPOSITION_MULTIPLIER:
        raise errors.CannotRepairError(
            errors.ErrorCode.REBUILD_SERIES_UNKNOWN, series=series
        )
    if series == "recorded":
        if period_s is not None:
            raise errors.CannotRepairError(
                errors.ErrorCode.RECORDED_RATE_REBUILD_GIVEN_PERIOD,
                period_s=period_s,
            )
        return np.diff(boundaries)
    if period_s is None:
        raise errors.CannotRepairError(
            errors.ErrorCode.NORMALIZED_RATE_REBUILD_WITHOUT_PERIOD
        )
    derived = decode_stage.derive_period(boundaries, decode_result)
    agreement = decode_stage.PERIOD_TEST_TOLERANCE_S
    if abs(float(period_s) - derived) > agreement:
        raise errors.CannotRepairError(
            errors.ErrorCode.REBUILD_PERIOD_DIFFERS,
            stated_us=float(period_s) * 1e6,
            derived_us=derived * 1e6,
        )
    return float(period_s)


def quaternion_track_from_repaired_rates(
    decode_result, repaired_rate_deg_s, *, series, period_s=None
):
    """The rebuilt track: every quaternion of ``decode_result``, in its order, rebuilt
    from the repaired rates ``repaired_rate_deg_s``.

    ``series`` names the rate series ``repaired_rate_deg_s`` is in, ``"normalized"`` or
    ``"recorded"``, and has no default, since it selects the time each rate sample is
    multiplied by. ``period_s`` is T, which the Normalized Rate requires; given
    with the recorded rate, it raises an error. Raises the errors of
    ``_composition_seconds``, and ValueError on repaired rates whose shape differs from
    the shape of the recorded rate of ``decode_result``.
    """
    recorded = np.asarray(decode_result["recorded_rate_deg_s"], dtype=np.float64)
    repaired = np.asarray(repaired_rate_deg_s, dtype=np.float64)
    if repaired.shape != recorded.shape:
        raise ValueError(
            f"the repaired rates are {repaired.shape} and the recorded rate of the "
            f"decode result is {recorded.shape}, but both must hold one rate sample for "
            f"each interval between consecutive measurement groups"
        )

    boundaries = decode_stage.group_boundary_times(decode_result)
    seconds = _composition_seconds(decode_result, boundaries, series, period_s)

    recorded_quaternions = np.asarray(
        decode_result["quaternions_wxyz"], dtype=np.float64
    )
    keep = np.asarray(decode_result["kept_after_time_collisions"], dtype=bool)
    group_opening_quaternion_index = np.asarray(
        decode_result["group_opening_quaternion_index"], dtype=int
    )
    kept_quaternions = recorded_quaternions[keep]

    rebuilt_orientations_at_group_boundaries = _composed_orientations(
        kept_quaternions[group_opening_quaternion_index[0]],
        repaired * np.reshape(seconds, (-1, 1)),
    )

    if spreads_one_measurement_across_quaternions(decode_result):
        rebuilt_kept_quaternions = redistribute_across_group_quaternions(
            kept_quaternions,
            group_opening_quaternion_index,
            rebuilt_orientations_at_group_boundaries,
        )
    else:
        # An air unit stores a measurement in one quaternion or in two identical ones,
        # so every quaternion of a group takes the rebuilt orientation of the group, and
        # each identical pair stays identical.
        run_lengths = np.diff(
            np.append(group_opening_quaternion_index, kept_quaternions.shape[0])
        )
        rebuilt_kept_quaternions = np.repeat(
            rebuilt_orientations_at_group_boundaries, run_lengths, axis=0
        )

    rebuilt = recorded_quaternions.copy()
    rebuilt[keep] = rebuilt_kept_quaternions
    # The derivation of the rates excludes the later quaternion of a time collision, so
    # it has no rate sample of its own. It takes the rebuilt orientation of the kept
    # quaternion before it, and with it the carry forward.
    if not keep.all():
        source = np.maximum.accumulate(np.where(keep, np.arange(keep.size), 0))
        rebuilt[~keep] = rebuilt[source[~keep]]
    return rebuilt


def round_trip_identity(decode_result):
    """Round trip identity: orientations rebuilt from the recorded rate of
    ``decode_result`` reproduce its recorded rate and its rate between consecutive
    stored quaternions within ``ROUND_TRIP_IDENTITY_TOLERANCE_DEG_S``.

    It shows that the rebuild inverts the derivation of the rates, in its composition
    order and in its placement of each group across its quaternions. Raises an error
    when either largest difference exceeds ``ROUND_TRIP_IDENTITY_TOLERANCE_DEG_S``.
    """
    rebuilt = quaternion_track_from_repaired_rates(
        decode_result, decode_result["recorded_rate_deg_s"], series="recorded"
    )
    keep = np.asarray(decode_result["kept_after_time_collisions"], dtype=bool)
    kept_times = np.asarray(decode_result["quaternion_times_s"], dtype=np.float64)[keep]
    _, consecutive_quaternion_rate = decode_stage.consecutive_quaternion_rates(
        kept_times, rebuilt[keep]
    )
    # The group openings come from the decode result and are never derived again here:
    # the air unit rule finds identical consecutive quaternions, and an Osmo Action 6
    # stores none.
    group_opening_quaternion_index = np.asarray(
        decode_result["group_opening_quaternion_index"], dtype=int
    )
    _, recorded_rate = decode_stage.interval_rates(
        kept_times[group_opening_quaternion_index],
        rebuilt[keep][group_opening_quaternion_index],
    )

    max_consecutive_quaternion_rate_difference = float(
        np.max(
            np.abs(
                consecutive_quaternion_rate
                - decode_result["consecutive_quaternion_rate_deg_s"]
            )
        )
    )
    max_recorded_rate_difference = float(
        np.max(np.abs(recorded_rate - decode_result["recorded_rate_deg_s"]))
    )
    result = {
        "name": "round trip identity",
        "max_consecutive_quaternion_rate_difference_deg_s": (
            max_consecutive_quaternion_rate_difference
        ),
        "max_recorded_rate_difference_deg_s": max_recorded_rate_difference,
        "tolerance_deg_s": ROUND_TRIP_IDENTITY_TOLERANCE_DEG_S,
    }
    if not (
        max_consecutive_quaternion_rate_difference
        <= ROUND_TRIP_IDENTITY_TOLERANCE_DEG_S
        and max_recorded_rate_difference <= ROUND_TRIP_IDENTITY_TOLERANCE_DEG_S
    ):
        raise errors.CannotRepairError(
            errors.ErrorCode.ROUND_TRIP_IDENTITY_FAILED,
            recorded_difference=max_recorded_rate_difference,
            consecutive_difference=max_consecutive_quaternion_rate_difference,
            tolerance=ROUND_TRIP_IDENTITY_TOLERANCE_DEG_S,
        )
    result["passed"] = True
    return result


def cadence_preservation(decode_result, rebuilt_track):
    """Cadence preservation: the rebuilt track ``rebuilt_track`` repeats an orientation
    at exactly the positions where the recorded track of ``decode_result`` repeats one.

    Where the camera stored a measurement in two identical quaternions, each rebuilt pair
    must stay identical, since the decode result recomputed on the rebuilt track finds its
    measurement groups from those pairs. Where the camera stored a different value in
    every quaternion, a repeat in the rebuilt track is a hold and a jump that the camera
    never recorded. Raises an error where the positions differ.
    """
    keep = np.asarray(decode_result["kept_after_time_collisions"], dtype=bool)
    recorded_repeats = decode_stage.repeats_previous_quaternion(
        np.asarray(decode_result["quaternions_wxyz"], dtype=np.float64)[keep]
    )
    rebuilt_repeats = decode_stage.repeats_previous_quaternion(
        np.asarray(rebuilt_track, dtype=np.float64)[keep]
    )
    result = {
        "name": "cadence preservation",
        "recorded_repeated_pairs": int(recorded_repeats.sum()),
        "rebuilt_repeated_pairs": int(rebuilt_repeats.sum()),
    }
    if not np.array_equal(recorded_repeats, rebuilt_repeats):
        raise errors.CannotRepairError(
            errors.ErrorCode.CADENCE_PRESERVATION_FAILED,
            rebuilt=int(rebuilt_repeats.sum()),
            recorded=int(recorded_repeats.sum()),
            differing=int(np.count_nonzero(recorded_repeats != rebuilt_repeats)),
        )
    result["passed"] = True
    return result
