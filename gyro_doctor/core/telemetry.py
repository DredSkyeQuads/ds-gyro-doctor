# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""The read of the gyro data track out of the container: every quaternion in file order,
converted into the camera frame of reference, with its time, and the test for a video
whose stored camera model name carries the model name marker. Every value is read by its
Protocol Buffers field number with the Python standard library.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gyro_doctor import errors
from gyro_doctor.core import mp4_writeback


@dataclass
class GyroDataTrack:
    """The gyro data track in file order: the time of each quaternion in seconds, each
    quaternion in the camera frame of reference as (w, x, y, z), the quaternion count of
    each telemetry frame, and the handler name of the track."""

    quaternion_times_s: np.ndarray
    quaternions_wxyz: np.ndarray
    quaternion_count_per_telemetry_frame: np.ndarray
    handler_name: str


def _varint_field(telemetry_frame_bytes, span, field_number):
    """The value of the varint field ``field_number`` inside ``span``, or None if
    absent."""
    if span is None:
        return None
    for number, wire_type, value, _value_span in mp4_writeback.protobuf_fields(
        telemetry_frame_bytes, *span
    ):
        if number == field_number and wire_type == 0:
            return value
    return None


def _float_field(telemetry_frame_bytes, span, field_number):
    """The value of the 32 bit fixed field ``field_number`` inside ``span``, or None if
    absent."""
    if span is None:
        return None
    for number, wire_type, _value, value_span in mp4_writeback.protobuf_fields(
        telemetry_frame_bytes, *span
    ):
        if number == field_number and wire_type == 5:
            return struct.unpack_from("<f", telemetry_frame_bytes, value_span[0])[0]
    return None


def _frame_rates_and_stored_camera_model_name(telemetry_frame_bytes):
    """``sensor_fps``, ``framerate``, and ``product_name`` from the first telemetry
    frame, through ``ProductMeta.clip_meta(1).sensor_fps(11).sensor_frame_rate(1)`` and
    ``ProductMeta.stream_meta(2).video_stream_meta(3).framerate(3)``. Raises an error
    carrying the field on a first telemetry frame that lacks one of the three."""
    whole = (0, len(telemetry_frame_bytes))
    clip_meta = mp4_writeback.submessage_span(telemetry_frame_bytes, whole, 1)
    sensor_fps = _float_field(
        telemetry_frame_bytes,
        None
        if clip_meta is None
        else mp4_writeback.submessage_span(telemetry_frame_bytes, clip_meta, 11),
        1,
    )
    stream_meta = mp4_writeback.submessage_span(telemetry_frame_bytes, whole, 2)
    framerate = _float_field(
        telemetry_frame_bytes,
        None
        if stream_meta is None
        else mp4_writeback.submessage_span(telemetry_frame_bytes, stream_meta, 3),
        3,
    )
    for name, value in (("sensor_fps", sensor_fps), ("framerate", framerate)):
        if value is None:
            raise errors.CannotReadError(
                errors.ErrorCode.FRAME_RATE_MISSING, field=name
            )
    return (
        sensor_fps,
        framerate,
        mp4_writeback.find_stored_camera_model_name(telemetry_frame_bytes)[2],
    )


def _telemetry_frame(telemetry_frame_bytes):
    """``frame_timestamp``, ``offset``, and the stored quaternions of one telemetry
    frame, through ``ProductMeta.frame_meta(3).frame_meta_header(1).frame_timestamp(2)``
    and ``ProductMeta.frame_meta(3).imu_frame_meta(3).imu_attitude_after_fusion(2)``.

    An absent ``frame_timestamp`` or ``offset`` reads as zero, since a Protocol Buffers
    message in the proto3 syntax omits a scalar field whose value is zero. Raises the
    errors of ``mp4_writeback.decode_quaternion_message``."""
    frame_meta = mp4_writeback.submessage_span(
        telemetry_frame_bytes, (0, len(telemetry_frame_bytes)), 3
    )
    header = (
        None
        if frame_meta is None
        else mp4_writeback.submessage_span(telemetry_frame_bytes, frame_meta, 1)
    )
    imu_frame_meta = (
        None
        if frame_meta is None
        else mp4_writeback.submessage_span(telemetry_frame_bytes, frame_meta, 3)
    )
    orientation_record = (
        None
        if imu_frame_meta is None
        else mp4_writeback.submessage_span(telemetry_frame_bytes, imu_frame_meta, 2)
    )
    frame_timestamp = _varint_field(telemetry_frame_bytes, header, 2)
    offset = _float_field(telemetry_frame_bytes, orientation_record, 4)
    quaternions = np.array(
        [
            mp4_writeback.decode_quaternion_message(telemetry_frame_bytes, span)
            for span in mp4_writeback.quaternion_message_spans(telemetry_frame_bytes)
        ],
        dtype=np.float64,
    ).reshape(-1, 4)
    return (
        0 if frame_timestamp is None else frame_timestamp,
        0.0 if offset is None else offset,
        quaternions,
    )


def _already_repaired_test(stored_camera_model_name):
    """Raise an error on a video whose stored camera model name carries the model name
    marker."""
    if stored_camera_model_name.endswith(
        mp4_writeback.MODEL_NAME_MARKER.encode("ascii")
    ):
        raise errors.CannotRepairError(
            errors.ErrorCode.ALREADY_REPAIRED,
            model_name=stored_camera_model_name.decode("latin1"),
        )


def already_repaired_test(path):
    """Raise an error on the video at ``path`` if its stored camera model name ends in
    ``"-Dr."``.

    Reads the first telemetry frame alone, so a run can stop on a repaired video before
    it reads a decode result from the cache. Also raises an error on a video with no gyro
    data track, no telemetry frame, or a first telemetry frame without the stored camera
    model name. Every error it raises carries ``path`` as ``file``."""
    path = Path(path)
    try:
        with path.open("rb") as mp4_file:
            track = mp4_writeback.locate_djmd_track(mp4_file)
            if track.telemetry_frame_offsets.size == 0:
                raise errors.CannotRepairError(
                    errors.ErrorCode.TELEMETRY_FRAMES_MISSING
                )
            mp4_file.seek(int(track.telemetry_frame_offsets[0]))
            telemetry_frame_bytes = bytearray(
                mp4_file.read(int(track.telemetry_frame_sizes[0]))
            )
        _already_repaired_test(
            mp4_writeback.find_stored_camera_model_name(telemetry_frame_bytes)[2]
        )
    except OSError as error:
        raise errors.CannotReadError(
            errors.ErrorCode.SYSTEM_ERROR, file=path, reason=error.strerror
        ) from error
    except errors.GyroDoctorError as error:
        error.values.setdefault("file", path)
        raise


def _quaternion_times(frame_timestamp, offset, counts, sensor_fps, framerate):
    """The time of each quaternion in seconds, in file order.

    The time on the gyro sensor clock is ``frame_timestamp`` less the first
    ``frame_timestamp``, plus the position of the quaternion in its telemetry frame less
    ``offset``, in quaternion durations. That time is rescaled by ``sensor_fps`` over
    ``framerate``. The arithmetic runs in milliseconds and in the order written here:
    another order changes the last bit of a time, and the order of two quaternions and
    the detection of a time collision both depend on that bit."""
    starts = np.cumsum(counts) - counts
    position = np.arange(counts.sum(), dtype=np.float64) - np.repeat(starts, counts)
    frame_timestamp_from_first_ms = np.repeat(
        (frame_timestamp - frame_timestamp[0]).astype(np.float64) / 1000.0, counts
    )
    frame_period_ms = 1000.0 / sensor_fps
    count = np.repeat(counts.astype(np.float64), counts)
    gyro_sensor_clock_ms = (
        frame_timestamp_from_first_ms
        + ((position - np.repeat(offset, counts)) / count) * frame_period_ms
    )
    return gyro_sensor_clock_ms / (framerate / sensor_fps) / 1000.0


def read_gyro_data_track(path, *, force):
    """The gyro data track of the video at ``path``, in file order.

    The camera stores every quaternion with w at or above zero, so where the orientation
    carries w below zero it stores the negation, which is the same rotation. The track
    therefore jumps to the opposite sign wherever w passes zero. The quaternions are
    returned as stored, jumps included; every consumer treats a quaternion and its
    negation as the same rotation.

    Raises an error on the video if it has no gyro data track, if its gyro data track
    stores more than one entry in a chunk or holds no orientations, if its stored camera
    model name ends in ``"-Dr."`` and ``force`` is not set, if its first telemetry frame
    lacks ``sensor_fps`` or ``framerate``, or if a quaternion is not a number or is all
    zeros. Every error it raises carries ``path`` as ``file``."""
    path = Path(path)
    try:
        return _read_gyro_data_track(path, force)
    except OSError as error:
        raise errors.CannotReadError(
            errors.ErrorCode.SYSTEM_ERROR, file=path, reason=error.strerror
        ) from error
    except errors.GyroDoctorError as error:
        error.values.setdefault("file", path)
        raise


def _read_gyro_data_track(path, force):
    with path.open("rb") as mp4_file:
        track = mp4_writeback.locate_djmd_track(mp4_file)
        telemetry_frames = []
        for position, size in zip(
            track.telemetry_frame_offsets, track.telemetry_frame_sizes, strict=True
        ):
            mp4_file.seek(int(position))
            telemetry_frame_bytes = bytearray(mp4_file.read(int(size)))
            if not telemetry_frames:
                sensor_fps, framerate, stored_camera_model_name = (
                    _frame_rates_and_stored_camera_model_name(telemetry_frame_bytes)
                )
                if not force:
                    _already_repaired_test(stored_camera_model_name)
            telemetry_frames.append(_telemetry_frame(telemetry_frame_bytes))

    counts = np.array(
        [len(telemetry_frame[2]) for telemetry_frame in telemetry_frames],
        dtype=np.int64,
    )
    if counts.sum() == 0:
        raise errors.CannotRepairError(errors.ErrorCode.STORED_ORIENTATIONS_MISSING)
    frame_timestamp = np.array(
        [telemetry_frame[0] for telemetry_frame in telemetry_frames], dtype=np.int64
    )
    offset = np.array(
        [telemetry_frame[1] for telemetry_frame in telemetry_frames], dtype=np.float64
    )
    stored = np.concatenate(
        [telemetry_frame[2] for telemetry_frame in telemetry_frames]
    )
    owning_telemetry_frame = np.repeat(np.arange(counts.size), counts)
    not_a_number = np.isnan(stored).any(axis=1)
    if not_a_number.any():
        raise errors.CannotReadError(
            errors.ErrorCode.STORED_ORIENTATION_NOT_A_NUMBER,
            telemetry_frame=int(owning_telemetry_frame[not_a_number][0]),
        )
    quaternions = mp4_writeback.stored_to_camera_frame(stored)
    all_zero = (quaternions == 0.0).all(axis=1)
    if all_zero.any():
        raise errors.CannotReadError(
            errors.ErrorCode.STORED_ORIENTATION_ALL_ZEROS,
            telemetry_frame=int(owning_telemetry_frame[all_zero][0]),
        )
    return GyroDataTrack(
        quaternion_times_s=_quaternion_times(
            frame_timestamp, offset, counts, sensor_fps, framerate
        ),
        quaternions_wxyz=quaternions,
        quaternion_count_per_telemetry_frame=counts,
        handler_name=track.handler_name,
    )
