# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""The container layer of the gyro data track: the walk of the MP4 boxes that locates
the track, the read and the in place overwrite of each quaternion message by Protocol
Buffers field number, the basis transform between a stored quaternion and the camera
frame of reference, and the write of the repaired video.

The write copies the input video. In the copy it overwrites in place the four float32
values of each quaternion message whose rotation from its stored orientation to its
repaired orientation exceeds the identity threshold, and it rewrites the stored camera
model name in the first telemetry frame with the model name marker, unless the name
already carries it. Each new value has the length of the value it replaces, so no
message, box, or index table changes length, and every other byte of the repaired video
is identical to the input video. Before it copies any byte, the write reads the input
video, runs each of its tests, and computes every new value, so a video on which a test
raises an error leaves the output path as it was. Only the equal length assertion runs
after the copy.
"""

import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gyro_doctor import errors

# The model name marker. The write derives the new stored camera model name from the
# stored one by removing the "DJI " prefix and appending the marker. Both are four
# bytes, so the new name has the length of the stored name for any model. Gyroflow
# strips any "DJI " prefix from the stored name and shows "DJI " followed by the
# result, for example "DJI O4-Dr.", and the marker lets the product recognize a video
# it has already repaired.
MODEL_NAME_MARKER = "-Dr."

_DJI_PREFIX = b"DJI "

# The handler names of the gyro data track: "DJI meta" on the air units and "CAM meta"
# on the Osmo Action 6. The rest of the layout this module walks is the same on every
# camera.
_GYRO_DATA_TRACK_HANDLER_NAMES = ("DJI meta", "CAM meta")

# The camera writes each quaternion as the two byte header 0x1A 0x14 followed by a 20
# byte quaternion message, which holds w, x, y, and z, each as a one byte tag and a four
# byte little endian float32 value. The span of a quaternion message leaves out the
# header.
_QUATERNION_MESSAGE_LENGTH = 20
# The tags of w, x, y, and z, at bytes 0, 5, 10, and 15 of the quaternion message.
_QUATERNION_MESSAGE_TAGS = (0x0D, 0x15, 0x1D, 0x25)

# The basis transform, in Hamilton products of quaternions written (w, x, y, z): a
# quaternion in the camera frame of reference is _BASIS_TRANSFORM_LEFT times the stored
# quaternion times _BASIS_TRANSFORM_RIGHT, and the stored quaternion is
# _BASIS_TRANSFORM_LEFT_INVERSE times it times _BASIS_TRANSFORM_RIGHT_INVERSE.
# _BASIS_TRANSFORM_LEFT and _BASIS_TRANSFORM_RIGHT are unit quaternions, so each inverse
# is the conjugate.
_BASIS_TRANSFORM_LEFT = np.array([0.0, 0.0, 1.0, 0.0])
_BASIS_TRANSFORM_RIGHT = np.array([0.5, -0.5, -0.5, 0.5])
_BASIS_TRANSFORM_LEFT_INVERSE = np.array([0.0, 0.0, -1.0, 0.0])
_BASIS_TRANSFORM_RIGHT_INVERSE = np.array([0.5, 0.5, 0.5, -0.5])

# The one to one correspondence: each quaternion message of the video, converted into
# the camera frame of reference, matches the decoded quaternion at its position in file
# order to at least this absolute dot product. The absolute value takes a quaternion
# and its negation as the same rotation.
_MINIMUM_ABSOLUTE_DOT_PRODUCT = 0.999

# Box types whose bodies hold only boxes.
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta", b"edts"}


@dataclass
class DjmdTrack:
    """The gyro data track as ``locate_djmd_track`` finds it in the container: the byte
    position in the file and the byte length of each telemetry frame, as int64 arrays in
    file order, and the handler name of the track, by which the product narrows the
    grouping pattern: ``DJI meta`` identifies an air unit and ``CAM meta`` an Osmo
    Action 6."""

    telemetry_frame_offsets: np.ndarray
    telemetry_frame_sizes: np.ndarray
    handler_name: str


def repaired_camera_model_name(stored_camera_model_name):
    """The stored camera model name that the write puts in place of
    ``stored_camera_model_name``, the stored camera model name of the video, as bytes:
    the camera model followed by the model name marker.

    The stored name takes one of two forms, ``"DJI " + model`` on a video the product
    has not repaired and ``model + "-Dr."`` on one it has. The function removes a
    trailing ``"-Dr."`` and then a leading ``"DJI "``, and appends the model name
    marker. ``b"DJI O4"`` and ``b"O4-Dr."`` both give ``b"O4-Dr."``, and
    ``b"DJI OsmoAction6"`` gives ``b"OsmoAction6-Dr."``. Both affixes are four bytes, so
    the new name has the length of a stored name in either form. A name that carries
    both affixes, such as ``b"DJI O4-Dr."``, gives a new name of another length,
    ``b"O4-Dr."``, on which the equal length test of the write raises an error.

    A name that carries the model name marker and not the prefix gives itself, so that a
    repair of a video the product has already repaired, which a run reaches only with
    ``--force``, leaves the model name marker as it is.

    Raises an error on a name that carries neither affix, since no model can be
    recovered from it."""
    suffix = MODEL_NAME_MARKER.encode("ascii")
    name = bytes(stored_camera_model_name)
    model = name.removesuffix(suffix).removeprefix(_DJI_PREFIX)
    if model == name:
        raise errors.CannotRepairError(
            errors.ErrorCode.CAMERA_MODEL_NAME_UNKNOWN_FORM,
            model_name=name.decode("latin1"),
        )
    return model + suffix


@dataclass
class WritebackReport:
    """The report of the write: the number of quaternion messages it overwrote, the
    number of telemetry frames it rewrote, and the size in bytes of the input video,
    which the equal length assertion makes the size of the repaired video too."""

    quaternion_messages_overwritten: int
    telemetry_frames_rewritten: int
    video_size_bytes: int


@dataclass
class RepairedOrientations:
    """What the write needs to write the repaired orientations into the video.

    - ``track_quaternions_wxyz`` (N, 4): the decoded quaternions of the gyro data track
      in file order, in the camera frame of reference, as (w, x, y, z). File order is
      the order in which the write walks the quaternion messages; in time order the
      quaternions of neighboring telemetry frames can interleave.
    - ``track_times`` (N,): the time of each of those quaternions in seconds.
    - ``repaired_orientation_times_s`` (M,): strictly increasing times in seconds, among
      which every time of ``track_times`` lies.
    - ``stored_orientations_xyzw`` and ``repaired_orientations_xyzw`` (M, 4): the stored
      orientation and the repaired orientation at each time of
      ``repaired_orientation_times_s``, in the camera frame of reference, in the order
      (x, y, z, w) of scipy. The write takes for each quaternion the rotation from the
      stored orientation to the repaired orientation at its time."""

    track_quaternions_wxyz: np.ndarray
    track_times: np.ndarray
    repaired_orientation_times_s: np.ndarray
    stored_orientations_xyzw: np.ndarray
    repaired_orientations_xyzw: np.ndarray


def _walk(mp4_file, start, end, path=""):
    """Yield ``(box_path, position, size, header_size)`` for every box from byte
    ``start`` up to, and not including, byte ``end``, descending into the boxes of
    ``_CONTAINERS``. Reads a 32 bit size, a 64 bit size (a size field of 1), and a box
    that runs to ``end`` (a size field of 0)."""
    position = start
    while position + 8 <= end:
        mp4_file.seek(position)
        header = mp4_file.read(8)
        if len(header) < 8:
            break
        size, box_type = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            size = struct.unpack(">Q", mp4_file.read(8))[0]
            header_size = 16
        elif size == 0:
            size = end - position
        if size < header_size or position + size > end:
            break
        name = box_type.decode("latin1")
        yield (path + "/" + name, position, size, header_size)
        if box_type in _CONTAINERS:
            yield from _walk(
                mp4_file, position + header_size, position + size, path + "/" + name
            )
        position += size


def _parse_tracks(mp4_file, boxes):
    """For each track among the walked ``boxes``: its handler type and handler name, the
    fourcc of its sample description table, and its sample size, chunk offset, and
    sample to chunk tables."""
    tracks = []
    track = None
    for box_path, position, size, header_size in boxes:
        leaf = box_path.rsplit("/", 1)[-1]
        if leaf == "trak":
            track = {}
            tracks.append(track)
        if track is None:
            continue
        # The sample size and chunk offset tables are read whole. Every other box is
        # read to at most 64 KiB, which holds the handler reference box, the sample
        # description table, and the sample to chunk table of the gyro data track.
        mp4_file.seek(position + header_size)
        if leaf in ("stsz", "stco", "co64"):
            body = mp4_file.read(size - header_size)
        else:
            body = mp4_file.read(min(size - header_size, 1 << 16))
        if leaf == "hdlr":
            track["handler_type"] = body[8:12].decode("latin1")
            track["handler_name"] = (
                body[24:].split(b"\x00")[0].decode("latin1", "replace")
            )
        elif leaf == "stsd":
            track["stsd_fourcc"] = body[12:16].decode("latin1")
        elif leaf == "stsz":
            fixed, count = struct.unpack_from(">II", body, 4)
            if fixed:
                track["sizes"] = np.full(count, fixed, dtype=np.int64)
            else:
                track["sizes"] = np.frombuffer(
                    body, dtype=">u4", count=count, offset=12
                ).astype(np.int64)
        elif leaf == "stco":
            entry_count = struct.unpack_from(">I", body, 4)[0]
            track["offsets"] = np.frombuffer(
                body, dtype=">u4", count=entry_count, offset=8
            ).astype(np.int64)
        elif leaf == "co64":
            entry_count = struct.unpack_from(">I", body, 4)[0]
            track["offsets"] = np.frombuffer(
                body, dtype=">u8", count=entry_count, offset=8
            ).astype(np.int64)
        elif leaf == "stsc":
            entry_count = struct.unpack_from(">I", body, 4)[0]
            track["stsc"] = [
                struct.unpack_from(">3I", body, 8 + 12 * i) for i in range(entry_count)
            ]
    return tracks


def locate_djmd_track(mp4_file):
    """Locate the gyro data track in the open MP4 file ``mp4_file`` by its handler name
    and its fourcc, and return its index tables as a ``DjmdTrack``.

    Raises an error on a video with no gyro data track, a track with a handler name of
    the gyro data track and another fourcc, an absent sample size or chunk offset table,
    sample size and chunk offset tables of different lengths, and a sample to chunk table
    with any row that states other than one entry per chunk or a sample description
    other than the first. Each row is tested, since a table of several rows can state one
    entry per chunk throughout. The sample description is tested since the fourcc is
    read from the first sample description alone, so another one could name a fourcc
    this function never read. The error of a video with no gyro data track carries the
    handler names of the tracks the video holds."""
    mp4_file.seek(0, 2)
    boxes = list(_walk(mp4_file, 0, mp4_file.tell()))
    tracks = _parse_tracks(mp4_file, boxes)
    for track in tracks:
        if (
            track.get("handler_type") == "meta"
            and track.get("handler_name") in _GYRO_DATA_TRACK_HANDLER_NAMES
        ):
            if track.get("stsd_fourcc") != "djmd":
                raise errors.CannotReadError(
                    errors.ErrorCode.GYRO_DATA_TRACK_UNKNOWN_FORMAT,
                    handler_name=track.get("handler_name"),
                    fourcc=track.get("stsd_fourcc"),
                )
            stsc = track.get("stsc")
            if not stsc or any(
                per_chunk != 1 or sample_description != 1
                for _first, per_chunk, sample_description in stsc
            ):
                raise errors.CannotReadError(
                    errors.ErrorCode.GYRO_DATA_TRACK_UNSUPPORTED_LAYOUT, stsc=stsc
                )
            sizes, offsets = track.get("sizes"), track.get("offsets")
            if sizes is None or offsets is None:
                raise errors.CannotReadError(
                    errors.ErrorCode.GYRO_DATA_TRACK_INDEX_MISSING
                )
            if len(sizes) != len(offsets):
                raise errors.CannotReadError(
                    errors.ErrorCode.GYRO_DATA_TRACK_INDEX_INVALID,
                    size_count=len(sizes),
                    offset_count=len(offsets),
                )
            return DjmdTrack(
                telemetry_frame_offsets=offsets,
                telemetry_frame_sizes=sizes,
                handler_name=track["handler_name"],
            )
    raise errors.CannotReadError(
        errors.ErrorCode.GYRO_DATA_TRACK_MISSING,
        handler_names=[track.get("handler_name") for track in tracks],
    )


def _read_varint(telemetry_frame_bytes, position):
    value = shift = 0
    while True:
        byte = telemetry_frame_bytes[position]
        value |= (byte & 0x7F) << shift
        position += 1
        if not byte & 0x80:
            return value, position
        shift += 7


def protobuf_fields(telemetry_frame_bytes, start, end):
    """Yield ``(field_number, wire_type, value, span)`` for each top level Protocol
    Buffers field in ``telemetry_frame_bytes[start:end]``. A varint, wire type 0, yields
    its value and no span; a field of wire type 1, 2, or 5 yields the byte span of its
    value and no value. Raises an error on any other wire type."""
    position = start
    while position < end:
        key, position = _read_varint(telemetry_frame_bytes, position)
        field_number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, position = _read_varint(telemetry_frame_bytes, position)
            yield field_number, wire_type, value, None
        elif wire_type == 1:
            yield field_number, wire_type, None, (position, position + 8)
            position += 8
        elif wire_type == 2:
            length, position = _read_varint(telemetry_frame_bytes, position)
            yield field_number, wire_type, None, (position, position + length)
            position += length
        elif wire_type == 5:
            yield field_number, wire_type, None, (position, position + 4)
            position += 4
        else:
            raise errors.CannotReadError(
                errors.ErrorCode.TELEMETRY_FRAME_INVALID,
                byte=position,
                wire_type=wire_type,
            )


def submessage_span(telemetry_frame_bytes, span, field_number):
    """The value span of the first field ``field_number`` of wire type 2 inside
    ``span``, or None if there is none."""
    for number, wire_type, _value, value_span in protobuf_fields(
        telemetry_frame_bytes, *span
    ):
        if number == field_number and wire_type == 2:
            return value_span
    return None


def quaternion_message_spans(telemetry_frame_bytes):
    """The byte span of each quaternion message of the telemetry frame
    ``telemetry_frame_bytes``, through
    ``ProductMeta.frame_meta(3).imu_frame_meta(3).imu_attitude_after_fusion(2)`` and its
    repeated field ``attitude(3)``. Returns [] when the telemetry frame holds no
    orientation record."""
    frame_meta = submessage_span(
        telemetry_frame_bytes, (0, len(telemetry_frame_bytes)), 3
    )
    if frame_meta is None:
        return []
    imu_frame_meta = submessage_span(telemetry_frame_bytes, frame_meta, 3)
    if imu_frame_meta is None:
        return []
    orientation_record = submessage_span(telemetry_frame_bytes, imu_frame_meta, 2)
    if orientation_record is None:
        return []
    return [
        value_span
        for field_number, wire_type, _value, value_span in protobuf_fields(
            telemetry_frame_bytes, *orientation_record
        )
        if field_number == 3 and wire_type == 2
    ]


def find_stored_camera_model_name(telemetry_frame_bytes):
    """``(start, end, value)`` of the stored camera model name in the first telemetry
    frame ``telemetry_frame_bytes``, with the span relative to
    ``telemetry_frame_bytes``, through
    ``ProductMeta.clip_meta(1).clip_meta_header(1).product_name(10)``. Raises an error
    carrying the absent field on a telemetry frame that lacks a field of that path."""
    clip_meta = submessage_span(
        telemetry_frame_bytes, (0, len(telemetry_frame_bytes)), 1
    )
    if clip_meta is None:
        raise errors.CannotReadError(
            errors.ErrorCode.CAMERA_MODEL_NAME_MISSING, field="clip_meta"
        )
    header = submessage_span(telemetry_frame_bytes, clip_meta, 1)
    if header is None:
        raise errors.CannotReadError(
            errors.ErrorCode.CAMERA_MODEL_NAME_MISSING, field="clip_meta_header"
        )
    for field_number, wire_type, _value, value_span in protobuf_fields(
        telemetry_frame_bytes, *header
    ):
        if field_number == 10 and wire_type == 2:
            start, end = value_span
            return start, end, bytes(telemetry_frame_bytes[start:end])
    raise errors.CannotReadError(
        errors.ErrorCode.CAMERA_MODEL_NAME_MISSING, field="product_name"
    )


def decode_quaternion_message(telemetry_frame_bytes, span):
    """The quaternion message at ``span`` of ``telemetry_frame_bytes`` as a (4,) float64
    array ``[w, x, y, z]``. Raises an error on a message other than 20 bytes or with
    tags other than 0x0D, 0x15, 0x1D, and 0x25."""
    start, end = span
    if end - start != _QUATERNION_MESSAGE_LENGTH:
        raise errors.CannotReadError(
            errors.ErrorCode.STORED_ORIENTATION_WRONG_SIZE,
            size=end - start,
            expected=_QUATERNION_MESSAGE_LENGTH,
        )
    tags = [telemetry_frame_bytes[start + 5 * k] for k in range(4)]
    if tuple(tags) != _QUATERNION_MESSAGE_TAGS:
        raise errors.CannotReadError(
            errors.ErrorCode.STORED_ORIENTATION_UNKNOWN_LAYOUT,
            tags=tags,
            expected_tags=list(_QUATERNION_MESSAGE_TAGS),
        )
    return np.array(
        [
            struct.unpack_from("<f", telemetry_frame_bytes, start + 1 + 5 * k)[0]
            for k in range(4)
        ]
    )


def encode_quaternion_message(telemetry_frame_bytes, span, quaternion_wxyz):
    """Overwrite in place the four float32 values of the quaternion message at ``span``
    of the bytearray ``telemetry_frame_bytes`` with ``quaternion_wxyz``, leaving its
    tags as they are."""
    start = span[0]
    for k in range(4):
        struct.pack_into(
            "<f", telemetry_frame_bytes, start + 1 + 5 * k, float(quaternion_wxyz[k])
        )


def _hamilton_product(left, right):
    """Hamilton product of quaternions written (w, x, y, z), broadcast over the leading
    axes."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    w0, x0, y0, z0 = left[..., 0], left[..., 1], left[..., 2], left[..., 3]
    w1, x1, y1, z1 = right[..., 0], right[..., 1], right[..., 2], right[..., 3]
    return np.stack(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        axis=-1,
    )


def stored_to_camera_frame(quaternions_wxyz):
    """Stored quaternions, (w, x, y, z), converted into the camera frame of reference by
    the basis transform."""
    return _hamilton_product(
        _BASIS_TRANSFORM_LEFT,
        _hamilton_product(quaternions_wxyz, _BASIS_TRANSFORM_RIGHT),
    )


def camera_frame_to_stored(quaternions_wxyz):
    """Quaternions in the camera frame of reference, (w, x, y, z), converted back into
    the stored form by the reverse of the basis transform."""
    return _hamilton_product(
        _BASIS_TRANSFORM_LEFT_INVERSE,
        _hamilton_product(quaternions_wxyz, _BASIS_TRANSFORM_RIGHT_INVERSE),
    )


# The identity threshold in radians: the write treats a rotation from the stored
# orientation to the repaired orientation by a smaller angle as the identity and leaves
# its quaternion message as stored. The test is by angle, since that rotation of an
# unchanged quaternion is the identity only to floating point precision.
_IDENTITY_THRESHOLD_RAD = 1e-6


def _xyzw_to_wxyz(quaternions_xyzw):
    """Quaternions in the order (x, y, z, w) of scipy, reordered to (w, x, y, z)."""
    return np.asarray(quaternions_xyzw, dtype=np.float64)[..., [3, 0, 1, 2]]


def _quaternion_conjugate(quaternions_wxyz):
    """The conjugate of quaternions written (w, x, y, z), which is the inverse of a
    unit quaternion."""
    conjugate = np.array(quaternions_wxyz, dtype=np.float64)
    conjugate[..., 1:] *= -1.0
    return conjugate


def _stored_to_repaired_rotation_at_times(
    repaired_orientation_times_s, stored_to_repaired_rotations_wxyz, times
):
    """For the quaternion at each of ``times``, the rotation from its stored orientation
    to its repaired orientation: the value of ``stored_to_repaired_rotations_wxyz`` at
    the same time of the strictly increasing ``repaired_orientation_times_s``, negated
    where its w is negative, and divided by its norm.

    The stored track jumps to the opposite sign wherever w passes zero, and a repaired
    orientation need not take the sign of its stored orientation, so such a rotation can
    come out near the negation of the identity. Negating it keeps each written
    quaternion on the sign of its stored quaternion. The division by the norm is needed
    since a quaternion decoded from float32 values has a norm that differs from one by
    about 1e-7, enough to put that rotation of an unchanged quaternion above the
    identity threshold.

    Raises an error on a time that is not a time of ``repaired_orientation_times_s``."""
    rotations_wxyz = np.array(stored_to_repaired_rotations_wxyz, dtype=np.float64)
    negative_w = rotations_wxyz[:, 0] < 0.0
    rotations_wxyz[negative_w] *= -1.0
    index = np.searchsorted(repaired_orientation_times_s, times)
    among_repaired = index < repaired_orientation_times_s.size
    among_repaired[among_repaired] = (
        repaired_orientation_times_s[index[among_repaired]] == times[among_repaired]
    )
    if not among_repaired.all():
        raise errors.CannotRepairError(
            errors.ErrorCode.ORIENTATION_TIMES_NOT_AMONG_REPAIRED,
            count=int(np.count_nonzero(~among_repaired)),
        )
    at_times = rotations_wxyz[index]
    return at_times / np.linalg.norm(at_times, axis=1, keepdims=True)


def _telemetry_frames_to_rewrite(
    video, decoded, quaternion_stored_to_repaired_rotations, changed
):
    """Read the video at ``video`` without writing, run the tests of the write on its
    gyro data track, and return the byte position and the new bytes of each telemetry
    frame that the write changes, with the number of quaternion messages it overwrites.

    ``decoded`` holds the decoded quaternions in file order, and
    ``quaternion_stored_to_repaired_rotations`` and ``changed`` hold, for each of them,
    the rotation from its stored orientation to its repaired orientation and whether
    that rotation exceeds the identity threshold. Raises the errors that
    ``write_repaired_mp4`` lists for the input video."""
    decoded_count = decoded.shape[0]
    new_telemetry_frames = []
    quaternion_messages_overwritten = 0
    quaternion_count_in_file = 0
    with (
        errors.operating_system_error_as(errors.CannotReadError, video),
        video.open("rb") as mp4_file,
    ):
        track = locate_djmd_track(mp4_file)
        for telemetry_frame, (position, size) in enumerate(
            zip(track.telemetry_frame_offsets, track.telemetry_frame_sizes, strict=True)
        ):
            position = int(position)
            mp4_file.seek(position)
            telemetry_frame_bytes = bytearray(mp4_file.read(int(size)))
            modified = False

            if telemetry_frame == 0:
                start, end, stored_name = find_stored_camera_model_name(
                    telemetry_frame_bytes
                )
                new_name = repaired_camera_model_name(stored_name)
                if end - start != len(new_name):
                    raise errors.CannotRepairError(
                        errors.ErrorCode.CAMERA_MODEL_NAME_LENGTH_DIFFERS,
                        new_name=new_name.decode("latin1"),
                        new_length=len(new_name),
                        length=end - start,
                        model_name=stored_name.decode("latin1"),
                    )
                if stored_name != new_name:
                    telemetry_frame_bytes[start:end] = new_name
                    modified = True

            spans = quaternion_message_spans(telemetry_frame_bytes)
            if spans:
                stored = np.array(
                    [
                        decode_quaternion_message(telemetry_frame_bytes, span)
                        for span in spans
                    ]
                )
                # A comparison with a value that is not a number is false, so the dot
                # product test below would pass such a value.
                if np.isnan(stored).any():
                    raise errors.CannotRepairError(
                        errors.ErrorCode.STORED_ORIENTATION_NOT_A_NUMBER,
                        telemetry_frame=telemetry_frame,
                    )
                camera_frame = stored_to_camera_frame(stored)
                if (camera_frame == 0.0).all(axis=1).any():
                    raise errors.CannotRepairError(
                        errors.ErrorCode.STORED_ORIENTATION_ALL_ZEROS,
                        telemetry_frame=telemetry_frame,
                    )
                count = len(spans)
                if quaternion_count_in_file + count > decoded_count:
                    raise errors.CannotRepairError(
                        errors.ErrorCode.STORED_ORIENTATIONS_MORE_THAN_DECODED,
                        decoded=decoded_count,
                        telemetry_frame=telemetry_frame,
                    )
                absolute_dot_product = np.abs(
                    np.sum(
                        camera_frame
                        * decoded[
                            quaternion_count_in_file : quaternion_count_in_file + count
                        ],
                        axis=1,
                    )
                )
                if absolute_dot_product.min() < _MINIMUM_ABSOLUTE_DOT_PRODUCT:
                    raise errors.CannotRepairError(
                        errors.ErrorCode.ORIENTATION_DIFFERS_FROM_DECODED,
                        telemetry_frame=telemetry_frame,
                        dot=float(absolute_dot_product.min()),
                        minimum=_MINIMUM_ABSOLUTE_DOT_PRODUCT,
                    )
                for index_in_telemetry_frame, span in enumerate(spans):
                    decoded_index = quaternion_count_in_file + index_in_telemetry_frame
                    if changed[decoded_index]:
                        repaired = _hamilton_product(
                            camera_frame[index_in_telemetry_frame],
                            quaternion_stored_to_repaired_rotations[decoded_index],
                        )
                        encode_quaternion_message(
                            telemetry_frame_bytes,
                            span,
                            camera_frame_to_stored(repaired),
                        )
                        quaternion_messages_overwritten += 1
                        modified = True
                quaternion_count_in_file += count

            if modified:
                new_telemetry_frames.append((position, bytes(telemetry_frame_bytes)))

    if quaternion_count_in_file == 0:
        raise errors.CannotRepairError(errors.ErrorCode.STORED_ORIENTATIONS_MISSING)
    if quaternion_count_in_file != decoded_count:
        raise errors.CannotRepairError(
            errors.ErrorCode.STORED_ORIENTATIONS_FEWER_THAN_DECODED,
            count=quaternion_count_in_file,
            decoded=decoded_count,
        )
    return new_telemetry_frames, quaternion_messages_overwritten


def write_repaired_mp4(video, output_path, repaired_orientations):
    """Write to ``output_path`` the repaired video of the video at ``video``, and return
    a ``WritebackReport``.

    ``repaired_orientations``, a ``RepairedOrientations``, gives the stored and the
    repaired orientation at each time of
    ``repaired_orientations.repaired_orientation_times_s``. The write takes the
    rotation of each quaternion of the video from its stored orientation to its
    repaired orientation at that quaternion's own time, so it writes each repaired
    orientation into the time slot of its quaternion and moves no quaternion in time. It
    overwrites each quaternion message where that rotation exceeds the identity
    threshold with its quaternion in the camera frame of reference times that rotation,
    converted back into the stored form, and it rewrites the stored camera model name
    with the model name marker unless the name already carries it. Any other file at
    ``output_path`` is replaced. After the write the equal length assertion tests that
    the repaired video has the size of the input video.

    The write raises an error on each of these before it writes any byte, leaving
    ``output_path`` as it was:

    - an output path that resolves to the input video;
    - a quaternion time that is not a time of
      ``repaired_orientations.repaired_orientation_times_s``;
    - the errors of ``locate_djmd_track``: a video with no gyro data track, a track
      with a handler name of the gyro data track and a fourcc other than djmd, an absent
      sample size or chunk offset table, sample size and chunk offset tables of
      different lengths, and an absent sample to chunk table or one with a row that
      states other than one telemetry frame per chunk or a sample description other
      than the first;
    - a Protocol Buffers wire type other than 0, 1, 2, or 5 in a telemetry frame;
    - a first telemetry frame that lacks ``clip_meta``, ``clip_meta_header``, or
      ``product_name``;
    - a stored camera model name that carries neither of its two affixes, or whose new
      name has another length;
    - a quaternion message other than 20 bytes or with an unexpected tag;
    - a value that is not a number, and a quaternion of all zeros;
    - a count mismatch between the quaternions of the file and the decoded quaternions,
      and a quaternion message that does not match its decoded quaternion;
    - a gyro data track that holds no orientations.

    What each error carries as ``file``:

    - an error of a test of the input video or of ``repaired_orientations`` carries no
      ``file``, and the caller adds the input video;
    - an error of the operating system while it reads the input video carries
      ``video``;
    - an error of the operating system while it copies the input video to
      ``output_path`` or writes into the copy, the error of an output path that resolves
      to the input video, and the error of the equal length assertion carry
      ``output_path``."""
    video = Path(video)
    output_path = Path(output_path)
    if video.resolve() == output_path.resolve():
        raise errors.CannotWriteError(
            errors.ErrorCode.OUTPUT_IS_VIDEO, file=output_path, video=video
        )

    decoded = np.asarray(repaired_orientations.track_quaternions_wxyz, dtype=np.float64)
    stored_to_repaired_rotations_wxyz = _hamilton_product(
        _quaternion_conjugate(
            _xyzw_to_wxyz(repaired_orientations.stored_orientations_xyzw)
        ),
        _xyzw_to_wxyz(repaired_orientations.repaired_orientations_xyzw),
    )
    quaternion_stored_to_repaired_rotations = _stored_to_repaired_rotation_at_times(
        np.asarray(
            repaired_orientations.repaired_orientation_times_s, dtype=np.float64
        ),
        stored_to_repaired_rotations_wxyz,
        np.asarray(repaired_orientations.track_times, dtype=np.float64),
    )
    angle = 2.0 * np.arccos(
        np.clip(np.abs(quaternion_stored_to_repaired_rotations[:, 0]), 0.0, 1.0)
    )
    changed = angle > _IDENTITY_THRESHOLD_RAD

    new_telemetry_frames, quaternion_messages_overwritten = (
        _telemetry_frames_to_rewrite(
            video, decoded, quaternion_stored_to_repaired_rotations, changed
        )
    )

    with errors.operating_system_error_as(errors.CannotReadError, video):
        video_size_bytes = video.stat().st_size
    with errors.operating_system_error_as(errors.CannotWriteError, output_path):
        shutil.copyfile(video, output_path)
        with output_path.open("r+b") as repaired_file:
            for position, new_bytes in new_telemetry_frames:
                repaired_file.seek(position)
                repaired_file.write(new_bytes)
        repaired_size_bytes = output_path.stat().st_size
    if repaired_size_bytes != video_size_bytes:
        raise errors.CannotWriteError(
            errors.ErrorCode.REPAIRED_VIDEO_SIZE_DIFFERS,
            file=output_path,
            size=repaired_size_bytes,
            expected=video_size_bytes,
        )

    return WritebackReport(
        quaternion_messages_overwritten=quaternion_messages_overwritten,
        telemetry_frames_rewritten=len(new_telemetry_frames),
        video_size_bytes=video_size_bytes,
    )
