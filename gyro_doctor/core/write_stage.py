# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The write stage, which writes the orientations of the repair result into a copy of the
video, each at the time slot of its quaternion, and the cache writer and reader of the write
record, which locates that copy for ``gyro-doctor-render``.
"""

import json
from pathlib import Path

import numpy as np

from gyro_doctor import errors
from gyro_doctor.core import decode_stage, mp4_writeback, telemetry


def _wxyz_to_xyzw(quaternions_wxyz):
    quaternions_wxyz = np.asarray(quaternions_wxyz, dtype=np.float64)
    return np.column_stack(
        [
            quaternions_wxyz[:, 1],
            quaternions_wxyz[:, 2],
            quaternions_wxyz[:, 3],
            quaternions_wxyz[:, 0],
        ]
    )


def write(video, repair_result, output_path, *, force):
    """The write stage: write the repaired video of the video at ``video`` to
    ``output_path``, from ``repair_result``, and return the report of the write.

    Raises an error on a repair result whose quaternion times differ from the times of
    the video's own quaternions ordered as the decode stage orders them, since such a
    repair result was not made from this video. Raises an error on a video whose stored
    camera model name ends in ``"-Dr."`` unless ``force`` is set. Every error it raises
    that concerns the video carries the path of the video as ``file``."""
    video = Path(video)
    track = telemetry.read_gyro_data_track(video, force=force)
    order = decode_stage.time_order(track.quaternion_times_s)
    track_times = track.quaternion_times_s[order]
    times = np.asarray(repair_result["quaternion_times_s"], dtype=np.float64)
    if track_times.size != times.size:
        raise errors.CannotRepairError(
            errors.ErrorCode.REPAIR_RESULT_ORIENTATION_COUNT_DIFFERS,
            file=video,
            result_count=times.size,
            track_count=track_times.size,
        )
    if not np.array_equal(track_times, times):
        raise errors.CannotRepairError(
            errors.ErrorCode.REPAIR_RESULT_ORIENTATION_TIMES_DIFFER,
            file=video,
            count=int(np.count_nonzero(track_times != times)),
        )

    keep = np.asarray(repair_result["kept_after_time_collisions"], dtype=bool)
    # The repair result holds the rebuilt track only, so the stored orientations come
    # from the video itself.
    stored = track.quaternions_wxyz[order][keep]
    repaired = np.asarray(repair_result["quaternions_wxyz"], dtype=np.float64)[keep]
    repaired_orientations = mp4_writeback.RepairedOrientations(
        track_quaternions_wxyz=track.quaternions_wxyz,
        track_times=track.quaternion_times_s,
        repaired_orientation_times_s=times[keep],
        stored_orientations_xyzw=_wxyz_to_xyzw(stored),
        repaired_orientations_xyzw=_wxyz_to_xyzw(repaired),
    )
    try:
        return mp4_writeback.write_repaired_mp4(
            video, output_path, repaired_orientations
        )
    except errors.GyroDoctorError as error:
        error.values.setdefault("file", video)
        raise


def write_record_path(video_cache_directory, video_name):
    """The path of the write record of the video with file stem ``video_name`` in
    ``video_cache_directory``."""
    return Path(video_cache_directory) / f"{video_name}-write-record.json"


def store_write_record(
    video_cache_directory, video_name, repaired_video, repair_result_path
):
    """The cache writer of the write record: once the write stage has written the repaired
    video at ``repaired_video``, store in ``video_cache_directory`` as
    ``<video file stem>-write-record.json`` the absolute path and the file stamp of the
    repaired video, and the file stamp of the repair result at ``repair_result_path`` it
    was written from.

    The write record is a file of its own because the write stage writes nothing into the
    repair record, and the repair record file is written before the repaired video
    exists. An error of the operating system while it makes the file stamps or writes
    the write record raises an error carrying the path of the write record."""
    path = write_record_path(video_cache_directory, video_name)
    with errors.operating_system_error_as(errors.CannotWriteError, path):
        record = {
            "repaired_video": str(Path(repaired_video).resolve()),
            "file_stamps": {
                "repaired_video": decode_stage.file_stamp(repaired_video),
                "repair_result": decode_stage.file_stamp(repair_result_path),
            },
        }
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def read_write_record(path):
    """The cache reader of the write record: the write record stored at ``path``, with the
    path of the repaired video under ``repaired_video`` and the file stamps of the repaired
    video and of the repair result under ``file_stamps``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
