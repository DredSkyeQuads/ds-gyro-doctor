# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""The container probe through PyAV: the video frame rate, and the video frames that are
keyframes, read without an external ffmpeg binary.
"""

from pathlib import Path

import av


def _video_frame_rate_of_stream(video_stream):
    rate = video_stream.average_rate or video_stream.guessed_rate
    if rate is None:
        return None
    return float(rate)


def keyframe_indices(path, video_frame_rate):
    """The video frame of every keyframe of the first video stream of the container at
    ``path``, in display order, from the keyframe flag of each packet, with no decode of
    the picture.

    The result depends on the file alone and assumes no keyframe interval.
    ``video_frame_rate`` maps the presentation time of each packet to its video frame.
    Raises ``av.FFmpegError`` on a container it cannot read."""
    keyframes = []
    with av.open(str(Path(path))) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        for packet in container.demux(stream):
            if packet.is_keyframe and packet.pts is not None:
                keyframes.append(
                    round(float(packet.pts * time_base) * video_frame_rate)
                )
    return sorted(keyframes)


def video_frame_rate(path):
    """The video frame rate of the first video stream of the container at ``path``; None
    where the container holds no video stream or the stream reports neither an average
    nor a guessed rate. Raises ``av.FFmpegError`` on a container it cannot read."""
    with av.open(str(Path(path))) as container:
        video_stream = next(
            (stream for stream in container.streams if stream.type == "video"), None
        )
        return (
            _video_frame_rate_of_stream(video_stream)
            if video_stream is not None
            else None
        )
