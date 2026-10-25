# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""Every error the product raises on purpose and its code, in the one module that holds
the words of every error, warning, and hint line that ``gyro-doctor``,
``gyro-doctor-chart``, and ``gyro-doctor-render`` print. An error is one of five
classes, one for each "what failed" phrase that the commands print, and carries a code,
a member of ``ErrorCode``, and every value that locates the problem in ``error.values``,
so a test asserts an error's class, code, and values, never its words. A value that
only the caller of the raising function knows, such as the video path for a test inside
a stage, the caller adds to ``error.values`` before it raises the error again.
"""

import contextlib
from enum import Enum, auto


class ErrorCode(Enum):
    """One member per distinct problem, each named by the problem. The comments group
    the codes by what they concern; the class of each error is chosen where it is
    raised, so one code can stand under more than one class."""

    # any file
    SYSTEM_ERROR = auto()

    # the video and its gyro data track
    VIDEO_NOT_A_FILE = auto()
    VIDEO_INVALID = auto()
    GYRO_DATA_TRACK_MISSING = auto()
    GYRO_DATA_TRACK_UNKNOWN_FORMAT = auto()
    GYRO_DATA_TRACK_UNSUPPORTED_LAYOUT = auto()
    GYRO_DATA_TRACK_INDEX_MISSING = auto()
    GYRO_DATA_TRACK_INDEX_INVALID = auto()
    TELEMETRY_FRAME_INVALID = auto()
    CAMERA_MODEL_NAME_MISSING = auto()
    FRAME_RATE_MISSING = auto()
    VIDEO_FRAME_RATE_MISSING = auto()
    STORED_ORIENTATION_WRONG_SIZE = auto()
    STORED_ORIENTATION_UNKNOWN_LAYOUT = auto()
    STORED_ORIENTATION_NOT_A_NUMBER = auto()
    STORED_ORIENTATION_ALL_ZEROS = auto()
    KEYFRAME_MISSING_BEFORE_VIDEO_FRAME = auto()
    SEEK_MISSED_KEYFRAME = auto()

    # a cache file, a Gyroflow project, or a Gyroflow render
    CACHED_RESULT_NOT_FOUND = auto()
    CACHE_FILE_INVALID = auto()
    VIDEO_CHANGED_SINCE_RESULT_CACHED = auto()
    CACHE_VERSION_DIFFERS = auto()
    CACHED_RESULT_PARTS_MISSING = auto()
    STAMPED_FILE_NOT_FOUND = auto()
    STAMPED_FILE_CHANGED = auto()
    STORED_PERIOD_DIFFERS = auto()
    RATE_SERIES_UNEQUAL = auto()
    CLASSIFICATION_UNKNOWN = auto()
    CLASSIFICATION_WITHOUT_COLOR = auto()
    SHOW_NOISE_RATIO_WITHOUT_DETECTION_RESULT = auto()
    NO_MODE_READY = auto()
    GYROFLOW_PROJECT_NOT_FOUND = auto()
    GYROFLOW_PROJECT_INVALID = auto()
    GYROFLOW_PROJECT_VIDEO_DIFFERS = auto()
    GYROFLOW_PROJECT_GYRO_SOURCE_DIFFERS = auto()
    GYROFLOW_PROJECT_INTEGRATION_METHOD_DIFFERS = auto()
    GYROFLOW_PROJECT_CARRIES_INLINE_MOTION = auto()
    GYROFLOW_PROJECT_TRIM_DIFFERS = auto()
    GYROFLOW_PROJECT_STABILIZATION_SETTINGS_DIFFER = auto()
    GYROFLOW_RENDER_TOO_FEW_VIDEO_FRAMES = auto()

    # the tests of the stages and of the write
    ALREADY_REPAIRED = auto()
    TELEMETRY_FRAMES_MISSING = auto()
    STORED_ORIENTATIONS_MISSING = auto()
    ORIENTATIONS_FEWER_THAN_TWO_GROUPS = auto()
    PERIOD_TEST_FAILED = auto()
    EXTENTS_OVERLAP = auto()
    EXTENTS_TOUCH = auto()
    PEAK_OUTSIDE_EXTENT = auto()
    EVENT_OUTSIDE_EXTENTS = auto()
    LOCAL_MEDIAN_WINDOWS_WITHOUT_WEIGHT = auto()
    LOCAL_MEDIAN_ZERO_SCALE = auto()
    FULL_TRACK_ROBUST_STANDARD_DEVIATION_NOT_ABOVE_ZERO = auto()
    RATE_SAMPLES_CHANGED_OUTSIDE_EXTENTS = auto()
    AXIS_OUTSIDE_REPAIR_AXES_CHANGED = auto()
    REMOVED_AXIS_NOT_ZERO = auto()
    EXTENT_END_CHANGED = auto()
    DEPARTURE_SCALED_OUTSIDE_GAIN = auto()
    INTERIOR_ABOVE_CEILING = auto()
    ROUND_TRIP_IDENTITY_FAILED = auto()
    CADENCE_PRESERVATION_FAILED = auto()
    REBUILD_SERIES_UNKNOWN = auto()
    RECORDED_RATE_REBUILD_GIVEN_PERIOD = auto()
    NORMALIZED_RATE_REBUILD_WITHOUT_PERIOD = auto()
    REBUILD_PERIOD_DIFFERS = auto()
    ORIENTATION_TIMES_NOT_AMONG_REPAIRED = auto()
    CAMERA_MODEL_NAME_UNKNOWN_FORM = auto()
    CAMERA_MODEL_NAME_LENGTH_DIFFERS = auto()
    STORED_ORIENTATIONS_MORE_THAN_DECODED = auto()
    STORED_ORIENTATIONS_FEWER_THAN_DECODED = auto()
    ORIENTATION_DIFFERS_FROM_DECODED = auto()
    REPAIR_RESULT_ORIENTATION_COUNT_DIFFERS = auto()
    REPAIR_RESULT_ORIENTATION_TIMES_DIFFER = auto()

    # an output: the repaired video, a cache file, a chart, a render, or a composite
    OUTPUT_IS_VIDEO = auto()
    OUTPUT_IS_DIRECTORY = auto()
    OUTPUT_DIRECTORY_NOT_FOUND = auto()
    OUTPUT_NOT_PNG = auto()
    REPAIRED_VIDEO_SIZE_DIFFERS = auto()
    GYROFLOW_NOT_SET = auto()
    GYROFLOW_NOT_A_FILE = auto()
    GYROFLOW_WROTE_NO_RENDER = auto()
    RECODE_STARTS_AT_OTHER_VIDEO_FRAME = auto()
    RECODE_VIDEO_FRAME_COUNT_DIFFERS = auto()

    # the --do-not-repair file
    DO_NOT_REPAIR_FILE_NOT_FOUND = auto()
    DO_NOT_REPAIR_FILE_IS_DIRECTORY = auto()
    DO_NOT_REPAIR_FILE_NOT_UTF8 = auto()

    # an option value, tested after parsing
    VALUE_NOT_ABOVE_ZERO = auto()
    TIME_NOT_ABOVE_ZERO = auto()
    TIME_BELOW_ZERO = auto()
    CEILING_BELOW_ZERO = auto()
    START_PAST_LAST_VIDEO_FRAME = auto()
    DURATION_WITHOUT_VIDEO_FRAMES = auto()
    DURATION_PAST_LAST_VIDEO_FRAME = auto()

    # the reason after "argument --OPTION:" that argparse prints for a type function
    VALUE_NOT_A_NUMBER = auto()
    VALUE_NOT_FINITE = auto()
    THRESHOLD_VALUE_COUNT_INVALID = auto()
    AXIS_UNKNOWN = auto()
    AXIS_REPEATED = auto()
    VALUE_NOT_A_POSITION = auto()
    VALUE_NOT_AN_IDENTIFIER = auto()
    VALUE_NOT_A_RANGE = auto()
    RANGE_REVERSED = auto()
    VALUE_NOT_A_DURATION = auto()
    VALUE_NOT_A_RATE_AXIS_SCALE = auto()
    VALUE_NOT_A_CLASSIFICATION_AND_COLOR = auto()

    # two options
    STOP_OPTION_WITHOUT_CACHE = auto()
    REPAIR_AXIS_NOT_DETECTED = auto()

    # warnings
    REPAIR_AXES_NARROWED_TO_DETECTED = auto()
    GYRO_MOTION_REMOVED_FROM_EVERY_AXIS = auto()
    DO_NOT_REPAIR_ELEMENT_INVALID = auto()
    DO_NOT_REPAIR_LINE_INVALID = auto()
    DO_NOT_REPAIR_ELEMENT_NAMES_NO_EXTENT = auto()
    DO_NOT_REPAIR_LINE_NAMES_NO_EXTENT = auto()
    DO_NOT_REPAIR_ELEMENT_ON_REMOVED_AXIS = auto()
    DO_NOT_REPAIR_LINE_ON_REMOVED_AXIS = auto()


MESSAGES = {
    ErrorCode.SYSTEM_ERROR: "{reason}",
    ErrorCode.VIDEO_NOT_A_FILE: "not a file",
    ErrorCode.VIDEO_INVALID: "invalid video, {reason}",
    ErrorCode.GYRO_DATA_TRACK_MISSING: "no gyro data track",
    ErrorCode.GYRO_DATA_TRACK_UNKNOWN_FORMAT: ("gyro data track in unknown format"),
    ErrorCode.GYRO_DATA_TRACK_UNSUPPORTED_LAYOUT: (
        "gyro data track stored in unsupported layout"
    ),
    ErrorCode.GYRO_DATA_TRACK_INDEX_MISSING: "gyro data track index missing",
    ErrorCode.GYRO_DATA_TRACK_INDEX_INVALID: ("gyro data track index invalid"),
    ErrorCode.TELEMETRY_FRAME_INVALID: "invalid telemetry frame",
    ErrorCode.CAMERA_MODEL_NAME_MISSING: "no camera model name in gyro data track",
    ErrorCode.FRAME_RATE_MISSING: "no frame rate in first telemetry frame",
    ErrorCode.VIDEO_FRAME_RATE_MISSING: "no video frame rate",
    ErrorCode.STORED_ORIENTATION_WRONG_SIZE: (
        "stored orientation of {size} bytes, expected {expected} bytes"
    ),
    ErrorCode.STORED_ORIENTATION_UNKNOWN_LAYOUT: "stored orientation in unknown layout",
    ErrorCode.STORED_ORIENTATION_NOT_A_NUMBER: (
        "stored orientation not a number in telemetry frame {telemetry_frame}"
    ),
    ErrorCode.STORED_ORIENTATION_ALL_ZEROS: (
        "stored orientation of all zeros in telemetry frame {telemetry_frame}"
    ),
    ErrorCode.KEYFRAME_MISSING_BEFORE_VIDEO_FRAME: (
        "no keyframe at or before video frame f{video_frame}"
    ),
    ErrorCode.SEEK_MISSED_KEYFRAME: (
        "seek to keyframe at video frame f{keyframe} landed on video frame f{landed}"
    ),
    ErrorCode.CACHED_RESULT_NOT_FOUND: "file not found",
    ErrorCode.CACHE_FILE_INVALID: "invalid cache file",
    ErrorCode.VIDEO_CHANGED_SINCE_RESULT_CACHED: "{video} changed since result cached",
    ErrorCode.CACHE_VERSION_DIFFERS: (
        "cache version {stored_version}, expected {expected_version}"
    ),
    ErrorCode.CACHED_RESULT_PARTS_MISSING: "{count} parts missing",
    ErrorCode.STAMPED_FILE_NOT_FOUND: "made from {stamped}, file not found",
    ErrorCode.STAMPED_FILE_CHANGED: "made from {stamped}, changed since",
    ErrorCode.STORED_PERIOD_DIFFERS: (
        "stored period T {stored_us:.4f} us, expected {derived_us:.4f} us"
    ),
    ErrorCode.RATE_SERIES_UNEQUAL: (
        "Normalized Rate and recorded rate cover different stored orientations"
    ),
    ErrorCode.CLASSIFICATION_UNKNOWN: "unknown classification {classifications}",
    ErrorCode.CLASSIFICATION_WITHOUT_COLOR: (
        "classification {classifications} without color"
    ),
    ErrorCode.SHOW_NOISE_RATIO_WITHOUT_DETECTION_RESULT: (
        "--show-noise-ratio needs detection result, none in cache passes tests"
    ),
    ErrorCode.NO_MODE_READY: "no detection result and no repair result",
    ErrorCode.GYROFLOW_PROJECT_NOT_FOUND: "file not found",
    ErrorCode.GYROFLOW_PROJECT_INVALID: "invalid Gyroflow project",
    ErrorCode.GYROFLOW_PROJECT_VIDEO_DIFFERS: (
        "names video file {named}, expected {video}"
    ),
    ErrorCode.GYROFLOW_PROJECT_GYRO_SOURCE_DIFFERS: (
        "reads gyro data from {named}, expected {video}"
    ),
    ErrorCode.GYROFLOW_PROJECT_INTEGRATION_METHOD_DIFFERS: (
        "integrates with method {method}, expected {expected}"
    ),
    ErrorCode.GYROFLOW_PROJECT_CARRIES_INLINE_MOTION: "carries inline motion data",
    ErrorCode.GYROFLOW_PROJECT_TRIM_DIFFERS: (
        "trims video to {trim} ms, expected {expected} ms"
    ),
    ErrorCode.GYROFLOW_PROJECT_STABILIZATION_SETTINGS_DIFFER: (
        "stabilization settings differ from preset {preset}"
    ),
    ErrorCode.GYROFLOW_RENDER_TOO_FEW_VIDEO_FRAMES: (
        "{count} video frames, expected at least {expected}"
    ),
    ErrorCode.ALREADY_REPAIRED: (
        "repaired by gyro-doctor before, camera model name {model_name}"
    ),
    ErrorCode.TELEMETRY_FRAMES_MISSING: "gyro data track empty",
    ErrorCode.STORED_ORIENTATIONS_MISSING: "no stored orientations in gyro data track",
    ErrorCode.ORIENTATIONS_FEWER_THAN_TWO_GROUPS: (
        "{count} stored orientations, expected at least {expected}"
    ),
    ErrorCode.PERIOD_TEST_FAILED: (
        "period T {period_us:.4f} us, expected {counted_us:.4f} us"
        " from counted measurement rate within {tolerance_ns:.0f} ns"
    ),
    ErrorCode.EXTENTS_OVERLAP: (
        "{axis} extents at rate samples {first}-{last} and {next_first}-{next_last}"
        " overlap"
    ),
    ErrorCode.EXTENTS_TOUCH: (
        "{axis} extents at rate samples {first}-{last} and {next_first}-{next_last}"
        " touch with same classification"
    ),
    ErrorCode.PEAK_OUTSIDE_EXTENT: (
        "{axis} extent at rate samples {first}-{last} has peak at rate sample {peak}"
    ),
    ErrorCode.EVENT_OUTSIDE_EXTENTS: (
        "{axis} gyro noise event found at rate sample {rate_sample} in no extent"
    ),
    ErrorCode.LOCAL_MEDIAN_WINDOWS_WITHOUT_WEIGHT: (
        "local median fit failed on {axis}"
    ),
    ErrorCode.LOCAL_MEDIAN_ZERO_SCALE: ("local median fit failed on {axis}"),
    ErrorCode.FULL_TRACK_ROBUST_STANDARD_DEVIATION_NOT_ABOVE_ZERO: (
        "{axis} full track Robust Standard Deviation {value}, expected more than 0"
    ),
    ErrorCode.RATE_SAMPLES_CHANGED_OUTSIDE_EXTENTS: (
        "repair changed {count} rate samples outside every extent,"
        " first at rate sample {rate_sample}"
    ),
    ErrorCode.AXIS_OUTSIDE_REPAIR_AXES_CHANGED: (
        "repair changed {count} rate samples of {axis} outside --repair-axes,"
        " first at rate sample {rate_sample}"
    ),
    ErrorCode.REMOVED_AXIS_NOT_ZERO: (
        "{count} rate samples of {axis} in --remove-gyro-motion not 0 degrees per"
        " second, first at rate sample {rate_sample}"
    ),
    ErrorCode.EXTENT_END_CHANGED: (
        "repair changed first or last rate sample of extent at rate sample"
        " {rate_sample}"
    ),
    ErrorCode.DEPARTURE_SCALED_OUTSIDE_GAIN: (
        "extent at rate sample {rate_sample} scaled by {low:.6f} to {high:.6f},"
        " expected {gain:.6f} to 1"
    ),
    ErrorCode.INTERIOR_ABOVE_CEILING: (
        "{count} extents above ceiling between fades, first at rate sample"
        " {rate_sample}"
    ),
    ErrorCode.ROUND_TRIP_IDENTITY_FAILED: ("orientation rebuild failed"),
    ErrorCode.CADENCE_PRESERVATION_FAILED: ("orientation rebuild failed"),
    ErrorCode.REBUILD_SERIES_UNKNOWN: "orientation rebuild failed",
    ErrorCode.RECORDED_RATE_REBUILD_GIVEN_PERIOD: ("orientation rebuild failed"),
    ErrorCode.NORMALIZED_RATE_REBUILD_WITHOUT_PERIOD: ("orientation rebuild failed"),
    ErrorCode.REBUILD_PERIOD_DIFFERS: ("orientation rebuild failed"),
    ErrorCode.ORIENTATION_TIMES_NOT_AMONG_REPAIRED: (
        "{count} stored orientation times not among repaired orientation times"
    ),
    ErrorCode.CAMERA_MODEL_NAME_UNKNOWN_FORM: (
        "camera model name {model_name} without DJI prefix or -Dr. marker"
    ),
    ErrorCode.CAMERA_MODEL_NAME_LENGTH_DIFFERS: (
        "new camera model name {new_name} {new_length} bytes, expected {length} bytes"
    ),
    ErrorCode.STORED_ORIENTATIONS_MORE_THAN_DECODED: (
        "more stored orientations than expected {decoded}, by telemetry frame"
        " {telemetry_frame}"
    ),
    ErrorCode.STORED_ORIENTATIONS_FEWER_THAN_DECODED: (
        "{count} stored orientations, expected {decoded} decoded"
    ),
    ErrorCode.ORIENTATION_DIFFERS_FROM_DECODED: (
        "stored orientation in telemetry frame {telemetry_frame} differs from decoded"
        " orientation"
    ),
    ErrorCode.REPAIR_RESULT_ORIENTATION_COUNT_DIFFERS: (
        "repair result holds {result_count} stored orientations, expected"
        " {track_count}, made from another video"
    ),
    ErrorCode.REPAIR_RESULT_ORIENTATION_TIMES_DIFFER: (
        "{count} orientation times of repair result differ from gyro data track"
    ),
    ErrorCode.OUTPUT_IS_VIDEO: "same file as video {video}",
    ErrorCode.OUTPUT_IS_DIRECTORY: "is a directory",
    ErrorCode.OUTPUT_DIRECTORY_NOT_FOUND: "directory {directory} not found",
    ErrorCode.OUTPUT_NOT_PNG: "expected file name ending in .png",
    ErrorCode.REPAIRED_VIDEO_SIZE_DIFFERS: "{size} bytes, expected {expected} bytes",
    ErrorCode.GYROFLOW_NOT_SET: "GYRO_DOCTOR_GYROFLOW not set",
    ErrorCode.GYROFLOW_NOT_A_FILE: "GYRO_DOCTOR_GYROFLOW names {path}, not a file",
    ErrorCode.GYROFLOW_WROTE_NO_RENDER: "Gyroflow wrote no file from {project}",
    ErrorCode.RECODE_STARTS_AT_OTHER_VIDEO_FRAME: (
        "first video frame f{first}, expected f{expected}"
    ),
    ErrorCode.RECODE_VIDEO_FRAME_COUNT_DIFFERS: (
        "{count} video frames, expected {expected}"
    ),
    ErrorCode.DO_NOT_REPAIR_FILE_NOT_FOUND: "file not found",
    ErrorCode.DO_NOT_REPAIR_FILE_IS_DIRECTORY: "is a directory",
    ErrorCode.DO_NOT_REPAIR_FILE_NOT_UTF8: "not UTF-8 text",
    ErrorCode.VALUE_NOT_ABOVE_ZERO: "expected more than 0",
    ErrorCode.TIME_NOT_ABOVE_ZERO: "expected more than 0 milliseconds",
    ErrorCode.TIME_BELOW_ZERO: "expected 0 milliseconds or more",
    ErrorCode.CEILING_BELOW_ZERO: "expected 0 or more, or inf",
    ErrorCode.START_PAST_LAST_VIDEO_FRAME: "past last video frame f{last}",
    ErrorCode.DURATION_WITHOUT_VIDEO_FRAMES: "no video frame",
    ErrorCode.DURATION_PAST_LAST_VIDEO_FRAME: (
        "ends at video frame f{end}, past last video frame f{last}"
    ),
    ErrorCode.VALUE_NOT_A_NUMBER: "{text} not a number",
    ErrorCode.VALUE_NOT_FINITE: "{text} not a finite number",
    ErrorCode.THRESHOLD_VALUE_COUNT_INVALID: (
        "{text} holds {count} values, expected 1, or 3 for pitch,yaw,roll"
    ),
    ErrorCode.AXIS_UNKNOWN: "{text} names {axes}, expected pitch, yaw, or roll",
    ErrorCode.AXIS_REPEATED: "{text} names {axes} more than once",
    ErrorCode.VALUE_NOT_A_POSITION: (
        "{text} not a position, expected time such as 75s or video frame such as f4500"
    ),
    ErrorCode.VALUE_NOT_AN_IDENTIFIER: (
        "{text} not an identifier, expected axis letter and video frame such as Y1502"
    ),
    ErrorCode.VALUE_NOT_A_RANGE: (
        "{text} not a range, expected two positions with hyphen between"
        " such as f4500-f4620"
    ),
    ErrorCode.RANGE_REVERSED: "{text} has second position before first",
    ErrorCode.VALUE_NOT_A_DURATION: (
        "{text} not a duration, expected time such as 15s"
        " or count of video frames such as 360f"
    ),
    ErrorCode.VALUE_NOT_A_RATE_AXIS_SCALE: (
        "{text} not a rate axis scale, expected 1 or 3 values in degrees per second"
        " above 0, or peraxis"
    ),
    ErrorCode.VALUE_NOT_A_CLASSIFICATION_AND_COLOR: (
        "{text} not a classification and color such as impulse event=#1f4fd6"
    ),
    ErrorCode.STOP_OPTION_WITHOUT_CACHE: "no cache to hold {stage} result",
    ErrorCode.REPAIR_AXIS_NOT_DETECTED: "{absent} not in detection result",
    ErrorCode.REPAIR_AXES_NARROWED_TO_DETECTED: (
        "{absent} not in detection result: --detect-axes {detect_axes};"
        " cannot repair {absent}"
    ),
    ErrorCode.GYRO_MOTION_REMOVED_FROM_EVERY_AXIS: (
        "--remove-gyro-motion names every axis: repaired video holds no gyro motion"
        " for Gyroflow to stabilize, removing all three"
    ),
    ErrorCode.DO_NOT_REPAIR_ELEMENT_INVALID: "invalid --do-not-repair-list: {reason}; ignored",
    ErrorCode.DO_NOT_REPAIR_LINE_INVALID: (
        "invalid --do-not-repair {file} line {line}: {reason}; ignored"
    ),
    ErrorCode.DO_NOT_REPAIR_ELEMENT_NAMES_NO_EXTENT: (
        "--do-not-repair-list: no extent {element}, ignored"
    ),
    ErrorCode.DO_NOT_REPAIR_LINE_NAMES_NO_EXTENT: (
        "--do-not-repair {file} line {line}: no extent {element}, ignored"
    ),
    ErrorCode.DO_NOT_REPAIR_ELEMENT_ON_REMOVED_AXIS: (
        "--do-not-repair-list: {element} on {axis} in --remove-gyro-motion, ignored"
    ),
    ErrorCode.DO_NOT_REPAIR_LINE_ON_REMOVED_AXIS: (
        "--do-not-repair {file} line {line}: {element} on {axis} in"
        " --remove-gyro-motion, ignored"
    ),
}

HINTS = {
    ErrorCode.ALREADY_REPAIRED: ("add --force to repair a repaired video",),
    ErrorCode.CACHED_RESULT_NOT_FOUND: ("run {command} first",),
    ErrorCode.CACHE_FILE_INVALID: ("run {command} again",),
    ErrorCode.VIDEO_CHANGED_SINCE_RESULT_CACHED: ("run {command} again",),
    ErrorCode.CACHE_VERSION_DIFFERS: ("run {command} again",),
    ErrorCode.CACHED_RESULT_PARTS_MISSING: ("run {command} again",),
    ErrorCode.STAMPED_FILE_NOT_FOUND: ("run {command} again",),
    ErrorCode.STAMPED_FILE_CHANGED: ("run {command} again",),
    ErrorCode.STORED_PERIOD_DIFFERS: ("run {command} again",),
    ErrorCode.RATE_SERIES_UNEQUAL: ("run {command} again",),
    ErrorCode.CLASSIFICATION_WITHOUT_COLOR: (
        "give each classification a color with --decorator-color",
    ),
    ErrorCode.SHOW_NOISE_RATIO_WITHOUT_DETECTION_RESULT: ("run {command} first",),
    ErrorCode.NO_MODE_READY: (
        "run {detect_command} for --mode raw-vs-stab",
        "run {repair_command} for --mode original-vs-repair",
    ),
    ErrorCode.GYROFLOW_NOT_SET: (
        "set GYRO_DOCTOR_GYROFLOW to path of Gyroflow executable",
    ),
    ErrorCode.GYROFLOW_NOT_A_FILE: (
        "set GYRO_DOCTOR_GYROFLOW to path of Gyroflow executable",
    ),
    ErrorCode.GYROFLOW_WROTE_NO_RENDER: (
        "open {project} in Gyroflow to see why Gyroflow wrote no file",
    ),
}

CACHED_RESULT_WARNING = "{problem}, running {stage} again"

DETECTION_RESULT_UNUSED_WARNING = "{problem}, drawing no marks"


def message(code, **values):
    """The words of ``code`` with ``values`` filled in."""
    return MESSAGES[code].format_map(values)


def hints(code, **values):
    """The hint lines of ``code`` with ``values`` filled in, none where it has none."""
    return [hint.format_map(values) for hint in HINTS.get(code, ())]


class GyroDoctorError(Exception):
    """The base of every error the product raises on purpose."""

    def __init__(self, code, **values):
        super().__init__(code)
        self.code = code
        self.values = values

    def __str__(self):
        return (
            f"{self.what_failed.format_map(self.values)}: "
            f"{message(self.code, **self.values)}"
        )

    def hints(self):
        return hints(self.code, **self.values)


class CannotReadError(GyroDoctorError):
    what_failed = "cannot read {file}"


class CannotRepairError(GyroDoctorError):
    what_failed = "cannot repair {file}"


class CannotWriteError(GyroDoctorError):
    what_failed = "cannot write {file}"


class InvalidOptionValueError(GyroDoctorError):
    what_failed = "invalid {option} {value}"


class CannotCombineOptionsError(GyroDoctorError):
    what_failed = "cannot combine {option} with {other_option}"


@contextlib.contextmanager
def operating_system_error_as(error_class, file):
    """Raise ``error_class`` with the code ``SYSTEM_ERROR``, ``file``, and the system's
    reason in place of an ``OSError`` that the block raises."""
    try:
        yield
    except OSError as error:
        raise error_class(
            ErrorCode.SYSTEM_ERROR, file=file, reason=error.strerror
        ) from error


def log_error(log, error):
    """Log the error line of ``error`` and each of its hint lines at ERROR level, each
    record carrying its code and its values."""
    extra = {"code": error.code, "values": error.values}
    log.error("error: %s", error, extra=extra)
    for hint in error.hints():
        log.error("hint: %s", hint, extra=extra)


def log_warning(log, code, **values):
    """Log the warning line of ``code`` at WARNING level, the record carrying the code
    and the values."""
    log.warning(
        "warning: %s",
        message(code, **values),
        extra={"code": code, "values": values},
    )


def log_detection_result_unused_warning(log, error):
    """Log the warning that the render cannot use a detection result, which failed with
    ``error``, and draws no marks, followed by each hint line of ``error``, each record
    carrying the code of ``error`` and its values."""
    extra = {"code": error.code, "values": error.values}
    log.warning(
        "warning: %s",
        DETECTION_RESULT_UNUSED_WARNING.format(problem=error),
        extra=extra,
    )
    for hint in error.hints():
        log.warning("hint: %s", hint, extra=extra)


def log_cached_result_warning(log, error, stage):
    """Log the warning that a cached result failed with ``error`` and that ``stage``
    runs again, the record carrying the code of ``error`` and its values with the
    stage."""
    values = {**error.values, "stage": stage}
    log.warning(
        "warning: %s",
        CACHED_RESULT_WARNING.format(problem=error, stage=stage),
        extra={"code": error.code, "values": values},
    )
