# Command reference: `gyro-doctor`

## Description

`gyro-doctor` repairs the gyro data track of a DJI O4 video and saves the repaired video
alongside the original without modifying the video track. Due to noise issues in certain O4
air units, the gyro data track of some O4 videos can contain incorrect motion data artifacts
referred to here as gyro noise events. When Gyroflow stabilizes a video containing these
noise events, the result is often an unusable video with erratic motion. `gyro-doctor`
detects and repairs these gyro noise events so that Gyroflow can produce smooth footage from
the repaired video.

`gyro-doctor` measures each axis independently and detects each noise event by its
signature of either a sharp spike or a span of noise many times the expected level of that
axis. Because every measurement is relative to the flight itself, similar detection
parameters work across different drones and flights, and noise events can be detected even
in the middle of a fast maneuver.

`gyro-doctor` then repairs each noise event by attenuating the noise so that its peak falls
below a maximum ceiling from the smoothed gyro data, a sloped line across the event, or
zero. All motion outside the noise events remains exactly as recorded, and no low pass
filter is used in `gyro-doctor`, nor is one necessary in Gyroflow. In the event of a false
positive, the pilot can add the identifier of that event to the do not repair list and
`gyro-doctor` leaves that event as recorded.

## Synopsis

<!-- Help output. Do not edit by hand. -->
```
usage: gyro-doctor [options] VIDEO

gyro-doctor repairs the gyro data track of a DJI O4 video and saves the
repaired video alongside the original.

positional arguments:
  VIDEO                 The video to repair

options:
  -h, --help            show this help message and exit
  --help-all            Show help message including advanced options
  --version             show program's version number and exit
  --output PATH         The path of the repaired video (default: adds the
                        suffix '-repaired' to the video name before the file
                        extension)
  --verbose             Print one line summary for each stage
  --force               Force repair of a video already marked as repaired

cache related & charting only run options:
  --no-cache            Disables caching. Charts and renders will be
                        unavailable.
  --decode-only         Stop the run after the decode stage
  --detect-only         Stop the run after the detection stage
  --repair-only         Stop the run after the repair stage
  --cache-dir PATH      The location of the cache (default: .gyro-doctor-cache
                        under the current working directory)

detection tuning parameters:
  --detect-axes LIST    The detect axes to process (default: pitch,yaw,roll)
  --detect-oscillation-threshold VALUE or LIST
                        Noise threshold for a local segment of the video. When
                        the noise of the gyro signal reaches this many times
                        the Robust Standard Deviation of the axis, Gyro Doctor
                        detects a large amplitude oscillation. Set one value
                        for all three axes, or three values in the order
                        pitch,yaw,roll (default: 100)
  --detect-burst-threshold VALUE or LIST
                        Noise threshold for a local segment of the video. When
                        the noise of the gyro signal reaches this many times
                        the Robust Standard Deviation of the axis, Gyro Doctor
                        detects a mid-amplitude burst. Set one value for all
                        three axes, or set each axis individually in the order
                        pitch,yaw,roll (default: 10)
  --detect-impulse-threshold VALUE or LIST
                        Noise threshold for a single sample of the gyro
                        signal. When a sample differs from the local median by
                        this many times the Robust Standard Deviation of the
                        axis, Gyro Doctor detects an impulse candidate, which
                        the advanced tuning parameters of --help-all qualify
                        as an impulse event. Set one value for all three axes,
                        or set each axis individually in the order
                        pitch,yaw,roll (default: 4.5)

repair tuning parameters:
  --repair-axes LIST    The repair axes to process (default: pitch,yaw,roll)
  --remove-gyro-motion LIST
                        Removes all the gyro motion of the axes in LIST from
                        the gyro data track of the repaired video. Comma
                        separated, such as yaw,pitch,roll (default: none)
  --do-not-repair-list LIST
                        Do not repair the named extents, time ranges, and
                        frame ranges in LIST. Comma separated, such as
                        Y1502,0:00-0:05,Rf4500-f4620. Frame numbers begin with
                        f. P, R, or Y indicates the axis of a range, and a
                        range without one applies to all axes (default: none)
  --do-not-repair FILE  A text file for the do not repair list, one item or
                        range on each line. A blank line or a line that begins
                        with # is ignored (default: none)

large amplitude oscillation repair parameters:
  --repair-oscillation-ceiling VALUE
                        Ceiling for the repair of a large amplitude
                        oscillation. The repair reduces the oscillation until
                        the gyro signal peak is VALUE times the noise of the
                        axis excluding all detected events. A value of 0
                        collapses the motion to the centerline, and inf leaves
                        the oscillation as recorded (default: 0)
  --repair-oscillation-centerline {local-median,chord,zero}
                        The centerline toward which a large amplitude
                        oscillation is attenuated (default: chord)

mid-amplitude burst repair parameters:
  --repair-burst-ceiling VALUE
                        Ceiling for the repair of a mid-amplitude burst. The
                        repair reduces the burst until the gyro signal peak is
                        VALUE times the noise of the axis excluding all
                        detected events. A value of 0 collapses the motion to
                        the centerline, and inf leaves the burst as recorded
                        (default: 2)
  --repair-burst-centerline {local-median,chord,zero}
                        The centerline toward which a mid-amplitude burst is
                        attenuated (default: local-median)

impulse repair parameters:
  --repair-impulse-ceiling VALUE
                        Ceiling for the repair of an impulse event. The repair
                        reduces the impulse event until the gyro signal peak
                        is VALUE times the noise of the axis excluding all
                        detected events. A value of 0 collapses the motion to
                        the centerline, and inf leaves the impulse event as
                        recorded (default: 0.5)
  --repair-impulse-centerline {local-median,chord,zero}
                        The centerline toward which an impulse event is
                        attenuated (default: local-median)
```
<!-- End of the help output. -->

## Concepts

### The four stages

`gyro-doctor` works in four stages, in this order. Each stage saves its results in the
cache.

- **Decode:** Gyro Doctor reads the gyro data from the video. After `--decode-only`, a chart
  can show the original gyro data. A render with `--mode raw-vs-stab` can show the video as
  shot beside the original video stabilized by Gyroflow.
- **Detection:** Gyro Doctor detects the gyro noise events. After `--detect-only`, a chart can
  show the original gyro data and the gyro noise events. A `raw-vs-stab` render can show the
  video as shot beside the original video stabilized by Gyroflow and mark the gyro noise
  events.
- **Repair:** Gyro Doctor repairs each gyro noise event. After `--repair-only`, a chart can
  show the original gyro data, the gyro noise events, and the repaired gyro data. Since the
  repaired video has not been written, a render can show only the same `raw-vs-stab`
  comparison as after the detection stage.
- **Write:** Gyro Doctor writes the repaired video. After a full run, all charts and videos
  can be rendered. An `original-vs-repair` render can show the original video
  and the repaired video side by side after Gyroflow stabilizes both.

### The cache

By default, `gyro-doctor` will save a cache for each of the stages described above. Subsequent
runs reuse the results of every stage whose settings did
not change. For example, changing a detection setting invalidates every part of the cache
after the decode stage. Changing a repair setting or the do not repair list allows the run to
begin from the repair stage by loading the detection from cache if it exists.

`gyro-doctor-chart` and `gyro-doctor-render` both use the cache results, so `gyro-doctor`
must be run before they can be used. The cache is relative to the working directory, so run
all three commands from the same folder.

You can delete the cache at any time. Use the `--no-cache` option to prevent caching, in
which case you will not be able to generate gyro charts or video renders.

## Usage

This reference assumes that you have set up Gyro Doctor as
[the quick start guide](../quick-start.md) describes, with the folder of Gyro Doctor on
`PATH`. Run each command in a terminal open in the folder that holds the video. On Windows,
navigate to that folder in File Explorer, then type `powershell` or `cmd` in the Explorer
address bar, and press Enter. Alternatively, right click in a folder contents area and
select "Open in Terminal".

The command below repairs the gyro data track of a video. The video path is the one input
that `gyro-doctor` requires.

```
gyro-doctor video_file.mp4
```

The command prints nothing unless it has an error, a warning, or a hint. It writes the
repaired video, `video_file-repaired.mp4`, in the same folder as the original video, and
builds a cache of the results for charting, video renders, and speed enhancements when
testing different parameters. To change a repair, run `gyro-doctor` again on the original video
with new settings. Each full run writes the repaired video again, unless you use one of the
cache or charting options to stop a run earlier in the process. Stopping a run early permits
you to chart the results of different parameters without writing the video file.

At this point, you can immediately stabilize the repaired video in Gyroflow, or use the
`gyro-doctor-chart` and `gyro-doctor-render` commands below to review the gyro data repairs
in charts and side-by-side videos comparing before and after. When stabilizing in Gyroflow,
do not use horizon lock. See
[How Gyro Doctor detects and repairs gyro noise events](../detection-and-repair.md) for
additional details.

The image below shows the repaired video file and `.gyro-doctor-cache` folder that are
created when you run `gyro-doctor`. The repaired video takes the name of the original video
with `-repaired` added, and the cache folder is created in the folder where you run the
command.

![PowerShell after a run of gyro-doctor.exe, beside File Explorer showing the cache folder and the repaired video](../images/shared/command_line_powershell.png)

### The do not repair list

The do not repair list allows the user to prevent the repair of false positives by
providing a list of extent identifiers or ranges that should not be repaired. You can find
the identifier of an extent in either the gyro chart or the video render. In addition to
specific extent identifiers, you can name time ranges or frame ranges to be ignored by the
repair:

- An identifier such as `Y1502` names an extent by the letter of its axis and the video
  frame of its peak.
- A time range such as `1:02-1:05` or a frame range such as `f3720-f3900` names every
  extent that it partly or fully overlaps on every axis.
- An axis letter before a range limits the range to that one axis, for example `R1:02-1:05`.

You can provide the do not repair list either on the command line using
`--do-not-repair-list` or in a file using `--do-not-repair`. When using the command line to
supply the do not repair list, provide identifiers and ranges as a comma separated list with
no spaces. When using a file to supply the do not repair list, enter one identifier or range
per line. Blank lines and lines that begin with a `#` symbol are ignored.

Here is a command line sample showing a valid do not repair list:

```
gyro-doctor --do-not-repair-list "Y1502,R1:02-1:05" video_file.mp4
```

Here is a sample file saved as `do-not-repair.txt` showing a valid do not repair list:

```
# Sample do not repair file showing a named extent and a roll time range to ignore
Y1502
R1:02-1:05
```

Here is a sample command that uses the do not repair file:

```
gyro-doctor --do-not-repair do-not-repair.txt video_file.mp4
```

### Removing all motion from an axis

In rare cases, an axis has so much gyro noise that removing all motion from the gyro on
that axis provides a better result than a repair can provide. In this case, you can use the
option `--remove-gyro-motion` to remove all gyro motion data from the axis, while still
repairing the remaining axes. Gyroflow will then stabilize the repaired video normally. To
remove the gyro motion of more than one axis, separate the axes with commas:

```
gyro-doctor --remove-gyro-motion pitch,roll video_file.mp4
```

## More examples

The command below raises the impulse threshold to 6 times the expected noise, attenuates
each mid-amplitude burst toward the chord, and leaves the extent `Y1502` as recorded.

```
gyro-doctor --detect-impulse-threshold 6 --repair-burst-centerline chord --do-not-repair-list "Y1502" video_file.mp4
```

The command below stops the run after the detection stage.

```
gyro-doctor --detect-only video_file.mp4
```

---

<div align="center">

[<< back](../detection-and-repair.md) | [Return to Table of Contents](../README.md) | [next >>](gyro-doctor-chart.md)

</div>
