# Command reference: `gyro-doctor-render`

## Description

`gyro-doctor-render` renders a composite video showing two versions of your video side by
side in columns A and B using Gyroflow. Each gyro noise event that `gyro-doctor` detected
appears as a shaded extent on the rate strip under the repaired video and is labeled with
its identifier, such as `Y1502`. The color of the shading shows whether the event is an
impulse event, a mid-amplitude burst, or a large amplitude oscillation.

The mode `original-vs-repair` shows the original video and the repaired video after
Gyroflow stabilizes both. Use it to see whether the repair removed the false motion and to
show off your results. The mode `raw-vs-stab` shows the original video without
stabilization beside the unrepaired video stabilized with Gyroflow. Use it to decide
whether motion in the stabilized video is real motion that your camera made or false motion
from gyro noise. The render can also be helpful to see the results of changing
`gyro-doctor` options to fine tune detection and repair. See
[the command reference of `gyro-doctor`](gyro-doctor.md) and `gyro-doctor --help-all` for
advanced tuning options.

Note: `gyro-doctor` creates the cache of the results for charting and video renders. The
cache must be available before using `gyro-doctor-render`. In order to use
`gyro-doctor-render`, you must have Gyroflow installed and the environment variable
`GYRO_DOCTOR_GYROFLOW` configured as described in
[the quick start guide](../quick-start.md#view-the-results).

![Composite of the original and the repaired video, each over a strip of the Normalized Rate, with gyro noise events shaded](../images/shared/original-vs-repair-composite-2.png)

## Synopsis

<!-- Help output. Do not edit by hand. -->
```
usage: gyro-doctor-render [options] VIDEO

gyro-doctor-render renders the stabilized video before and after the repair
side by side using Gyroflow.

positional arguments:
  VIDEO                 The original video

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --mode MODE           The render to make: original-vs-repair, the original
                        video against the repaired video, or raw-vs-stab, the
                        video as shot against the video stabilized by Gyroflow
                        (default: original-vs-repair when the cache holds a
                        repair result of the video, otherwise raw-vs-stab when
                        it holds a detection result)
  --output PATH         The complete path of the composite (default: the video
                        stem, the mode, the center video frame of the video
                        frames that --start and --duration select, and their
                        count, with the extension .MP4, in the current
                        directory, such as DJI_0002-original-vs-
                        repair-f4950-900f.MP4)
  --start POSITION      The first video frame to render: a time such as 75s,
                        1:15, or 1:02:15, or a video frame such as f4500
                        (default: the first video frame)
  --duration DURATION   The duration to render from --start: a time such as
                        15s, or a count of video frames such as 900f (default:
                        the rest of the video)
  --cache-dir PATH      The cache, as gyro-doctor takes it (default: .gyro-
                        doctor-cache under the current working directory)
  --preset NAME         The Gyroflow dials of every Gyroflow render, standard
                        or zooming (default: zooming)
  --rate-axis-scale VALUE or LIST
                        The rate axis of the three panels, in degrees per
                        second: one value sets plus and minus that value on
                        all three panels; three values, as one comma separated
                        token, set pitch, yaw, and roll; peraxis gives each
                        panel its own rate axis at 1.25 times the largest
                        absolute rate on its axis that is not a transient
                        outlier (default: one rate axis at 1.25 times the
                        largest absolute rate of the series drawn in the
                        composite, over the three axes, that is not a
                        transient outlier)
  --decorator-color CLASS=#RRGGBB
                        The color of the marks of one classification, such as
                        "impulse event=#1f4fd6". Repeatable (default: the
                        color of each classification)
  --verbose             Print what the render drew and the Gyroflow renders it
                        made or reused
```
<!-- End of the help output. -->

## Concepts

[How Gyro Doctor detects and repairs gyro noise events](../detection-and-repair.md)
explains the three classifications of gyro noise events displayed in the composite.

### Cache and run requirements from `gyro-doctor`

Different stages of the `gyro-doctor` command create the results required by
`gyro-doctor-render`. If you run `gyro-doctor` without using any of the options
`--no-cache`, `--decode-only`, `--detect-only`, or `--repair-only`, then all render modes
will be available. Since the cache is required for rendering, calls to `gyro-doctor-render`
without a valid cache will fail.

In addition to the `gyro-doctor` data, the render keeps each Gyroflow render in the cache. A
later render of the same window of the original video with the same preset reuses a
cached render. Since changes to repair parameters change the repaired video, the repaired
video renders are not reused. However, they do remain in the cache, which may need to be
cleaned to free up the space.

| stage | available modes to render | description |
|---|---|---|
| Decode | `raw-vs-stab` | A side: The original video without stabilization is shown over the original rates in orange.<br>B side: The video stabilized by Gyroflow is shown over the original rates in orange. |
| Detection | `raw-vs-stab` | A side: The original video without stabilization is shown over the original rates in orange and the detected noise events.<br>B side: The video stabilized by Gyroflow is shown over the original rates in orange and the detected noise events. |
| Repair | `raw-vs-stab` | Same as detection. Note: `original-vs-repair` is not available yet for rendering since the repaired video doesn't exist. |
| Write | `raw-vs-stab`, `original-vs-repair` | `raw-vs-stab`: Same as above.<br>`original-vs-repair`:<br>A side: The original video stabilized by Gyroflow is shown over the original rates in orange as series A. Expect to see the noise events exhibited on the A side.<br>B side: The repaired video stabilized by Gyroflow is shown over the original rates in orange as series A and the repaired rates in cyan as series B. |

The default value of `--mode` changes depending on which stages have run. When the cache
holds a repair result of the video, the default is `original-vs-repair`. Otherwise, when the
cache holds a detection result, the default is `raw-vs-stab`.

The shading of an extent is strong over the video frames where the repair changed the rate
by more than 0.5 degrees per second and faint over the rest. Faint shading shows an extent
that the repair barely changed or left as recorded. In `raw-vs-stab` every shaded extent is
faint. The shading comes from the result of the detection stage, and a composite made
without that result has no shaded extents.

## Usage

This reference assumes that you have set up Gyro Doctor and `GYRO_DOCTOR_GYROFLOW` as
[the quick start guide](../quick-start.md) describes, with the folder of Gyro Doctor on
`PATH`, and that you have run `gyro-doctor` on the video. Run each command in a terminal
open in the folder that holds the video. On Windows, navigate to that folder in File
Explorer, then type `powershell` or `cmd` in the Explorer address bar, and press Enter.
Alternatively, right click in a folder contents area and select "Open in Terminal".

For consistent rendering, two Gyroflow presets are provided, `standard` and `zooming`. The
following values are set for each preset:

| Gyroflow setting | `standard` | `zooming` (default) |
|---|---|---|
| smoothness | 0.5 | 0.15 |
| field of view | 0.9 | 1.0 |
| lens correction | 25% | 20% |
| adaptive zoom | off | on, with a zooming speed of 2.0 seconds |
| max zoom | unused | 130% |

A Gyroflow setting that the render does not set, such as the maximum smoothness of the
smoothing method, takes the value of your own Gyroflow settings.

The command below renders the 15 seconds that begin 40 seconds into a video.

```
gyro-doctor-render --start 40s --duration 15s video_file.mp4
```

Open the composite in a video player to see the original video and the repaired video side
by side, as shown in the image above.

By default, the three panels share one rate axis at 1.25 times the largest absolute rate in
the composite that is not a transient outlier. A taller peak runs beyond the rate axis, and
a label beside each of the largest such peaks indicates its size. Use `--rate-axis-scale`
to set the rate axis yourself. One value, such as `--rate-axis-scale 200`, sets +/- 200
degrees per second on all three panels. Three values, such as
`--rate-axis-scale 50,100,30`, set pitch, yaw, and roll independently. Use
`--rate-axis-scale peraxis` to allow each axis to scale automatically independent of one
another.

## More examples

If you want to start a render several video frames before a particular event, subtract
those video frames from the video frame of its identifier. The command below starts 90
video frames before the extent `Y1502` at video frame 1412 and renders an
`original-vs-repair` composite of 180 video frames.

```
gyro-doctor-render --start f1412 --duration 180f video_file.mp4
```

The command below renders a `raw-vs-stab` composite of 900 video frames starting at video
frame 4500.

```
gyro-doctor-render --mode raw-vs-stab --start f4500 --duration 900f video_file.mp4
```

Both `--start` and `--duration` can be given in video frames or as a time, independently of
each other. The command below renders an `original-vs-repair` composite of 900 video frames
starting 1:15 into the video.

```
gyro-doctor-render --start 1:15 --duration 900f video_file.mp4
```

---

<div align="center">

[<< back](gyro-doctor-chart.md) | [Return to Table of Contents](../README.md) | [next >>](../how-gyro-doctor-differs.md)

</div>
