# Command reference: `gyro-doctor-chart`

## Description

`gyro-doctor-chart` charts the rotational rate of the camera on each axis before and after
the repair. It creates a chart depicting a portion of the video around a moment that you
choose and saves the chart as a PNG file. The chart shows each gyro noise event that
`gyro-doctor` detected as a shaded extent on the panel of its axis. Each extent is labeled
with its identifier, such as `P2300`, and its color shows whether it is an impulse event, a
mid-amplitude burst, or a large amplitude oscillation.

After `gyro-doctor` repairs the gyro noise events, the chart draws the original rates in
orange as series A and the repaired rates in green as series B, so you can visualize the
results of the repair. The chart can be helpful to see the results of changing
`gyro-doctor` options to fine tune detection and repair. See
[the command reference of `gyro-doctor`](gyro-doctor.md) and `gyro-doctor --help-all` for
advanced tuning options.

Note: `gyro-doctor` creates the cache of the results for charting and video renders. The
cache must be available before using `gyro-doctor-chart`.

![Normalized Rate chart of three axes, original series in orange, repaired series in green, gyro noise events shaded](../images/shared/3-O4Lite-Gyro-Glitch-Ciotti-DJI_20260425224918_0007_D-chart-f2300-24f.png)

## Synopsis

<!-- Help output. Do not edit by hand. -->
```
usage: gyro-doctor-chart [options] VIDEO

gyro-doctor-chart charts the rotational rate of the camera on each axis before
and after the repair.

positional arguments:
  VIDEO                 The video whose cached results the chart draws

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --center POSITION     The middle of the drawn window: a time such as 75s,
                        1:15, or 1:02:15, or a video frame such as f4500. The
                        window slides to fit at the ends of the video
  --output PATH         The complete path of the PNG (default: the video stem,
                        chart, the center video frame, and the duration of the
                        window in video frames, in the current directory, such
                        as DJI_0002-chart-f4500-24f.png)
  --series {original}   original draws the original Normalized Rate alone
                        (default: every series the cache holds)
  --cache-dir PATH      The cache, as gyro-doctor takes it (default: .gyro-
                        doctor-cache under the current working directory)
  --view NAME           The duration of the window: normal, 24 video frames;
                        wide, 120 video frames; ultra-wide, 15 seconds
                        (default: normal)
  --rate-axis-scale VALUE or LIST
                        The rate axis of the three panels, in degrees per
                        second: one value sets plus and minus that value on
                        all three panels; three values, as one comma separated
                        token, set pitch, yaw, and roll; peraxis gives each
                        panel its own rate axis at 1.3 times the largest
                        absolute rate on its axis that is not a transient
                        outlier (default: one rate axis at 1.3 times the
                        largest absolute rate in the window, over the three
                        axes and every series drawn, that is not a transient
                        outlier)
  --show-noise-ratio    Draw the local Robust Standard Deviation divided by
                        the full track Robust Standard Deviation using a right
                        hand axis for each panel, with lines at the
                        oscillation threshold and the burst threshold, in
                        every window (default: only in a window that contains
                        a large amplitude oscillation or a mid-amplitude
                        burst). Requires a detection result in the cache
  --title-prefix WORDS  Words placed before the title
  --verbose             Print what the chart drew
```
<!-- End of the help output. -->

## Concepts

[How Gyro Doctor detects and repairs gyro noise events](../detection-and-repair.md)
explains the three classifications of gyro noise event, the local median, and the Robust
Standard Deviation that the chart draws.

### Cache and run requirements from `gyro-doctor`

Different stages of the `gyro-doctor` command create the necessary series required by
`gyro-doctor-chart`. If you run `gyro-doctor` without using any of the options
`--no-cache`, `--decode-only`, or `--detect-only`, then all chart series will be available.
Since the cache is required for charting, calls to `gyro-doctor-chart` without a valid
cache will fail.

| stage | available to the chart | description |
|---|---|---|
| Decode | the original rates | The original rates are drawn in orange. This is the only series available after the decode stage. |
| Detection | the original rates, the extents of the gyro noise events, the local median, the thresholds, the ratio axis, and the Robust Standard Deviation of each axis | In addition to the above, each extent is shaded in the color of its classification and labeled with its identifier. The local median is drawn as a dashed purple curve, and the threshold lines are drawn. The ratio axis is drawn in a window that holds a large amplitude oscillation or a mid-amplitude burst. The Measurements box at the top right gives the Robust Standard Deviation values of each axis. |
| Repair | the original rates, the extents of the gyro noise events, the local median, the thresholds, the ratio axis, the Robust Standard Deviation of each axis, and the repaired rates | In addition to the above, once the repaired rates are available, the original rates are always drawn as series A in orange and the repaired rates as series B in green. The green appears only inside the extents. |
| Write | Same as repair | Same as repair |

An extent is the span of time on one axis that Gyro Doctor found to hold a gyro noise
event. Its identifier is the letter of the axis followed by the video frame of its peak,
such as `Y1502`. The letter is `P` for pitch, `Y` for yaw, or `R` for roll.

### The threshold lines

With the result of the detection stage, the chart draws the thresholds that the run of
`gyro-doctor` used:

- The impulse threshold as a solid pair of lines on either side of the local median.
- The estimated peak height at the burst threshold as a dashed pair of lines. The lines
  estimate how high the peaks of a stretch reach when the stretch just meets the burst
  threshold.
- In a window that holds a large amplitude oscillation or a mid-amplitude burst, a ratio
  axis on the right of each panel. The ratio axis draws `Local / Full Track RSD` with a
  line at the oscillation threshold and a line at the burst threshold.
  `Local / Full Track RSD` shows how much noisier the rate is at each moment than the
  whole axis.

## Usage

This reference assumes that you have set up Gyro Doctor as
[the quick start guide](../quick-start.md) describes, with the folder of Gyro Doctor on
`PATH`, and that you have run `gyro-doctor` on the video. Run each command in a terminal
open in the folder that holds the video. On Windows, navigate to that folder in File
Explorer, then type `powershell` or `cmd` in the Explorer address bar, and press Enter.
Alternatively, right click in a folder contents area and select "Open in Terminal".

The command below charts the 24 video frames centered around 46 seconds into a video. To
chart a known extent, you can provide the video frame of its identifier as seen in the
render, such as `f1502` for extent `Y1502`. Both the video path and `--center` are required
inputs. `--center` can be provided as a frame number using the `f` prefix. For example,
`--center f4300` centers the chart on that frame.

```
gyro-doctor-chart --center 46s video_file.mp4
```

Use the `--view` parameter as shown in the Synopsis to get wider rate charts.

By default, the three panels share one rate axis at 1.3 times the largest absolute rate in
the window that is not a transient outlier. A taller peak runs beyond the rate axis, and a
label above its panel indicates the largest peak. Use `--rate-axis-scale` to set the rate
axis yourself. One value, such as `--rate-axis-scale 200`, sets +/- 200 degrees
per second on all three panels. Three values, such as `--rate-axis-scale 50,100,30`, set
pitch, yaw, and roll independently. Use `--rate-axis-scale peraxis` to allow each axis to
scale automatically independent of one another.

## More examples

The command below creates a chart centered on the peak of the extent `Y1502`.

```
gyro-doctor-chart --center f1502 video_file.mp4
```

The command below creates a wide chart of 120 video frames centered on video frame 4500 and
shows how much noisier the rate is at each moment than the whole axis.

```
gyro-doctor-chart --center f4500 --view wide --show-noise-ratio video_file.mp4
```

The command below creates an ultra-wide chart of 15 seconds centered on 1:15 into the video.
The ultra-wide chart spans many page widths and will require zooming in to be able to read.

```
gyro-doctor-chart --center 1:15 --view ultra-wide video_file.mp4
```

---

<div align="center">

[<< back](gyro-doctor.md) | [Return to Table of Contents](../README.md) | [next >>](gyro-doctor-render.md)

</div>
