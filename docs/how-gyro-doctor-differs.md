# How Gyro Doctor differs from other solutions

Gyro Doctor is one of several public projects that address the gyro noise
events of DJI videos. Each project takes its own approach and so changes
different parts of the gyro data. The descriptions below hold for each project
as it was in October 2026.

## The other solutions

The following projects also address gyro noise events in DJI videos.

### Gyroflow - <https://github.com/gyroflow/gyroflow>

In its nightly builds, Gyroflow itself has a record of two features intended to
repair gyro noise events:

- Initially, Gyroflow created Glitch filtering in July 2026 to "detect and
  repair short bursts of corrupt gyro data". The feature used only the gyro data
  track and had a strength slider. Gyroflow removed the feature in September
  2026 in favor of Optical correction.
- Gyroflow's current Optical correction feature uses optical flow to measure the
  camera's rotation from the video frames and repairs the motion data where the motion data
  differs from that rotation. The feature blends the gyro data with the measured
  rotation by frequency. Gyroflow keeps the slow motion of the gyro data and
  repairs the faster motion to match the measured rotation. The feature needs a build of Gyroflow with the
  OpenCV library and stops when Gyroflow cannot track enough points in the video
  frames.

Neither feature is in a production release of Gyroflow as of October 2026. The
latest release at this time is v1.6.3. That release does not include Optical
correction.

### fpvflow - <https://github.com/rodrigosclosa/fpvflow>

fpvflow is a fork of Gyroflow "for FPV pilots: gyro-based video stabilization
with a DJI O4 Gyrogate fix, .cube LUT colour grading and external audio that
syncs automatically". fpvflow builds on the method of o4fix described below and
not on the Optical correction feature of the nightly builds of Gyroflow.
Therefore, fpvflow uses only the gyro data track for its fix. After each burst,
the method also removes the slow return that the author of o4fix calls a
"pull-back". fpvflow then runs a low pass filter over the whole gyro data track.
fpvflow applies the fix in its own preview and render and can save a repaired
copy of the video.

### o4fix - <https://github.com/vadimbryl/o4fix>

o4fix provides a "fix and analysis of false gyro readings in DJI O4 Air Unit
(Pro) units with the ICM-40609-D". o4fix is a command line tool that analyzes the
gyro data track and writes a repaired copy of the video. According to the author's research, the
ICM-40609-D gyro is at the root of the problem for O4 Lite and O4 Pro air units,
and DJI has built that gyro into O4 Pro air units since about February 2026.
The author measured the false rotation against a
Betaflight blackbox log. o4fix is the only project with an independent record of
the true motion.

One interesting result of the author's research is a bench test of what the
author calls the "pull-back". The author tested a drone with an O4 Pro air
unit built in 2026. With the drone running on the bench and its
propellers removed, the gyro recorded a false rotation of 11.84 degrees in 0.22 seconds, with a peak rate of
about 450 degrees per second on one axis, while the
video frames only indicated a rotation of about 0.08 degrees. The recorded
orientation then returned toward its starting orientation with a time constant
of 12.1 to 12.8 seconds. The author attributes this slow return to the air
unit's own attitude fusion.

### djgyrofix - <https://github.com/steamvogue/djgyrofix>

djgyrofix is a command line tool to "detect and fix high-frequency gyro
artifacts in DJI MP4 metadata, in place, with exact revert". djgyrofix repairs
the gyro data track of the video in place or writes a repaired copy. djgyrofix
saves a journal from which it restores the original video exactly. djgyrofix
only reports the changes it would make until you tell it to apply them.

djgyrofix compares the rate with its average over the 60 milliseconds on either
side of each point. djgyrofix then measures the size of the difference in each
10 millisecond interval, and detects an event where that size exceeds a
threshold. The threshold is the largest of three values. The first is 60
degrees per second. The second is the median size over the 12 seconds on either
side, plus 5 times its Robust Standard Deviation. The third is 3 times that
median size. djgyrofix divides all three values by its sensitivity setting. The
default sensitivity is 1. In a quiet video, the threshold is then 60 degrees per
second. In a noisier video, the threshold rises with the noise.

### DJIFix - <https://github.com/parhipov/DJIFix>

DJIFix is a Python script to "fix DJI O4 / O4 Pro 'GyroGate' telemetry for
Gyroflow: spikes and roll jitter repaired against the video itself". The script
uses optical flow to measure the camera's real rotation from the video frames
and repairs the gyro data track against that measurement. DJIFix states that
this measurement understates pitch and yaw by 25 to 50 percent through
parallax. The script writes a small separate MP4 that holds only the gyro data
track. Gyroflow loads that MP4 as motion data. DJIFix leaves the
video itself unchanged. Its batch file states that its image pass takes about 8
minutes for each minute of 4K video at 60 frames per second.

## Summary of detection methods

Each solution detects gyro noise events either by optical flow or by numerical
analysis of the gyro data track alone. The solutions also differ in how much of
the gyro data track they change. Some change the angular rate only inside the
gyro noise events that they detect. Others also change the angular rate outside
those events, in some cases over the whole video.

The following use optical flow:

- Gyroflow's Optical correction changes the angular rate over the whole trim
  range of the video.
- DJIFix changes the angular rate over the whole video.

The following use analysis of the gyro data track alone:

- Gyro Doctor changes the angular rate only inside the gyro noise events that it
  detects.
- Gyroflow's Glitch filtering changed the angular rate only inside the gyro
  noise events that it detected.
- o4fix runs a low pass filter over the whole gyro data track.
  The cutoff of the filter is 20 Hz and drops toward 3 Hz near a heavy repair.
- fpvflow runs the same 20 Hz to 3 Hz low pass filter as o4fix over the whole
  gyro data track.
- djgyrofix runs a low pass filter inside each event and changes the angular
  rate up to 100 milliseconds beyond each event.

## Unique methods of Gyro Doctor

Gyro Doctor detects gyro noise events in three passes, one for each
classification. The first pass detects large amplitude oscillations, the second
detects mid-amplitude bursts, and the third detects impulse events. Each pass
measures against a Robust Standard Deviation that leaves out the events of the
earlier passes. Gyro Doctor thereby keeps the larger events out of the noise
floor against which it detects the smaller events.

Gyro Doctor then attenuates each event toward a centerline until its peak is at
or below the ceiling of its classification.
Within each event, Gyro Doctor keeps the real motion that the centerline follows
and suppresses the peaks of the noise. Additionally, Gyro Doctor provides
commands that draw charts and render videos to assist with analysis and tuning.

## Reflections from the author

This project was originally an optical flow based solution. It worked quite
well. There's real value in using optical flow, particularly for detection;
validating what motion is real with optical flow can be more precise than with a
statistical algorithm. But it came at a significant speed cost. Within weeks of
being release ready, I started over. What if we can detect *everything* from
gyro data? Can it work and be faster? Does it matter? I decided that speed of
detection and repair WAS important, particularly if you have a slow computer.
It's one thing to wait for a Gyroflow render for 5 to 10 minutes, but it's
another thing to wait that long just to see *if* the fix works, and how adjustments to tuning parameters change
the outcome. So this iteration of Gyro Doctor operates solely on the gyro track and currently has
zero optical flow algorithms.

I didn't read the code of each of the projects mentioned, and I honestly wasn't even
aware of any of them when I started working on Gyro Doctor. So none of their work is
represented in my code here. However, one thing I do find really interesting
is the discovery of the 12.5 s pull-back after a glitch. It makes sense.
Gyrofusion is a combination of the accelerometer's absolute "down" reference and
the gyro, while a gyro only measures radial change in motion. It makes sense
that when the two don't agree, DJI's algorithm would pull the absolute reference
back into proper perspective. Of all the things I've seen, this is probably the
first enhancement I'd look into. I can live without horizon leveling, but I
don't know if YOU can. So I'll look into that.

Finally, maybe this software was written for nobody, and that's OK. I had fun
and learned an awful lot. Ultimately, Gyroflow will release their Optical
correction. And from what I've tested, it's really good. Nobody should expect
otherwise. But if you find this software useful, I sure would appreciate some
encouragement, and if you're willing, drop by my Patreon (Jamie Anne Harrell),
help support my work, and I'll keep writing solutions to weird and fun problems.
In the meantime, fly happy and publish your FPV videos on YouTube to keep the
hobby and inspiration alive!

---

<div align="center">

[<< back](commands/gyro-doctor-render.md) | [Return to Table of Contents](README.md)

</div>
