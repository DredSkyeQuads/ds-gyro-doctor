# Gyro Doctor

Gyro Doctor™ is a software utility that repairs the data track of a DJI O4
video that exhibits incorrect motion when stabilized by Gyroflow. Gyro
Doctor operates on the gyro track directly and saves the repaired version
into a new file alongside the original. It leaves the video track itself
unmodified.

[Click here to go straight to downloads](https://github.com/DredSkyeQuads/ds-gyro-doctor/releases/latest).

[Click here for the quick start guide](docs/quick-start.md).

Gyro Doctor uses noise attenuation algorithms designed to recover as much
of the real motion of the camera as possible while removing the noise bursts.
Since Gyro Doctor works at the measurement rate of the gyro rather than the
frame rate of the camera and requires no low pass filter, it can often reduce
the noise spikes without degrading other stabilization properties such as
rolling shutter correction, possibly improving stabilization results as
compared to other solutions that require low pass filters.

Gyro Doctor has three commands, the `gyro-doctor` command itself and two
diagnostic tools that assist with visualizing and fine-tuning the repair. The
two diagnostic tools, `gyro-doctor-chart` and `gyro-doctor-render`, both draw
the rotational rate of the camera on each axis computed from the gyro data
track, and, when available, also the repaired rates for each axis.
`gyro-doctor-chart` draws charts, while `gyro-doctor-render` creates a side by
side video to visualize before and after.

| command | what it does |
|---|---|
| `gyro-doctor` | `gyro-doctor` repairs the gyro data track of a DJI O4 video and saves the repaired video alongside the original. |
| `gyro-doctor-chart` | `gyro-doctor-chart` charts the rotational rate of the camera on each axis before and after the repair. |
| `gyro-doctor-render` | `gyro-doctor-render` renders the stabilized video before and after the repair side by side using Gyroflow. |

The two diagnostics draw from the results that `gyro-doctor` keeps for the video
in its cache, `.gyro-doctor-cache/` under the current working directory, so run
`gyro-doctor` on a video before either of them.

## How Gyro Doctor differs from other solutions

As of October 2026, Gyro Doctor is one of several projects that address the gyro
noise events of DJI videos. Gyroflow's "Optical correction", available only in
pre-release code, and DJIFix measure the camera's rotation from the video using
optical flow algorithms that cost an analysis of every video
frame. Gyro Doctor, like o4fix, fpvflow, and djgyrofix, reads only the gyro data
track. Both o4fix and fpvflow draw a straight line across each gyro noise event,
and djgyrofix fills or smooths it.

Gyro Doctor instead uses a highly configurable ceiling attenuator that allows
it to retain some of the real motion while rejecting the noise as much as
possible, without using a low pass filter. It changes the rotational rate only
inside the gyro noise events that it detects, while o4fix and fpvflow run a low
pass filter over the whole gyro data track.

Gyro Doctor carries the rotation of the repair forward into the rest of the
video as a permanent offset, which currently makes horizon lock in Gyroflow
unusable on a repaired video.
In contrast, o4fix, fpvflow, and DJIFix bring the orientation back to the
original orientation over a short period of time. djgyrofix keeps the net rotation across the event, so nothing carries
forward.

[How Gyro Doctor differs from other solutions](docs/how-gyro-doctor-differs.md)
compares Gyro Doctor with each of these projects in detail.

## Download and install

1. For the latest release, visit
   <https://github.com/DredSkyeQuads/ds-gyro-doctor/releases/latest>.
2. Download the zip for your system. The table below indicates the zip of each
   supported platform and the minimum version requirements, where X.Y.Z is the
   version of the release.

   | system | zip | runs on |
   |---|---|---|
   | Windows x64 | `gyro-doctor-vX.Y.Z-windows-x64.zip` | Windows 10 or later |
   | macOS on Apple Silicon | `gyro-doctor-vX.Y.Z-macos-arm64.zip` | macOS 14 or later |
   | macOS on Intel | `gyro-doctor-vX.Y.Z-macos-x64.zip` | macOS 14 or later |
   | Linux x64 | `gyro-doctor-vX.Y.Z-linux-x64.zip` | glibc 2.35 or later |

3. Unzip the archive and follow the [quick start guide](docs/quick-start.md),
   which provides essential first run steps for Windows users.

## Run from source

To run Gyro Doctor from source, you need Python v3.14 or later with pip, and git
to clone this repository. pip installs every runtime dependency from PyPI.

Steps to get Gyro Doctor running from source:

1. Clone the repository and change into its working directory.

   ```
   git clone https://github.com/DredSkyeQuads/ds-gyro-doctor.git
   cd ds-gyro-doctor
   ```

2. Create a virtual environment. On Windows, run this command.

   ```
   py -3.14 -m venv .venv
   ```

   On macOS and Linux, run this command.

   ```
   python3.14 -m venv .venv
   ```

3. Activate the virtual environment. In PowerShell, run this command.

   ```
   .\.venv\Scripts\Activate.ps1
   ```

   In Command Prompt, run this command.

   ```
   .venv\Scripts\activate.bat
   ```

   On macOS and Linux, run this command.

   ```
   source .venv/bin/activate
   ```

4. Install Gyro Doctor in the virtual environment. pip puts `gyro-doctor`,
   `gyro-doctor-chart`, and `gyro-doctor-render` in the virtual environment.

   ```
   pip install .
   ```

The Gyro Doctor commands will work when your virtual environment is activated.
When you open a new terminal, activate the virtual environment before you run
any of the commands.

To test the commands, run these steps with the virtual environment activated. Git
ignores the folder `test-media/`, so your videos and every file that the commands
write there stay out of git.

1. Make the folder `test-media` in the working directory of the repository. You
   make it once. Run this command.

   ```
   mkdir test-media
   ```

2. Copy a video into `test-media`. Add more videos whenever you want to run Gyro
   Doctor on them.

3. Change into the folder. Run this command.

   ```
   cd test-media
   ```

4. Repair a video, using the name of your video in place of `video_file.mp4`.
   Run this command.

   ```
   gyro-doctor video_file.mp4
   ```

5. Draw its chart. `gyro-doctor-chart` takes the name of the original video,
   `video_file.mp4`, not the name of the repaired video. Run this command.

   ```
   gyro-doctor-chart video_file.mp4 --center 30s
   ```

6. Print the help of `gyro-doctor-render`, which needs Gyroflow to render. A
   render also takes the name of the original video, not the name of the
   repaired video. Run this command.

   ```
   gyro-doctor-render --help
   ```

   See [Usage](#usage) below on adding the `GYRO_DOCTOR_GYROFLOW` variable. If
   you already have this set, create a video render with this command.

   ```
   gyro-doctor-render video_file.mp4 --start 30s --duration 15s
   ```

7. If you'd like to test the repaired video directly in Gyroflow, open
   `video_file-repaired.mp4` in Gyroflow.

## Usage

See the [quick start guide](docs/quick-start.md) for a walkthrough, or follow the
brief instructions below for the minimum setup necessary for every Gyro Doctor
command to work.

The following usage commands assume you either have a working binary
distribution or are running in an activated Python virtual environment. For a
binary distribution, put the folder of the executables on `PATH`. On Windows,
each command is the same in PowerShell and in Command Prompt.

The command below repairs a video using the default options.

```
gyro-doctor DJI_0002.MP4
```

`gyro-doctor` saves the repaired video alongside the original, with the suffix
`-repaired` added to the file name, and also saves a cache of the repair for the
diagnostic tools to read. In the example above, the repaired video would be
named `DJI_0002-repaired.MP4`.

*Note: The chart and render commands take the path of the original video, since
they compare the original and repaired gyro data. Gyro Doctor locates the cache
in the working directory based on the original video's file name.*

The command below draws a gyro rate chart showing the 24 video frames centered
near the time 1:15 and saves it as a PNG in the current working directory.

```
gyro-doctor-chart --center 1:15 DJI_0002.MP4
```

In addition to the default gyro rate chart width, see the usage information to
draw both wide and ultra-wide charts to examine gyro rates from larger segments
of the video.

The command below renders the original video side by side against the repaired
video over the 15 seconds that begin at 1:15 and saves the render as an MP4 in
the current working directory.

`gyro-doctor-render` runs [Gyroflow](https://github.com/gyroflow/gyroflow),
which you must install separately. Additionally, `gyro-doctor-render` requires
the path of your Gyroflow executable in the user environment variable
`GYRO_DOCTOR_GYROFLOW`.

```
gyro-doctor-render --start 1:15 --duration 15s DJI_0002.MP4
```

To set `GYRO_DOCTOR_GYROFLOW` on Windows, run these commands in PowerShell,
replacing the path with the path of your Gyroflow executable:

```
$env:GYRO_DOCTOR_GYROFLOW = "path\to\gyroflow.exe"
[Environment]::SetEnvironmentVariable("GYRO_DOCTOR_GYROFLOW", "path\to\gyroflow.exe", "User")
```

On macOS, run these commands:

```
export GYRO_DOCTOR_GYROFLOW="/path/to/gyroflow"
echo 'export GYRO_DOCTOR_GYROFLOW="/path/to/gyroflow"' >> ~/.zshrc
```

On Linux, run these commands:

```
export GYRO_DOCTOR_GYROFLOW="/path/to/gyroflow"
echo 'export GYRO_DOCTOR_GYROFLOW="/path/to/gyroflow"' >> ~/.bashrc
```

## Documentation

The full documentation of Gyro Doctor is in `docs/`. It explains how Gyro
Doctor detects and repairs gyro noise events, shows how to tune the detection and
repair parameters to improve the results, and describes the options of the three
commands. [The Table of Contents](docs/README.md) lists each page for quick navigation.

Each command has its own reference page:

- [`gyro-doctor`](docs/commands/gyro-doctor.md)
- [`gyro-doctor-chart`](docs/commands/gyro-doctor-chart.md)
- [`gyro-doctor-render`](docs/commands/gyro-doctor-render.md)

## Build the executables

You do not need to build the executables to use Gyro Doctor within a virtual
environment, as [Run from source](#run-from-source) shows, or to run it from one
of our binary distributions. This section is for anyone who wants to build the
executables themselves. Run each command of this section in the terminal of VS Code.

The release executables bundle the Python runtime and every required library,
so an end user can run them without installing Python or running from
source. The table below states what a build from source needs on each system.

| system | what a build from source needs |
|---|---|
| Windows x64 | Python v3.14 with the Python launcher, which runs it as `py -3.14` |
| macOS on Apple Silicon | Python v3.14, run as `python3.14` |
| macOS on Intel | Python v3.14, run as `python3.14` |
| Linux x64 | Python v3.14, run as `python3.14`. The executables run on a Linux system whose glibc is at least as new as that of the system that built them. |

This repository holds the Python package of Gyro Doctor, `gyro_doctor/`, and its
`pyproject.toml`. The `pyproject.toml` declares the three commands, the runtime
dependencies, and the optional dependency group `build`, which includes
PyInstaller. The build script, `build.py`, builds the executables from the
package that you install in the virtual environment at the versions that the
manifest `dist-manifest.txt` pins. The build will not work if Python or a
package of the group `build` is at any other version.

To build the executables on your system:

1. Clone the repository and change into its working directory.

   ```
   git clone https://github.com/DredSkyeQuads/ds-gyro-doctor.git
   cd ds-gyro-doctor
   ```

2. Create a virtual environment with Python v3.14 that the manifest pins. On
   Windows, run this command.

   ```
   py -3.14 -m venv .venv
   ```

   On macOS and Linux, run this command.

   ```
   python3.14 -m venv .venv
   ```

3. Activate the virtual environment. In PowerShell, run this command.

   ```
   .\.venv\Scripts\Activate.ps1
   ```

   In Command Prompt, run this command.

   ```
   .venv\Scripts\activate.bat
   ```

   On macOS and Linux, run this command.

   ```
   source .venv/bin/activate
   ```

4. Install Gyro Doctor with the group `build` in the virtual environment, with
   every package at the version that the release was built and tested with. The
   file `constraints.txt` pins those versions. Run this command.

   ```
   pip install --constraint constraints.txt --build-constraint constraints.txt ".[build]"
   ```

5. Run the build script.

   ```
   python build.py
   ```

   When the build is done, the script prints the path of each executable. The
   three executables are in `build/pyinstaller/dist/`: `gyro-doctor`,
   `gyro-doctor-chart`, and `gyro-doctor-render`, each with the extension
   `.exe` on Windows.

## Test the executables

After you build the executables, test them on a video from your camera. Git
ignores the folder `test-media/`, so your videos and every file that the
executables write there stay out of git.

1. Make the folder `test-media` in the working directory of the repository. You
   make it once. Run this command.

   ```
   mkdir test-media
   ```

2. Copy a video into `test-media`. Add more videos whenever you want to run Gyro
   Doctor on them.

3. Change into the folder. Run this command.

   ```
   cd test-media
   ```

4. Repair a video, using the name of your video in place of `video_file.mp4`. On
   Windows, run this command.

   ```
   ..\build\pyinstaller\dist\gyro-doctor.exe video_file.mp4
   ```

   On macOS and Linux, run this command.

   ```
   ../build/pyinstaller/dist/gyro-doctor video_file.mp4
   ```

5. Draw its chart. `gyro-doctor-chart` takes the name of the original video,
   `video_file.mp4`, not the name of the repaired video. On Windows, run this
   command.

   ```
   ..\build\pyinstaller\dist\gyro-doctor-chart.exe video_file.mp4 --center 30s
   ```

   On macOS and Linux, run this command.

   ```
   ../build/pyinstaller/dist/gyro-doctor-chart video_file.mp4 --center 30s
   ```

6. Print the help of `gyro-doctor-render`, which needs Gyroflow to render. A
   render also takes the name of the original video, not the name of the
   repaired video. On Windows, run this command.

   ```
   ..\build\pyinstaller\dist\gyro-doctor-render.exe --help
   ```

   On macOS and Linux, run this command.

   ```
   ../build/pyinstaller/dist/gyro-doctor-render --help
   ```

   See [Usage](#usage) above on adding the `GYRO_DOCTOR_GYROFLOW` variable. If
   you already have this set, create a video render. On Windows, run this
   command.

   ```
   ..\build\pyinstaller\dist\gyro-doctor-render.exe video_file.mp4 --start 30s --duration 15s
   ```

   On macOS and Linux, run this command.

   ```
   ../build/pyinstaller/dist/gyro-doctor-render video_file.mp4 --start 30s --duration 15s
   ```

7. If you'd like to test the repaired video directly in Gyroflow, open
   `video_file-repaired.mp4` in Gyroflow.

## Licenses and third party libraries

Gyro Doctor is licensed under GPL-3.0-or-later (the GNU General Public License,
version 3 or later). [LICENSE](LICENSE) holds the text of the license.

The release executables bundle the third party libraries in the table
below, and the libraries that these depend on, each under its own license.

| library | use in Gyro Doctor | license |
|---|---|---|
| Python | the Python runtime | [the Python license](https://docs.python.org/3/license.html) |
| numpy | matrix arithmetic | [the numpy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| scipy | rotation arithmetic | [the scipy license](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| PyAV, the Python package `av` | video input and output | [the PyAV license](https://github.com/PyAV-Org/PyAV/blob/main/LICENSE.txt), the BSD 3-Clause License |
| FFmpeg, as the PyAV wheel carries it | video decoding and encoding | [the FFmpeg license](https://ffmpeg.org/legal.html) |
| matplotlib | charting | [the matplotlib license](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) |
| Pillow | image drawing | [the Pillow license](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| the PyInstaller bootloader | executable startup | [the PyInstaller license](https://github.com/pyinstaller/pyinstaller/blob/develop/COPYING.txt) |

The third party notices file in each zip names every library that the build of
that zip bundles, the libraries that the libraries of the table depend on
included, with its version and its license. It also states where to get the source code
of Gyro Doctor, which is this repository, and of FFmpeg and each library under
the GNU General Public License or the GNU Lesser General Public License that the
zip bundles, at the version that the zip bundles.

"Gyro Doctor" is a trademark of DredSkye LLC.
