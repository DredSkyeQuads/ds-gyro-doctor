# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""The launcher that PyInstaller builds into the executable gyro-doctor."""

import sys

from gyro_doctor.cli import main

sys.exit(main())
