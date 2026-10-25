# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""The launcher that PyInstaller builds into the executable gyro-doctor-render."""

import sys

from gyro_doctor.tools.render import main

sys.exit(main())
