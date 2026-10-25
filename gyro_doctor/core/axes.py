# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""The axis order pitch, yaw, roll, which every per axis array of the product follows:
pitch in column 0, yaw in column 1, and roll in column 2.
"""

AXES = ("pitch", "yaw", "roll")


def axis_column(axis):
    """The column of an axis in every per axis array: 0 for ``pitch``, 1 for ``yaw``,
    and 2 for ``roll``. Raises ValueError on an unknown axis."""
    try:
        return AXES.index(axis)
    except ValueError:
        raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}") from None
