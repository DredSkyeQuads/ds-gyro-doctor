# SPDX-FileCopyrightText: 2026 DredSkye LLC
# SPDX-FileContributor: Jamie Anne Harrell
# SPDX-License-Identifier: GPL-3.0-or-later

"""The conversion of quaternions to scipy rotations. A quaternion and its negation are
the same rotation, so each quaternion first takes the sign that agrees with the
quaternion before it.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def canonicalize(quaternions_wxyz):
    """A copy of the quaternions, written (w, x, y, z), with the sign of each one
    flipped where its dot product with the quaternion before it is negative."""
    quaternions_wxyz = np.array(quaternions_wxyz, dtype=np.float64, copy=True)
    for i in range(1, len(quaternions_wxyz)):
        if np.dot(quaternions_wxyz[i], quaternions_wxyz[i - 1]) < 0:
            quaternions_wxyz[i] = -quaternions_wxyz[i]
    return quaternions_wxyz


def to_rotations(quaternions_wxyz):
    """The scipy rotations of quaternions written (w, x, y, z), after
    ``canonicalize``."""
    canonical_wxyz = canonicalize(quaternions_wxyz)
    return Rotation.from_quat(
        np.column_stack(
            [
                canonical_wxyz[:, 1],
                canonical_wxyz[:, 2],
                canonical_wxyz[:, 3],
                canonical_wxyz[:, 0],
            ]
        )
    )
