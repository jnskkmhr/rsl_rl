# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from rsl_rl.algorithms.l2t import L2T


class RecurrentL2T(L2T):
    """Alias of :class:`L2T` for recurrent L2T configurations.

    The :class:`L2T` implementation already supports recurrent teacher/critic/student
    models through rollout storage recurrent mini-batches.
    """

    pass
