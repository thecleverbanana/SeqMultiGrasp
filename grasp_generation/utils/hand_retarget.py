"""
Allegro-to-LEAP retargeting via anatomical joint correspondence.

Joint layout per finger:
    Allegro: [abduction, MCP_flex, PIP, DIP]  (joints 4i+0 .. 4i+3)
    LEAP:    [MCP_flex, abduction, PIP, DIP]  (joints 4i+0 .. 4i+3)
    → first two joints are swapped between the two hands.

Thumb (joints 12–15): mapped directly in order.

Joint values are copied directly (no limit normalization).

Usage:
    retargeter = AllegroToLeapRetargeter(allegro_model, leap_model, device=device)
    leap_joints = retargeter.retarget(allegro_full_joints)    # [23]
    leap_batch  = retargeter.retarget_batch(allegro_batch)    # [B, 23]
"""

import torch

N_ARM  = 7   # xarm7_joint1–7
N_HAND = 16  # leap_joint0–15 / allegro joint_0.0–15.0

# Source (Allegro) hand-joint indices read in anatomical order for LEAP
# For each long finger i (0,1,2): swap positions 0 and 1 within the group
#   Allegro group [abduction=4i, MCP=4i+1, PIP=4i+2, DIP=4i+3]
#   → read as   [MCP=4i+1, abduction=4i, PIP=4i+2, DIP=4i+3]  for LEAP slot
# Thumb (i=3, joints 12-15): same order
_SRC_IDX = []
for _i in range(3):
    _b = 4 * _i
    _SRC_IDX.extend([_b + 1, _b + 0, _b + 2, _b + 3])
_SRC_IDX.extend([12, 13, 14, 15])          # thumb — direct
_SRC_IDX = torch.tensor(_SRC_IDX)          # [16]


class AllegroToLeapRetargeter:
    """
    Retarget Allegro hand joint positions to LEAP hand.

    Strategy: reorder Allegro joints into LEAP anatomical order and copy
    values directly (no limit normalization).  No FK / optimisation needed.
    """

    def __init__(self, allegro_model, leap_model, device='cpu'):
        self.device  = device
        self.src_idx = _SRC_IDX.to(device)          # [16] — index into Allegro hand joints

    # ------------------------------------------------------------------
    def retarget(self, allegro_full_joints):
        """Single-sample retargeting.  Input / output: [23] tensor."""
        return self.retarget_batch(allegro_full_joints.unsqueeze(0))[0]

    # ------------------------------------------------------------------
    def retarget_batch(self, allegro_batch):
        """
        Batch retargeting.

        Args:
            allegro_batch: [B, 23]  (7 arm + 16 Allegro hand joints)

        Returns:
            leap_batch: [B, 23]  (7 arm + 16 LEAP hand joints)
        """
        arm_joints   = allegro_batch[:, :N_ARM]     # [B, 7]
        allegro_hand = allegro_batch[:, N_ARM:]     # [B, 16]

        # Reorder Allegro joints into LEAP anatomical order (swap abduction↔MCP per finger)
        leap_hand = allegro_hand[:, self.src_idx]   # [B, 16]

        return torch.cat([arm_joints, leap_hand], dim=1).detach()
