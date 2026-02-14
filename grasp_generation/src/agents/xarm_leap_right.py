import os
from typing import Dict, List

import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import BaseAgent, DictControllerConfig, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, deepcopy_dict
from mani_skill.agents.registration import register_agent
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs import Actor, Link

from src.consts import _ASSET_PATH


@register_agent()
class XArm7LeapRight(BaseAgent):
    uid = "xarm7_leap_right"
    urdf_path = os.path.join(
        _ASSET_PATH, "urdf/xarm7_dexhand/xarm7_leap_right.urdf"
    )

    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        ),
        link={
            "fingertip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "fingertip_2": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "fingertip_3": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "thumb_fingertip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )

    # Debug / default pose
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                # xArm7 neutral-ish pose (edit to taste)
                [0.0, -0.6, 0.0, 1.2, 0.0, 1.2, 0.0]
                # Leap hand pose (open hand)
                + [0.0] * 16,
                dtype=np.float32,
            ),
            pose=sapien.Pose(p=np.array([-0.5, 0.0, 0.0])),
        )
    )

    arm_joint_names = [
        "xarm7_joint1",
        "xarm7_joint2",
        "xarm7_joint3",
        "xarm7_joint4",
        "xarm7_joint5",
        "xarm7_joint6",
        "xarm7_joint7",
    ]

    qmin_arm = torch.tensor(
        [
            -6.28318530718,
            -2.059,
            -6.28318530718,
            -0.19198,
            -6.28318530718,
            -1.69297,
            -6.28318530718,
        ]
    )
    qmax_arm = torch.tensor(
        [
            6.28318530718,
            2.0944,
            6.28318530718,
            3.927,
            6.28318530718,
            3.14159265359,
            6.28318530718,
        ]
    )

    qmin_hand = torch.tensor(
        [
            -1.047, -0.314, -0.506, -0.366,
            -1.047, -0.314, -0.506, -0.366,
            -1.047, -0.314, -0.506, -0.366,
            -0.349, -0.47, -1.20, -1.34,
        ]
    )
    qmax_hand = torch.tensor(
        [
            1.047, 2.23, 1.885, 2.042,
            1.047, 2.23, 1.885, 2.042,
            1.047, 2.23, 1.885, 2.042,
            2.094, 2.443, 1.90, 1.88,
        ]
    )

    hand_joint_names = [
        "leap_joint0",
        "leap_joint1",
        "leap_joint2",
        "leap_joint3",
        "leap_joint4",
        "leap_joint5",
        "leap_joint6",
        "leap_joint7",
        "leap_joint8",
        "leap_joint9",
        "leap_joint10",
        "leap_joint11",
        "leap_joint12",
        "leap_joint13",
        "leap_joint14",
        "leap_joint15",
    ]

    _finger_contact_link_names = [
        "thumb_fingertip", "thumb_dip", "thumb_pip", "pip_4",  # thumb
        "fingertip_1", "dip_1", "pip_1", "mcp_joint_1",        # index
        "fingertip_2", "dip_2", "pip_2", "mcp_joint_2",        # middle
        "fingertip_3", "dip_3", "pip_3", "mcp_joint_3",        # ring
    ]

    _palm_link_name = "palm_link"

    @property
    def _finger_contact_links(self) -> List[Link]:
        return [self.robot.find_link_by_name(name) for name in self._finger_contact_link_names]

    @property
    def _palm_link(self) -> Link:
        return self.robot.find_link_by_name(self._palm_link_name)

    def _get_net_boolean_contact(self, impulse_threshold: float = 1e-2) -> torch.Tensor:
        t = []
        for link in self._finger_contact_links:
            forces = link.get_net_contact_forces()
            t.append(torch.norm(forces, dim=-1) > impulse_threshold)

        t.append(torch.norm(self._palm_link.get_net_contact_forces(), dim=-1) > impulse_threshold)
        return torch.stack(t, dim=-1)  # (n, 16)

    def _get_boolean_contact(
        self,
        scene: ManiSkillScene,
        obj: Actor,
        impulse_threshold: float = 1e-2,
    ) -> torch.Tensor:
        t = []
        for link in self._finger_contact_links:
            impulse = scene.get_pairwise_contact_impulses(obj, link)
            t.append(torch.norm(impulse, dim=-1) > impulse_threshold)

        impulse = scene.get_pairwise_contact_impulses(obj, self._palm_link)
        t.append(torch.norm(impulse, dim=-1) > impulse_threshold)
        return torch.stack(t, dim=-1)  # (n, 16)

    def __init__(
        self,
        scene: ManiSkillScene,
        control_freq: int,
        control_mode: str = None,
        agent_idx: int = None,
        initial_pose=None,
        *args,
        **kwargs,
    ):
        self.arm_joint_pos_stiffness = 1000
        self.arm_joint_pos_damping = 100
        self.arm_force_limit = 100

        self.hand_stiffness = 4e2
        self.hand_damping = 1e1
        self.hand_force_limit = 5e1

        self.ee_link_name = "link7"

        super().__init__(scene, control_freq, control_mode, agent_idx, initial_pose, *args, **kwargs)

    def _make_pd_joint_pos_hand_config(self):
        return PDJointPosControllerConfig(
            self.hand_joint_names,
            None,
            None,
            self.hand_stiffness,
            self.hand_damping,
            self.hand_force_limit,
            use_delta=False,
            normalize_action=False,
        )

    def _make_pd_joint_pos_config(self) -> Dict[str, PDJointPosControllerConfig]:
        """
        Absolute joint position control for both arm and hand.
        """
        arm_pd_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_joint_pos_stiffness,
            self.arm_joint_pos_damping,
            self.arm_force_limit,
            use_delta=False,
            normalize_action=False,
        )
        hand_pd_pos = self._make_pd_joint_pos_hand_config()

        return dict(arm=arm_pd_pos, gripper=hand_pd_pos)

    @property
    def _controller_configs(self) -> Dict[str, DictControllerConfig]:
        controller_configs = dict(
            pd_joint_pos=self._make_pd_joint_pos_config(),
        )
        return deepcopy_dict(controller_configs)
