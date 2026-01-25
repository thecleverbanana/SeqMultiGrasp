from src.agents import XArm7AllegoRight
import os
from src.utils import torch_3d_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs import Actor, SimConfig, GPUMemoryConfig, SceneConfig, Pose
from mani_skill.envs.sapien_env import BaseEnv

from typing import Optional

import numpy as np
import torch

_MAX_EPISODE_STEPS = 50

_DELTA_XYZ_THRESHOLD = 0.02
_DELTA_RPY_THRESHOLD = 0.3


class ValidatorBase(BaseEnv):

    SUPPORTED_ROBOTS = [XArm7AllegoRight.uid]
    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(self, *args, **kwargs):
        self._initialized = False
        super().__init__(robot_uids=XArm7AllegoRight.uid, *args, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.0, 0.3, 0.2], [0.0, 0.0, 0.05])
        return [
            CameraConfig(
                uid="render_camera",
                pose=pose,
                width=1280,
                height=960,
                fov=1,
                near=0.01,
                far=100,
                shader_pack="default",
            )
        ]

    @property
    def _default_sim_config(self):
        gpu_memory_config = GPUMemoryConfig(
            max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 8,
            max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 2,
            found_lost_pairs_capacity=2**26,
        )
        scene_config = SceneConfig(
            contact_offset=0.02,
            solver_position_iterations=25,
            cpu_workers=min(os.cpu_count(), 4),
        )
        return SimConfig(
            sim_freq=200,
            control_freq=20,
            scene_config=scene_config,
            gpu_memory_config=gpu_memory_config,
        )

    def _load_agent(self, options):
        # Place robot base similarly to your other envs
        initial_agent_poses = Pose.create_from_pq(p=[-0.5, 0.0, 0.0])
        return super()._load_agent(options, initial_agent_poses)

    def _first_initialize_episode(self, env_idx: torch.Tensor, options: dict):
        assert self.control_mode == "pd_joint_pos"
        self._episode_steps = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.int32)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if not self._initialized:
            self._first_initialize_episode(env_idx, options)
            self._initialized = True
        self._episode_steps[env_idx] = 0

    def _after_control_step(self):
        self._episode_steps += 1

    # --- Contact helpers (hand-only, recommended) ---
    def _hand_contact_links(self):
        """
        Returns the Allegro finger contact links + base_link (palm) from the agent.
        Assumes XArm7AllegoRight defines:
            _finger_contact_link_names
            _palm_link_name
        """
        links = []
        for name in self.agent._finger_contact_link_names:
            links.append(self.agent.robot.find_link_by_name(name))
        links.append(self.agent.robot.find_link_by_name(self.agent._palm_link_name))
        return links

    def _is_hand_in_contact_with_object(self, obj: Actor, impulse_threshold: float = 1e-2) -> torch.Tensor:
        in_contact = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        for link in self._hand_contact_links():
            impulse = self.scene.get_pairwise_contact_impulses(link, obj)
            in_contact |= torch.norm(impulse, dim=-1) > impulse_threshold

        return in_contact

    def _is_successful(self, obj: Actor, initial_object_pose_in_hand_frame: Pose) -> torch.Tensor:
        # Choose a "hand frame": Allegro base_link pose (not the robot root).
        # This keeps the meaning consistent even though the arm moves.
        hand_base_link = self.agent.robot.find_link_by_name("base_link")
        hand_pose = hand_base_link.pose  # Pose over envs
        object_pose = obj.pose

        object_pose_in_hand_frame: Pose = hand_pose.inv() * object_pose
        delta_object_pose_in_hand_frame = object_pose_in_hand_frame * initial_object_pose_in_hand_frame.inv()

        delta_xyz = torch.norm(delta_object_pose_in_hand_frame.p, dim=-1)
        rpy = torch_3d_utils.quaternion_to_rpy(delta_object_pose_in_hand_frame.q)
        delta_rpy = torch.norm(rpy, p=1, dim=-1)

        success = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        success = success & self._is_hand_in_contact_with_object(obj)

        # Optional stricter checks (uncomment if needed)
        # success = success & (delta_xyz < _DELTA_XYZ_THRESHOLD)
        # success = success & (delta_rpy < _DELTA_RPY_THRESHOLD)

        return success
