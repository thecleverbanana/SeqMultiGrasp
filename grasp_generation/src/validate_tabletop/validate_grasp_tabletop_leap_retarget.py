"""
Tabletop validation for LEAP hand — simulation version.

Mirrors validate_grasp_tabletop.py but:
  - Retargets Allegro hand joints → LEAP hand joints
  - Spawns the "xarm7_leap_right" robot in ManiSkill
  - Skips WidthMapper (LEAP open/close handled directly)

Trajectory phases (same as Allegro):
  1. Arm moves to pre-grasp pose, hand open  (zeros)
  2. Arm at grasp pose,  hand closes to retargeted joints
  3. Arm at grasp pose,  hand holds
  4. Arm lifts up,       hand holds

Usage (library):
    from src.validate_tabletop.validate_grasp_tabletop_leap_retarget import (
        validate_one_object_tabletop_leap_retarget,
    )
    indices = validate_one_object_tabletop_leap_retarget(data_path, debug=True)

Usage (script):
    python -m src.validate_tabletop.validate_grasp_tabletop_leap_retarget \
        --data_path <path.npy> [--debug]
"""

import copy
import os
import sys
import time
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
import transforms3d
import yaml

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Pose as ManiSkillPose
from mani_skill.utils.wrappers import RecordEpisode

from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuRoboPose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.util.logger import setup_logger

setup_logger('warning')

from src.consts import get_object_mesh_path
from src.utils.misc import maniskill_transform_translation_rpy_batched
from src.utils import torch_3d_utils

from .utils import (
    _load_data,
    compute_qpos_and_success,
    make_world_config_empty,
    normalize,
    run_interpolated_trajectory,
)
from .validate_grasp_tabletop import make_world_config_object

# ---- paths for retargeter -----------------------------------------------------
_HERE        = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
_ASSET_DIR   = os.path.join(_REPO_ROOT, 'grasp_generation', 'assets', 'urdf', 'xarm7_dexhand')
_CUROBO_ROOT = os.path.join(_REPO_ROOT, 'third-party', 'curobo', 'src', 'curobo', 'content')

ALLEGRO_URDF    = os.path.join(_ASSET_DIR, 'xarm7_allegro_right.urdf')
LEAP_URDF       = os.path.join(_ASSET_DIR, 'xarm7_leap_right.urdf')
ALLEGRO_SPHERES = os.path.join(_CUROBO_ROOT, 'configs', 'robot', 'spheres', 'xarm7_allegro_right.yml')
LEAP_SPHERES    = os.path.join(_CUROBO_ROOT, 'configs', 'robot', 'spheres', 'xarm7_leap_right_spheres.yml')

N_ARM  = 7
N_HAND = 16

# LEAP open-hand pose used during arm approach (all joints at 0 = fingers extended)
_LEAP_HAND_OPEN = [0.0] * N_HAND


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_one_object_tabletop_leap_retarget(
    data_path: str,
    debug: bool = False,
    rotate_hand: bool = False,
    vis: bool = False,
) -> List[int]:
    """
    Args:
        data_path:   .npy or .h5 validated grasp file (Allegro format).
        debug:       If True, process only 1 grasp and save a debug video.
        rotate_hand: Rotate hand after lifting (matches Allegro validation).
        vis:         If True, open an interactive ManiSkill viewer to watch
                     the simulation replay (implies single-grasp mode).

    Returns:
        List of sample indices where the LEAP hand successfully lifted the object.
    """
    from src.envs.evaluator_xarm import OneObjectV0  # registers env with gym

    object_name, hand_pose_relative_to_object, allegro_hand_qpos = _load_data(
        data_path)
    if object_name is None:
        return []

    if debug or vis:
        hand_pose_relative_to_object = hand_pose_relative_to_object[:1]
        allegro_hand_qpos            = allegro_hand_qpos[:1]

    tensor_args = TensorDeviceType()
    hand_pose_relative_to_object = hand_pose_relative_to_object.to(
        tensor_args.dtype).to(tensor_args.device)
    allegro_hand_qpos = allegro_hand_qpos.to(
        tensor_args.dtype).to(tensor_args.device)
    batch_size = hand_pose_relative_to_object.shape[0]

    # ------------------------------------------------------------------ #
    # 1. World config  (identical to Allegro validation)                  #
    # ------------------------------------------------------------------ #
    initial_agent_poses = ManiSkillPose.create_from_pq(p=[-0.5, 0, 0]).to(
        tensor_args.device)
    z_top           = 0.083
    static_box_pos  = np.array([0.0, 0.0, z_top / 2])
    static_box_dims = np.array([0.5, 0.8, z_top])
    object_xy       = [0.0, 0.0]

    import trimesh as _trimesh
    object_mesh = _trimesh.load(get_object_mesh_path(object_name))
    object_z    = -object_mesh.bounding_box.bounds[0, 2]

    initial_object_pose = ManiSkillPose.create_from_pq(
        p=[object_xy[0], object_xy[1], object_z + z_top])
    object_pose_in_root_frame = initial_agent_poses.inv() * initial_object_pose

    static_box_pose_in_root_frame = (
        initial_agent_poses.inv()
        * ManiSkillPose.create_from_pq(p=static_box_pos)
    ).raw_pose.squeeze(0).cpu().numpy().tolist()

    table_pose_world = ManiSkillPose.create_from_pq(
        p=[0.38, 0, -0.9196429 / 2],
        q=transforms3d.euler.euler2quat(0, 0, np.pi / 2),
    )
    table_pose_in_root_frame = (
        initial_agent_poses.inv() * table_pose_world
    ).raw_pose.squeeze(0).cpu().numpy().tolist()

    world_cfg_object = make_world_config_object(
        object_file_path=get_object_mesh_path(object_name),
        object_pose=object_pose_in_root_frame.raw_pose.squeeze(0).cpu().numpy().tolist(),
        table_pose=table_pose_in_root_frame,
        sponge_pose=static_box_pose_in_root_frame,
        sponge_dims=static_box_dims.tolist(),
    )

    # ------------------------------------------------------------------ #
    # 2. IK  (Allegro config — same arm approach trajectory)              #
    # ------------------------------------------------------------------ #
    ik_solver = _initialize_ik_solver(world_cfg_object, tensor_args)

    hand_pose_xyz, hand_pose_rpy = hand_pose_relative_to_object.split([3, 3], dim=-1)
    hand_pose_xyz, hand_pose_rpy = maniskill_transform_translation_rpy_batched(
        hand_pose_xyz, hand_pose_rpy)

    rpy_offset = [
        float(os.getenv("TABLETOP_GRASP_ROLL_OFFSET",  "0.0")),
        float(os.getenv("TABLETOP_GRASP_PITCH_OFFSET", "0.0")),
        float(os.getenv("TABLETOP_GRASP_YAW_OFFSET",   "0.0")),
    ]
    if any(abs(v) > 0 for v in rpy_offset):
        hand_pose_rpy = hand_pose_rpy + torch.tensor(
            rpy_offset, device=hand_pose_rpy.device, dtype=hand_pose_rpy.dtype)

    hand_pose_quat = torch_3d_utils.rpy_to_quaternion(hand_pose_rpy)
    hand_pose_ms   = ManiSkillPose.create_from_pq(hand_pose_xyz, hand_pose_quat)

    object_pose_root_dev = ManiSkillPose(
        object_pose_in_root_frame.raw_pose.to(tensor_args.dtype).to(tensor_args.device))
    hand_grasp_pose = object_pose_root_dev * hand_pose_ms
    hand_grasp_pose.p[:, 0] += float(os.getenv("TABLETOP_GRASP_X_OFFSET", "-0.05"))

    norm_object = normalize(hand_pose_ms.get_p())

    hand_pre_grasp_pose = copy.deepcopy(hand_grasp_pose)
    pregrasp_offset = float(os.getenv("TABLETOP_PREGRASP_OFFSET", "0.05"))
    hand_pre_grasp_pose.set_p(
        hand_grasp_pose.get_p() + pregrasp_offset * norm_object)

    qpos_pre_grasp, pre_grasp_ok = compute_qpos_and_success(
        ik_solver, hand_pre_grasp_pose)
    print(f"Pre-grasp IK: {pre_grasp_ok.sum().item()} / {batch_size}")

    world_cfg_no_object = make_world_config_empty(
        table_pose=table_pose_in_root_frame,
        table_dims=[2.418, 1.209, 0.9196429],
        sponge_pose=static_box_pose_in_root_frame,
        sponge_dims=static_box_dims.tolist(),
    )
    if ik_solver.world_coll_checker is not None:
        ik_solver.world_coll_checker.clear_cache()
        ik_solver.update_world(world_cfg_no_object)

    qpos_grasp, grasp_ok = compute_qpos_and_success(ik_solver, hand_grasp_pose)
    print(f"Grasp IK:     {grasp_ok.sum().item()} / {batch_size}")

    hand_lift_pose = copy.deepcopy(hand_grasp_pose)
    hand_lift_pose.p[:, 2] += 0.2
    qpos_lift, lift_ok = compute_qpos_and_success(ik_solver, hand_lift_pose)

    success_mask = pre_grasp_ok & grasp_ok & lift_ok
    print(f"Combined IK:  {success_mask.sum().item()} / {batch_size}")

    if not success_mask.any():
        return []

    if rotate_hand:
        hand_rotated_pose = copy.deepcopy(hand_grasp_pose)
        hand_rotated_pose.q = torch_3d_utils.rpy_to_quaternion(
            torch.tensor([torch.pi / 2, 0, torch.pi / 2],
                         dtype=tensor_args.dtype,
                         device=tensor_args.device).unsqueeze(0).repeat(batch_size, 1))
        qpos_rotated, rotated_ok = compute_qpos_and_success(
            ik_solver, hand_rotated_pose)
        success_mask = success_mask & rotated_ok
    else:
        qpos_rotated = None

    # ------------------------------------------------------------------ #
    # 3. Retarget Allegro → LEAP                                          #
    # ------------------------------------------------------------------ #
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'grasp_generation'))
    from utils.robot_model_lite import RobotModelURDFLite
    from utils.hand_retarget import AllegroToLeapRetargeter

    with open(ALLEGRO_SPHERES) as f:
        allegro_spheres = yaml.safe_load(f)['collision_spheres']
    with open(LEAP_SPHERES) as f:
        leap_spheres = yaml.safe_load(f)['collision_spheres']

    a_model = RobotModelURDFLite(ALLEGRO_URDF, device='cpu', collision_spheres=allegro_spheres)
    l_model = RobotModelURDFLite(LEAP_URDF,    device='cpu', collision_spheres=leap_spheres)
    retargeter = AllegroToLeapRetargeter(a_model, l_model, device='cpu')

    dummy_arm      = torch.zeros(batch_size, N_ARM)
    allegro_cpu    = allegro_hand_qpos.cpu()
    leap_full      = retargeter.retarget_batch(torch.cat([dummy_arm, allegro_cpu], dim=1))
    leap_hand_qpos = leap_full[:, N_ARM:].to(tensor_args.device).to(tensor_args.dtype)

    # ------------------------------------------------------------------ #
    # 4. Simulation                                                       #
    # ------------------------------------------------------------------ #
    leap_hand_open = torch.tensor(
        _LEAP_HAND_OPEN, device=tensor_args.device,
        dtype=tensor_args.dtype).unsqueeze(0).repeat(batch_size, 1)

    initial_joint_positions = torch.cat([qpos_pre_grasp, leap_hand_open], dim=-1)

    static_box_pose_ms = (
        initial_agent_poses
        * ManiSkillPose.create_from_pq(
            p=static_box_pose_in_root_frame[:3],
            q=static_box_pose_in_root_frame[3:])
    )
    static_box_pos_world = static_box_pose_ms.p.squeeze(0).cpu().numpy()

    env = gym.make(
        "OneObject-v0",
        num_envs=batch_size,
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode="pd_joint_pos",
        render_mode="human" if vis else "rgb_array",
        sim_backend="cpu" if vis else "gpu",
        object_name=object_name,
        object_xy=object_xy,
        robot_uid="xarm7_leap_right",
        initial_joint_positions=initial_joint_positions.clone(),
        initial_agent_poses=initial_agent_poses,
        static_box_pos=static_box_pos_world,
        static_box_dims=static_box_dims,
    )

    if debug and not vis:
        env = RecordEpisode(
            env,
            output_dir='mp_logs_leap',
            save_trajectory=False,
            save_video=True,
            max_steps_per_video=500,
            video_fps=int(env.unwrapped.control_freq),
        )

    env.reset(seed=0)
    print(f"LEAP hand joint names (sim): {env.unwrapped.agent.hand_joint_names}")

    # Open interactive SAPIEN viewer when vis=True
    viewer = None
    if vis:
        viewer = env.unwrapped.render_human()

    def _refresh_viewer():
        if viewer is not None:
            env.unwrapped.render_human()
            time.sleep(1.0 / 30)  # ~30 fps so the animation is visible

    run_interpolated_trajectory(
        env=env,
        success_indices=success_mask,
        qpos_hand_pre_grasp_pose=qpos_pre_grasp,
        qpos_hand_grasp_pose=qpos_grasp,
        qpos_hand_lift_pose=qpos_lift,
        qpos_hand_rotated_pose=qpos_rotated,
        # LEAP: open → retargeted → retargeted (no WidthMapper squeeze)
        pre_grasp_hand_qpos=leap_hand_open,
        grasp_hand_qpos=leap_hand_qpos,
        target_grasp_hand_qpos=leap_hand_qpos,
        initial_joint_positions=initial_joint_positions,
        action_reorder_idx=None,
        on_step=_refresh_viewer if vis else None,
        steps_per_phase=50,
        rotate_hand=rotate_hand,
    )

    info    = env.unwrapped.evaluate()
    lifted  = (info["object_lifted_height"] > 0.1).to(success_mask.device)
    n_contact = info["n_contact"]

    print(f"Lifted:    {lifted.sum().item()} / {batch_size}")
    print(f"n_contact: {n_contact.min().item()} – {n_contact.max().item()}")

    success_mask = success_mask & lifted

    if vis and viewer is not None:
        print("\n[vis] Simulation finished. Interact with the viewer.")
        print("[vis] Press Enter in the terminal to close...")
        try:
            input()
        except EOFError:
            pass

    env.close()

    return success_mask.cpu().nonzero().squeeze(1).tolist()


# ---------------------------------------------------------------------------
# IK solver (Allegro config — same arm kinematics)
# ---------------------------------------------------------------------------

def _initialize_ik_solver(world_cfg: WorldConfig,
                           tensor_args: TensorDeviceType) -> IKSolver:
    robot_cfg_data = load_yaml(
        join_path(get_robot_configs_path(), "xarm7_allegro_right.yml"))
    robot_cfg = RobotConfig.from_dict(robot_cfg_data["robot_cfg"])

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg, world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        store_debug=True,
        self_collision_check=False,
        self_collision_opt=False,
        collision_activation_distance=0.0,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    return IKSolver(ik_config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import tyro
    from dataclasses import dataclass

    @dataclass
    class Args:
        data_path: str
        debug: bool = False
        rotate_hand: bool = False
        vis: bool = False

    args = tyro.cli(Args)
    indices = validate_one_object_tabletop_leap_retarget(
        args.data_path,
        debug=args.debug,
        rotate_hand=args.rotate_hand,
        vis=args.vis,
    )
    print(f"\nSuccess indices ({len(indices)}): {indices}")
