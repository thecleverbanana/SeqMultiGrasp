"""
Direct replay of grasps in ManiSkill simulation with interactive viewer.

Supports both original Allegro and retargeted LEAP hands.

Usage:
    # Replay with LEAP hand (default)
    python leap_sim_replay.py \
        --data-path ../data/experiments/.../cylinder_..._tabletop_validated.npy

    # Replay with original Allegro hand
    python leap_sim_replay.py --data-path <path.npy> --hand allegro

    # Pick a specific grasp index
    python leap_sim_replay.py --data-path <path.npy> --index 3

    # Slow-motion (lower fps = slower)
    python leap_sim_replay.py --data-path <path.npy> --fps 10
"""

import copy
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

import gymnasium as gym
import numpy as np
import torch
import transforms3d
import yaml
import tyro
import trimesh as _trimesh

# ---- project paths -------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))

from mani_skill.utils.structs import Pose as ManiSkillPose

from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuRoboPose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.util.logger import setup_logger

setup_logger("warning")

from src.consts import get_object_mesh_path
from src.utils.misc import maniskill_transform_translation_rpy_batched
from src.utils import torch_3d_utils
from src.validate_tabletop.utils import (
    _load_data,
    normalize,
    run_interpolated_trajectory,
)
from src.validate_tabletop.validate_grasp_tabletop import make_world_config_object
from src.validate_tabletop.utils import make_world_config_empty, compute_qpos_and_success

# ---- retargeter paths -----------------------------------------------------
_ASSET_DIR = os.path.join(_HERE, 'assets', 'urdf', 'xarm7_dexhand')
_CUROBO_ROOT = os.path.join(_HERE, '..', 'third-party', 'curobo',
                             'src', 'curobo', 'content')

ALLEGRO_URDF = os.path.join(_ASSET_DIR, 'xarm7_allegro_right.urdf')
LEAP_URDF = os.path.join(_ASSET_DIR, 'xarm7_leap_right.urdf')
ALLEGRO_SPHERES = os.path.join(_CUROBO_ROOT, 'configs', 'robot', 'spheres',
                                'xarm7_allegro_right.yml')
LEAP_SPHERES = os.path.join(_CUROBO_ROOT, 'configs', 'robot', 'spheres',
                             'xarm7_leap_right_spheres.yml')

N_ARM = 7
N_HAND = 16


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class Args:
    data_path: str
    index: int = 0
    hand: Literal["leap", "allegro"] = "leap"
    fps: int = 30
    steps_per_phase: int = 50


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args):
    from src.envs.evaluator_xarm import OneObjectV0  # registers gym env

    data_path = args.data_path
    idx = args.index
    fps = args.fps
    use_leap = args.hand == "leap"

    # ---- Load grasp data -------------------------------------------------
    object_name, hand_pose_rel, allegro_hand_qpos = _load_data(data_path)
    if object_name is None:
        print("No data found in", data_path)
        return
    n_total = hand_pose_rel.shape[0]
    print(f"Loaded {n_total} grasps for '{object_name}', replaying index {idx}")
    print(f"  Hand: {args.hand}")

    # pick single grasp
    hand_pose_rel = hand_pose_rel[idx : idx + 1]
    allegro_hand_qpos = allegro_hand_qpos[idx : idx + 1]

    tensor_args = TensorDeviceType()
    hand_pose_rel = hand_pose_rel.to(tensor_args.dtype).to(tensor_args.device)
    allegro_hand_qpos = allegro_hand_qpos.to(tensor_args.dtype).to(tensor_args.device)
    batch_size = 1

    # ---- Determine hand joints to use ------------------------------------
    if use_leap:
        from utils.robot_model_lite import RobotModelURDFLite
        from utils.hand_retarget import AllegroToLeapRetargeter

        with open(ALLEGRO_SPHERES) as f:
            allegro_spheres = yaml.safe_load(f)['collision_spheres']
        with open(LEAP_SPHERES) as f:
            leap_spheres = yaml.safe_load(f)['collision_spheres']

        a_model = RobotModelURDFLite(ALLEGRO_URDF, device='cpu',
                                      collision_spheres=allegro_spheres)
        l_model = RobotModelURDFLite(LEAP_URDF, device='cpu',
                                      collision_spheres=leap_spheres)
        retargeter = AllegroToLeapRetargeter(a_model, l_model, device='cpu')

        dummy_arm = torch.zeros(batch_size, N_ARM)
        leap_full = retargeter.retarget_batch(
            torch.cat([dummy_arm, allegro_hand_qpos.cpu()], dim=1))
        hand_qpos = leap_full[:, N_ARM:].to(tensor_args.device).to(tensor_args.dtype)
        robot_uid = "xarm7_leap_right"
        hand_open = torch.zeros(batch_size, N_HAND,
                                device=tensor_args.device, dtype=tensor_args.dtype)

        print(f"  Allegro hand: {allegro_hand_qpos[0].cpu().numpy().round(3).tolist()}")
        print(f"  LEAP hand:    {hand_qpos[0].cpu().numpy().round(3).tolist()}")
    else:
        hand_qpos = allegro_hand_qpos
        robot_uid = "xarm7_allegro_right"
        hand_open = torch.zeros(batch_size, N_HAND,
                                device=tensor_args.device, dtype=tensor_args.dtype)

        print(f"  Allegro hand: {hand_qpos[0].cpu().numpy().round(3).tolist()}")

    # ---- Scene geometry (same as validation) ------------------------------
    initial_agent_poses = ManiSkillPose.create_from_pq(p=[-0.5, 0, 0]).to(
        tensor_args.device)
    z_top = 0.083
    static_box_pos = np.array([0.0, 0.0, z_top / 2])
    static_box_dims = np.array([0.5, 0.8, z_top])
    object_xy = [0.0, 0.0]

    object_mesh = _trimesh.load(get_object_mesh_path(object_name))
    object_z = -object_mesh.bounding_box.bounds[0, 2]

    initial_object_pose = ManiSkillPose.create_from_pq(
        p=[object_xy[0], object_xy[1], object_z + z_top])
    object_pose_in_root = initial_agent_poses.inv() * initial_object_pose

    static_box_pose_root = (
        initial_agent_poses.inv()
        * ManiSkillPose.create_from_pq(p=static_box_pos)
    ).raw_pose.squeeze(0).cpu().numpy().tolist()

    table_pose_world = ManiSkillPose.create_from_pq(
        p=[0.38, 0, -0.9196429 / 2],
        q=transforms3d.euler.euler2quat(0, 0, np.pi / 2))
    table_pose_root = (
        initial_agent_poses.inv() * table_pose_world
    ).raw_pose.squeeze(0).cpu().numpy().tolist()

    # ---- IK for arm poses ------------------------------------------------
    world_cfg_obj = make_world_config_object(
        object_file_path=get_object_mesh_path(object_name),
        object_pose=object_pose_in_root.raw_pose.squeeze(0).cpu().numpy().tolist(),
        table_pose=table_pose_root,
        sponge_pose=static_box_pose_root,
        sponge_dims=static_box_dims.tolist(),
    )

    robot_cfg_data = load_yaml(
        join_path(get_robot_configs_path(), "xarm7_allegro_right.yml"))
    robot_cfg = RobotConfig.from_dict(robot_cfg_data["robot_cfg"])
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg, world_cfg_obj,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=False,
        self_collision_opt=False,
        collision_activation_distance=0.0,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)

    # Compute hand poses
    hand_xyz, hand_rpy = hand_pose_rel.split([3, 3], dim=-1)
    hand_xyz, hand_rpy = maniskill_transform_translation_rpy_batched(hand_xyz, hand_rpy)
    hand_quat = torch_3d_utils.rpy_to_quaternion(hand_rpy)
    hand_ms = ManiSkillPose.create_from_pq(hand_xyz, hand_quat)

    obj_pose_dev = ManiSkillPose(
        object_pose_in_root.raw_pose.to(tensor_args.dtype).to(tensor_args.device))
    hand_grasp_pose = obj_pose_dev * hand_ms
    hand_grasp_pose.p[:, 0] += float(os.getenv("TABLETOP_GRASP_X_OFFSET", "-0.05"))

    norm_obj = normalize(hand_ms.get_p())

    hand_pre_grasp_pose = copy.deepcopy(hand_grasp_pose)
    pregrasp_offset = float(os.getenv("TABLETOP_PREGRASP_OFFSET", "0.05"))
    hand_pre_grasp_pose.set_p(
        hand_grasp_pose.get_p() + pregrasp_offset * norm_obj)

    # Solve IK
    qpos_pre, ok_pre = compute_qpos_and_success(ik_solver, hand_pre_grasp_pose)
    print(f"  Pre-grasp IK: {'OK' if ok_pre.all() else 'FAIL'}")

    world_cfg_empty = make_world_config_empty(
        table_pose=table_pose_root,
        table_dims=[2.418, 1.209, 0.9196429],
        sponge_pose=static_box_pose_root,
        sponge_dims=static_box_dims.tolist(),
    )
    if ik_solver.world_coll_checker is not None:
        ik_solver.world_coll_checker.clear_cache()
        ik_solver.update_world(world_cfg_empty)

    qpos_grasp, ok_grasp = compute_qpos_and_success(ik_solver, hand_grasp_pose)
    print(f"  Grasp IK:     {'OK' if ok_grasp.all() else 'FAIL'}")

    hand_lift_pose = copy.deepcopy(hand_grasp_pose)
    hand_lift_pose.p[:, 2] += 0.2
    qpos_lift, ok_lift = compute_qpos_and_success(ik_solver, hand_lift_pose)
    print(f"  Lift IK:      {'OK' if ok_lift.all() else 'FAIL'}")

    ik_ok = ok_pre & ok_grasp & ok_lift
    if not ik_ok.all():
        print("WARNING: IK failed for some poses — continuing anyway for debugging")

    # ---- Create ManiSkill env with viewer --------------------------------
    initial_joints = torch.cat([qpos_pre, hand_open], dim=-1)

    static_box_pose_ms = (
        initial_agent_poses
        * ManiSkillPose.create_from_pq(
            p=static_box_pose_root[:3], q=static_box_pose_root[3:])
    )
    static_box_pos_world = static_box_pose_ms.p.squeeze(0).cpu().numpy()

    print(f"\nCreating ManiSkill env (CPU, human viewer, robot={robot_uid})...")
    env = gym.make(
        "OneObject-v0",
        num_envs=1,
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode="pd_joint_pos",
        render_mode="human",
        sim_backend="cpu",
        object_name=object_name,
        object_xy=object_xy,
        robot_uid=robot_uid,
        initial_joint_positions=initial_joints.clone(),
        initial_agent_poses=initial_agent_poses,
        static_box_pos=static_box_pos_world,
        static_box_dims=static_box_dims,
    )

    env.reset(seed=0)

    # ---- Print joint ordering diagnostics --------------------------------
    active_joint_names = [j.get_name() for j in env.unwrapped.agent.robot.get_active_joints()]
    print(f"\n  Active joints (SAPIEN order): {active_joint_names}")
    print(f"  Agent hand_joint_names:       {env.unwrapped.agent.hand_joint_names}")

    actual_qpos = env.unwrapped.agent.robot.get_qpos()
    print(f"  Actual qpos after reset:      {actual_qpos[0].cpu().numpy().round(3).tolist()}")
    print(f"  Desired initial joints:       {initial_joints[0].cpu().numpy().round(3).tolist()}")

    # ---- Open viewer -----------------------------------------------------
    viewer = env.unwrapped.render_human()
    print("\n--- Viewer is open. Press Enter here to start the grasp trajectory ---")
    try:
        input()
    except EOFError:
        pass

    # ---- Run trajectory with viewer refresh ------------------------------
    success_mask = torch.ones(batch_size, dtype=torch.bool, device=tensor_args.device)
    frame_dt = 1.0 / fps

    phase_names = [
        "1/6  Approach (arm → grasp, hand open)",
        "2/6  Pre-close (hand open → retargeted)",
        "3/6  Grasp close",
        "4/6  Hold grasp",
        "5/6  Lift",
        "6/6  Final hold",
    ]
    phase_idx = [0]

    def on_step():
        env.unwrapped.render_human()
        time.sleep(frame_dt)

    steps = args.steps_per_phase
    phases = [
        # (arm_start, arm_end, hand_start, hand_end)
        (qpos_pre, qpos_grasp, hand_open, hand_open),        # approach
        (qpos_grasp, qpos_grasp, hand_open, hand_qpos),      # pre-close
        (qpos_grasp, qpos_grasp, hand_qpos, hand_qpos),      # grasp close
        (qpos_grasp, qpos_grasp, hand_qpos, hand_qpos),      # hold
        (qpos_grasp, qpos_lift, hand_qpos, hand_qpos),       # lift
        (qpos_lift, qpos_lift, hand_qpos, hand_qpos),        # final hold
    ]

    for i, (arm_s, arm_e, hand_s, hand_e) in enumerate(phases):
        print(f"\n  Phase {phase_names[i]}")
        _run_phase(env, success_mask, arm_s, arm_e, hand_s, hand_e,
                   initial_joints, steps, on_step)

        # Print diagnostics after each phase
        info = env.unwrapped.evaluate()
        height = info["object_lifted_height"]
        n_c = info["n_contact"]
        print(f"    lifted_height={height[0].item():.4f}  n_contact={n_c[0].item()}")

        # Pause between phases so user can inspect
        print("    Press Enter for next phase...")
        try:
            input()
        except EOFError:
            pass

    # ---- Final result ----------------------------------------------------
    info = env.unwrapped.evaluate()
    lifted = info["object_lifted_height"][0].item()
    n_contact = info["n_contact"][0].item()
    print(f"\n=== RESULT: lifted_height={lifted:.4f}  n_contact={n_contact}")
    print(f"=== {'SUCCESS' if lifted > 0.1 else 'FAIL'}")

    print("\nViewer still open. Press Enter to close.")
    try:
        input()
    except EOFError:
        pass

    env.close()


def _run_phase(env, success_mask, arm_start, arm_end, hand_start, hand_end,
               initial_joints, n_steps, on_step_fn):
    """Run one interpolation phase."""
    device = initial_joints.device
    dtype = initial_joints.dtype
    batch_size = initial_joints.shape[0]
    arm_dof = N_ARM
    hand_dof = N_HAND
    initial_arm = initial_joints[:, :arm_dof]
    initial_hand = initial_joints[:, arm_dof:arm_dof + hand_dof]

    for i in range(n_steps):
        alpha = i / max(n_steps - 1, 1)
        action = torch.zeros(batch_size, arm_dof + hand_dof,
                             device=device, dtype=dtype)

        arm_interp = arm_start + alpha * (arm_end - arm_start)
        hand_interp = hand_start + alpha * (hand_end - hand_start)

        action[success_mask, :arm_dof] = arm_interp[success_mask]
        action[~success_mask, :arm_dof] = initial_arm[~success_mask]
        action[success_mask, arm_dof:] = hand_interp[success_mask]
        action[~success_mask, arm_dof:] = initial_hand[~success_mask]

        env.step(action)
        if on_step_fn is not None:
            on_step_fn()


if __name__ == "__main__":
    main(tyro.cli(Args))
