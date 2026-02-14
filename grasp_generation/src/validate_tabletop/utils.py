import torch
import numpy as np

import copy
import h5py
import os

import transforms3d
import transforms3d.quaternions

import gymnasium as gym

from mani_skill.utils.structs import Pose as ManiSkillPose
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers import RecordEpisode

from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuRoboPose
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionQueryBuffer
from curobo.types.robot import JointState

from src.utils.width_mapper import WidthMapper
from src.utils import torch_3d_utils
from src.utils.misc import maniskill_transform_translation_rpy_batched
# from src.envs.evaluator import OneObjectV0
from src.envs.evaluator_xarm import OneObjectV0


from typing import Any, Dict, List, Optional, Union, Callable, Tuple


def run_interpolated_trajectory(
    env: gym.Env,
    success_indices: torch.Tensor,
    qpos_hand_pre_grasp_pose: torch.Tensor,  # pre-grasp arm pose
    qpos_hand_grasp_pose: torch.Tensor,      # grasp arm pose
    qpos_hand_lift_pose: torch.Tensor,       # lift arm pose
    qpos_hand_rotated_pose: Optional[torch.Tensor],    # rotated arm pose

    pre_grasp_hand_qpos: torch.Tensor,
    grasp_hand_qpos: torch.Tensor,
    target_grasp_hand_qpos: torch.Tensor,

    initial_joint_positions: torch.Tensor,
    action_reorder_idx: Optional[torch.Tensor] = None,
    debug_hook: Optional[Callable[[str], None]] = None,
    on_step: Optional[Callable[[], None]] = None,

    steps_per_phase: int = 50,
    rotate_hand: bool = False,
):


    device = initial_joint_positions.device
    dtype = initial_joint_positions.dtype
    batch_size = initial_joint_positions.shape[0]

    arm_dof = 7
    hand_dof = 16

    initial_arm_qpos = initial_joint_positions[:, :arm_dof]
    initial_hand_qpos = initial_joint_positions[:, arm_dof:arm_dof+hand_dof]

    def linear_interpolate(start: torch.Tensor, end: torch.Tensor, alpha: float) -> torch.Tensor:
        return start + alpha * (end - start)

    def step_env_with_interpolation(
        arm_start: torch.Tensor,
        arm_end: torch.Tensor,
        hand_start: torch.Tensor,
        hand_end: torch.Tensor,
        n_steps: int
    ):

        for i in range(n_steps):
            alpha = i / max(n_steps - 1, 1)

            action = torch.zeros(batch_size, arm_dof +
                                 hand_dof, device=device, dtype=dtype)

            arm_interp = linear_interpolate(arm_start, arm_end, alpha)
            hand_interp = linear_interpolate(hand_start, hand_end, alpha)

            success_mask = success_indices

            # arm
            action[success_mask, :arm_dof] = arm_interp[success_mask]
            action[~success_mask, :arm_dof] = initial_arm_qpos[~success_mask]

            # hand
            action[success_mask, arm_dof:] = hand_interp[success_mask]
            action[~success_mask, arm_dof:] = initial_hand_qpos[~success_mask]

            if action_reorder_idx is not None:
                action = action[:, action_reorder_idx]
            env.step(action)
            if on_step is not None:
                on_step()

    step_env_with_interpolation(
        arm_start=qpos_hand_pre_grasp_pose,
        arm_end=qpos_hand_grasp_pose,
        hand_start=initial_hand_qpos,
        hand_end=initial_hand_qpos,
        n_steps=steps_per_phase
    )
    if debug_hook is not None:
        debug_hook("after_pregrasp_pose")

    step_env_with_interpolation(
        arm_start=qpos_hand_grasp_pose,
        arm_end=qpos_hand_grasp_pose,
        hand_start=initial_hand_qpos,
        hand_end=pre_grasp_hand_qpos,
        n_steps=steps_per_phase
    )
    if debug_hook is not None:
        debug_hook("after_pregrasp_close")

    step_env_with_interpolation(
        arm_start=qpos_hand_grasp_pose,
        arm_end=qpos_hand_grasp_pose,
        hand_start=pre_grasp_hand_qpos,
        hand_end=grasp_hand_qpos,
        n_steps=steps_per_phase
    )
    if debug_hook is not None:
        debug_hook("after_grasp_close")

    step_env_with_interpolation(
        arm_start=qpos_hand_grasp_pose,
        arm_end=qpos_hand_grasp_pose,
        hand_start=grasp_hand_qpos,
        hand_end=target_grasp_hand_qpos,
        n_steps=steps_per_phase
    )
    if debug_hook is not None:
        debug_hook("after_target_close")

    # Hold target grasp to let contacts settle before lifting.
    step_env_with_interpolation(
        arm_start=qpos_hand_grasp_pose,
        arm_end=qpos_hand_grasp_pose,
        hand_start=target_grasp_hand_qpos,
        hand_end=target_grasp_hand_qpos,
        n_steps=steps_per_phase
    )
    if debug_hook is not None:
        debug_hook("after_target_hold")

    step_env_with_interpolation(
        arm_start=qpos_hand_grasp_pose,
        arm_end=qpos_hand_lift_pose,
        hand_start=target_grasp_hand_qpos,
        hand_end=target_grasp_hand_qpos,
        n_steps=steps_per_phase
    )
    if debug_hook is not None:
        debug_hook("after_lift")

    arm_end_qpos = qpos_hand_lift_pose

    if rotate_hand:
        step_env_with_interpolation(
            arm_start=qpos_hand_lift_pose,
            arm_end=qpos_hand_rotated_pose,
            hand_start=target_grasp_hand_qpos,
            hand_end=target_grasp_hand_qpos,
            n_steps=steps_per_phase
        )
        arm_end_qpos = qpos_hand_rotated_pose

    step_env_with_interpolation(
        arm_start=arm_end_qpos,
        arm_end=arm_end_qpos,
        hand_start=target_grasp_hand_qpos,
        hand_end=target_grasp_hand_qpos,
        n_steps=steps_per_phase
    )



def _validate_tabletop(
    ik_solver: IKSolver,
    width_mapper: WidthMapper,
    tensor_args: TensorDeviceType,
    object_name: str,
    object_xy: List[float],
    hand_initial_joint_positions: List[float],
    table_pose_in_root_frame: List[float],
    table_dims: List[float],
    static_box_pose_in_root_frame: np.ndarray,
    static_box_dims: np.ndarray,
    initial_agent_poses: ManiSkillPose,
    object_pose_in_root_frame: ManiSkillPose,
    hand_pose_relative_to_object: torch.Tensor,
    hand_qpos: torch.Tensor,
    output_dir: Optional[str] = None,
    save_video: bool = False,
    rotate_hand: bool = False,
) -> List[int]:

    object_pose_in_root_frame = ManiSkillPose(object_pose_in_root_frame.raw_pose.to(
        tensor_args.device).to(tensor_args.dtype))

    assert hand_pose_relative_to_object.shape[0] == hand_qpos.shape[0]
    batch_size = hand_pose_relative_to_object.shape[0]
    if save_video:
        assert batch_size == 1

    hand_pose_relative_to_object_xyz, hand_pose_relative_to_object_rpy = hand_pose_relative_to_object.split([
        3, 3], dim=-1)
    hand_pose_relative_to_object_xyz, hand_pose_relative_to_object_rpy = maniskill_transform_translation_rpy_batched(
        hand_pose_relative_to_object_xyz, hand_pose_relative_to_object_rpy)

    rpy_offset = [
        float(os.getenv("TABLETOP_GRASP_ROLL_OFFSET", "0.0")),
        float(os.getenv("TABLETOP_GRASP_PITCH_OFFSET", "0.0")),
        float(os.getenv("TABLETOP_GRASP_YAW_OFFSET", "0.0")),
    ]
    if any(abs(val) > 0 for val in rpy_offset):
        offset_tensor = torch.tensor(
            rpy_offset, device=hand_pose_relative_to_object_rpy.device, dtype=hand_pose_relative_to_object_rpy.dtype
        )
        hand_pose_relative_to_object_rpy = hand_pose_relative_to_object_rpy + offset_tensor

    if os.getenv("TABLETOP_GRASP_STATS", "0") == "1":
        with torch.no_grad():
            rel_norm = torch.linalg.norm(hand_pose_relative_to_object_xyz, dim=-1)
            rel_dir = hand_pose_relative_to_object_xyz / rel_norm.clamp_min(1e-6).unsqueeze(-1)
            z_ratio = rel_dir[:, 2].abs()
            print("grasp rel xyz min/max:", hand_pose_relative_to_object_xyz.min(0).values.tolist(),
                  hand_pose_relative_to_object_xyz.max(0).values.tolist())
            print("grasp rel norm min/mean/max:", rel_norm.min().item(), rel_norm.mean().item(), rel_norm.max().item())
            print("grasp rel z_ratio min/mean/max:", z_ratio.min().item(), z_ratio.mean().item(), z_ratio.max().item())

    side_max_ratio = float(os.getenv("TABLETOP_SIDE_GRASP_MAX_Z_RATIO", "-1"))
    if side_max_ratio > 0:
        rel_norm = torch.linalg.norm(hand_pose_relative_to_object_xyz, dim=-1)
        rel_dir = hand_pose_relative_to_object_xyz / rel_norm.clamp_min(1e-6).unsqueeze(-1)
        side_mask = rel_dir[:, 2].abs() <= side_max_ratio
        if side_mask.sum().item() == 0:
            print("No grasps left after side-grasp filter. side_max_ratio:", side_max_ratio)
            return []
        hand_pose_relative_to_object_xyz = hand_pose_relative_to_object_xyz[side_mask]
        hand_pose_relative_to_object_rpy = hand_pose_relative_to_object_rpy[side_mask]
        hand_qpos = hand_qpos[side_mask]
        batch_size = hand_pose_relative_to_object_xyz.shape[0]
        if save_video:
            assert batch_size == 1

    success_indices = torch.ones(
        batch_size, dtype=torch.bool, device=tensor_args.device)

    hand_pose_relative_to_object_quat = torch_3d_utils.rpy_to_quaternion(
        hand_pose_relative_to_object_rpy)

    hand_pose_relative_to_object: ManiSkillPose = ManiSkillPose.create_from_pq(
        hand_pose_relative_to_object_xyz, hand_pose_relative_to_object_quat)

    hand_grasp_pose = object_pose_in_root_frame*hand_pose_relative_to_object

    # root_pose = initial_agent_poses  # or env root pose if available
    # hand_grasp_pose = root_pose * hand_pose_relative_to_object

    # X/Y/Z offsets for xarm tuning (env overridable).
    x_offset = float(os.getenv("TABLETOP_GRASP_X_OFFSET", "-0.05"))
    y_offset = float(os.getenv("TABLETOP_GRASP_Y_OFFSET", "0.0"))
    z_offset = float(os.getenv("TABLETOP_GRASP_Z_OFFSET", "0.0"))
    hand_grasp_pose.p[:, 0] += x_offset
    hand_grasp_pose.p[:, 1] += y_offset
    hand_grasp_pose.p[:, 2] += z_offset

    norm_object = normalize(hand_pose_relative_to_object.get_p())
    grasp_approach = float(os.getenv("TABLETOP_GRASP_APPROACH", "0.0"))
    if grasp_approach != 0.0:
        # Move grasp pose toward the object along the approach direction.
        hand_grasp_pose.set_p(hand_grasp_pose.get_p() - grasp_approach * norm_object)

    if os.getenv("TABLETOP_GRASP_STATS", "0") == "1":
        with torch.no_grad():
            rel_vec = hand_grasp_pose.get_p() - object_pose_in_root_frame.get_p()
            approach_gap = (rel_vec * norm_object).sum(-1)
            rel_norm = torch.linalg.norm(rel_vec, dim=-1)
            print("grasp approach gap min/mean/max:",
                  approach_gap.min().item(),
                  approach_gap.mean().item(),
                  approach_gap.max().item())
            print("grasp rel dist min/mean/max:",
                  rel_norm.min().item(),
                  rel_norm.mean().item(),
                  rel_norm.max().item())

    hand_pre_grasp_pose = copy.deepcopy(hand_grasp_pose)
    pregrasp_offset = float(os.getenv("TABLETOP_PREGRASP_OFFSET", "0.05"))
    hand_pre_grasp_pose.set_p(hand_grasp_pose.get_p() + pregrasp_offset * norm_object)

    if os.getenv("CUROBO_IK_COLLISION_DEBUG", "0") == "1":
        table_top = table_pose_in_root_frame[2] + table_dims[2] / 2
        grasp_p = hand_grasp_pose.get_p()
        pre_p = hand_pre_grasp_pose.get_p()
        print("Grasp pose p (first):", grasp_p[0].detach().cpu().numpy().tolist())
        print("Pre-grasp pose p (first):", pre_p[0].detach().cpu().numpy().tolist())
        print("Table top z:", table_top)
        print("Grasp z min/max:", grasp_p[:, 2].min().item(), grasp_p[:, 2].max().item())
        print("Pre-grasp z min/max:", pre_p[:, 2].min().item(), pre_p[:, 2].max().item())

    qpos_hand_pre_grasp_pose, hand_pre_grasp_pose_success = compute_qpos_and_success(
        ik_solver, hand_pre_grasp_pose)

    print("pre_grasp IK success:", hand_pre_grasp_pose_success.sum().item(), "/", batch_size)

    world_cfg_no_object = make_world_config_empty(
        table_pose=table_pose_in_root_frame,
        table_dims=table_dims,
        sponge_pose=static_box_pose_in_root_frame,
        sponge_dims=static_box_dims.tolist(),
    )

    if ik_solver.world_coll_checker is not None:
        ik_solver.world_coll_checker.clear_cache()
        ik_solver.update_world(world_cfg_no_object)

    qpos_hand_grasp_pose, hand_grasp_pose_success = compute_qpos_and_success(
        ik_solver, hand_grasp_pose)

    print("grasp IK success:", hand_grasp_pose_success.sum().item(), "/", batch_size)

    if os.getenv("TABLETOP_PREGRASP_FALLBACK", "0") == "1":
        fallback_mask = (~hand_pre_grasp_pose_success) & hand_grasp_pose_success
        if fallback_mask.any():
            qpos_hand_pre_grasp_pose = qpos_hand_pre_grasp_pose.clone()
            qpos_hand_pre_grasp_pose[fallback_mask] = qpos_hand_grasp_pose[fallback_mask]
            hand_pre_grasp_pose_success = hand_pre_grasp_pose_success | fallback_mask
            print("pre_grasp fallback used:", fallback_mask.sum().item(), "/", batch_size)

    success_indices = success_indices & hand_pre_grasp_pose_success & hand_grasp_pose_success

    if os.getenv("CUROBO_IK_COLLISION_DEBUG", "0") == "1":
        table_top = table_pose_in_root_frame[2] + table_dims[2] / 2
        try:
            _, _, _, _, _, _, link_spheres = ik_solver.kinematics.forward(
                qpos_hand_grasp_pose[:1]
            )
            # link_spheres: (B, n_spheres, 4) -> (x, y, z, r)
            spheres = link_spheres[0]
            sphere_bottom = spheres[:, 2] - spheres[:, 3]
            min_bottom = sphere_bottom.min().item()
            below = (sphere_bottom < table_top).sum().item()
            print("Robot sphere min bottom z:", min_bottom)
            print("Table top z:", table_top)
            print("Spheres below table top:", below, "/", spheres.shape[0])

            if ik_solver.world_coll_checker is not None:
                sph = link_spheres[:1].view(1, 1, -1, 4)
                weight = torch.ones(1, device=sph.device, dtype=sph.dtype)
                activation = torch.zeros(1, device=sph.device, dtype=sph.dtype)
                cq = CollisionQueryBuffer()
                cq.update_buffer_shape(
                    sph.shape, tensor_args, ik_solver.world_coll_checker.collision_types
                )
                dist_all = ik_solver.world_coll_checker.get_sphere_distance(
                    sph, cq, weight, activation, sum_collisions=False
                )
                print("World min distance (all obstacles):", dist_all.min().item())

                env_obb_names = getattr(ik_solver.world_coll_checker, "_env_obbs_names", None)
                if env_obb_names is not None and len(env_obb_names) > 0:
                    for name in env_obb_names[0]:
                        try:
                            ik_solver.world_coll_checker.enable_obstacle(
                                name=name, enable=False
                            )
                            dist = ik_solver.world_coll_checker.get_sphere_distance(
                                sph, cq, weight, activation, sum_collisions=False
                            )
                            print(
                                f"World min distance without obstacle '{name}':",
                                dist.min().item(),
                            )
                        finally:
                            ik_solver.world_coll_checker.enable_obstacle(
                                name=name, enable=True
                            )

            # Manual signed distance to table OBB
            try:
                tx, ty, tz, qw, qx, qy, qz = table_pose_in_root_frame
                R = torch.tensor(
                    transforms3d.quaternions.quat2mat([qw, qx, qy, qz]),
                    device=spheres.device,
                    dtype=spheres.dtype,
                )
                t = torch.tensor([tx, ty, tz], device=spheres.device, dtype=spheres.dtype)
                local = (spheres[:, :3] - t) @ R.T
                half = torch.tensor(table_dims, device=spheres.device, dtype=spheres.dtype) / 2
                d = local.abs() - half
                outside = torch.clamp(d, min=0.0)
                outside_dist = torch.linalg.norm(outside, dim=1)
                inside_dist = torch.minimum(d.max(dim=1).values, torch.zeros_like(outside_dist))
                signed_dist = outside_dist + inside_dist
                print("Manual table OBB min signed dist:", signed_dist.min().item())
                print("Manual table OBB inside count:", (signed_dist < 0).sum().item(), "/", signed_dist.numel())
            except Exception as exc:
                print("Manual table OBB distance failed:", exc)

            # Compare against collision checker's internal table pose/dims
            try:
                env_obb_names = getattr(ik_solver.world_coll_checker, "_env_obbs_names", None)
                cube_list = getattr(ik_solver.world_coll_checker, "_cube_tensor_list", None)
                if env_obb_names is not None and cube_list is not None and len(env_obb_names) > 0:
                    if "table" in env_obb_names[0]:
                        idx = env_obb_names[0].index("table")
                        dims = cube_list[0][0, idx, :3].detach().cpu().numpy().tolist()
                        pose = cube_list[1][0, idx, :7].detach().cpu().numpy().tolist()
                        print("World checker table dims:", dims)
                        print("World checker table pose:", pose)
                        tx, ty, tz, qw, qx, qy, qz = pose
                        R = torch.tensor(
                            transforms3d.quaternions.quat2mat([qw, qx, qy, qz]),
                            device=spheres.device,
                            dtype=spheres.dtype,
                        )
                        t = torch.tensor([tx, ty, tz], device=spheres.device, dtype=spheres.dtype)
                        local = (spheres[:, :3] - t) @ R.T
                        half = torch.tensor(dims, device=spheres.device, dtype=spheres.dtype) / 2
                        d = local.abs() - half
                        outside = torch.clamp(d, min=0.0)
                        outside_dist = torch.linalg.norm(outside, dim=1)
                        inside_dist = torch.minimum(d.max(dim=1).values, torch.zeros_like(outside_dist))
                        signed_dist = outside_dist + inside_dist
                        print("Checker table min signed dist:", signed_dist.min().item())
                        print("Checker table inside count:", (signed_dist < 0).sum().item(), "/", signed_dist.numel())
                    else:
                        print("World checker table not found in OBB names.")
            except Exception as exc:
                print("Failed to read world checker table pose:", exc)
        except Exception as exc:
            print("Failed to compute robot spheres:", exc)

    hand_lift_pose = copy.deepcopy(hand_grasp_pose)
    hand_lift_pose.p[:, 2] += 0.2

    qpos_hand_lift_pose, hand_lift_pose_success = compute_qpos_and_success(
        ik_solver, hand_lift_pose)

    success_indices = success_indices & hand_lift_pose_success

    if rotate_hand:
        hand_rotated_pose = copy.deepcopy(hand_grasp_pose)
        hand_rotated_pose.q = torch_3d_utils.rpy_to_quaternion(
            torch.tensor([torch.pi/2, 0, torch.pi/2], dtype=tensor_args.dtype, device=tensor_args.device).unsqueeze(0)).repeat(batch_size, 1)

        qpos_rotated_hand_pose, hand_rotated_pose_success = compute_qpos_and_success(
            ik_solver, hand_rotated_pose)
        success_indices = success_indices & hand_rotated_pose_success
    else:
        qpos_rotated_hand_pose = None

    if not success_indices.any():
        return []


    hand_qpos_dict = hand_qpos_to_dict(hand_qpos)

    pre_grasp_hand_qpos_dict, _ = width_mapper.squeeze_fingers(hand_qpos_dict,
                                                               delta_width_thumb=-0.025,
                                                               delta_width_others=-0.025,
                                                               keep_z=True,
                                                               )

    pre_grasp_hand_qpos = dict_to_hand_qpos(pre_grasp_hand_qpos_dict)

    target_grasp_hand_qpos_dict, _ = width_mapper.squeeze_fingers(
        hand_qpos_dict,
        delta_width_thumb=0.06,
        delta_width_others=0.06,
        keep_z=True,
    )
    target_grasp_hand_qpos = dict_to_hand_qpos(target_grasp_hand_qpos_dict)

    sim_backend = "gpu"

    env_seed = 0

    initial_joint_positions = torch.cat(
        [qpos_hand_pre_grasp_pose,
         torch.tensor(hand_initial_joint_positions).to(tensor_args.device).to(tensor_args.dtype).unsqueeze(0).repeat(
             batch_size, 1)],
        dim=-1
    )


    static_box_pose = initial_agent_poses*ManiSkillPose.create_from_pq(
        p=static_box_pose_in_root_frame[:3], q=static_box_pose_in_root_frame[3:])
    static_box_pos = static_box_pose.p.squeeze(
        0).cpu().numpy()

    env = gym.make(
        "OneObject-v0",
        num_envs=batch_size,
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend=sim_backend,
        object_name=object_name,
        object_xy=object_xy,
        # robot_uid="franka_allegro_right",
        robot_uid="xarm7_allegro_right",
        initial_joint_positions=initial_joint_positions.clone(),
        initial_agent_poses=initial_agent_poses,
        static_box_pos=static_box_pos,
        static_box_dims=static_box_dims,
    )
    unwrapped_env: BaseEnv = env.unwrapped
    env_agent = unwrapped_env.agent
    robot = env_agent.robot
    control_freq = int(unwrapped_env.control_freq)

    if output_dir is not None:
        env = RecordEpisode(
            env,
            output_dir=output_dir,
            save_trajectory=False,
            save_video=save_video,
            max_steps_per_video=500,
            video_fps=control_freq,
        )

    env.reset(seed=env_seed)

    root_pose = env.unwrapped.agent.robot.get_root_pose()
    hand_base = env.unwrapped.agent.robot.find_link_by_name("base_link")
    object_pose = env.unwrapped.object.pose

    def print_contact_link_distances(tag: str):
        try:
            links = env.unwrapped.agent._finger_contact_links + [env.unwrapped.agent._palm_link]
            obj_p = object_pose.p
            min_dist = None
            min_name = None
            for link in links:
                lp = link.pose.p
                dist = torch.norm(lp - obj_p, dim=-1)
                dmin = dist.min().item()
                if min_dist is None or dmin < min_dist:
                    min_dist = dmin
                    min_name = link.get_name()
            if min_dist is not None:
                print(f"{tag} min link->object dist:", min_dist, "link:", min_name)
            palm_link = env.unwrapped.agent.robot.find_link_by_name("palm")
            if palm_link is not None:
                palm_dist = torch.norm(palm_link.pose.p - obj_p, dim=-1).min().item()
                print(f"{tag} palm->object dist:", palm_dist)
        except Exception as exc:
            print(f"Failed to compute {tag} link distances:", exc)

    active_joint_names = []
    try:
        active_joint_names = [j.get_name() for j in robot.get_active_joints()]
    except Exception as exc:
        print("Failed to read active joint names:", exc)

    desired_joint_names = (
        [f"xarm7_joint{i}" for i in range(1, 8)]
        + [f"joint_{i}.0" for i in range(16)]
    )
    desired_to_active = None
    active_to_desired = None
    if active_joint_names:
        try:
            active_name_to_idx = {name: i for i, name in enumerate(active_joint_names)}
            desired_to_active = torch.tensor(
                [active_name_to_idx[name] for name in desired_joint_names],
                device=tensor_args.device,
                dtype=torch.long,
            )
            active_to_desired = torch.tensor(
                [desired_joint_names.index(name) for name in active_joint_names],
                device=tensor_args.device,
                dtype=torch.long,
            )
        except ValueError as exc:
            print("Unknown joint name in active list:", exc)
        except KeyError as exc:
            print("Unknown desired joint name:", exc)

    if not getattr(_validate_tabletop, "_printed_collision_info", False):
        try:
            print("active joint names:", [j.get_name() for j in robot.get_active_joints()])
        except Exception as exc:
            print("Failed to read active joint names:", exc)
        try:
            contact_links = env.unwrapped.agent._finger_contact_links + [env.unwrapped.agent._palm_link]
            for link in contact_links:
                try:
                    shapes = link.get_collision_shapes()
                    print(f"collision shapes for {link.get_name()}: {len(shapes)}")
                except Exception as exc:
                    print(f"Failed to read collision shapes for {link.get_name()}: {exc}")
        except Exception as exc:
            print("Failed to read contact link collision shapes:", exc)
        _validate_tabletop._printed_collision_info = True

    target_root = hand_grasp_pose  # target pose in root frame
    actual_world = hand_base.pose
    actual_root = root_pose.inv() * actual_world
    target_world = root_pose * target_root

    delta_root_p = actual_root.p - target_root.p
    delta_world_p = actual_world.p - target_world.p
    print("hand pose error root (m) [pre]:", delta_root_p)
    print("hand pose error world (m) [pre]:", delta_world_p)

    object_pose_root = root_pose.inv() * object_pose
    print("root pose p:", root_pose.p, "q:", root_pose.q)
    print("hand base pose p:", actual_world.p, "q:", actual_world.q)
    print("hand base pose (root) p:", actual_root.p, "q:", actual_root.q)
    print("object pose p:", object_pose.p, "q:", object_pose.q)
    print("object pose (root) p:", object_pose_root.p, "q:", object_pose_root.q)
    print("hand joint names:", env.unwrapped.agent.hand_joint_names)
    print("hand qpos input (first):", hand_qpos[0].detach().cpu().numpy())
    print_contact_link_distances("pre")

    def debug_phase(tag: str):
        print_contact_link_distances(tag)
        try:
            scene = env.unwrapped.scene
            impulses = []
            for link in env.unwrapped.agent._finger_contact_links:
                imp = scene.get_pairwise_contact_impulses(env.unwrapped.object, link)
                impulses.append(torch.norm(imp, dim=-1))
            imp_palm = scene.get_pairwise_contact_impulses(
                env.unwrapped.object, env.unwrapped.agent._palm_link
            )
            impulses.append(torch.norm(imp_palm, dim=-1))
            impulse_stack = torch.stack(impulses, dim=-1)
            print(f"{tag} contact impulse max:", impulse_stack.max().item())
        except Exception as exc:
            print(f"{tag} contact impulse read failed:", exc)

    run_interpolated_trajectory(
        env=env,
        success_indices=success_indices,
        qpos_hand_pre_grasp_pose=qpos_hand_pre_grasp_pose,
        qpos_hand_grasp_pose=qpos_hand_grasp_pose,
        qpos_hand_lift_pose=qpos_hand_lift_pose,
        qpos_hand_rotated_pose=qpos_rotated_hand_pose,

        pre_grasp_hand_qpos=pre_grasp_hand_qpos,
        grasp_hand_qpos=hand_qpos,
        target_grasp_hand_qpos=target_grasp_hand_qpos,

        initial_joint_positions=initial_joint_positions,
        action_reorder_idx=None,
        debug_hook=debug_phase,
        steps_per_phase=50,
        rotate_hand=rotate_hand,
    )

    info = unwrapped_env.evaluate()
    object_lifted_height = info["object_lifted_height"]
    n_contact = info["n_contact"]
    print_contact_link_distances("post")
    desired_arm_qpos = qpos_hand_lift_pose
    if rotate_hand and qpos_rotated_hand_pose is not None:
        desired_arm_qpos = qpos_rotated_hand_pose
    desired_qpos = torch.cat([desired_arm_qpos, target_grasp_hand_qpos], dim=-1)
    try:
        actual_qpos = robot.get_qpos()
        actual_qpos = torch.as_tensor(
            actual_qpos, device=desired_qpos.device, dtype=desired_qpos.dtype
        )
        qpos_diff = actual_qpos - desired_qpos
        print("qpos diff abs max:", qpos_diff.abs().max().item())
        print("qpos diff abs mean:", qpos_diff.abs().mean().item())
        try:
            diff_abs = qpos_diff.abs()
            top_vals, top_idx = torch.topk(diff_abs.max(dim=0).values, k=5)
            active_names = active_joint_names if active_joint_names else [f"idx_{i}" for i in range(diff_abs.shape[1])]
            top_names = [active_names[i] for i in top_idx.tolist()]
            print("top qpos diffs:", list(zip(top_names, top_vals.tolist())))
        except Exception as exc:
            print("Failed to summarize qpos diffs:", exc)
    except Exception as exc:
        print("Failed to read robot qpos:", exc)
    root_pose = env.unwrapped.agent.robot.get_root_pose()
    hand_base = env.unwrapped.agent.robot.find_link_by_name("base_link")
    object_pose = env.unwrapped.object.pose
    target_root = hand_grasp_pose
    actual_world = hand_base.pose
    actual_root = root_pose.inv() * actual_world
    target_world = root_pose * target_root
    delta_root_p = actual_root.p - target_root.p
    delta_world_p = actual_world.p - target_world.p
    print("hand pose error root (m) [post]:", delta_root_p)
    print("hand pose error world (m) [post]:", delta_world_p)
    object_pose_root = root_pose.inv() * object_pose
    print("object pose (root) [post] p:", object_pose_root.p, "q:", object_pose_root.q)

    try:
        scene = env.unwrapped.scene
        impulses = []
        for link in env.unwrapped.agent._finger_contact_links:
            imp = scene.get_pairwise_contact_impulses(env.unwrapped.object, link)
            impulses.append(torch.norm(imp, dim=-1))
        imp_palm = scene.get_pairwise_contact_impulses(
            env.unwrapped.object, env.unwrapped.agent._palm_link
        )
        impulses.append(torch.norm(imp_palm, dim=-1))
        impulse_stack = torch.stack(impulses, dim=-1)
        print("contact impulse max:", impulse_stack.max().item())
        for thresh in (0.0, 1e-4, 1e-3, 1e-2):
            count = (impulse_stack > thresh).sum(dim=-1)
            print(f"n_contact@{thresh} min/max:", count.min().item(), count.max().item())
    except Exception as exc:
        print("Failed to read contact impulses:", exc)
    print("finger contact links:", env.unwrapped.agent._finger_contact_link_names)
    print("palm link name:", env.unwrapped.agent._palm_link_name)

    print(
        "object_lifted_height min/max:",
        object_lifted_height.min().item(),
        object_lifted_height.max().item(),
    )
    print("n_contact min/max:", n_contact.min().item(), n_contact.max().item())
    print("n_contact > 0:", (n_contact > 0).sum().item(), "/", batch_size)

    lifted = object_lifted_height > 0.1
    print("lifted:", lifted.sum().item(), "/", batch_size)
    lifted = lifted.to(success_indices.device)

    success_indices: torch.Tensor = success_indices & lifted  # (B,)

    env.close()

    return success_indices.cpu().nonzero().squeeze(1).tolist()


def hand_qpos_to_dict(qpos: torch.Tensor) -> Dict[str, torch.Tensor]:
    assert len(qpos.shape) == 2 and qpos.shape[-1] == 16

    d = {
        f"joint_{i}.0": qpos[:, i] for i in range(16)
    }

    return d


def dict_to_hand_qpos(d: Dict[str, torch.Tensor]) -> torch.Tensor:
    qpos = torch.stack([d[f"joint_{i}.0"] for i in range(16)], dim=-1)
    return qpos


def normalize(v: torch.Tensor):
    return v / torch.norm(v, dim=-1, keepdim=True)

def compute_qpos_and_success(
    ik_solver: IKSolver,
    pose: ManiSkillPose,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (b, 7) (b, )
    """
    result = ik_solver.solve_batch(CuRoboPose(pose.p, pose.q))
    qpos = result.solution.squeeze(1)
    success = result.success.squeeze(1)

    if not success.all() and not getattr(compute_qpos_and_success, "_printed_ik_failure", False):
        joint_limits = ik_solver.robot_config.kinematics.get_joint_limits().position
        low = joint_limits[0]
        high = joint_limits[1]
        out_of_limits = ((qpos < low) | (qpos > high)).any(dim=-1)
        idx = (~success).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() > 0:
            first = idx[0].item()
            print(
                "IK failed:",
                "pos_err=", result.position_error[first].item(),
                "rot_err=", result.rotation_error[first].item(),
                "out_of_limits=", bool(out_of_limits[first].item()),
            )
            print(
                "IK result:",
                "success=", bool(success[first].item()),
                "error=", result.error[first].item(),
            )
            if os.getenv("CUROBO_IK_COLLISION_DEBUG", "0") == "1":
                if ik_solver.world_coll_checker is None:
                    print("IK collision debug: world collision checker is disabled.")
                else:
                    js = JointState(
                        qpos[first].unsqueeze(0),
                        joint_names=ik_solver.joint_names,
                    )
                    try:
                        metrics = ik_solver.check_constraints(js)
                        feasible = (
                            bool(metrics.feasible[0].item())
                            if metrics.feasible is not None
                            else None
                        )
                        print("IK constraint feasible (all obstacles):", feasible)
                    except Exception as exc:
                        print("IK collision debug failed to evaluate constraints:", exc)
                        metrics = None

                    obb_names = None
                    env_obb_names = getattr(ik_solver.world_coll_checker, "_env_obbs_names", None)
                    if env_obb_names is not None and len(env_obb_names) > 0:
                        obb_names = env_obb_names[0]

                    if obb_names:
                        fixed_by_obb = False
                        for name in obb_names:
                            try:
                                ik_solver.world_coll_checker.enable_obstacle(
                                    name=name, enable=False
                                )
                                metrics = ik_solver.check_constraints(js)
                                feasible = (
                                    bool(metrics.feasible[0].item())
                                    if metrics.feasible is not None
                                    else None
                                )
                                print(
                                    f"IK constraint feasible without obstacle '{name}':",
                                    feasible,
                                )
                                fixed_by_obb = fixed_by_obb or bool(feasible)
                            finally:
                                ik_solver.world_coll_checker.enable_obstacle(
                                    name=name, enable=True
                                )
                        if not fixed_by_obb:
                            print(
                                "IK collision debug: disabling OBB obstacles did not help. "
                                "Likely colliding with mesh obstacle (object) or mesh cache."
                            )
                    else:
                        print("IK collision debug: no OBB obstacles found.")
            compute_qpos_and_success._printed_ik_failure = True

    return qpos, success


def make_world_config_empty(
    table_pose: List[float],
    table_dims: List[float],
    sponge_pose: List[float],
    sponge_dims: List[float],
):
    world_dict = {
        "cuboid": {
            "table": {
                "dims": table_dims,
                "pose": table_pose,
            },
            'sponge': {
                "dims": sponge_dims,
                "pose": sponge_pose,
            }
        },
    }

    world_cfg = WorldConfig.from_dict(world_dict)
    return world_cfg


def _load_data(data_path: str):
    if data_path.endswith('.npy'):
        return _load_data_npy(data_path)
    elif data_path.endswith('.h5'):
        return _load_data_hdf5(data_path)
    else:
        raise ValueError(f"Unsupported data path: {data_path}")


_translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
_rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
_joint_names = [
    'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0',
    'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0',
    'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0',
    'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0'
]


def get_translation_from_qpos(qpos: Dict[str, float]) -> List[float]:
    return [qpos[name] for name in _translation_names]


def get_rpy_from_qpos(qpos: Dict[str, float]) -> List[float]:
    return [qpos[name] for name in _rot_names]


def get_joint_positions_from_qpos(qpos: Dict[str, float]) -> List[float]:
    return [qpos[name] for name in _joint_names]


def _load_data_npy(data_path: str):
    data_list = np.load(data_path, allow_pickle=True)

    if len(data_list) == 0:
        return None, None, None

    object_name = data_list[0]['object_code']

    n = len(data_list)

    hand_pose_relative_to_object = []
    hand_qpos = []

    for data_item in data_list:
        qpos_dict = data_item['qpos']
        trans = get_translation_from_qpos(qpos_dict)
        rpy = get_rpy_from_qpos(qpos_dict)
        joints = get_joint_positions_from_qpos(qpos_dict)

        hand_pose_relative_to_object.append(trans+rpy)
        hand_qpos.append(joints)

    hand_pose_relative_to_object = np.array(hand_pose_relative_to_object)
    hand_pose_relative_to_object = torch.from_numpy(
        hand_pose_relative_to_object)

    hand_qpos = np.array(hand_qpos)
    hand_qpos = torch.from_numpy(hand_qpos)

    return object_name, hand_pose_relative_to_object, hand_qpos


def _load_data_hdf5(data_path: str):
    with h5py.File(data_path, 'r') as f:

        if len(f['object_name']) == 0:
            return None, None, None
        object_name = f['object_name'][0].decode('utf-8')

        hand_pose_relative_to_object = f["pose"]
        hand_pose_relative_to_object = np.array(hand_pose_relative_to_object)
        hand_pose_relative_to_object = torch.from_numpy(
            hand_pose_relative_to_object)

        hand_qpos = f["qpos"]
        hand_qpos = np.array(hand_qpos)
        hand_qpos = torch.from_numpy(hand_qpos)

    return object_name, hand_pose_relative_to_object, hand_qpos
