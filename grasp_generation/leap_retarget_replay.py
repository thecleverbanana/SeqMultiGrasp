"""
Replay Allegro grasps retargeted to LEAP hand on xArm.

Loads validated grasp data (.npy or .h5), retargets the 16 Allegro
hand joints to LEAP via anatomical joint mapping, and visualises the
result using RobotModelURDFLite.

Usage:
    # Show LEAP retargeted grasp at index 5
    python leap_retarget_replay.py --data_path <path.npy> --index 5

    # Side-by-side: LEAP (left) + original Allegro (right)
    python leap_retarget_replay.py --data_path <path.npy> --index 5 --show_allegro

    # Also render the grasped object (npy files only)
    python leap_retarget_replay.py --data_path <path.npy> --index 5 --show_object

    # Override arm joints (comma-separated 7 values)
    python leap_retarget_replay.py --data_path <path.npy> --arm_joints "0,-0.6,0,1.2,0,1.2,0"
"""

import os
import sys

import h5py
import numpy as np
import torch
import trimesh
import yaml
import tyro

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.robot_model_lite import RobotModelURDFLite
from utils.hand_retarget import AllegroToLeapRetargeter
from utils.misc import get_joint_positions_from_qpos
from src.consts import MESHDATA_PATH

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ASSET_DIR   = os.path.join(_HERE, 'assets', 'urdf', 'xarm7_dexhand')
_CUROBO_ROOT = os.path.join(_HERE, '..', 'third-party', 'curobo',
                             'src', 'curobo', 'content')

ALLEGRO_URDF    = os.path.join(_ASSET_DIR, 'xarm7_allegro_right.urdf')
LEAP_URDF       = os.path.join(_ASSET_DIR, 'xarm7_leap_right.urdf')

ALLEGRO_SPHERES = os.path.join(_CUROBO_ROOT, 'configs', 'robot', 'spheres',
                                'xarm7_allegro_right.yml')
LEAP_SPHERES    = os.path.join(_CUROBO_ROOT, 'configs', 'robot', 'spheres',
                                'xarm7_leap_right_spheres.yml')

# xArm retract pose used when no override is given
ARM_RETRACT = [0.0, -0.6, 0.0, 1.2, 0.0, 1.2, 0.0]

N_ARM  = 7
N_HAND = 16


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_npy(data_path: str, index: int):
    """Return (allegro_hand_joints [16 float32], record dict)."""
    data   = np.load(data_path, allow_pickle=True)
    record = data[index]
    joints = get_joint_positions_from_qpos(record['qpos'])   # list[16]
    return np.array(joints, dtype=np.float32), record


def _load_h5(data_path: str, index: int):
    """Return (allegro_hand_joints [16 float32], None)."""
    with h5py.File(data_path, 'r') as f:
        joints = f['qpos'][index].astype(np.float32)         # (16,)
    return joints, None


def load_grasp(data_path: str, index: int):
    if data_path.endswith('.h5'):
        return _load_h5(data_path, index)
    return _load_npy(data_path, index)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _full_joints(arm: list, hand: np.ndarray, device) -> torch.Tensor:
    """Build [1, 23] joint tensor from arm (7) + hand (16)."""
    return torch.cat([
        torch.tensor(arm,  dtype=torch.float, device=device),
        torch.tensor(hand, dtype=torch.float, device=device),
    ]).unsqueeze(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_path: str,
    index: int = 0,
    device: str = 'cpu',
    show_allegro: bool = False,
    show_object: bool = False,
    arm_joints: str = '',
) -> None:
    """
    Retarget and visualise a single grasp from the dataset.

    Args:
        data_path:    Path to .npy or .h5 grasp file.
        index:        Which grasp sample to load (0-based).
        device:       'cpu' or 'cuda'.
        show_allegro: Side-by-side comparison with original Allegro grasp.
        show_object:  Overlay the grasped object mesh (npy files only).
        arm_joints:   Comma-separated 7 xArm joint values.
                      Defaults to the retract pose.
    """
    dev = torch.device(device)

    # ---- Robot models ----------------------------------------------------------------
    with open(ALLEGRO_SPHERES) as f:
        allegro_spheres = yaml.safe_load(f)['collision_spheres']
    with open(LEAP_SPHERES) as f:
        leap_spheres = yaml.safe_load(f)['collision_spheres']

    allegro_model = RobotModelURDFLite(ALLEGRO_URDF, device=device,
                                       collision_spheres=allegro_spheres)
    leap_model    = RobotModelURDFLite(LEAP_URDF,    device=device,
                                       collision_spheres=leap_spheres)
    retargeter    = AllegroToLeapRetargeter(allegro_model, leap_model, device=device)

    # ---- Arm joints ------------------------------------------------------------------
    arm = [float(v) for v in arm_joints.split(',')] if arm_joints else ARM_RETRACT
    assert len(arm) == N_ARM, f"Expected 7 arm joints, got {len(arm)}"

    # ---- Load grasp ------------------------------------------------------------------
    allegro_hand_joints, record = load_grasp(data_path, index)
    print(f"[{os.path.basename(data_path)}]  index={index}")
    print(f"  Allegro: {np.round(allegro_hand_joints, 3).tolist()}")

    # ---- Retarget -------------------------------------------------------------------
    allegro_batch = _full_joints(arm, allegro_hand_joints, dev)  # [1, 23]
    leap_batch    = retargeter.retarget_batch(allegro_batch)      # [1, 23]
    print(f"  LEAP:    {np.round(leap_batch[0, N_ARM:].cpu().numpy(), 3).tolist()}")

    # ---- LEAP mesh ------------------------------------------------------------------
    leap_model.set_parameters(leap_batch)
    leap_mesh = leap_model.get_trimesh_data(i=0)
    meshes = [leap_mesh]

    # ---- Optional: Allegro side-by-side ---------------------------------------------
    if show_allegro:
        allegro_model.set_parameters(allegro_batch)
        allegro_mesh = allegro_model.get_trimesh_data(i=0)
        allegro_mesh.apply_translation([0.0, 0.9, 0.0])
        meshes.append(allegro_mesh)
        print("  Allegro shown offset +0.9 m in Y")

    # ---- Optional: object mesh ------------------------------------------------------
    if show_object and record is not None:
        obj_code  = record.get('object_code')
        obj_scale = record.get('scale', 1.0)
        mesh_path = os.path.join(str(MESHDATA_PATH), obj_code,
                                 'coacd', 'decomposed.obj')
        if os.path.exists(mesh_path):
            obj_mesh = trimesh.load(mesh_path)
            obj_mesh.apply_scale(obj_scale)
            obj_mesh.visual.face_colors = [180, 180, 180, 160]
            meshes.append(obj_mesh)
            print(f"  Object: {obj_code}  scale={obj_scale:.3f}")
        else:
            print(f"  Warning: object mesh not found: {mesh_path}")

    # ---- Show -----------------------------------------------------------------------
    trimesh.Scene(meshes).show()


if __name__ == '__main__':
    tyro.cli(main)
