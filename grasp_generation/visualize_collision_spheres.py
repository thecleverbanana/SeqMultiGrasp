import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import trimesh
import yaml
import xml.etree.ElementTree as ET


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_spheres_path(robot_yaml_path: str, spheres_entry: str) -> str:
    if os.path.isabs(spheres_entry):
        return spheres_entry
    robot_dir = os.path.dirname(robot_yaml_path)
    return os.path.join(robot_dir, spheres_entry)


def _color_for_link(link_name: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(link_name)) % (2**32))
    rgb = rng.uniform(0.2, 0.9, size=3)
    return (rgb * 255).astype(np.uint8)


def _resolve_urdf_path(robot_yaml_path: str, robot_kin: Dict) -> Optional[str]:
    urdf_path = robot_kin.get("urdf_path")
    if not urdf_path:
        return None
    if os.path.isabs(urdf_path):
        return urdf_path
    external_asset_path = robot_kin.get("external_asset_path")
    if external_asset_path:
        return os.path.join(external_asset_path, urdf_path)
    robot_dir = os.path.dirname(robot_yaml_path)
    return os.path.join(robot_dir, urdf_path)


def _resolve_mesh_path(
    urdf_dir: str,
    filename: str,
    search_roots: List[str],
) -> Optional[str]:
    if filename.startswith("package://"):
        filename = filename.replace("package://", "", 1)
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    candidates = [os.path.normpath(os.path.join(urdf_dir, filename))]
    for root in search_roots:
        candidates.append(os.path.normpath(os.path.join(root, filename)))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_visual_meshes(
    urdf_path: str,
    link_filter: List[str],
    search_roots: List[str],
) -> List[trimesh.Trimesh]:
    meshes: List[trimesh.Trimesh] = []
    urdf_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        if link_filter and link_name not in link_filter:
            continue
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh_el = geom.find("mesh")
            if mesh_el is None:
                continue
            filename = mesh_el.attrib.get("filename")
            if not filename:
                continue
            mesh_path = _resolve_mesh_path(urdf_dir, filename, search_roots)
            if not mesh_path:
                continue
            try:
                loaded = trimesh.load(mesh_path, process=False)
            except Exception:
                continue
            if isinstance(loaded, trimesh.Scene):
                loaded_meshes = list(loaded.geometry.values())
            else:
                loaded_meshes = [loaded]

            scale_attr = mesh_el.attrib.get("scale")
            scale = None
            if scale_attr:
                scale = np.array([float(v) for v in scale_attr.split()], dtype=np.float32)

            origin = visual.find("origin")
            transform = None
            if origin is not None:
                xyz = origin.attrib.get("xyz", "0 0 0")
                rpy = origin.attrib.get("rpy", "0 0 0")
                xyz = np.array([float(v) for v in xyz.split()], dtype=np.float32)
                rpy = np.array([float(v) for v in rpy.split()], dtype=np.float32)
                transform = trimesh.transformations.euler_matrix(rpy[0], rpy[1], rpy[2], axes="sxyz")
                transform[:3, 3] = xyz

            for mesh in loaded_meshes:
                if scale is not None:
                    mesh.apply_scale(scale)
                if transform is not None:
                    mesh.apply_transform(transform)
                mesh.visual.face_colors = np.array([180, 180, 180, 60], dtype=np.uint8)
                meshes.append(mesh)
    return meshes


def build_scene(
    spheres_config: Dict,
    link_filter: List[str],
    visual_meshes: List[trimesh.Trimesh],
) -> trimesh.Scene:
    scene = trimesh.Scene()
    link_map = spheres_config.get("collision_spheres", {})

    for mesh in visual_meshes:
        scene.add_geometry(mesh)

    for link_name, spheres in link_map.items():
        if link_filter and link_name not in link_filter:
            continue
        color = _color_for_link(link_name)
        for s in spheres:
            radius = float(s["radius"])
            center = np.array(s["center"], dtype=np.float32)
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
            mesh.apply_translation(center)
            mesh.visual.face_colors = np.array([*color, 160], dtype=np.uint8)
            scene.add_geometry(mesh, node_name=f"{link_name}_sphere")

    scene.add_geometry(trimesh.creation.axis(origin_size=0.01, axis_radius=0.002))
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize collision spheres from a CuRobo robot config."
    )
    parser.add_argument(
        "--robot-yaml",
        default=os.path.join(
            "third-party",
            "curobo",
            "src",
            "curobo",
            "content",
            "configs",
            "robot",
            "xarm7_allegro_right.yml",
        ),
        help="Path to the robot YAML (e.g., xarm7_allegro_right.yml).",
    )
    parser.add_argument(
        "--spheres-yaml",
        default=None,
        help="Optional path to the collision spheres YAML (overrides robot YAML).",
    )
    parser.add_argument(
        "--link",
        action="append",
        default=[],
        help="Filter to specific link names (repeatable).",
    )
    parser.add_argument(
        "--show-mesh",
        action="store_true",
        help="Overlay URDF visual meshes.",
    )
    parser.add_argument(
        "--urdf-only",
        action="store_true",
        help="Show only URDF visual meshes (no spheres).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path to export the scene (e.g., .glb).",
    )
    args = parser.parse_args()

    robot_cfg = _load_yaml(args.robot_yaml)
    robot_kin = robot_cfg["robot_cfg"]["kinematics"]
    spheres_entry = robot_kin.get("collision_spheres")
    if args.spheres_yaml:
        spheres_path = args.spheres_yaml
    else:
        if not spheres_entry:
            raise ValueError("collision_spheres not found in robot YAML.")
        spheres_path = _resolve_spheres_path(args.robot_yaml, spheres_entry)

    urdf_path = _resolve_urdf_path(args.robot_yaml, robot_kin)
    search_roots: List[str] = []
    if robot_kin.get("external_asset_path"):
        search_roots.append(robot_kin["external_asset_path"])
    if robot_kin.get("asset_root_path") and robot_kin.get("external_asset_path"):
        search_roots.append(
            os.path.join(robot_kin["external_asset_path"], robot_kin["asset_root_path"])
        )

    visual_meshes: List[trimesh.Trimesh] = []
    if (args.show_mesh or args.urdf_only) and urdf_path and os.path.exists(urdf_path):
        visual_meshes = _load_visual_meshes(urdf_path, args.link, search_roots)

    if args.urdf_only:
        scene = trimesh.Scene()
        for mesh in visual_meshes:
            scene.add_geometry(mesh)
    else:
        spheres_cfg = _load_yaml(spheres_path)
        scene = build_scene(spheres_cfg, args.link, visual_meshes)
    print(f"Loaded spheres from: {spheres_path}")
    print(f"Links shown: {args.link if args.link else 'ALL'}")
    if args.show_mesh:
        print(f"Visual meshes loaded: {len(visual_meshes)}")
    if args.output:
        scene.export(args.output)
        print(f"Scene exported to: {args.output}")
    else:
        scene.show()


if __name__ == "__main__":
    main()
