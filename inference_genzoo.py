import os
os.environ.setdefault("PYGLET_HEADLESS", "1")        # pyglet 완전 headless
os.environ.setdefault("PYOPENGL_PLATFORM", "egl") # 또는 "egl"

import pyglet
pyglet.options['headless'] = True

import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib
import sys
import types
from typing import Optional, Tuple

# Provide a no-op wandb module for lightweight environments.
try:
    import wandb  # type: ignore
except ModuleNotFoundError:
    wandb = types.ModuleType("wandb")

    def _noop(*args, **kwargs):
        return None

    wandb.Image = lambda *args, **kwargs: None
    wandb.init = _noop
    wandb.log = _noop
    wandb.watch = _noop
    wandb.unwatch = _noop
    wandb.finish = _noop
    wandb.run = None
    sys.modules["wandb"] = wandb

matplotlib.use("Agg")

# Ensure project root (contains both genzoo and nature3d packages) is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from hmr2.models import load_hmr2, SMPL
from hmr2.configs import get_config
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from utils import (
    weak_perspective_project,
    convert_to_pixel_coords,
    save_vertices_obj,
    MeshRenderer,
    overlay_rgba_on_rgb,
)
from nature3d.joint_utils import (
    SMAL_JOINT_NAMES,
    SMAL_JOINT_PARENTS,
    export_joints_to_json,
    draw_skeleton,
)
from nature3d.utils import convert_opencv_to_unity

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def build_smal_model(device: torch.device) -> SMPL:
    """
    Instantiate the SMAL model used by GenZoo without loading the full HMR2 network.
    """
    cfg_path = Path(__file__).resolve().parent / 'data' / 'genzoo_1M_config.yaml'
    model_cfg = get_config(str(cfg_path), update_cachedir=True)
    smpl_kwargs = {k.lower(): v for k, v in dict(model_cfg.SMPL).items()}
    model = SMPL(**smpl_kwargs)
    model.to(device)
    model.eval()
    return model


def estimate_weak_perspective_params(verts: np.ndarray, img_size: Tuple[int, int]) -> Tuple[float, float, float]:
    """
    Derive deterministic weak-perspective parameters that center the projected vertices.
    Returns normalized (scale, tx, ty) compatible with convert_to_pixel_coords.
    """
    width, height = img_size
    if width <= 0 or height <= 0:
        return 1.0, 0.0, 0.0

    min_xy = verts[:, :2].min(axis=0)
    max_xy = verts[:, :2].max(axis=0)
    extent = np.maximum(max_xy - min_xy, 1e-6)

    scale_px = 0.95 * min(width / extent[0], height / extent[1])
    tx_px = (width - scale_px * (min_xy[0] + max_xy[0])) * 0.5
    ty_px = (height - scale_px * (min_xy[1] + max_xy[1])) * 0.5

    scale = float(scale_px / width)
    tx = float((tx_px / width) - 0.5)
    ty = float((ty_px / width) - 0.5)
    return scale, tx, ty


def load_combined_parameters(
    stem: str,
    param_root: Path,
    smal_model: SMPL,
    device: torch.device,
    img_size: Tuple[int, int],
) -> dict:
    """
    Load combined AniMer/AWOL parameters from disk and convert them into SMAL outputs.
    """
    candidates = [
        param_root / stem / "all_parameters.npz",
        param_root / f"{stem}.npz",
        param_root / f"{stem}_all_parameters.npz",
    ]
    param_path = next((p for p in candidates if p.exists()), None)
    if param_path is None:
        raise FileNotFoundError(f"No combined parameter npz found for '{stem}' in {param_root}")

    with np.load(param_path) as params:
        pose_body = params.get("combined_pose")
        global_orient = params.get("combined_global_orient")
        betas = params.get("combined_betas")
        transl = params.get("combined_transl")

    if pose_body is None or global_orient is None or betas is None:
        raise ValueError(f"Combined AniMer parameters missing expected keys in {param_path}")

    pose_body = pose_body.astype(np.float32)
    global_orient = global_orient.astype(np.float32)
    betas = betas.astype(np.float32)
    transl = transl.astype(np.float32) if transl is not None else None

    if global_orient.ndim == 2:
        global_orient = global_orient.reshape(1, 3, 3)
    if pose_body.ndim == 2:
        pose_body = pose_body.reshape(1, 3, 3)

    with torch.no_grad():
        body_pose_tensor = torch.tensor(pose_body, dtype=torch.float32, device=device).unsqueeze(0)
        global_orient_tensor = torch.tensor(global_orient, dtype=torch.float32, device=device)
        if global_orient_tensor.ndim == 3:
            global_orient_tensor = global_orient_tensor.unsqueeze(0)
        betas_tensor = torch.tensor(betas, dtype=torch.float32, device=device).reshape(1, -1)

        smal_inputs = {
            "body_pose": body_pose_tensor,
            "global_orient": global_orient_tensor,
            "betas": betas_tensor,
            "pose2rot": False,
        }
        if transl is not None:
            smal_inputs["transl"] = torch.tensor(transl, dtype=torch.float32, device=device).reshape(1, -1)

        smal_out = smal_model(**smal_inputs)
        verts = smal_out.vertices[0].detach().cpu().numpy()
        joints = smal_out.joints[0].detach().cpu().numpy()

    scale, tx, ty = estimate_weak_perspective_params(verts, img_size)
    pose_stack = np.concatenate([global_orient, pose_body], axis=0).astype(np.float32)

    return {
        "pose_mats": pose_stack,
        "body_pose": pose_body.astype(np.float32),
        "global_orient": global_orient.astype(np.float32),
        "betas": betas.astype(np.float32),
        "vertices": verts.astype(np.float32),
        "joints": joints.astype(np.float32),
        "scale": float(scale),
        "tx": float(tx),
        "ty": float(ty),
        "transl": transl.reshape(-1).astype(np.float32) if transl is not None else None,
        "param_path": str(param_path),
    }

def collect_valid_images(input_paths):
    valid_images = []
    for path_str in input_paths:
        path = Path(path_str)
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                with Image.open(path) as img:
                    valid_images.append(str(path))
            except Exception:
                print(f"Skipping {path} - invalid image")
        elif path.is_dir():
            for img_file in path.iterdir():
                if (
                    img_file.is_file()
                    and img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ):
                    try:
                        with Image.open(img_file) as img:
                            valid_images.append(str(img_file))
                    except Exception:
                        print(f"Skipping {img_file} - invalid image")
    return valid_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", nargs="+", help="Input image paths (files or directories)", default=["./example_input"]
    )
    parser.add_argument(
        "--checkpoint",
        default="./data/genzoo_1M.ckpt",
        help="Model checkpoint path",
    )
    parser.add_argument("--output", default="./output", help="Output folder path")
    parser.add_argument("--render", action="store_true", help="Render samples")
    parser.add_argument(
        "--export-unity",
        action="store_true",
        help="Also export Unity coordinate meshes and joint jsons",
    )
    parser.add_argument(
        "--animer-params-dir",
        type=str,
        default=None,
        help="Directory containing AniMer/AWOL combined parameter outputs (all_parameters.npz). "
             "When provided, skips GenZoo HMR2 inference and uses these parameters instead.",
    )
    args = parser.parse_args()

    valid_images = collect_valid_images(args.input)
    if not valid_images:
        print("No valid images found")
        return

    print(f"Processing {len(valid_images)} images")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_precomputed = args.animer_params_dir is not None
    param_root: Optional[Path] = None

    if use_precomputed:
        param_root = Path(args.animer_params_dir).expanduser()
        if not param_root.exists():
            print(f"❌ Parameter directory not found: {param_root}")
            return
        print(f"Using AniMer/AWOL parameters from {param_root}")
        hmr2 = None
        hmr2_cfg = None
        smal_model = build_smal_model(device)
        mesh_faces = smal_model.faces
    else:
        hmr2, hmr2_cfg = load_hmr2(args.checkpoint)
        hmr2 = hmr2.to(device).eval()
        smal_model = hmr2.smpl
        mesh_faces = hmr2.smpl.faces
    mesh_faces = np.asarray(mesh_faces)

    output_folder = Path(args.output)
    obj_folder = output_folder / "obj"
    data_folder = output_folder / "data"
    pose_folder = output_folder / "poses"
    data_folder.mkdir(parents=True, exist_ok=True)
    obj_folder.mkdir(parents=True, exist_ok=True)
    pose_folder.mkdir(parents=True, exist_ok=True)
    unity_obj_folder = None
    unity_data_folder = None
    if args.export_unity:
        unity_obj_folder = output_folder / "obj_unity"
        unity_data_folder = output_folder / "data_unity"
        unity_obj_folder.mkdir(parents=True, exist_ok=True)
        unity_data_folder.mkdir(parents=True, exist_ok=True)
    else:
        unity_obj_folder = unity_data_folder = None

    all_params = []

    for img_path in tqdm(valid_images, desc="Processing images"):
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        H, W = img_np.shape[:2]

        base_name = Path(img_path).stem

        if use_precomputed and param_root is not None:
            try:
                inferred = load_combined_parameters(base_name, param_root, smal_model, device, (W, H))
            except Exception as exc:
                print(f"⚠️  Skipping {base_name}: {exc}")
                continue

            pose_rotmat = inferred["pose_mats"]
            betas = inferred["betas"]
            body_pose = inferred["body_pose"]
            global_orient = inferred["global_orient"]
            verts = inferred["vertices"]
            keypoints_3d = inferred["joints"]
            s = np.array([inferred["scale"]], dtype=np.float32)
            tx = np.array([inferred["tx"]], dtype=np.float32)
            ty = np.array([inferred["ty"]], dtype=np.float32)
            s_px, tx_px, ty_px = convert_to_pixel_coords(s, tx, ty, resolution=W)
            keypoints_2d = weak_perspective_project(keypoints_3d, s_px, tx_px, ty_px)
            vertices_2d = weak_perspective_project(verts, s_px, tx_px, ty_px)
            transl = inferred["transl"]
        else:
            bbox = [0, 0, W, H]
            dataset = ViTDetDataset(hmr2_cfg, img_np, np.array([bbox]))
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=0
            )
            batch = recursive_to(next(iter(dataloader)), device)

            with torch.no_grad():
                out = hmr2(batch)

            keypoints_3d = out["pred_keypoints_3d"][0].cpu().numpy()
            s = out["scale"].cpu().numpy().astype(np.float32)
            tx = out["tx"].cpu().numpy().astype(np.float32)
            ty = out["ty"].cpu().numpy().astype(np.float32)
            verts = out["pred_vertices"][0].cpu().numpy()
            betas = out["pred_smpl_params"]["betas"][0].cpu().numpy()
            body_pose = out["pred_smpl_params"]["body_pose"][0].cpu().numpy()
            global_orient = out["pred_smpl_params"]["global_orient"][0].cpu().numpy()
            pose_rotmat = np.concatenate([global_orient, body_pose], axis=0)
            s_px, tx_px, ty_px = convert_to_pixel_coords(s, tx, ty, resolution=W)
            keypoints_2d = weak_perspective_project(keypoints_3d, s_px, tx_px, ty_px)
            vertices_2d = weak_perspective_project(verts, s_px, tx_px, ty_px)
            transl = None

        betas = betas.astype(np.float32)
        body_pose = body_pose.astype(np.float32)
        global_orient = global_orient.astype(np.float32)
        pose_rotmat = pose_rotmat.astype(np.float32)

        data_payload = {
            "pose": pose_rotmat,
            "pose_rotmat": pose_rotmat,
            "pose_body": body_pose,
            "body_pose": body_pose,
            "global_orient": global_orient,
            "betas": betas,
            "beta": betas,
            "scale": s.astype(np.float32),
            "tx": tx.astype(np.float32),
            "ty": ty.astype(np.float32),
            "keypoints_3d": keypoints_3d.astype(np.float32),
            "keypoints_2d": keypoints_2d.astype(np.float32),
            "vertices_3d": verts.astype(np.float32),
            "vertices_2d": vertices_2d.astype(np.float32),
        }
        if transl is not None:
            data_payload["transl"] = transl.astype(np.float32)

        np.savez(
            data_folder / f"{base_name}.npz",
            **data_payload,
        )
        save_vertices_obj(verts, mesh_faces, obj_folder / f"{base_name}.obj")

        # Export SMAL+ joint data and visualization alongside mesh assets
        joints = keypoints_3d
        joint_count = joints.shape[0]
        if joint_count != len(SMAL_JOINT_NAMES):
            print(
                f"Warning: expected {len(SMAL_JOINT_NAMES)} joints, got {joint_count}. "
                "Joint metadata will be truncated."
            )
        joint_names = SMAL_JOINT_NAMES[:joint_count]
        parents = SMAL_JOINT_PARENTS[:joint_count]

        joints_json_path = data_folder / f"{base_name}_joints.json"
        try:
            export_joints_to_json(joints, str(joints_json_path), joint_names, parents)
        except Exception as exc:
            print(f"Failed to export joint json for {base_name}: {exc}")
        else:
            pass
            # print(f"Joint json saved to {joints_json_path}")

        pose_image_path = pose_folder / f"{base_name}_pose.png"
        try:
            draw_skeleton(joints, output_path=str(pose_image_path), title=f"{base_name} 3D Pose")
        except Exception as exc:
            print(f"Failed to render 3D pose for {base_name}: {exc}")
        else:
            pass
            # print(f"3D pose visualization saved to {pose_image_path}")

        if args.export_unity:
            root_joint = joints[0] if joints.size else np.zeros(3)
            try:
                unity_vertices = convert_opencv_to_unity(verts, root_joint)
                unity_joints = convert_opencv_to_unity(joints, root_joint)

                unity_obj_path = unity_obj_folder / f"{base_name}_unity.obj"
                save_vertices_obj(
                    unity_vertices,
                    mesh_faces,
                    unity_obj_path,
                )
                # print(f"Unity mesh saved to {unity_obj_path}")

                unity_joints_json = unity_data_folder / f"{base_name}_joints_unity.json"
                export_joints_to_json(unity_joints, str(unity_joints_json), joint_names, parents)
                # print(f"Unity joints json saved to {unity_joints_json}")
            except Exception as exc:
                print(f"Failed to export Unity assets for {base_name}: {exc}")

        if args.render:
            # Store parameters for rendering
            all_params.append({
                'base_name': base_name,
                'img': img_np,
                'img_size': (W, H),
                'verts': verts,
                's': float(np.asarray(s).reshape(-1)[0]),
                'tx': float(np.asarray(tx).reshape(-1)[0]),
                'ty': float(np.asarray(ty).reshape(-1)[0]),
            })

    if args.render and all_params:
        render_folder = output_folder / "renders"
        render_folder.mkdir(parents=True, exist_ok=True)
        overlay_folder = output_folder / "overlays"
        overlay_folder.mkdir(parents=True, exist_ok=True)

        # Render in high resolution to avoid artifacts
        renderer = MeshRenderer(mesh_faces, resolution=(1024, 1024))
        for i, params in enumerate(tqdm(all_params, desc="Rendering samples")):
            s_val = max(params['s'], 1e-6)
            camera = [2 * s_val, 2 * s_val, params['tx']/s_val, params['ty']/s_val]
            if i == 0:
                # warm-up render (headless workaround)
                _ = renderer.render(params['verts'], camera, color=LIGHT_BLUE)
            animal_render = renderer.render(params['verts'], camera, color=LIGHT_BLUE)

            # Downsample render to original resolution (img_size stored as (W, H))
            animal_render = cv2.resize(animal_render, (params['img_size'][0], params['img_size'][1]))

            Image.fromarray(animal_render).save(render_folder / f"{params['base_name']}.png")
            overlay = overlay_rgba_on_rgb(animal_render, params['img'])
            Image.fromarray(overlay).save(overlay_folder / f"{params['base_name']}.png")

if __name__ == "__main__":
    main()
