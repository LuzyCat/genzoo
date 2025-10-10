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

matplotlib.use("Agg")

# Ensure project root (contains both genzoo and nature3d packages) is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from hmr2.models import load_hmr2
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
    args = parser.parse_args()

    valid_images = collect_valid_images(args.input)
    if not valid_images:
        print("No valid images found")
        return

    print(f"Processing {len(valid_images)} images")

    hmr2, hmr2_cfg = load_hmr2(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hmr2 = hmr2.to(device).eval()

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
        
    if args.render:
        all_params = []

    for img_path in tqdm(valid_images, desc="Processing images"):
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        H, W = img.shape[:2]
        bbox = [0, 0, W, H]

        dataset = ViTDetDataset(hmr2_cfg, img, np.array([bbox]))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )

        batch = recursive_to(next(iter(dataloader)), device)


        with torch.no_grad():
            out = hmr2(batch)

        keypoints_3d = out["pred_keypoints_3d"][0].cpu().numpy()
        s, tx, ty = (
            out["scale"].cpu().numpy(),
            out["tx"].cpu().numpy(),
            out["ty"].cpu().numpy(),
        )
        verts = out["pred_vertices"][0].cpu().numpy()
        betas = out["pred_smpl_params"]["betas"][0].cpu().numpy()
        body_pose = out["pred_smpl_params"]["body_pose"][0].cpu().numpy()
        global_orient = out["pred_smpl_params"]["global_orient"][0].cpu().numpy()
        thetas = np.concatenate([global_orient, body_pose], axis=0)
        s_px, tx_px, ty_px = convert_to_pixel_coords(s, tx, ty, resolution=W)
        keypoints_2d = weak_perspective_project(keypoints_3d, s_px, tx_px, ty_px)
        vertices_2d = weak_perspective_project(verts, s_px, tx_px, ty_px)

        base_name = Path(img_path).stem
        np.savez(
            data_folder / f"{base_name}.npz",
            pose=thetas,
            beta=betas,
            scale=s,
            tx=tx,
            ty=ty,
            keypoints_3d=keypoints_3d,
            keypoints_2d=keypoints_2d,
            vertices_3d=verts,
            vertices_2d=vertices_2d,
        )
        save_vertices_obj(verts, hmr2.smpl.faces, obj_folder / f"{base_name}.obj")

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
                save_vertices_obj(unity_vertices, hmr2.smpl.faces, unity_obj_path)
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
                'img': img,
                'img_size': (W, H),
                'verts': verts,
                's': s,
                'tx': tx,
                'ty': ty,
            })

    if args.render:
        render_folder = output_folder / "renders"
        render_folder.mkdir(parents=True, exist_ok=True)
        overlay_folder = output_folder / "overlays"
        overlay_folder.mkdir(parents=True, exist_ok=True)

        # Render in high resolution to avoid artifacts
        renderer = MeshRenderer(hmr2.smpl.faces, resolution=(1024, 1024))
    for i, params in enumerate(tqdm(all_params, desc="Rendering samples")):
        camera = [2 * params['s'], 2 * params['s'], params['tx']/params['s'], params['ty']/params['s']]
        if i == 0:
            # warm-up render (headless workaround)
            _ = renderer.render(params['verts'], camera, color=LIGHT_BLUE)
        animal_render = renderer.render(params['verts'], camera, color=LIGHT_BLUE)
            
        # Downsample render to original resolution
        animal_render = cv2.resize(animal_render, (params['img_size'][0], params['img_size'][1]))
        
        Image.fromarray(animal_render).save(render_folder / f"{params['base_name']}.png")
        overlay = overlay_rgba_on_rgb(animal_render, params['img'])
        Image.fromarray(overlay).save(overlay_folder / f"{params['base_name']}.png")

if __name__ == "__main__":
    main()
