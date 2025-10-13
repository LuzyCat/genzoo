import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

import sys

# Ensure project root (contains both genzoo and nature3d packages) is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from nature3d.joint_utils import (
    SMAL_JOINT_NAMES,
    SMAL_JOINT_PARENTS,
    export_joints_to_json,
)
from nature3d.utils import convert_opencv_to_unity
from genzoo.utils import save_vertices_obj, SMALLayer
from smplx.lbs import batch_rodrigues
from pytorch3d.transforms import matrix_to_axis_angle


def unity_to_opencv(points: np.ndarray) -> np.ndarray:
    """Invert ``convert_opencv_to_unity`` (mirror Y and Z without recentering)."""
    return convert_opencv_to_unity(points, np.zeros(3, dtype=np.float32))


def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Simple symmetric Chamfer distance for point clouds (N,3) and (M,3)."""
    dists = torch.cdist(x, y)
    loss_xy = dists.min(dim=1)[0].mean()
    loss_yx = dists.min(dim=0)[0].mean()
    return loss_xy + loss_yx


def mesh_to_points(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    if not mesh.is_watertight:
        # Try to fill holes to stabilise sampling (in-place modification)
        mesh = mesh.copy()
        try:
            mesh.fill_holes()
        except Exception:
            pass
    return mesh.sample(num_points)


def load_baseline_parameters(npz_path: str, smal_model: SMALLayer):
    data = np.load(npz_path)
    if "pose" not in data or "beta" not in data:
        raise KeyError("Baseline npz must contain 'pose' and 'beta'.")

    pose = torch.tensor(data["pose"], dtype=torch.float32)  # (N,3,3)
    betas = torch.tensor(data["beta"], dtype=torch.float32).view(1, -1)

    if pose.ndim != 3 or pose.shape[1:] != (3, 3):
        raise ValueError("Baseline pose must have shape (J,3,3).")

    global_orient_mat = pose[0].unsqueeze(0).unsqueeze(0).contiguous()
    body_pose_mats = pose[1: 1 + smal_model.NUM_BODY_JOINTS].clone()
    if body_pose_mats.shape[0] < smal_model.NUM_BODY_JOINTS:
        missing = smal_model.NUM_BODY_JOINTS - body_pose_mats.shape[0]
        identity = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(missing, 1, 1)
        body_pose_mats = torch.cat([body_pose_mats, identity], dim=0)
    body_pose_mat = body_pose_mats.unsqueeze(0)

    return betas, body_pose_mat, global_orient_mat


def estimate_scale_from_bbox(
    mesh_points_cv: torch.Tensor,
    baseline_vertices: np.ndarray,
) -> float:
    target_extents = (mesh_points_cv.cpu().numpy().max(axis=0) - mesh_points_cv.cpu().numpy().min(axis=0))
    baseline_extents = baseline_vertices.max(axis=0) - baseline_vertices.min(axis=0)
    target_norm = np.linalg.norm(target_extents)
    baseline_norm = np.linalg.norm(baseline_extents)
    if baseline_norm < 1e-6:
        return 1.0
    return float(target_norm / baseline_norm)


def align_forward_to_axis(vertices: np.ndarray, joints: np.ndarray, head_index: int = 16, target_axis: str = "z"):
    """Rotate the mesh so that the head direction aligns with the target axis (default +Z)."""
    if joints.shape[0] <= head_index:
        return vertices, joints, 0.0

    root = joints[0]
    head = joints[head_index]
    forward_vec = head - root
    forward_vec[1] = 0  # project onto XZ plane
    norm = np.linalg.norm(forward_vec)
    if norm < 1e-8:
        return vertices, joints, 0.0

    forward_vec /= norm
    # Compute yaw angle relative to +X axis
    angle_x = np.arctan2(forward_vec[2], forward_vec[0])

    if target_axis.lower() == "x":
        yaw = -angle_x
    elif target_axis.lower() == "z":
        # Additional rotation of +90° (pi/2) around Y to map X -> Z
        yaw = -(angle_x - np.pi / 2.0)
    else:
        yaw = 0.0

    cos_a = np.cos(yaw)
    sin_a = np.sin(yaw)
    rot = np.array(
        [
            [cos_a, 0.0, -sin_a],
            [0.0, 1.0, 0.0],
            [sin_a, 0.0, cos_a],
        ],
        dtype=np.float32,
    )
    vertices_rot = vertices @ rot.T
    joints_rot = joints @ rot.T
    return vertices_rot, joints_rot, yaw


def run_similarity_alignment(
    mesh_points: torch.Tensor,
    smal_model: SMALLayer,
    betas: torch.Tensor,
    body_pose_mat: torch.Tensor,
    global_orient_mat: torch.Tensor,
    scale_init: float,
    iterations: int = 200,
    point_subset: int = 2048,
    device: torch.device = torch.device("cpu"),
):
    """Optimise scale/rotation/translation while keeping pose/shape fixed."""
    vert_count = smal_model.v_template.shape[-2]
    subset = np.random.choice(vert_count, size=min(point_subset, vert_count), replace=False)
    subset = torch.tensor(subset, device=device, dtype=torch.long)

    betas = betas.to(device)
    body_pose_mat = body_pose_mat.to(device)
    global_orient_init = matrix_to_axis_angle(global_orient_mat[:, 0]).view(1, 3)
    global_orient = global_orient_init.clone().detach().to(device).requires_grad_(True)
    transl = torch.zeros(1, 3, device=device, requires_grad=True)
    log_scale = torch.tensor([[np.log(max(scale_init, 1e-6))]], dtype=torch.float32, device=device, requires_grad=True)

    optimiser = torch.optim.Adam([global_orient, transl, log_scale], lr=0.02)

    target_points = mesh_points

    best_state = None
    best_loss = float("inf")

    for step in range(iterations):
        optimiser.zero_grad()

        scale = torch.exp(log_scale)
        global_orient_mat = batch_rodrigues(global_orient).view(1, 1, 3, 3)
        smal_out = smal_model(
            betas=betas,
            body_pose=body_pose_mat,
            global_orient=global_orient_mat,
            pose2rot=False,
        )
        pred_vertices = smal_out.vertices * scale + transl
        pred_subset = pred_vertices[0, subset]

        loss_data = chamfer_distance(pred_subset, target_points)
        loss_reg = 5e-5 * global_orient.pow(2).mean()
        loss_reg = loss_reg + 5e-2 * (scale - scale_init) ** 2
        loss = loss_data + loss_reg
        loss.backward()
        optimiser.step()

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {
                "betas": betas.detach().cpu().clone(),
                "body_pose": body_pose_mat.detach().cpu().clone(),
                "global_orient": global_orient.detach().cpu().clone(),
                "transl": transl.detach().cpu().clone(),
                "scale": torch.exp(log_scale.detach().cpu()).clone(),
                "loss": current_loss,
            }

        if step % 25 == 0 or step == iterations - 1:
            print(
                f"[SIM {step:03d}/{iterations}] loss={current_loss:.5f}, "
                f"data={loss_data.item():.5f}"
            )

    print(f"[SIM] Best loss: {best_loss:.6f}")
    return best_state


def run_optimization(
    mesh_points: torch.Tensor,
    smal_model: SMALLayer,
    betas_init: torch.Tensor,
    body_pose_mat: torch.Tensor,
    global_orient_init: torch.Tensor,
    transl_init: torch.Tensor,
    scale_init: torch.Tensor,
    iterations: int = 400,
    point_subset: int = 2048,
    device: torch.device = torch.device("cpu"),
    train_global: bool = True,
    betas_prior: torch.Tensor | None = None,
):
    vert_count = smal_model.v_template.shape[-2]
    subset = np.random.choice(vert_count, size=min(point_subset, vert_count), replace=False)
    subset = torch.tensor(subset, device=device, dtype=torch.long)

    betas = betas_init.to(device).clone().detach().requires_grad_(True)
    body_pose_mat = body_pose_mat.to(device)
    global_orient = global_orient_init.to(device).clone().detach().requires_grad_(train_global)
    transl = transl_init.to(device).clone().detach().requires_grad_(train_global)
    scale_tensor = torch.tensor([[scale_init]], dtype=torch.float32, device=device)
    log_scale = scale_tensor.log().detach().requires_grad_(train_global)

    params = [betas]
    if train_global:
        params.extend([global_orient, transl, log_scale])

    optimiser = torch.optim.Adam(params, lr=0.01)

    target_points = mesh_points

    best_state = None
    best_loss = float("inf")

    for step in range(iterations):
        optimiser.zero_grad()

        scale = torch.exp(log_scale)
        body_pose_eval = body_pose_mat
        global_orient_mat = batch_rodrigues(global_orient).view(1, 1, 3, 3)
        smal_out = smal_model(
            betas=betas,
            body_pose=body_pose_eval,
            global_orient=global_orient_mat,
            pose2rot=False,
        )
        pred_vertices = smal_out.vertices * scale + transl
        pred_subset = pred_vertices[0, subset]

        loss_data = chamfer_distance(pred_subset, target_points)
        loss_reg = 0.0
        if betas_prior is not None:
            loss_reg = 5e-4 * (betas - betas_prior.to(device)).pow(2).mean()
        if train_global:
            loss_reg = loss_reg + 5e-5 * global_orient.pow(2).mean()
        loss = loss_data + loss_reg
        loss.backward()
        optimiser.step()

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {
                "betas": betas.detach().cpu().clone(),
                "body_pose": body_pose_eval.detach().cpu().clone(),
                "global_orient": global_orient.detach().cpu().clone(),
                "transl": transl.detach().cpu().clone(),
                "scale": torch.exp(log_scale.detach().cpu()).clone(),
                "loss": current_loss,
            }

        if step % 50 == 0 or step == iterations - 1:
            print(
                f"[{step:03d}/{iterations}] loss={current_loss:.5f}, "
                f"data={loss_data.item():.5f}"
            )

    print(f"Best loss: {best_loss:.6f}")
    return best_state


def export_results(
    smal_model: SMALLayer,
    state: dict,
    output_dir: Path,
    device: torch.device,
    prefix: str = "smal_fit",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        betas = state["betas"].to(device)
        body_pose = state["body_pose"].to(device)
        global_orient = state["global_orient"].to(device)
        transl = state["transl"].to(device)
        scale = state["scale"].to(device)

        if body_pose.ndim == 4:
            body_pose_mat = body_pose
        else:
            body_pose_mat = batch_rodrigues(body_pose.view(-1, 3)).view(
                1, smal_model.NUM_BODY_JOINTS, 3, 3
            )
        global_orient_mat = batch_rodrigues(global_orient).view(1, 1, 3, 3)

        smal_out = smal_model(
            betas=betas,
            body_pose=body_pose_mat,
            global_orient=global_orient_mat,
            pose2rot=False,
        )
        verts = smal_out.vertices * scale + transl
        joints = smal_out.joints * scale + transl

    verts_np = verts[0].cpu().numpy()
    joints_np = joints[0].cpu().numpy()

    verts_unity = convert_opencv_to_unity(verts_np, np.zeros(3, dtype=np.float32))
    joints_unity = convert_opencv_to_unity(joints_np, np.zeros(3, dtype=np.float32))
    # verts_unity, joints_unity, yaw_rad = align_forward_to_axis(verts_unity, joints_unity, target_axis="z")

    faces = smal_model.faces_tensor.cpu().numpy()
    obj_path = output_dir / f"{prefix}_unity.obj"
    save_vertices_obj(verts_unity, faces, obj_path)

    joints_json = output_dir / f"{prefix}_unity_joints.json"
    export_joints_to_json(joints_unity, str(joints_json), SMAL_JOINT_NAMES, SMAL_JOINT_PARENTS)

    params_path = output_dir / f"{prefix}_params.npz"
    np.savez(
        params_path,
        betas=betas.cpu().numpy(),
        body_pose=body_pose.cpu().numpy(),
        global_orient=global_orient.cpu().numpy(),
        transl=transl.cpu().numpy(),
        scale=scale.cpu().numpy(),
    )

    meta = {
        "obj_path": str(obj_path),
        "joints_json": str(joints_json),
        "params_path": str(params_path),
        "best_loss": float(state.get("loss", 0.0)),
        # "yaw_applied_deg": float(np.degrees(yaw_rad)),
    }
    meta_path = output_dir / f"{prefix}_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved SMAL fit OBJ to {obj_path}")
    print(f"Saved joints JSON to {joints_json}")
    print(f"Saved parameter cache to {params_path}")


def main():
    parser = argparse.ArgumentParser(description="Fit SMAL pose/shape to a Unity mesh.")
    parser.add_argument("mesh_path", type=str, help="Target mesh (Unity coordinate system).")
    parser.add_argument(
        "--model-path",
        type=str,
        default="genzoo/data/smal_plus.pkl",
        help="Path to SMAL+ model pickle compatible with smplx.SMPLLayer.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="genzoo/smal_fit_output",
        help="Directory to save fitted assets.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=400,
        help="Number of optimisation iterations.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4096,
        help="Number of surface samples from the input mesh.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for optimisation.",
    )
    parser.add_argument(
        "--baseline-npz",
        type=str,
        help="Baseline SMAL prediction (.npz) with pose/betas.",
    )
    parser.add_argument(
        "--shape-only",
        action="store_true",
        help="Freeze pose/global motion and optimise shape (betas) only.",
    )

    args = parser.parse_args()
    mesh_path = Path(args.mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input mesh must be a single Trimesh.")

    mesh_points_np = mesh_to_points(mesh, args.samples)
    mesh_points_cv = unity_to_opencv(mesh_points_np.astype(np.float32))
    mesh_points = torch.tensor(mesh_points_cv, device=device, dtype=torch.float32)

    smal_model = SMALLayer(
        model_path=args.model_path,
        num_betas=SMALLayer.SHAPE_SPACE_DIM,
    ).to(device)

    if args.shape_only and not args.baseline_npz:
        raise ValueError("--baseline-npz must be provided when using --shape-only.")

    betas_init, body_pose_mat, global_orient_mat = load_baseline_parameters(args.baseline_npz, smal_model) if args.baseline_npz else (
        torch.zeros(1, SMALLayer.SHAPE_SPACE_DIM),
        torch.eye(3).view(1, 1, 3, 3).repeat(1, smal_model.NUM_BODY_JOINTS, 1, 1),
        torch.eye(3).view(1, 1, 3, 3),
    )

    with torch.no_grad():
        baseline_out = smal_model(
            betas=betas_init.to(device),
            body_pose=body_pose_mat.to(device),
            global_orient=global_orient_mat.to(device),
            pose2rot=False,
        )
    baseline_vertices = baseline_out.vertices[0].cpu().numpy()
    scale_guess = estimate_scale_from_bbox(mesh_points, baseline_vertices)

    print("Running similarity-only alignment (scale + rotation + translation)...")
    similarity_state = run_similarity_alignment(
        mesh_points=mesh_points,
        smal_model=smal_model,
        betas=betas_init,
        body_pose_mat=body_pose_mat,
        global_orient_mat=global_orient_mat,
        scale_init=scale_guess,
        iterations=200,
        point_subset=min(2048, args.samples),
        device=device,
    )

    export_results(
        smal_model=smal_model,
        state=similarity_state,
        output_dir=Path(args.output_dir),
        device=device,
        prefix="similarity_only",
    )

    global_orient_init = similarity_state["global_orient"]
    transl_init = similarity_state["transl"]
    scale_init = float(similarity_state["scale"].item())

    if args.shape_only:
        shape_state = run_optimization(
            mesh_points=mesh_points,
            smal_model=smal_model,
            betas_init=betas_init,
            body_pose_mat=body_pose_mat,
            global_orient_init=global_orient_init,
            transl_init=transl_init,
            scale_init=scale_init,
            iterations=args.iterations,
            point_subset=min(2048, args.samples),
            device=device,
            train_global=False,
            betas_prior=betas_init,
        )

        export_results(
            smal_model=smal_model,
            state=shape_state,
            output_dir=Path(args.output_dir),
            device=device,
            prefix="shape_fit",
        )
    else:
        full_state = run_optimization(
            mesh_points=mesh_points,
            smal_model=smal_model,
            betas_init=betas_init,
            body_pose_mat=body_pose_mat,
            global_orient_init=global_orient_init,
            transl_init=transl_init,
            scale_init=scale_init,
            iterations=args.iterations,
            point_subset=min(2048, args.samples),
            device=device,
            train_global=True,
            betas_prior=betas_init,
        )

        export_results(
            smal_model=smal_model,
            state=full_state,
            output_dir=Path(args.output_dir),
            device=device,
        )


if __name__ == "__main__":
    main()
