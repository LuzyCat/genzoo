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
    JOINT_CATEGORIES,
    SYMMETRIC_JOINT_PAIRS,
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

    body_pose_axis = matrix_to_axis_angle(body_pose_mat.view(-1, 3, 3)).view(
        1, smal_model.NUM_BODY_JOINTS, 3
    )
    global_orient_axis = matrix_to_axis_angle(global_orient_mat.view(-1, 3, 3)).view(1, 3)

    return {
        "betas": betas,
        "body_pose_axis": body_pose_axis,
        "body_pose_mat": body_pose_mat,
        "global_orient_axis": global_orient_axis,
        "global_orient_mat": global_orient_mat,
    }


HEAD_ANCHOR_JOINTS = sorted(
    set(JOINT_CATEGORIES.get("head", []) + [32, 33, 34])
)
FRONT_LEG_JOINTS = JOINT_CATEGORIES.get("front_legs", [])
BACK_LEG_JOINTS = JOINT_CATEGORIES.get("back_legs", [])

ANCHOR_CONFIG = [
    {"name": "head", "joints": HEAD_ANCHOR_JOINTS, "weight": 2.5},
    {"name": "ears", "joints": [33, 34], "weight": 3.0},
    {"name": "front_legs", "joints": FRONT_LEG_JOINTS, "weight": 1.6},
    {"name": "back_legs", "joints": BACK_LEG_JOINTS, "weight": 1.6},
]

ANCHOR_SIGMA = 0.08
SYMMETRY_WEIGHT = 0.02
SMAL_JOINT_PARENTS_TENSOR = torch.tensor(SMAL_JOINT_PARENTS, dtype=torch.long)


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


def compute_anchor_weights(
    points: torch.Tensor,
    joints: torch.Tensor,
    anchor_cfg = None,
    sigma: float = ANCHOR_SIGMA,
) -> torch.Tensor:
    if anchor_cfg is None or len(anchor_cfg) == 0:
        return torch.ones(points.shape[0], device=points.device, dtype=points.dtype)

    weights = torch.ones(points.shape[0], device=points.device, dtype=points.dtype)
    sigma_tensor = torch.tensor(sigma, device=points.device, dtype=points.dtype)

    for cfg in anchor_cfg:
        joint_ids = cfg.get("joints", [])
        if not joint_ids:
            continue
        weight_val = cfg.get("weight", 1.0)
        joint_positions = joints[joint_ids]
        dists = torch.cdist(points, joint_positions)
        influence = torch.exp(-dists / sigma_tensor)
        weights = weights + weight_val * influence.max(dim=1)[0]

    return weights


def compute_symmetry_penalty(joints: torch.Tensor) -> torch.Tensor:
    parents = SMAL_JOINT_PARENTS_TENSOR.to(joints.device)
    penalties = []
    for left, right in SYMMETRIC_JOINT_PAIRS:
        left_parent = parents[left]
        right_parent = parents[right]
        if left_parent < 0 or right_parent < 0:
            continue
        left_vec = joints[left] - joints[left_parent]
        right_vec = joints[right] - joints[right_parent]
        penalties.append((left_vec.norm() - right_vec.norm()).abs())
    if not penalties:
        return torch.tensor(0.0, device=joints.device, dtype=joints.dtype)
    return torch.stack(penalties).mean()


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

        joints_current = (smal_out.joints * scale + transl)[0]
        weights_pred = compute_anchor_weights(pred_subset, joints_current, ANCHOR_CONFIG, ANCHOR_SIGMA)
        weights_target = compute_anchor_weights(target_points, joints_current, ANCHOR_CONFIG, ANCHOR_SIGMA)
        dists = torch.cdist(pred_subset, target_points)
        loss_xy = (weights_pred * dists.min(dim=1)[0]).sum() / weights_pred.sum()
        loss_yx = (weights_target * dists.min(dim=0)[0]).sum() / weights_target.sum()
        loss_data = loss_xy + loss_yx
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
                "body_pose_mat": body_pose_mat.detach().cpu().clone(),
                "body_pose_axis": matrix_to_axis_angle(body_pose_mat.view(-1, 3, 3)).view(
                    1, smal_model.NUM_BODY_JOINTS, 3
                ).detach().cpu().clone(),
                "global_orient_axis": global_orient.detach().cpu().clone(),
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
    body_pose_axis_init: torch.Tensor,
    global_orient_axis_init: torch.Tensor,
    transl_init: torch.Tensor,
    scale_init: float,
    iterations: int = 400,
    point_subset: int = 2048,
    device: torch.device = torch.device("cpu"),
    train_betas: bool = True,
    train_pose: bool = False,
    train_global: bool = True,
    anchor_cfg=None,
    anchor_sigma: float = ANCHOR_SIGMA,
    betas_prior: torch.Tensor | None = None,
    pose_prior: torch.Tensor | None = None,
    symmetry_weight: float = SYMMETRY_WEIGHT,
):
    vert_count = smal_model.v_template.shape[-2]
    subset = np.random.choice(vert_count, size=min(point_subset, vert_count), replace=False)
    subset = torch.tensor(subset, device=device, dtype=torch.long)

    betas = betas_init.to(device).clone().detach()
    if train_betas:
        betas.requires_grad_(True)

    body_pose_axis = body_pose_axis_init.to(device).clone().detach()
    if train_pose:
        body_pose_axis.requires_grad_(True)

    global_orient_axis = global_orient_axis_init.to(device).clone().detach()
    if train_global:
        global_orient_axis.requires_grad_(True)

    transl = transl_init.to(device).clone().detach()
    if train_global:
        transl.requires_grad_(True)

    scale_tensor = torch.tensor([[scale_init]], dtype=torch.float32, device=device)
    log_scale = scale_tensor.log().detach()
    if train_global:
        log_scale.requires_grad_(True)

    params = []
    if train_betas:
        params.append(betas)
    if train_pose:
        params.append(body_pose_axis)
    if train_global:
        params.extend([global_orient_axis, transl, log_scale])

    if params:
        optimiser = torch.optim.Adam(params, lr=0.01)
    else:
        optimiser = None

    target_points = mesh_points

    best_state = None
    best_loss = float("inf")

    def _record_state(loss_value: float):
        nonlocal best_state, best_loss
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                "betas": betas.detach().cpu().clone(),
                "body_pose_axis": body_pose_axis.detach().cpu().clone(),
                "global_orient_axis": global_orient_axis.detach().cpu().clone(),
                "transl": transl.detach().cpu().clone(),
                "scale": torch.exp(log_scale.detach().cpu()).clone(),
                "loss": loss_value,
            }

    if optimiser is None:
        with torch.no_grad():
            scale = torch.exp(log_scale)
            body_pose_mat = batch_rodrigues(body_pose_axis.view(-1, 3)).view(1, smal_model.NUM_BODY_JOINTS, 3, 3)
            global_orient_mat = batch_rodrigues(global_orient_axis).view(1, 1, 3, 3)
            smal_out = smal_model(
                betas=betas,
                body_pose=body_pose_mat,
                global_orient=global_orient_mat,
                pose2rot=False,
            )
            pred_vertices = smal_out.vertices * scale + transl
            pred_subset = pred_vertices[0, subset]
            joints_current = (smal_out.joints * scale + transl)[0]
            weights_pred = compute_anchor_weights(pred_subset, joints_current, anchor_cfg, anchor_sigma)
            weights_target = compute_anchor_weights(target_points, joints_current, anchor_cfg, anchor_sigma)
            dists = torch.cdist(pred_subset, target_points)
            loss_xy = (weights_pred * dists.min(dim=1)[0]).sum() / weights_pred.sum()
            loss_yx = (weights_target * dists.min(dim=0)[0]).sum() / weights_target.sum()
            loss_total = (loss_xy + loss_yx).item()
            best_loss = loss_total
            best_state = {
                "betas": betas.detach().cpu().clone(),
                "body_pose_axis": body_pose_axis.detach().cpu().clone(),
                "global_orient_axis": global_orient_axis.detach().cpu().clone(),
                "transl": transl.detach().cpu().clone(),
                "scale": torch.exp(log_scale.detach().cpu()).clone(),
                "loss": loss_total,
            }
        print(f"Best loss: {best_loss:.6f}")
        return best_state

    for step in range(iterations):
        optimiser.zero_grad()

        scale = torch.exp(log_scale)
        body_pose_mat = batch_rodrigues(body_pose_axis.view(-1, 3)).view(1, smal_model.NUM_BODY_JOINTS, 3, 3)
        global_orient_mat = batch_rodrigues(global_orient_axis).view(1, 1, 3, 3)
        smal_out = smal_model(
            betas=betas,
            body_pose=body_pose_mat,
            global_orient=global_orient_mat,
            pose2rot=False,
        )
        pred_vertices = smal_out.vertices * scale + transl
        pred_subset = pred_vertices[0, subset]
        joints_current = (smal_out.joints * scale + transl)[0]

        weights_pred = compute_anchor_weights(pred_subset, joints_current, anchor_cfg, anchor_sigma)
        weights_target = compute_anchor_weights(target_points, joints_current, anchor_cfg, anchor_sigma)
        dists = torch.cdist(pred_subset, target_points)
        loss_xy = (weights_pred * dists.min(dim=1)[0]).sum() / weights_pred.sum()
        loss_yx = (weights_target * dists.min(dim=0)[0]).sum() / weights_target.sum()
        loss_data = loss_xy + loss_yx

        loss_reg = torch.tensor(0.0, device=device, dtype=loss_data.dtype)
        if betas_prior is not None and train_betas:
            loss_reg = loss_reg + 5e-4 * (betas - betas_prior.to(device)).pow(2).mean()
        if pose_prior is not None and train_pose:
            loss_reg = loss_reg + 2e-4 * (body_pose_axis - pose_prior.to(device)).pow(2).mean()
        if train_global:
            loss_reg = loss_reg + 5e-5 * global_orient_axis.pow(2).mean()
        if symmetry_weight > 0:
            loss_reg = loss_reg + symmetry_weight * compute_symmetry_penalty(joints_current)

        loss = loss_data + loss_reg
        loss.backward()
        optimiser.step()

        current_loss = loss.item()
        _record_state(current_loss)

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
        betas = torch.as_tensor(state["betas"], device=device, dtype=torch.float32)

        if "body_pose_axis" in state:
            body_pose_axis = torch.as_tensor(state["body_pose_axis"], device=device, dtype=torch.float32)
        elif "body_pose" in state:
            body_pose_tensor = torch.as_tensor(state["body_pose"], device=device, dtype=torch.float32)
            if body_pose_tensor.ndim == 4:
                body_pose_axis = matrix_to_axis_angle(body_pose_tensor.view(-1, 3, 3)).view(
                    1, smal_model.NUM_BODY_JOINTS, 3
                )
            else:
                body_pose_axis = body_pose_tensor.view(1, smal_model.NUM_BODY_JOINTS, 3)
        else:
            raise KeyError("State dict missing body pose information")

        if "global_orient_axis" in state:
            global_orient_axis = torch.as_tensor(state["global_orient_axis"], device=device, dtype=torch.float32)
        elif "global_orient" in state:
            global_orient_axis = torch.as_tensor(state["global_orient"], device=device, dtype=torch.float32)
        else:
            raise KeyError("State dict missing global orientation information")

        transl = torch.as_tensor(state["transl"], device=device, dtype=torch.float32)
        scale = torch.as_tensor(state["scale"], device=device, dtype=torch.float32)

        body_pose_mat = batch_rodrigues(body_pose_axis.view(-1, 3)).view(
            1, smal_model.NUM_BODY_JOINTS, 3, 3
        )
        global_orient_mat = batch_rodrigues(global_orient_axis.view(-1, 3)).view(1, 1, 3, 3)

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
        body_pose_axis=body_pose_axis.cpu().numpy(),
        global_orient_axis=global_orient_axis.cpu().numpy(),
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

    if args.baseline_npz:
        baseline_params = load_baseline_parameters(args.baseline_npz, smal_model)
        betas_init = baseline_params["betas"]
        body_pose_axis_init = baseline_params["body_pose_axis"]
        body_pose_mat_init = baseline_params["body_pose_mat"]
        global_orient_axis_init = baseline_params["global_orient_axis"]
        global_orient_mat_init = baseline_params["global_orient_mat"]
    else:
        betas_init = torch.zeros(1, SMALLayer.SHAPE_SPACE_DIM)
        body_pose_axis_init = torch.zeros(1, smal_model.NUM_BODY_JOINTS, 3)
        body_pose_mat_init = torch.eye(3).view(1, 1, 3, 3).repeat(1, smal_model.NUM_BODY_JOINTS, 1, 1)
        global_orient_axis_init = torch.zeros(1, 3)
        global_orient_mat_init = torch.eye(3).view(1, 1, 3, 3)

    with torch.no_grad():
        baseline_out = smal_model(
            betas=betas_init.to(device),
            body_pose=body_pose_mat_init.to(device),
            global_orient=global_orient_mat_init.to(device),
            pose2rot=False,
        )
    baseline_vertices = baseline_out.vertices[0].cpu().numpy()
    scale_guess = estimate_scale_from_bbox(mesh_points, baseline_vertices)

    print("Running similarity-only alignment (scale + rotation + translation)...")
    similarity_state = run_similarity_alignment(
        mesh_points=mesh_points,
        smal_model=smal_model,
        betas=betas_init,
        body_pose_mat=body_pose_mat_init,
        global_orient_mat=global_orient_mat_init,
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

    global_orient_axis_state = similarity_state["global_orient_axis"]
    transl_init = similarity_state["transl"]
    scale_init = float(similarity_state["scale"].item())

    pose_state = run_optimization(
        mesh_points=mesh_points,
        smal_model=smal_model,
        betas_init=betas_init,
        body_pose_axis_init=body_pose_axis_init,
        global_orient_axis_init=global_orient_axis_state,
        transl_init=transl_init,
        scale_init=scale_init,
        iterations=args.iterations,
        point_subset=min(2048, args.samples),
        device=device,
        train_betas=False,
        train_pose=True,
        train_global=False,
        anchor_cfg=ANCHOR_CONFIG,
        anchor_sigma=ANCHOR_SIGMA,
        pose_prior=body_pose_axis_init,
        symmetry_weight=SYMMETRY_WEIGHT,
    )

    export_results(
        smal_model=smal_model,
        state=pose_state,
        output_dir=Path(args.output_dir),
        device=device,
        prefix="pose_fit",
    )

    body_pose_axis_refined = pose_state["body_pose_axis"]

    if args.shape_only:
        shape_state = run_optimization(
            mesh_points=mesh_points,
            smal_model=smal_model,
            betas_init=betas_init,
            body_pose_axis_init=body_pose_axis_refined,
            global_orient_axis_init=global_orient_axis_state,
            transl_init=transl_init,
            scale_init=scale_init,
            iterations=args.iterations,
            point_subset=min(2048, args.samples),
            device=device,
            train_betas=True,
            train_pose=False,
            train_global=False,
            anchor_cfg=ANCHOR_CONFIG,
            anchor_sigma=ANCHOR_SIGMA,
            betas_prior=betas_init,
            symmetry_weight=SYMMETRY_WEIGHT,
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
            body_pose_axis_init=body_pose_axis_refined,
            global_orient_axis_init=global_orient_axis_state,
            transl_init=transl_init,
            scale_init=scale_init,
            iterations=args.iterations,
            point_subset=min(2048, args.samples),
            device=device,
            train_betas=True,
            train_pose=True,
            train_global=True,
            anchor_cfg=ANCHOR_CONFIG,
            anchor_sigma=ANCHOR_SIGMA,
            betas_prior=betas_init,
            pose_prior=body_pose_axis_init,
            symmetry_weight=SYMMETRY_WEIGHT,
        )

        export_results(
            smal_model=smal_model,
            state=full_state,
            output_dir=Path(args.output_dir),
            device=device,
        )


if __name__ == "__main__":
    main()
