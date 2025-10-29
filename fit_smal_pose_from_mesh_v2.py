"""
GenZoo 베이스라인을 사용한 SMAL 메시 최적화 스크립트

타겟 3D 메시에 SMAL 모델을 최적화하여 맞춤

=== 사용법 ===

최소 사용 (OBJ만):
    python genzoo/fit_smal_pose_from_mesh.py target.obj

OBJ + NPZ (베이스라인):
    python genzoo/fit_smal_pose_from_mesh.py \
        target.obj \
        --baseline-npz genzoo_baseline.npz

실전 예제 (Roe Deer):
    python genzoo/fit_smal_pose_from_mesh.py \
        Roe_Deer/Roe_Deer_unity.obj \
        --baseline-npz genzoo_output/smal_parameters.npz \
        --output-dir fitted_results/Roe_Deer \
        --iterations 400 \
        --samples 4096

Shape만 최적화 (Pose 고정):
    python genzoo/fit_smal_pose_from_mesh.py \
        target.obj \
        --baseline-npz baseline.npz \
        --shape-only

=== 입력 ===

필수:
    - target.obj: 타겟 3D 메시 (Unity 좌표계)

선택:
    - --baseline-npz: GenZoo로 생성한 초기 파라미터
      (없으면 제로 초기화, 수렴 느림)

=== 출력 ===

{output_dir}/
    ├── similarity_only_unity.obj       # 1단계: 위치/크기만 맞춤
    ├── pose_fit_unity.obj              # 2단계: Pose 최적화
    ├── smal_fit_unity.obj              # 3단계: 전체 최적화 (최종)
    ├── smal_fit_params.npz             # 최적화된 파라미터
    └── smal_fit_summary.json           # 요약 정보

=== 최적화 프로세스 ===

3단계 최적화:
1. Similarity Alignment (200회)
   - Scale, Rotation, Translation만 조정
   - Shape와 Pose는 고정

2. Pose Optimization (400회)
   - Pose만 최적화
   - Shape 고정, Scale/Rotation/Translation 고정

3. Full Optimization (400회)
   - Shape, Pose, Global 모두 최적화
   (--shape-only 옵션 시 Shape만 최적화)

Loss 구성:
- Chamfer Distance: 메시 표면 유사도
- Anchor Weights: 중요 부위(머리, 귀, 다리) 가중치
- Symmetry Loss: 좌우 대칭 유지
- Prior Loss: 베이스라인과의 차이 제한
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

import sys
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 프로젝트 루트를 Python path에 추가 (genzoo, nature3d 패키지 import 위해)
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
    """
    Unity → OpenCV 좌표계 변환

    Unity (Y-up, Z-forward) → OpenCV (Y-down, Z-forward)
    Y와 Z축을 뒤집음 (중심점 이동 없이)
    """
    return convert_opencv_to_unity(points, np.zeros(3, dtype=np.float32))


def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    대칭 Chamfer Distance 계산

    두 점군 (N,3)과 (M,3) 사이의 유사도 측정
    - X→Y: x의 각 점에서 가장 가까운 y 점까지 거리 평균
    - Y→X: y의 각 점에서 가장 가까운 x 점까지 거리 평균
    - 총 거리 = X→Y + Y→X (대칭적)
    """
    dists = torch.cdist(x, y)  # (N, M) 거리 행렬
    loss_xy = dists.min(dim=1)[0].mean()  # X → Y
    loss_yx = dists.min(dim=0)[0].mean()  # Y → X
    return loss_xy + loss_yx


def mesh_to_points(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """
    메시 표면에서 균일하게 점 샘플링

    Args:
        mesh: 입력 3D 메시
        num_points: 샘플링할 점의 개수

    Returns:
        (num_points, 3) 샘플 포인트 배열

    Note:
        - 메시가 watertight가 아니면 구멍을 메우려고 시도
        - 샘플링 안정성 향상을 위함
    """
    if not mesh.is_watertight:
        # 구멍 메우기 시도 (샘플링 안정화)
        mesh = mesh.copy()
        try:
            mesh.fill_holes()
        except Exception:
            pass
    return mesh.sample(num_points)


def load_baseline_parameters(npz_path: str, smal_model: SMALLayer):
    """
    GenZoo 베이스라인 NPZ 파일에서 SMAL 파라미터 로드

    Args:
        npz_path: GenZoo로 생성한 .npz 파일 경로
        smal_model: SMAL 모델 인스턴스

    Returns:
        dict: 다음 키를 포함하는 파라미터 딕셔너리
            - betas: Shape 파라미터 (1, 41)
            - body_pose_axis: Body pose axis-angle (1, 33, 3)
            - body_pose_mat: Body pose 회전 행렬 (1, 33, 3, 3)
            - global_orient_axis: Global orientation axis-angle (1, 3)
            - global_orient_mat: Global orientation 회전 행렬 (1, 1, 3, 3)

    NPZ 형식:
        - 'pose': (J, 3, 3) 회전 행렬 배열
        - 'beta': (41,) shape 파라미터 배열
    """
    data = np.load(npz_path)
    if "pose" not in data or "beta" not in data:
        raise KeyError("Baseline npz must contain 'pose' and 'beta'.")

    pose = torch.tensor(data["pose"], dtype=torch.float32)  # (N,3,3)
    betas_raw = torch.tensor(data["beta"], dtype=torch.float32).view(1, -1)
    num_betas_model = getattr(smal_model, "SHAPE_SPACE_DIM", None)
    if num_betas_model is None:
        num_betas_model = getattr(smal_model, "num_betas", betas_raw.shape[-1])
    betas = torch.zeros(1, num_betas_model, dtype=torch.float32)
    n_copy = min(betas_raw.numel(), num_betas_model)
    betas[:, :n_copy] = betas_raw[:, :n_copy]

    if pose.ndim != 3 or pose.shape[1:] != (3, 3):
        raise ValueError("Baseline pose must have shape (J,3,3).")

    # Joint 0: Global orientation
    global_orient_mat = pose[0].unsqueeze(0).unsqueeze(0).contiguous()

    # Joints 1~33: Body pose
    body_pose_mats = pose[1: 1 + smal_model.NUM_BODY_JOINTS].clone()
    if body_pose_mats.shape[0] < smal_model.NUM_BODY_JOINTS:
        # 부족한 joint는 identity로 채움
        missing = smal_model.NUM_BODY_JOINTS - body_pose_mats.shape[0]
        identity = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(missing, 1, 1)
        body_pose_mats = torch.cat([body_pose_mats, identity], dim=0)
    body_pose_mat = body_pose_mats.unsqueeze(0)

    # 회전 행렬 → Axis-angle 변환
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


# ====================================================================
# Anchor 기반 가중치 설정
# ====================================================================
# 최적화 시 중요한 신체 부위에 더 높은 가중치를 부여
# 머리, 귀, 다리 등 특징적인 부분을 정확하게 맞추는 것이 목표

HEAD_ANCHOR_JOINTS = sorted(
    set(JOINT_CATEGORIES.get("head", []) + [32, 33, 34])
)
FRONT_LEG_JOINTS = JOINT_CATEGORIES.get("front_legs", [])
BACK_LEG_JOINTS = JOINT_CATEGORIES.get("back_legs", [])
TAIL_JOINTS = JOINT_CATEGORIES.get("tail", [])
FOOT_JOINTS = [10, 14, 20, 24]  # L/R front & back feet indices
LEG_JOINTS = sorted(set(FRONT_LEG_JOINTS + BACK_LEG_JOINTS))

ANCHOR_CONFIG = [
    {"name": "head", "joints": HEAD_ANCHOR_JOINTS, "weight": 2.5},  # 머리: 높은 가중치
    {"name": "ears", "joints": [33, 34], "weight": 3.0},            # 귀: 매우 높은 가중치
    {"name": "front_legs", "joints": FRONT_LEG_JOINTS, "weight": 1.6},  # 앞다리
    {"name": "back_legs", "joints": BACK_LEG_JOINTS, "weight": 1.6},    # 뒷다리
]

ANCHOR_SIGMA = 0.08  # Anchor 영향 범위 (작을수록 영향 범위 좁음)
SYMMETRY_WEIGHT = 0.01  # 좌우 대칭 loss 가중치
SMAL_JOINT_PARENTS_TENSOR = torch.tensor(SMAL_JOINT_PARENTS, dtype=torch.long)
FOOT_HEIGHT_WEIGHT = 0.001  # 발 위치를 타겟 메시 바닥에 맞추는 가중치
LEG_POSE_WEIGHT = 5e-3  # 다리 관절 과도 회전 방지


def estimate_scale_from_bbox(
    mesh_points_cv: torch.Tensor,
    baseline_vertices: np.ndarray,
) -> float:
    """
    Bounding box 크기를 비교하여 초기 스케일 추정

    Args:
        mesh_points_cv: 타겟 메시 포인트 (OpenCV 좌표계)
        baseline_vertices: 베이스라인 SMAL 메시 vertices

    Returns:
        float: 추정된 스케일 팩터 (target_size / baseline_size)

    Note:
        타겟 메시와 베이스라인 메시의 크기 차이를 계산하여
        초기 최적화를 빠르게 수렴시킴
    """
    target_extents = (mesh_points_cv.cpu().numpy().max(axis=0) - mesh_points_cv.cpu().numpy().min(axis=0))
    baseline_extents = baseline_vertices.max(axis=0) - baseline_vertices.min(axis=0)
    target_norm = np.linalg.norm(target_extents)
    baseline_norm = np.linalg.norm(baseline_extents)
    if baseline_norm < 1e-6:
        return 1.0
    return float(target_norm / baseline_norm)


def align_forward_to_axis(
    vertices: np.ndarray,
    joints: np.ndarray,
    head_index: int = 16,
    target_axis: str = "z",
    target_points: Optional[np.ndarray] = None,
):
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
    yaw_to_x = -np.arctan2(forward_vec[2], forward_vec[0])
    cos_yaw = np.cos(yaw_to_x)
    sin_yaw = np.sin(yaw_to_x)
    rot_yaw = np.array(
        [
            [cos_yaw, 0.0, -sin_yaw],
            [0.0, 1.0, 0.0],
            [sin_yaw, 0.0, cos_yaw],
        ],
        dtype=np.float32,
    )
    vertices_rot = vertices @ rot_yaw.T
    joints_rot = joints @ rot_yaw.T

    head_dir = joints_rot[head_index] - joints_rot[0]
    if head_dir[0] < 0:
        flip_mat = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        vertices_rot = vertices_rot @ flip_mat.T
        joints_rot = joints_rot @ flip_mat.T
        yaw_to_x += np.pi

    target_axis = target_axis.lower()
    axis_map = {"x": 0, "y": 1, "z": 2}
    if target_axis == "x":
        total_yaw = yaw_to_x
    elif target_axis == "z":
        rot_map = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        vertices_rot = vertices_rot @ rot_map.T
        joints_rot = joints_rot @ rot_map.T
        total_yaw = yaw_to_x - np.pi / 2.0
    elif target_axis == "y":
        rot_map = np.array(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        vertices_rot = vertices_rot @ rot_map.T
        joints_rot = joints_rot @ rot_map.T
        total_yaw = yaw_to_x + np.pi / 2.0
    else:
        total_yaw = yaw_to_x

    axis_idx = axis_map.get(target_axis, 2)

    def _flip_forward(v, j):
        flip_mat = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        return v @ flip_mat.T, j @ flip_mat.T

    head_mean = joints_rot[HEAD_ANCHOR_JOINTS].mean(axis=0)
    tail_ids = [idx for idx in TAIL_JOINTS if idx < joints_rot.shape[0]]
    tail_mean = joints_rot[tail_ids].mean(axis=0) if tail_ids else joints_rot[0]

    if target_points is not None and target_points.size > 0:
        front_val = target_points[:, axis_idx].max()
        back_val = target_points[:, axis_idx].min()

        score_current = abs(head_mean[axis_idx] - front_val) + abs(tail_mean[axis_idx] - back_val)

        flipped_vertices, flipped_joints = _flip_forward(vertices_rot, joints_rot)
        flipped_head = flipped_joints[HEAD_ANCHOR_JOINTS].mean(axis=0)
        flipped_tail = flipped_joints[tail_ids].mean(axis=0) if tail_ids else flipped_joints[0]
        score_flipped = abs(flipped_head[axis_idx] - front_val) + abs(flipped_tail[axis_idx] - back_val)

        if score_flipped < score_current:
            vertices_rot, joints_rot = flipped_vertices, flipped_joints
            total_yaw += np.pi
    else:
        if head_mean[axis_idx] < tail_mean[axis_idx]:
            vertices_rot, joints_rot = _flip_forward(vertices_rot, joints_rot)
            total_yaw += np.pi

    return vertices_rot, joints_rot, total_yaw


def compute_anchor_weights(
    points: torch.Tensor,
    joints: torch.Tensor,
    anchor_cfg = None,
    sigma: float = ANCHOR_SIGMA,
) -> torch.Tensor:
    """
    Anchor 기반 가중치 계산

    중요한 신체 부위(머리, 귀, 다리) 근처의 점들에 높은 가중치 부여

    Args:
        points: 메시 포인트 (N, 3)
        joints: SMAL joint 위치 (J, 3)
        anchor_cfg: Anchor 설정 리스트 (ANCHOR_CONFIG 참조)
        sigma: Gaussian 영향 범위 파라미터 (작을수록 영향 범위 좁음)

    Returns:
        torch.Tensor: 각 점의 가중치 (N,)

    동작 원리:
        1. 각 점에서 anchor joint까지의 거리 계산
        2. exp(-dist/sigma)로 거리 기반 영향도 계산
        3. 여러 anchor의 영향을 합산하여 최종 가중치 생성
        4. 머리/귀 근처 점들이 더 높은 가중치를 받음
    """
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
        dists = torch.cdist(points, joint_positions)  # (N, num_joints)
        influence = torch.exp(-dists / sigma_tensor)  # Gaussian 영향도
        weights = weights + weight_val * influence.max(dim=1)[0]  # 가장 가까운 joint의 영향

    return weights


def compute_symmetry_penalty(joints: torch.Tensor) -> torch.Tensor:
    """
    좌우 대칭 Loss 계산

    대칭되는 joint 쌍(왼쪽 앞다리-오른쪽 앞다리 등)의
    bone 길이가 같도록 제약

    Args:
        joints: SMAL joint 위치 (J, 3)

    Returns:
        torch.Tensor: 대칭 penalty (스칼라)

    동작 원리:
        - 대칭 joint 쌍에 대해 (left_bone_length - right_bone_length)의 절댓값 계산
        - 모든 쌍의 평균을 반환
        - 이 loss를 최소화하면 좌우 대칭이 유지됨
    """
    parents = SMAL_JOINT_PARENTS_TENSOR.to(joints.device)
    penalties = []
    for left, right in SYMMETRIC_JOINT_PAIRS:
        left_parent = parents[left]
        right_parent = parents[right]
        if left_parent < 0 or right_parent < 0:
            continue
        # 각 joint에서 부모 joint까지의 벡터 (bone)
        left_vec = joints[left] - joints[left_parent]
        right_vec = joints[right] - joints[right_parent]
        # Bone 길이 차이
        penalties.append((left_vec.norm() - right_vec.norm()).abs())
    if not penalties:
        return torch.tensor(0.0, device=joints.device, dtype=joints.dtype)
    return torch.stack(penalties).mean()


def compute_pose_regularization(body_pose_axis: torch.Tensor) -> torch.Tensor:
    """
    Pose 각도 제약 (다리 꼬임 방지)

    각 joint의 rotation angle을 제한하여 비현실적인 자세 방지
    특히 다리 joint에 대한 강한 제약 적용

    Args:
        body_pose_axis: Body pose axis-angle (1, J, 3)

    Returns:
        torch.Tensor: Pose regularization penalty
    """
    # Axis-angle의 norm = rotation angle (radians)
    angles = body_pose_axis.norm(dim=-1)  # (1, J)

    # 다리 joint 인덱스 (앞다리, 뒷다리)
    leg_joints = [
        6, 7, 8, 9, 10,    # 왼쪽 앞다리
        11, 12, 13, 14, 15, # 오른쪽 앞다리
        17, 18, 19, 20,    # 왼쪽 뒷다리
        21, 22, 23, 24,    # 오른쪽 뒷다리
    ]

    penalties = []

    # 다리 joint는 강한 제약 (90도 이상 회전하면 penalty)
    max_leg_angle = torch.tensor(np.pi / 2, device=angles.device, dtype=angles.dtype)  # 90도
    for idx in leg_joints:
        if idx < angles.shape[1]:
            excess = torch.relu(angles[0, idx] - max_leg_angle)
            penalties.append(excess ** 2)

    # 전체 joint는 약한 제약 (180도 이상 회전하면 penalty)
    max_general_angle = torch.tensor(np.pi, device=angles.device, dtype=angles.dtype)  # 180도
    excess_general = torch.relu(angles - max_general_angle)
    penalties.append(excess_general.pow(2).mean())

    if not penalties:
        return torch.tensor(0.0, device=angles.device, dtype=angles.dtype)

    return torch.stack(penalties).mean()


def compute_foot_height_penalty(
    joints: torch.Tensor,
    target_points: torch.Tensor,
    foot_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    if target_points is None or target_points.numel() == 0:
        return torch.tensor(0.0, device=joints.device, dtype=joints.dtype)

    if foot_indices is None:
        foot_indices = FOOT_JOINTS

    valid_indices = [idx for idx in foot_indices if idx < joints.shape[0]]
    if not valid_indices:
        return torch.tensor(0.0, device=joints.device, dtype=joints.dtype)

    max_y = target_points[:, 1].max()
    foot_y = joints[valid_indices, 1]
    return ((foot_y - max_y) ** 2).mean()


def save_debug_alignment_plot(
    stage_name: str,
    target_points: np.ndarray,
    smal_vertices: np.ndarray,
    output_dir: Path,
    sample_size: int = 8000,
) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        tgt = target_points
        pred = smal_vertices
        if tgt.shape[0] > sample_size:
            idx = np.random.choice(tgt.shape[0], sample_size, replace=False)
            tgt = tgt[idx]
        if pred.shape[0] > sample_size:
            idx = np.random.choice(pred.shape[0], sample_size, replace=False)
            pred = pred[idx]

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2], s=1, c='gray', alpha=0.3, label='Target')
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=1, c='red', alpha=0.7, label='SMAL')
        ax.set_title(f"{stage_name} Alignment (Y-up)")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='upper right')
        max_range = np.array([
            pred[:, 0].max() - pred[:, 0].min(),
            pred[:, 1].max() - pred[:, 1].min(),
            pred[:, 2].max() - pred[:, 2].min(),
            tgt[:, 0].max() - tgt[:, 0].min(),
            tgt[:, 1].max() - tgt[:, 1].min(),
            tgt[:, 2].max() - tgt[:, 2].min()
        ]).max() / 2.0
        mid = np.array([
            np.concatenate([pred[:, 0], tgt[:, 0]]).mean(),
            np.concatenate([pred[:, 1], tgt[:, 1]]).mean(),
            np.concatenate([pred[:, 2], tgt[:, 2]]).mean(),
        ])
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        fig.tight_layout()
        fig.savefig(output_dir / f"{stage_name}_alignment.png", dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"⚠️  Failed to save debug plot for {stage_name}: {exc}")




def compute_leg_pose_penalty(body_pose_axis: torch.Tensor, leg_indices: List[int]) -> torch.Tensor:
    if body_pose_axis.ndim == 2:
        body_pose_axis = body_pose_axis.unsqueeze(0)
    penalties = []
    for idx in leg_indices:
        if idx < body_pose_axis.shape[1]:
            angle = body_pose_axis[:, idx].norm(dim=-1)
            penalties.append(angle.pow(2))
    if not penalties:
        return torch.tensor(0.0, device=body_pose_axis.device, dtype=body_pose_axis.dtype)
    return torch.stack(penalties).mean()


def map_forward_axis_to_opencv(axis: str) -> np.ndarray:
    axis = axis.lower()
    if axis == "x":
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif axis == "y":
        vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:  # "z"
        vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # Unity(+Y up) → OpenCV(+Y down, +Z forward)
    conversion = np.array([1.0, -1.0, -1.0], dtype=np.float32)
    return vec * conversion


def align_global_orientation_to_forward(
    joints_cv: np.ndarray,
    global_orient_mat: torch.Tensor,
    forward_axis: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if joints_cv.shape[0] <= 16:
        return matrix_to_axis_angle(global_orient_mat.view(-1, 3, 3)).view(1, 3), global_orient_mat

    root = joints_cv[0]
    head_vec = joints_cv[16] - root
    if np.linalg.norm(head_vec) < 1e-6:
        return matrix_to_axis_angle(global_orient_mat.view(-1, 3, 3)).view(1, 3), global_orient_mat
    forward_current = head_vec / np.linalg.norm(head_vec)

    # Belly(배) 방향 계산: 발 중심에서 척추 방향으로
    foot_indices = [idx for idx in FOOT_JOINTS if idx < joints_cv.shape[0]]
    if foot_indices:
        feet_center = joints_cv[foot_indices].mean(axis=0)
        # Belly는 발에서 척추를 향하는 반대 방향 (down 방향)
        down_vec = feet_center - root
    else:
        down_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if np.linalg.norm(down_vec) < 1e-6:
        down_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    down_current = down_vec / np.linalg.norm(down_vec)

    # Right = forward × down (외적)
    right_current = np.cross(forward_current, down_current)
    if np.linalg.norm(right_current) < 1e-6:
        right_current = np.cross(forward_current, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    if np.linalg.norm(right_current) < 1e-6:
        right_current = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right_current = right_current / np.linalg.norm(right_current)

    # Down = right × forward (재계산하여 직교 보장)
    down_current_ortho = np.cross(right_current, forward_current)
    down_current_ortho = down_current_ortho / np.linalg.norm(down_current_ortho)

    # 재계산된 down이 원래 down과 같은 방향인지 확인 (부호 체크)
    if np.dot(down_current_ortho, down_current) < 0:
        # 반대 방향이면 right를 뒤집음
        right_current = -right_current
        down_current_ortho = np.cross(right_current, forward_current)
        down_current_ortho = down_current_ortho / np.linalg.norm(down_current_ortho)

    current_basis = np.stack([forward_current, right_current, down_current_ortho], axis=1)

    target_vec = map_forward_axis_to_opencv(forward_axis)
    if np.linalg.norm(target_vec) < 1e-6:
        target_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    forward_target = target_vec / np.linalg.norm(target_vec)

    # 타겟의 down 방향: OpenCV에서 -Y는 위, +Y는 아래
    down_target = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # 배가 +Y(아래)를 향함
    if np.linalg.norm(np.cross(forward_target, down_target)) < 1e-6:
        down_target = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    right_target = np.cross(forward_target, down_target)
    right_target = right_target / np.linalg.norm(right_target)
    down_target = np.cross(right_target, forward_target)
    down_target = down_target / np.linalg.norm(down_target)
    target_basis = np.stack([forward_target, right_target, down_target], axis=1)

    rot_delta_np = target_basis @ current_basis.T
    rot_delta = torch.from_numpy(rot_delta_np).to(global_orient_mat.device, dtype=global_orient_mat.dtype)
    new_orient_mat = rot_delta @ global_orient_mat.squeeze(0)

    # 정렬 후 belly 방향이 올바른지 확인
    # down_current_ortho가 down_target([0, 1, 0])과 같은 방향이어야 함
    # 만약 down이 -Y(위쪽)를 향하면 잘못 정렬된 것 -> 180도 roll 추가
    final_down = rot_delta_np @ down_current_ortho
    if final_down[1] < 0:  # Y가 음수 = 위쪽을 향함
        print(f"[WARN] Belly pointing up after alignment (Y={final_down[1]:.3f}), applying 180° roll correction")
        # Forward axis를 기준으로 180도 회전
        roll_180 = np.array([[1, 0, 0],
                             [0, -1, 0],
                             [0, 0, -1]], dtype=np.float32)
        if np.abs(forward_target[0]) > 0.9:  # X축 forward
            roll_180 = np.array([[-1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, 1]], dtype=np.float32)
        elif np.abs(forward_target[2]) > 0.9:  # Z축 forward
            roll_180 = np.array([[-1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, 1]], dtype=np.float32)
        roll_180_torch = torch.from_numpy(roll_180).to(new_orient_mat.device, dtype=new_orient_mat.dtype)
        new_orient_mat = roll_180_torch @ new_orient_mat

    new_axis = matrix_to_axis_angle(new_orient_mat.unsqueeze(0)).view(1, 3)
    return new_axis, new_orient_mat.unsqueeze(0)


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
    """
    1단계 최적화: Similarity Alignment (위치/크기/회전만 조정)

    Shape와 Pose는 고정하고 Scale, Global Rotation, Translation만 최적화
    타겟 메시의 위치와 크기에 대략적으로 맞춤

    Args:
        mesh_points: 타겟 메시 포인트 (N, 3)
        smal_model: SMAL 모델 인스턴스
        betas: 초기 shape 파라미터 (고정)
        body_pose_mat: 초기 body pose (고정)
        global_orient_mat: 초기 global orientation (최적화)
        scale_init: 초기 스케일 추정값
        iterations: 반복 횟수 (기본 200)
        point_subset: 서브샘플링 포인트 수 (속도 향상)
        device: torch device

    Returns:
        dict: 최적화된 파라미터
            - betas, body_pose_mat, body_pose_axis (고정된 값 그대로)
            - global_orient_axis (최적화됨)
            - transl (최적화됨)
            - scale (최적화됨)
            - loss (최종 loss 값)
    """
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

    tail_indices = JOINT_CATEGORIES.get("tail", [])
    valid_indices = [idx for idx in tail_indices if idx < smal_model.NUM_BODY_JOINTS]
    anchor_cfg = [
        {"name": "head", "joints": HEAD_ANCHOR_JOINTS, "weight": 2.5},
        {"name": "ears", "joints": [33, 34], "weight": 3.0},
        {"name": "front_legs", "joints": FRONT_LEG_JOINTS, "weight": 1.6},
        {"name": "back_legs", "joints": BACK_LEG_JOINTS, "weight": 1.6},
    ]

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
        weights_pred = compute_anchor_weights(pred_subset, joints_current, anchor_cfg, ANCHOR_SIGMA)
        weights_target = compute_anchor_weights(target_points, joints_current, anchor_cfg, ANCHOR_SIGMA)
        dists = torch.cdist(pred_subset, target_points)
        loss_xy = (weights_pred * dists.min(dim=1)[0]).sum() / weights_pred.sum()
        loss_yx = (weights_target * dists.min(dim=0)[0]).sum() / weights_target.sum()
        loss_data = loss_xy + loss_yx
        foot_penalty = compute_foot_height_penalty(joints_current, target_points)
        loss_reg = 5e-5 * global_orient.pow(2).mean()
        loss_reg = loss_reg + 5e-2 * (scale - scale_init) ** 2
        # 초기 정렬에서 너무 멀어지지 않도록 제약 추가 (강화: 0.01 -> 0.5)
        loss_reg = loss_reg + 0.5 * (global_orient - global_orient_init.to(device)).pow(2).mean()
        loss = loss_data + loss_reg + FOOT_HEIGHT_WEIGHT * foot_penalty

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
    """
    2단계/3단계 최적화: Pose 최적화 또는 Full 최적화

    train_* 플래그로 최적화할 파라미터 선택 가능

    2단계 (Pose only):
        train_betas=False, train_pose=True, train_global=False
        → Pose만 최적화, Shape/Scale/Translation 고정

    3단계 (Full optimization):
        train_betas=True, train_pose=True, train_global=True
        → Shape, Pose, Global 모두 최적화

    Args:
        mesh_points: 타겟 메시 포인트 (N, 3)
        smal_model: SMAL 모델 인스턴스
        betas_init: 초기 shape 파라미터
        body_pose_axis_init: 초기 body pose (axis-angle)
        global_orient_axis_init: 초기 global orientation (axis-angle)
        transl_init: 초기 translation
        scale_init: 초기 scale
        iterations: 반복 횟수 (기본 400)
        point_subset: 서브샘플링 포인트 수
        device: torch device
        train_betas: Shape 최적화 여부
        train_pose: Pose 최적화 여부
        train_global: Global rotation/translation/scale 최적화 여부
        anchor_cfg: Anchor 가중치 설정
        anchor_sigma: Anchor 영향 범위
        betas_prior: Shape prior (None이면 prior loss 없음)
        pose_prior: Pose prior (None이면 prior loss 없음)
        symmetry_weight: 대칭 loss 가중치

    Returns:
        dict: 최적화된 파라미터
            - betas, body_pose_axis, global_orient_axis
            - transl, scale
            - loss (최종 loss 값)

    Loss 구성:
        1. Data loss: Anchor-weighted Chamfer distance
        2. Prior loss: 베이스라인과의 차이 제한
        3. Regularization: 과도한 변형 방지
        4. Symmetry loss: 좌우 대칭 유지
    """
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
            foot_penalty = compute_foot_height_penalty(joints_current, target_points)
            leg_penalty = compute_leg_pose_penalty(body_pose_axis, LEG_JOINTS)
            loss_total = (loss_xy + loss_yx + FOOT_HEIGHT_WEIGHT * foot_penalty + LEG_POSE_WEIGHT * leg_penalty).item()
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

        foot_penalty = compute_foot_height_penalty(joints_current, target_points)
        leg_penalty = compute_leg_pose_penalty(body_pose_axis, LEG_JOINTS) if train_pose else torch.tensor(0.0, device=device, dtype=loss_data.dtype)
        loss_reg = torch.tensor(0.0, device=device, dtype=loss_data.dtype)
        if betas_prior is not None and train_betas:
            loss_reg = loss_reg + 5e-4 * (betas - betas_prior.to(device)).pow(2).mean()
        if pose_prior is not None and train_pose:
            loss_reg = loss_reg + 2e-4 * (body_pose_axis - pose_prior.to(device)).pow(2).mean() 
        if train_global:
            loss_reg = loss_reg + 5e-5 * global_orient_axis.pow(2).mean()
        if symmetry_weight > 0:
            loss_reg = loss_reg + symmetry_weight * compute_symmetry_penalty(joints_current)

        loss = loss_data + loss_reg + FOOT_HEIGHT_WEIGHT * foot_penalty + LEG_POSE_WEIGHT * leg_penalty
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
    target_points_cv: torch.Tensor,
    forward_axis: str,
    debug_plot: bool,
    prefix: str = "smal_fit",
):
    """
    최적화 결과를 파일로 출력

    출력 파일:
        1. {prefix}_unity.obj: Unity 좌표계 메시
        2. {prefix}_unity_joints.json: Joint 위치 (Unity)
        3. {prefix}_params.npz: 최적화된 SMAL 파라미터
        4. {prefix}_summary.json: 요약 정보 (loss, metrics)

    Args:
        smal_model: SMAL 모델 인스턴스
        state: 최적화된 파라미터 딕셔너리
        output_dir: 출력 디렉토리
        device: torch device
        target_points_cv: 타겟 메시 포인트 (OpenCV 좌표계)
        prefix: 출력 파일명 접두사

    생성되는 메트릭:
        - chamfer_to_mesh: 타겟 메시와의 Chamfer distance
        - anchor_surface_distance: 주요 joint와 메시 표면 간 거리
    """
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
    # target_points_unity = convert_opencv_to_unity(
    #     target_points_cv.cpu().numpy(), np.zeros(3, dtype=np.float32)
    # )
    # verts_unity, joints_unity, yaw_rad = align_forward_to_axis(
    #     verts_unity,
    #     joints_unity,
    #     target_axis=forward_axis,
    #     target_points=target_points_unity,
    # )

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

    metrics = {}
    try:
        pred_points = torch.tensor(verts_np, device=device, dtype=target_points_cv.dtype)
        metrics["chamfer_to_mesh"] = float(chamfer_distance(pred_points, target_points_cv).item())
    except Exception as exc:
        print(f"⚠️  Chamfer evaluation failed: {exc}")
        metrics["chamfer_to_mesh"] = None
    # metrics["yaw_applied_deg"] = float(np.degrees(yaw_rad))

    anchor_joint_ids = {
        "root": 0,
        "head": 16,
        "pelvis": 1,
        "tail_base": 25,
    }
    target_np = target_points_cv.cpu().numpy()
    anchor_errors = {}
    for name, idx in anchor_joint_ids.items():
        if idx < joints_np.shape[0]:
            anchor = joints_np[idx]
            diffs = target_np - anchor
            dists = np.linalg.norm(diffs, axis=1)
            anchor_errors[name] = float(dists.min())
        else:
            anchor_errors[name] = None
    metrics["anchor_surface_distance"] = anchor_errors

    meta = {
        "obj_path": str(obj_path),
        "joints_json": str(joints_json),
        "params_path": str(params_path),
        "best_loss": float(state.get("loss", 0.0)),
        "metrics": metrics,
    }
    meta_path = output_dir / f"{prefix}_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved SMAL fit OBJ to {obj_path}")
    print(f"Saved joints JSON to {joints_json}")
    print(f"Saved parameter cache to {params_path}")

    if debug_plot:
        try:
            save_debug_alignment_plot(
                stage_name=prefix,
                target_points=target_points_cv.cpu().numpy(),
                smal_vertices=verts_np,
                output_dir=output_dir,
            )
        except Exception as exc:
            print(f"⚠️  Debug plot failed for {prefix}: {exc}")


def main():
    """
    메인 실행 함수: 타겟 메시에 SMAL 최적화

    전체 파이프라인:
        1. 타겟 메시 로드 및 샘플링
        2. SMAL 모델 초기화
        3. 베이스라인 파라미터 로드 (있으면)
        4. 3단계 최적화:
           - 1단계: Similarity alignment (위치/크기/회전)
           - 2단계: Pose optimization
           - 3단계: Full optimization (shape + pose + global)
        5. 각 단계별 결과 저장
    """
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
        "--forward-axis",
        type=str,
        choices=["x", "y", "z"],
        default="z",
        help="Target forward axis for Unity export alignment.",
    )
    parser.add_argument(
        "--debug-plot",
        action="store_true",
        help="Save debug alignment plots comparing target mesh and SMAL output.",
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

    # ====================================================================
    # 1. 타겟 메시 로드 및 샘플링
    # ====================================================================
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input mesh must be a single Trimesh.")

    # 메시 표면에서 균일하게 포인트 샘플링
    mesh_points_np = mesh_to_points(mesh, args.samples)
    # Unity → OpenCV 좌표계 변환 (최적화는 OpenCV 좌표계에서 수행)
    mesh_points_cv = unity_to_opencv(mesh_points_np.astype(np.float32))
    mesh_points = torch.tensor(mesh_points_cv, device=device, dtype=torch.float32)

    # ====================================================================
    # 2. SMAL 모델 초기화
    # ====================================================================
    smal_model = SMALLayer(
        model_path=args.model_path,
        num_betas=SMALLayer.SHAPE_SPACE_DIM,
    ).to(device)

    if args.shape_only and not args.baseline_npz:
        raise ValueError("--baseline-npz must be provided when using --shape-only.")

    # ====================================================================
    # 3. 베이스라인 파라미터 로드 (또는 제로 초기화)
    # ====================================================================
    if args.baseline_npz:
        # GenZoo/AniMer로 생성한 초기 파라미터 사용
        baseline_params = load_baseline_parameters(args.baseline_npz, smal_model)
        betas_init = baseline_params["betas"]
        body_pose_axis_init = baseline_params["body_pose_axis"]
        body_pose_mat_init = baseline_params["body_pose_mat"]
        global_orient_axis_init = baseline_params["global_orient_axis"]
        global_orient_mat_init = baseline_params["global_orient_mat"]
    else:
        # 베이스라인 없으면 제로 초기화 (수렴 느림)
        betas_init = torch.zeros(1, SMALLayer.SHAPE_SPACE_DIM)
        body_pose_axis_init = torch.zeros(1, smal_model.NUM_BODY_JOINTS, 3)
        body_pose_mat_init = torch.eye(3).view(1, 1, 3, 3).repeat(1, smal_model.NUM_BODY_JOINTS, 1, 1)
        global_orient_axis_init = torch.zeros(1, 3)
        global_orient_mat_init = torch.eye(3).view(1, 1, 3, 3)

    # ====================================================================
    # SMAL 모델을 원점에서 시작하도록 초기 translation 설정
    # ====================================================================
    # 타겟 메시는 이미 원점에 있다고 가정
    mesh_center = mesh_points.mean(dim=0).cpu().numpy()
    mesh_extents = mesh_points.cpu().numpy().max(axis=0) - mesh_points.cpu().numpy().min(axis=0)
    print(f"Target mesh center: {mesh_center}")
    print(f"Target mesh extents (X, Y, Z): {mesh_extents}")
    print(f"Target mesh longest axis: {['X', 'Y', 'Z'][np.argmax(mesh_extents)]}")

    # SMAL 베이스라인 생성 (초기 translation 없음 = 원점에서 시작)
    with torch.no_grad():
        baseline_out = smal_model(
            betas=betas_init.to(device),
            body_pose=body_pose_mat_init.to(device),
            global_orient=global_orient_mat_init.to(device),
            pose2rot=False,
        )
    baseline_vertices = baseline_out.vertices[0].cpu().numpy()
    baseline_joints_cv = baseline_out.joints[0].detach().cpu().numpy()

    # 정렬 전 방향 확인
    head_vec_before = baseline_joints_cv[16] - baseline_joints_cv[0]
    belly_vec_before = (baseline_joints_cv[FOOT_JOINTS].mean(axis=0) - baseline_joints_cv[0])
    print(f"[DEBUG] Before alignment - head direction (OpenCV): {head_vec_before}")
    print(f"[DEBUG] Before alignment - belly direction (OpenCV): {belly_vec_before}")

    # 정렬 전 메시 저장 및 시각화 (디버깅용)
    baseline_vertices_unity_before = convert_opencv_to_unity(baseline_vertices, np.zeros(3, dtype=np.float32))
    baseline_joints_unity_before = convert_opencv_to_unity(baseline_joints_cv, np.zeros(3, dtype=np.float32))
    out_dir_debug = Path(args.output_dir) / "debug"
    out_dir_debug.mkdir(parents=True, exist_ok=True)
    save_vertices_obj(baseline_vertices_unity_before, smal_model.faces_tensor.cpu().numpy(),
                      out_dir_debug / "baseline_before_alignment.obj")
    export_joints_to_json(baseline_joints_unity_before, str(out_dir_debug / "baseline_before_alignment_joints.json"),
                         SMAL_JOINT_NAMES, SMAL_JOINT_PARENTS)

    # 3D 플롯 생성 (정렬 전)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(15, 5))

    # Unity 좌표계로 플롯
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(baseline_vertices_unity_before[:, 0], baseline_vertices_unity_before[:, 1],
                baseline_vertices_unity_before[:, 2], c='blue', s=0.1, alpha=0.3, label='SMAL mesh')
    ax1.scatter(baseline_joints_unity_before[:, 0], baseline_joints_unity_before[:, 1],
                baseline_joints_unity_before[:, 2], c='red', s=20, label='Joints')
    # Root 좌표계 축 표시
    root = baseline_joints_unity_before[0]
    axis_length = 0.5
    ax1.quiver(root[0], root[1], root[2], axis_length, 0, 0, color='red', arrow_length_ratio=0.2, linewidth=2, label='Root X')
    ax1.quiver(root[0], root[1], root[2], 0, axis_length, 0, color='green', arrow_length_ratio=0.2, linewidth=2, label='Root Y')
    ax1.quiver(root[0], root[1], root[2], 0, 0, axis_length, color='blue', arrow_length_ratio=0.2, linewidth=2, label='Root Z')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Before Alignment (Unity coords)')
    ax1.legend(fontsize=8)
    # Equal aspect ratio
    all_coords = baseline_vertices_unity_before
    max_range = np.array([all_coords[:,0].max()-all_coords[:,0].min(),
                          all_coords[:,1].max()-all_coords[:,1].min(),
                          all_coords[:,2].max()-all_coords[:,2].min()]).max() / 2.0
    mid_x = (all_coords[:,0].max()+all_coords[:,0].min()) * 0.5
    mid_y = (all_coords[:,1].max()+all_coords[:,1].min()) * 0.5
    mid_z = (all_coords[:,2].max()+all_coords[:,2].min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    print(f"[DEBUG] Saved baseline BEFORE alignment to {out_dir_debug}")

    global_orient_axis_init, global_orient_mat_init = align_global_orientation_to_forward(
        baseline_joints_cv,
        global_orient_mat_init.to(device),
        args.forward_axis,
    )
    global_orient_axis_init = global_orient_axis_init.to(device)
    global_orient_mat_init = global_orient_mat_init.to(device)

    # 정렬 후 확인
    with torch.no_grad():
        aligned_out = smal_model(
            betas=betas_init.to(device),
            body_pose=body_pose_mat_init.to(device),
            global_orient=global_orient_mat_init,
            pose2rot=False,
        )
    aligned_joints = aligned_out.joints[0].detach().cpu().numpy()
    aligned_vertices = aligned_out.vertices[0].detach().cpu().numpy()
    head_vec_after = aligned_joints[16] - aligned_joints[0]
    belly_vec_after = (aligned_joints[FOOT_JOINTS].mean(axis=0) - aligned_joints[0])
    print(f"[DEBUG] After alignment - head direction (OpenCV): {head_vec_after}")
    print(f"[DEBUG] After alignment - belly direction (OpenCV): {belly_vec_after}")
    print(f"[DEBUG] Target forward axis: {args.forward_axis} -> OpenCV: {map_forward_axis_to_opencv(args.forward_axis)}")

    # 정렬 후 메시 저장 및 플롯 (디버깅용)
    aligned_vertices_unity = convert_opencv_to_unity(aligned_vertices, np.zeros(3, dtype=np.float32))
    aligned_joints_unity = convert_opencv_to_unity(aligned_joints, np.zeros(3, dtype=np.float32))
    save_vertices_obj(aligned_vertices_unity, smal_model.faces_tensor.cpu().numpy(),
                      out_dir_debug / "baseline_after_alignment.obj")
    export_joints_to_json(aligned_joints_unity, str(out_dir_debug / "baseline_after_alignment_joints.json"),
                         SMAL_JOINT_NAMES, SMAL_JOINT_PARENTS)

    # 정렬 후 플롯
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(aligned_vertices_unity[:, 0], aligned_vertices_unity[:, 1],
                aligned_vertices_unity[:, 2], c='blue', s=0.1, alpha=0.3, label='SMAL mesh')
    ax2.scatter(aligned_joints_unity[:, 0], aligned_joints_unity[:, 1],
                aligned_joints_unity[:, 2], c='red', s=20, label='Joints')
    # Root 좌표계 축 표시
    root = aligned_joints_unity[0]
    ax2.quiver(root[0], root[1], root[2], axis_length, 0, 0, color='red', arrow_length_ratio=0.2, linewidth=2, label='Root X')
    ax2.quiver(root[0], root[1], root[2], 0, axis_length, 0, color='green', arrow_length_ratio=0.2, linewidth=2, label='Root Y')
    ax2.quiver(root[0], root[1], root[2], 0, 0, axis_length, color='blue', arrow_length_ratio=0.2, linewidth=2, label='Root Z')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('After Alignment (Unity coords)')
    ax2.legend(fontsize=8)
    # Equal aspect ratio
    all_coords = aligned_vertices_unity
    max_range = np.array([all_coords[:,0].max()-all_coords[:,0].min(),
                          all_coords[:,1].max()-all_coords[:,1].min(),
                          all_coords[:,2].max()-all_coords[:,2].min()]).max() / 2.0
    mid_x = (all_coords[:,0].max()+all_coords[:,0].min()) * 0.5
    mid_y = (all_coords[:,1].max()+all_coords[:,1].min()) * 0.5
    mid_z = (all_coords[:,2].max()+all_coords[:,2].min()) * 0.5
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    print(f"[DEBUG] Saved baseline AFTER alignment to {out_dir_debug}")

    print(f"Baseline SMAL center: {baseline_vertices.mean(axis=0)}")

    # 스케일 추정
    scale_guess = estimate_scale_from_bbox(mesh_points, baseline_vertices)

    # ====================================================================
    # 4. 1단계 최적화: Similarity Alignment
    # ====================================================================
    # Shape와 Pose 고정, Scale/Rotation/Translation만 최적화
    # 타겟 메시의 대략적인 위치와 크기에 맞춤
    print("Running similarity-only alignment (scale + rotation + translation)...")
    similarity_state = run_similarity_alignment(
        mesh_points=mesh_points,  # 타겟 메시 그대로 사용 (이미 원점)
        smal_model=smal_model,
        betas=betas_init,
        body_pose_mat=body_pose_mat_init,
        global_orient_mat=global_orient_mat_init,
        scale_init=scale_guess,
        iterations=200,
        point_subset=min(2048, args.samples),
        device=device,
    )

    # 1단계 결과 저장
    export_results(
        smal_model=smal_model,
        state=similarity_state,
        output_dir=Path(args.output_dir),
        device=device,
        target_points_cv=mesh_points,
        forward_axis=args.forward_axis,
        debug_plot=args.debug_plot,
        prefix="similarity_only",
    )

    # Similarity alignment 후 방향 확인 (디버깅)
    with torch.no_grad():
        sim_out = smal_model(
            betas=similarity_state["betas"].to(device),
            body_pose=batch_rodrigues(similarity_state["body_pose_axis"].to(device).view(-1, 3)).view(1, smal_model.NUM_BODY_JOINTS, 3, 3),
            global_orient=batch_rodrigues(similarity_state["global_orient_axis"].to(device).view(-1, 3)).view(1, 1, 3, 3),
            pose2rot=False,
        )
    sim_vertices = (sim_out.vertices[0] * similarity_state["scale"].to(device) + similarity_state["transl"].to(device)).detach().cpu().numpy()
    sim_joints = (sim_out.joints[0] * similarity_state["scale"].to(device) + similarity_state["transl"].to(device)).detach().cpu().numpy()
    sim_head_vec = sim_joints[16] - sim_joints[0]
    sim_belly_vec = (sim_joints[FOOT_JOINTS].mean(axis=0) - sim_joints[0])
    print(f"[DEBUG] After similarity alignment - head direction (OpenCV): {sim_head_vec}")
    print(f"[DEBUG] After similarity alignment - belly direction (OpenCV): {sim_belly_vec}")
    print(f"[DEBUG] Global orient change: {similarity_state['global_orient_axis'].cpu().numpy() - global_orient_axis_init.cpu().numpy()}")

    # Similarity 후 플롯 (OpenCV 좌표계에서 변환)
    sim_vertices_unity = convert_opencv_to_unity(sim_vertices, np.zeros(3, dtype=np.float32))
    sim_joints_unity = convert_opencv_to_unity(sim_joints, np.zeros(3, dtype=np.float32))

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(sim_vertices_unity[:, 0], sim_vertices_unity[:, 1],
                sim_vertices_unity[:, 2], c='blue', s=0.1, alpha=0.3, label='SMAL mesh')
    ax3.scatter(sim_joints_unity[:, 0], sim_joints_unity[:, 1],
                sim_joints_unity[:, 2], c='red', s=20, label='Joints')
    # Root 좌표계 축 표시
    root = sim_joints_unity[0]
    ax3.quiver(root[0], root[1], root[2], axis_length, 0, 0, color='red', arrow_length_ratio=0.2, linewidth=2, label='Root X')
    ax3.quiver(root[0], root[1], root[2], 0, axis_length, 0, color='green', arrow_length_ratio=0.2, linewidth=2, label='Root Y')
    ax3.quiver(root[0], root[1], root[2], 0, 0, axis_length, color='blue', arrow_length_ratio=0.2, linewidth=2, label='Root Z')
    # 타겟 메시도 표시
    mesh_points_unity = convert_opencv_to_unity(mesh_points.cpu().numpy(), np.zeros(3, dtype=np.float32))
    ax3.scatter(mesh_points_unity[:, 0], mesh_points_unity[:, 1],
                mesh_points_unity[:, 2], c='orange', s=0.1, alpha=0.2, label='Target mesh')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('After Similarity Alignment (Unity coords)')
    ax3.legend(fontsize=8)
    # Equal aspect ratio
    all_coords = np.vstack([sim_vertices_unity, mesh_points_unity])
    max_range = np.array([all_coords[:,0].max()-all_coords[:,0].min(),
                          all_coords[:,1].max()-all_coords[:,1].min(),
                          all_coords[:,2].max()-all_coords[:,2].min()]).max() / 2.0
    mid_x = (all_coords[:,0].max()+all_coords[:,0].min()) * 0.5
    mid_y = (all_coords[:,1].max()+all_coords[:,1].min()) * 0.5
    mid_z = (all_coords[:,2].max()+all_coords[:,2].min()) * 0.5
    ax3.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    fig.savefig(out_dir_debug / "alignment_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[DEBUG] Saved alignment comparison plot to {out_dir_debug / 'alignment_comparison.png'}")

    # 1단계 결과를 2단계의 초기값으로 사용
    global_orient_axis_state = similarity_state["global_orient_axis"]
    transl_init = similarity_state["transl"]
    scale_init = float(similarity_state["scale"].item())

    # ====================================================================
    # 5. 2단계 최적화: Pose Optimization
    # ====================================================================
    # Shape/Scale/Translation 고정, Pose만 최적화
    # 관절의 자세를 타겟 메시에 맞춤
    pose_state = run_optimization(
        mesh_points=mesh_points,  # 타겟 메시 그대로 사용
        smal_model=smal_model,
        betas_init=betas_init,
        body_pose_axis_init=body_pose_axis_init,
        global_orient_axis_init=global_orient_axis_state,
        transl_init=transl_init,
        scale_init=scale_init,
        iterations=args.iterations,
        point_subset=min(2048, args.samples),
        device=device,
        train_betas=False,  # Shape 고정
        train_pose=True,    # Pose 최적화
        train_global=False, # Scale/Translation 고정
        anchor_cfg=ANCHOR_CONFIG,
        anchor_sigma=ANCHOR_SIGMA,
        pose_prior=body_pose_axis_init,  # Prior로 베이스라인과 크게 벗어나지 않도록
        symmetry_weight=SYMMETRY_WEIGHT,
    )

    # 2단계 결과 저장
    export_results(
        smal_model=smal_model,
        state=pose_state,
        output_dir=Path(args.output_dir),
        device=device,
        target_points_cv=mesh_points,
        forward_axis=args.forward_axis,
        debug_plot=args.debug_plot,
        prefix="pose_fit",
    )

    # 2단계 결과를 3단계의 초기값으로 사용
    body_pose_axis_refined = pose_state["body_pose_axis"]

    # ====================================================================
    # 6. 3단계 최적화: Full Optimization (또는 Shape Only)
    # ====================================================================
    if args.shape_only:
        # --shape-only 옵션: Pose/Global 고정, Shape만 최적화
        shape_state = run_optimization(
            mesh_points=mesh_points,  # 타겟 메시 그대로 사용
            smal_model=smal_model,
            betas_init=betas_init,
            body_pose_axis_init=body_pose_axis_refined,
            global_orient_axis_init=global_orient_axis_state,
            transl_init=transl_init,
            scale_init=scale_init,
            iterations=args.iterations,
            point_subset=min(2048, args.samples),
            device=device,
            train_betas=True,   # Shape 최적화
            train_pose=False,   # Pose 고정
            train_global=False, # Scale/Translation 고정
            anchor_cfg=ANCHOR_CONFIG,
            anchor_sigma=ANCHOR_SIGMA,
            betas_prior=betas_init,
            symmetry_weight=SYMMETRY_WEIGHT,
        )

        # Shape-only 결과 저장
        export_results(
            smal_model=smal_model,
            state=shape_state,
            output_dir=Path(args.output_dir),
            device=device,
            target_points_cv=mesh_points,
            forward_axis=args.forward_axis,
            debug_plot=args.debug_plot,
            prefix="shape_fit",
        )
    else:
        # 기본: Shape, Pose, Global 모두 최적화
        full_state = run_optimization(
            mesh_points=mesh_points,  # 타겟 메시 그대로 사용
            smal_model=smal_model,
            betas_init=betas_init,
            body_pose_axis_init=body_pose_axis_refined,
            global_orient_axis_init=global_orient_axis_state,
            transl_init=transl_init,
            scale_init=scale_init,
            iterations=args.iterations,
            point_subset=min(2048, args.samples),
            device=device,
            train_betas=True,  # Shape 최적화
            train_pose=True,   # Pose 최적화
            train_global=True, # Scale/Translation 최적화
            anchor_cfg=ANCHOR_CONFIG,
            anchor_sigma=ANCHOR_SIGMA,
            betas_prior=betas_init,  # Prior로 과도한 변형 방지
            pose_prior=body_pose_axis_init,
            symmetry_weight=SYMMETRY_WEIGHT,
        )

        # 최종 결과 저장 (smal_fit_unity.obj가 최종 결과)
        export_results(
            smal_model=smal_model,
            state=full_state,
            output_dir=Path(args.output_dir),
            device=device,
            target_points_cv=mesh_points,
            forward_axis=args.forward_axis,
            debug_plot=args.debug_plot,
        )


if __name__ == "__main__":
    main()
