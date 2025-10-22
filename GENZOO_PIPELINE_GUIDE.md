# GenZoo 파이프라인 가이드

GenZoo는 이미지에서 동물의 3D 형태(shape)와 자세(pose)를 추정하는 파이프라인입니다.

## 📋 목차

1. [개요](#개요)
2. [파이프라인 구조](#파이프라인-구조)
3. [inference_genzoo.py - SMAL 추정](#inference_genzoospy---smal-추정)
4. [fit_smal_pose_from_mesh.py - 메시 최적화](#fit_smal_pose_from_meshpy---메시-최적화)
5. [실전 워크플로우](#실전-워크플로우)
6. [출력 파일 형식](#출력-파일-형식)

---

## 개요

GenZoo 파이프라인은 두 가지 주요 기능을 제공합니다:

1. **이미지 → SMAL 파라미터** (`inference_genzoo.py`)
   - 동물 이미지에서 SMAL shape과 pose 추정
   - GenZoo HMR2 모델 또는 AniMer/AWOL 사전 생성 파라미터 사용
   - 578종의 동물 지원

2. **3D 메시 → SMAL 최적화** (`fit_smal_pose_from_mesh.py`)
   - 타겟 3D 메시에 SMAL 모델을 정밀하게 맞춤
   - 3단계 최적화: Similarity → Pose → Full
   - Anchor-based weighting으로 중요 부위 강조

---

## 파이프라인 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                        GenZoo 파이프라인                          │
└─────────────────────────────────────────────────────────────────┘

방법 1: 이미지에서 시작
┌──────────────┐
│ 동물 이미지   │
│ (JPG/PNG)    │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  inference_genzoo.py                     │
│  - GenZoo HMR2 모델로 shape/pose 추정    │
│  - 또는 AniMer/AWOL 파라미터 사용        │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  출력: SMAL 파라미터                      │
│  - pose (J, 3, 3): Joint 회전 행렬       │
│  - beta (41,): Shape 파라미터            │
│  - scale, tx, ty: Weak perspective       │
│  - vertices_3d: 메시 정점                │
│  - keypoints_3d: Joint 3D 위치           │
└──────┬───────────────────────────────────┘
       │
       └─────────────────┐
                         │
방법 2: 3D 메시에서 시작│
┌──────────────┐        │
│ 타겟 메시     │        │
│ (OBJ 파일)    │        │
└──────┬───────┘        │
       │                │
       │    ┌───────────┴────────────────────────────┐
       │    │ 선택적: 베이스라인 파라미터 (NPZ)        │
       │    └───────────┬────────────────────────────┘
       │                │
       ▼                ▼
┌──────────────────────────────────────────────────────┐
│  fit_smal_pose_from_mesh.py                          │
│  - 3단계 최적화로 SMAL을 타겟 메시에 정밀 맞춤         │
│                                                      │
│  1단계: Similarity Alignment (200회)                 │
│    → Scale, Rotation, Translation 조정               │
│                                                      │
│  2단계: Pose Optimization (400회)                    │
│    → Joint pose만 최적화                             │
│                                                      │
│  3단계: Full Optimization (400회)                    │
│    → Shape + Pose + Global 모두 최적화                │
└──────┬───────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  출력: 최적화된 SMAL 메시                  │
│  - similarity_only_unity.obj             │
│  - pose_fit_unity.obj                    │
│  - smal_fit_unity.obj (최종)              │
│  - smal_fit_params.npz                   │
│  - smal_fit_summary.json                 │
└──────────────────────────────────────────┘
```

---

## inference_genzoo.py - SMAL 추정

### 기능

이미지에서 동물의 SMAL 파라미터를 추정하는 스크립트입니다.

### 입력

- **이미지 파일**: JPG, JPEG, PNG 형식
- **모델 체크포인트**: GenZoo HMR2 학습 모델 (`genzoo_1M.ckpt`)
- **(선택) AniMer/AWOL 파라미터**: 사전 생성된 shape/pose 파라미터

### 사용법

#### 기본 사용 (GenZoo HMR2 모델)

```bash
python genzoo/inference_genzoo.py \
    example_input/ \
    --checkpoint data/genzoo_1M.ckpt \
    --output output/genzoo_results \
    --render \
    --export-unity
```

#### AniMer/AWOL 파라미터 사용

```bash
python genzoo/inference_genzoo.py \
    example_input/ \
    --animer-params-dir smal_generation_results/leopard_cat/animer \
    --output output/genzoo_results \
    --export-unity
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `input` | 입력 이미지 경로 (파일 또는 디렉토리) | `./example_input` |
| `--checkpoint` | GenZoo 모델 체크포인트 경로 | `./data/genzoo_1M.ckpt` |
| `--output` | 출력 디렉토리 | `./output` |
| `--render` | 메시 렌더링 및 오버레이 생성 | False |
| `--export-unity` | Unity 좌표계 메시/Joint 출력 | False |
| `--animer-params-dir` | AniMer/AWOL 파라미터 디렉토리 | None |

### 출력 구조

```
output/
├── obj/                        # OpenCV 좌표계 메시
│   ├── image1.obj
│   └── image2.obj
├── data/                       # SMAL 파라미터 (NPZ)
│   ├── image1.npz
│   ├── image1_joints.json      # Joint 3D 위치
│   └── image2.npz
├── poses/                      # 3D Pose 시각화
│   ├── image1_pose.png
│   └── image2_pose.png
├── obj_unity/                  # Unity 좌표계 메시 (--export-unity)
│   ├── image1_unity.obj
│   └── image2_unity.obj
├── data_unity/                 # Unity Joint JSON (--export-unity)
│   ├── image1_joints_unity.json
│   └── image2_joints_unity.json
├── renders/                    # 렌더링 결과 (--render)
│   ├── image1.png
│   └── image2.png
└── overlays/                   # 원본 이미지 오버레이 (--render)
    ├── image1.png
    └── image2.png
```

### 동작 원리

#### 1. 이미지 로딩 및 전처리
- 입력 이미지를 PIL로 로드
- RGB 변환 및 크기 확인

#### 2. SMAL 파라미터 추정

**GenZoo HMR2 모드:**
- ViTDet으로 bounding box 검출
- HMR2 네트워크로 shape/pose 추정
- Weak perspective 파라미터 계산

**AniMer/AWOL 모드:**
- `all_parameters.npz`에서 사전 생성된 파라미터 로드
- Combined shape/pose 적용
- Deterministic weak perspective 추정

#### 3. SMAL 모델 순전파
```python
smal_out = smal_model(
    body_pose=body_pose_tensor,      # (1, 33, 3, 3)
    global_orient=global_orient,     # (1, 1, 3, 3)
    betas=betas,                     # (1, 41)
    pose2rot=False
)
vertices = smal_out.vertices         # (1, 3889, 3)
joints = smal_out.joints             # (1, 35, 3)
```

#### 4. 출력 생성
- **OpenCV 좌표계**: Y-down, Z-forward (원본)
- **Unity 좌표계**: Y-up, Z-forward (변환)
- Joint JSON, 3D pose 시각화, 렌더링

---

## fit_smal_pose_from_mesh.py - 메시 최적화

### 기능

타겟 3D 메시에 SMAL 모델을 정밀하게 최적화하여 맞추는 스크립트입니다.

### 입력

- **필수**: 타겟 메시 (OBJ 파일, Unity 좌표계)
- **선택**: 베이스라인 SMAL 파라미터 (NPZ 파일)
  - GenZoo 또는 AniMer로 생성한 초기 파라미터
  - 없으면 제로 초기화 (수렴 느림)

### 사용법

#### 최소 사용 (OBJ만)

```bash
python genzoo/fit_smal_pose_from_mesh.py \
    Roe_Deer/Roe_Deer_unity.obj
```

#### 권장 사용 (베이스라인 포함)

```bash
python genzoo/fit_smal_pose_from_mesh.py \
    Roe_Deer/Roe_Deer_unity.obj \
    --baseline-npz output/genzoo_results/data/roe_deer.npz \
    --output-dir fitted_results/Roe_Deer \
    --iterations 400 \
    --samples 4096
```

#### Shape만 최적화 (Pose 고정)

```bash
python genzoo/fit_smal_pose_from_mesh.py \
    target.obj \
    --baseline-npz baseline.npz \
    --shape-only
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `mesh_path` | 타겟 메시 경로 (Unity 좌표계) | 필수 |
| `--baseline-npz` | 베이스라인 파라미터 NPZ | None |
| `--output-dir` | 출력 디렉토리 | `genzoo/smal_fit_output` |
| `--iterations` | 최적화 반복 횟수 | 400 |
| `--samples` | 메시 표면 샘플 포인트 수 | 4096 |
| `--device` | 디바이스 (cuda/cpu) | 자동 감지 |
| `--shape-only` | Shape만 최적화 (Pose 고정) | False |
| `--forward-axis` | Unity export에서 머리가 향해야 할 축 (`x`,`y`,`z`) | `z` |
| `--model-path` | SMAL+ 모델 경로 | `genzoo/data/smal_plus.pkl` |

> 💡 **Forward axis**: Korean Goral처럼 원본 OBJ의 전방이 +Y인 경우 `--forward-axis y`를 지정하세요. 기본값(`z`)은 머리를 +Z 방향(정면)으로 맞춥니다.

### 출력 구조

```
fitted_results/
├── similarity_only_unity.obj       # 1단계: 위치/크기 정렬
├── similarity_only_unity_joints.json
├── similarity_only_params.npz
├── similarity_only_summary.json
│
├── pose_fit_unity.obj              # 2단계: Pose 최적화
├── pose_fit_unity_joints.json
├── pose_fit_params.npz
├── pose_fit_summary.json
│
├── smal_fit_unity.obj              # 3단계: 최종 결과
├── smal_fit_unity_joints.json
├── smal_fit_params.npz
└── smal_fit_summary.json
```

### 최적화 프로세스

#### 1단계: Similarity Alignment (200회)

**목적**: 타겟 메시의 대략적인 위치와 크기에 맞춤

**최적화 파라미터**:
- ✅ Scale (스케일)
- ✅ Global Rotation (전역 회전)
- ✅ Translation (이동)

**고정 파라미터**:
- ❌ Shape (beta)
- ❌ Body Pose

**Loss 구성**:
```python
loss = loss_data + loss_reg
# loss_data = anchor_weighted_chamfer_distance
# loss_reg = global_orient_regularization + scale_regularization
```

#### 2단계: Pose Optimization (400회)

**목적**: 관절의 자세를 타겟 메시에 맞춤

**최적화 파라미터**:
- ✅ Body Pose (33개 joint)

**고정 파라미터**:
- ❌ Shape (beta)
- ❌ Scale, Rotation, Translation (1단계 결과 고정)

**Loss 구성**:
```python
loss = loss_data + loss_prior + loss_symmetry
# loss_data = anchor_weighted_chamfer_distance
# loss_prior = (pose - baseline_pose)^2  # 베이스라인과 크게 벗어나지 않도록
# loss_symmetry = symmetric_bone_length_penalty
```

#### 3단계: Full Optimization (400회)

**목적**: Shape, Pose, Global 모두 최적화

**최적화 파라미터**:
- ✅ Shape (beta, 41차원)
- ✅ Body Pose (33개 joint)
- ✅ Scale, Rotation, Translation

**Loss 구성**:
```python
loss = loss_data + loss_prior_shape + loss_prior_pose + loss_reg + loss_symmetry
# loss_data = anchor_weighted_chamfer_distance
# loss_prior_shape = (beta - baseline_beta)^2
# loss_prior_pose = (pose - baseline_pose)^2
# loss_reg = global_orient_regularization
# loss_symmetry = symmetric_bone_length_penalty
```

### Anchor-based Weighting

중요한 신체 부위에 높은 가중치를 부여하여 정밀도 향상:

```python
ANCHOR_CONFIG = [
    {"name": "head", "joints": [...], "weight": 2.5},   # 머리
    {"name": "ears", "joints": [33, 34], "weight": 3.0}, # 귀 (최고 가중치)
    {"name": "front_legs", "joints": [...], "weight": 1.6},
    {"name": "back_legs", "joints": [...], "weight": 1.6},
]
```

**동작 원리**:
1. 각 메시 포인트에서 anchor joint까지 거리 계산
2. Gaussian 함수로 영향도 계산: `exp(-dist/sigma)`
3. 머리/귀 근처 점들이 높은 가중치를 받음
4. Chamfer distance 계산 시 가중치 적용

### Symmetry Loss

좌우 대칭을 유지하기 위한 제약:

```python
# 대칭 joint 쌍 예시: (왼쪽 앞다리, 오른쪽 앞다리)
for left, right in SYMMETRIC_JOINT_PAIRS:
    left_bone = joints[left] - joints[left_parent]
    right_bone = joints[right] - joints[right_parent]
    penalty = abs(||left_bone|| - ||right_bone||)
```

---

## 실전 워크플로우

### 워크플로우 1: 이미지에서 SMAL 생성

```bash
# 1단계: 이미지에서 SMAL 추정
python genzoo/inference_genzoo.py \
    leopard_cat_images/ \
    --checkpoint data/genzoo_1M.ckpt \
    --output results/leopard_cat \
    --render \
    --export-unity

# 결과:
# - results/leopard_cat/obj_unity/image1_unity.obj
# - results/leopard_cat/data/image1.npz
```

### 워크플로우 2: 3D 메시를 SMAL로 변환

```bash
# 1단계: 타겟 메시 준비
# - Leopard_cat_unity.obj (Unity 좌표계)

# 2단계: SMAL 최적화 (베이스라인 없이)
python genzoo/fit_smal_pose_from_mesh.py \
    Leopard_cat/Leopard_cat_unity.obj \
    --output-dir fitted_results/leopard_cat \
    --iterations 400 \
    --samples 4096

# 결과:
# - fitted_results/leopard_cat/smal_fit_unity.obj (최종)
```

### 워크플로우 3: AniMer 파라미터 + 메시 최적화

```bash
# 1단계: AniMer로 이미지에서 파라미터 생성
# (AniMer 파이프라인 실행 - 별도 문서 참조)
# 결과: smal_generation_results/leopard_cat/animer/DSC06311/all_parameters.npz

# 2단계: GenZoo로 AniMer 파라미터 활용
python genzoo/inference_genzoo.py \
    leopard_cat_images/ \
    --animer-params-dir smal_generation_results/leopard_cat/animer \
    --output results/leopard_cat_animer \
    --export-unity

# 3단계: 타겟 메시에 최적화 (베이스라인 포함)
python genzoo/fit_smal_pose_from_mesh.py \
    Leopard_cat/Leopard_cat_unity.obj \
    --baseline-npz results/leopard_cat_animer/data/DSC06311.npz \
    --output-dir fitted_results/leopard_cat_final \
    --iterations 400

# 결과:
# - fitted_results/leopard_cat_final/smal_fit_unity.obj (최종, 베이스라인 기반)
```

### 워크플로우 4: 형태 변형 (Morphology)

```bash
# 1단계: SMAL 파라미터 생성 (워크플로우 1-3 중 하나)

# 2단계: 형태 변형 적용
python morph_smal.py \
    --input fitted_results/leopard_cat/smal_fit_params.npz \
    --output morphed_results \
    --tall 1.3 \
    --fat 1.4

# 또는 고급 변형 (다리 길이 독립 제어)
python morph_smal.py \
    --input fitted_results/leopard_cat/smal_fit_params.npz \
    --output morphed_results \
    --long_legs 1.4 \
    --slim 0.7

# 결과:
# - morphed_results/leopard_cat_tall_fat.obj
# - morphed_results/leopard_cat_long_legs_slim.obj
```

---

## 출력 파일 형식

### NPZ 파일 (SMAL 파라미터)

#### inference_genzoo.py 출력

```python
data = np.load("output/data/image.npz")

# 주요 키:
data["pose"]          # (35, 3, 3) - Joint 회전 행렬 (global_orient + body_pose)
data["beta"]          # (41,) - Shape 파라미터
data["scale"]         # (1,) - Weak perspective scale
data["tx"]            # (1,) - Translation X
data["ty"]            # (1,) - Translation Y
data["vertices_3d"]   # (3889, 3) - 메시 정점 (OpenCV 좌표계)
data["keypoints_3d"]  # (35, 3) - Joint 3D 위치
data["keypoints_2d"]  # (35, 2) - Joint 2D 투영
data["vertices_2d"]   # (3889, 2) - 정점 2D 투영
data["transl"]        # (3,) - 3D Translation (AniMer/AWOL만)
```

#### fit_smal_pose_from_mesh.py 출력

```python
params = np.load("fitted_results/smal_fit_params.npz")

# 주요 키:
params["betas"]              # (1, 41) - 최적화된 Shape
params["body_pose_axis"]     # (1, 33, 3) - Body pose (axis-angle)
params["global_orient_axis"] # (1, 3) - Global orientation (axis-angle)
params["transl"]             # (1, 3) - Translation
params["scale"]              # (1, 1) - Scale
```

### JSON 파일 (Joint 메타데이터)

```json
{
  "joints": [
    {"name": "root", "position": [0.0, 0.5, 0.0], "parent": -1},
    {"name": "pelvis", "position": [0.0, 0.6, -0.1], "parent": 0},
    {"name": "spine_1", "position": [0.0, 0.7, 0.0], "parent": 1},
    ...
  ],
  "metadata": {
    "coordinate_system": "Unity (Y-up, Z-forward)",
    "unit": "meters",
    "joint_count": 35
  }
}
```

### Summary JSON (최적화 결과)

```json
{
  "obj_path": "fitted_results/smal_fit_unity.obj",
  "joints_json": "fitted_results/smal_fit_unity_joints.json",
  "params_path": "fitted_results/smal_fit_params.npz",
  "best_loss": 0.0234,
  "metrics": {
    "chamfer_to_mesh": 0.0189,
    "anchor_surface_distance": {
      "root": 0.012,
      "head": 0.008,
      "pelvis": 0.015,
      "tail_base": 0.011
    }
  }
}
```

---

## 좌표계 변환

### OpenCV 좌표계
- **X**: 오른쪽
- **Y**: 아래
- **Z**: 앞 (forward)

### Unity 좌표계
- **X**: 오른쪽
- **Y**: 위
- **Z**: 앞 (forward)

### 변환 공식

```python
def convert_opencv_to_unity(points_cv, root_joint):
    """
    OpenCV → Unity 변환
    1. Root joint를 원점으로 이동
    2. Y와 Z 축 뒤집기
    """
    points_centered = points_cv - root_joint
    points_unity = points_centered.copy()
    points_unity[:, 1] = -points_centered[:, 1]  # Y 뒤집기
    points_unity[:, 2] = -points_centered[:, 2]  # Z 뒤집기
    return points_unity
```

---

## 성능 최적화 팁

### inference_genzoo.py

1. **GPU 사용**: CUDA 가능 시 자동으로 GPU 사용
2. **배치 크기**: 현재 단일 이미지 처리 (배치 처리 미지원)
3. **렌더링 속도**: 첫 렌더링은 warm-up으로 느림 (정상)

### fit_smal_pose_from_mesh.py

1. **샘플 포인트 수**: `--samples 4096` (기본값)
   - 더 높이면 정확도 증가, 속도 감소
   - 권장 범위: 2048-8192

2. **반복 횟수**: `--iterations 400` (기본값)
   - Stage 2, 3에 적용
   - 더 높이면 수렴 가능성 증가

3. **Point subset**: 내부적으로 2048 포인트만 샘플링하여 최적화 속도 향상

4. **디바이스**: CUDA 사용 시 10-20배 빠름

---

## 문제 해결

### inference_genzoo.py

**문제**: `No valid images found`
- 해결: 이미지 경로 확인, JPG/PNG 형식인지 확인

**문제**: `FileNotFoundError: checkpoint not found`
- 해결: `--checkpoint` 경로 확인, `genzoo_1M.ckpt` 다운로드 확인

**문제**: AniMer 파라미터 로드 실패
- 해결: `all_parameters.npz` 파일 존재 확인, 파일명이 이미지 stem과 일치하는지 확인

### fit_smal_pose_from_mesh.py

**문제**: `Mesh not found`
- 해결: OBJ 파일 경로 확인, Unity 좌표계인지 확인

**문제**: 수렴 실패 (loss가 높음)
- 해결:
  1. 베이스라인 NPZ 제공 (필수는 아니지만 강력히 권장)
  2. `--iterations` 증가 (600-800)
  3. `--samples` 증가 (6144-8192)

**문제**: 메시가 뒤집힘
- 해결: 입력 메시가 Unity 좌표계인지 확인, OpenCV 좌표계면 변환 필요

**문제**: Shape만 최적화 시 `--baseline-npz must be provided`
- 해결: `--shape-only` 사용 시 반드시 베이스라인 제공 필요

---

## 참고 자료

- **SMAL 모델**: [Zuffi et al. 2017] - 3D Shape Model and Articulated Gaussian
- **GenZoo**: 578종 동물 지원하는 HMR2 기반 모델
- **AniMer**: AWOL (shape) + AniMer (pose) 파이프라인
- **형태 변형 가이드**: `MORPH_GUIDE_SUMMARY.md` 참조

---

## 라이선스 및 인용

이 코드는 연구 목적으로 제공됩니다. 상업적 사용 시 별도 라이선스 필요.

```bibtex
@inproceedings{genzoo2024,
  title={GenZoo: Generalizable 3D Animal Models},
  author={...},
  year={2024}
}
```
