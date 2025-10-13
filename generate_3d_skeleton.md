목표: 3d obj 모델의 스켈레톤 json 생성하기

1. [Roe Deer 3D model](../origin/Roe_Deer/origin.obj)은 노루 이미지(/workspace/genzoo/input/Roe_deer.JPG)에 포함된 동물을 3d reconstruction한 결과이다.
2. 노루 이미지를 /workspace/genzoo/inference_genzoo.py를 통해 SMAL plus 모델과 3D obj, 3d skeleton 등을 생성한 결과는 output 폴더에 Roe_deer_*로 생성되었다.
3. 3D obj는 이미지로부터 생성한 SMAL plus 파라미터를 기반으로 생성하며 /workspace/nature3d/predictor.py에 자세한 방법에 대한 설명이 있다.
4. 나는 Roe Deer 3D model에 맞는 SMAL plus 파라미터를 알고싶다.
5. Roe Deer 3D model은 unity 좌표계(y-up)를 따른다.
6. 이미지로부터 생성한 SMAL plus 파라미터를 unity 좌표계에 맞추고 3d obj를 생성하는 방법은 /workspace/nature3d/joint_utils.py에 있다.
7. SMAL plus 파라미터를 이용하여 3D 모델에서 3D joint의 위치와 hierarchy 구조(parent-children 관계)를 알아내는 코드는 joint_utils.py에 있다.
8. 하지만 진짜 원하는 건 Roe Deer 3D model의 스켈레톤 구조를 알아내는 것이다.
9. 이를 위하여 이미지로부터 생성한 unity 3D 모델(/workspace/genzoo/output/obj_unity/Roe_deer_unity.obj)을 Roe Deer 3D model에 맞춰 크기, 회전을 변형하여 SMAL plus 파라미터에 반영하고 pose, shape parameter를 모델에 피팅하고 싶다.
10. 스켈레톤 구조는 joint_utils.py에서 생성하는 json 구조와 동일하게 35개의 joint여야 한다.
11. 이 과정에서 3D 모델의 멀티뷰 이미지를 촬영하여 정확도를 높이려는 시도도 해보았다.(/workspace/MULTIVIEW_SMAL_README.md)

---

## 2024-XX-XX 추가 관찰 및 시도

### 좌표계 정합
- Unity 메시와 SMAL 좌표계(OpenCV, -Y up) 사이의 축 방향이 일치하지 않아 앞/뒤 다리 판단이 거꾸로 되는 문제가 있었다.  
- `convert_opencv_to_unity` 변환이 Y만 반전시키고 있었는데, 실제로는 Z 축도 함께 반전해야 앞 방향이 유지된다.  
- `nature3d/utils.py`와 `fit_smal_parameters.py`에 동일한 보정을 적용하여 Unity ↔ OpenCV 전환이 대칭이 되도록 수정하였다.

### SMAL 파라미터 직접 피팅 시도
- 이미지 추론 없이 **기성 Unity 메시에서 SMAL 파라미터(포즈/쉐이프/스케일/필요한 평행이동)를 역으로 찾는** 실험을 진행했다.
- 새 스크립트 `genzoo/fit_smal_pose_from_mesh.py` 를 작성하였다.
  - 입력: Unity 좌표계 OBJ (예: `origin/Roe_Deer/origin.obj`)
  - 내부 단계  
    1. 메시 표면을 샘플링하고 OpenCV 좌표계로 변환  
    2. SMAL 파라미터(베타, 포즈, 전역 회전, 스케일, 평행이동)를 PyTorch로 초기화  
    3. Chamfer 거리(표면 근접도) + L2 정규화를 손실로 하여 Adam으로 최적화  
    4. 최적화된 파라미터로 생성된 SMAL 메시와 조인트를 Unity 좌표계로 다시 변환하여 저장
  - 출력:  
    - `smal_fit_unity.obj` : SMAL 메시가 원본과 정렬된 결과  
    - `smal_fit_unity_joints.json` : 35개 조인트 JSON (joint_utils 포맷)  
    - `smal_fit_params.npz` : 최적화된 pose/shape/scale/transl 기록  
    - `smal_fit_summary.json` : 손실 값 요약
- Chamfer 거리 기반 단일 뷰 정합이라 다리/꼬리 등의 미세 관절은 여전히 지역 최적해에 빠질 수 있음.  
- 손실을 줄이기 위해 더 많은 표면 샘플·멀티뷰·관절 priors 등을 추가하는 확장이 필요하다.

### 다음 개선 아이디어
- Chamfer 대신 최근접대응을 안정화하기 위해 PyTorch3D의 메시에 대한 ICP 혹은 고정 앵커 포인트(머리, 꼬리, 발굽 등)를 이용하는 방법 검토.  
- 멀티 뷰 렌더링 데이터(/workspace/MULTIVIEW_SMAL_README.md)와 결합하여 관절 위치에 대한 추가 관측(2D keypoint) 제약을 주는 방향.

### 2024-XX-XX 테스트 로그
- Unity OBJ의 전방 축이 뒤틀리는 문제 디버깅을 위해 `fit_smal_pose_from_mesh.py`에 전체 회전 보정(`align_forward_to_x`)을 추가했다. 변환 후 머리 방향이 +X 축을 향하도록 Yaw를 자동 보정한다.
- 테스트:  
  ```
  python - <<'PY'
  import numpy as np
  from nature3d.utils import convert_opencv_to_unity
  from genzoo.fit_smal_pose_from_mesh import align_forward_to_x
  npz = np.load('genzoo/output/data/DSC00530.npz')
  verts = npz['vertices_3d']; joints = npz['keypoints_3d']; root = joints[0]
  verts_u = convert_opencv_to_unity(verts, root); joints_u = convert_opencv_to_unity(joints, root)
  _, joints_rot, yaw = align_forward_to_x(verts_u, joints_u)
  print('yaw_deg', np.degrees(yaw)); print('dir', joints_rot[16]-joints_rot[0])
  PY
  ```
- 결과: `yaw_deg ≈ -27.53`, `dir ≈ [0.91, 0.12, ~0]` 으로, 회전 후 전방 벡터가 X 축과 정렬됨을 확인했다.
- SMAL pose는 baseline 이미지 추론 결과로 고정하고 shape만 피팅하는 파이프라인을 시험했다.  
  ```
  python genzoo/fit_smal_pose_from_mesh.py \
      origin/Roe_Deer/origin.obj \
      --baseline-npz genzoo/output/data/Roe_deer.npz \
      --shape-only \
      --output-dir genzoo/smal_fit_roe_deer \
      --iterations 200 \
      --samples 4096
  ```
  - similarity 단계 Chamfer ≈ 0.1486 → shape 단계 후 0.0883으로 감소
  - 산출물: `genzoo/smal_fit_roe_deer/similarity_only_unity.obj`, `shape_fit_unity.obj`, 대응하는 joints/params npz
- 좌표계를 Unity(+Z forward, +Y up)로 재정의한 후 동일 파이프라인을 반복 실행하였다.  
  ```
  python genzoo/fit_smal_pose_from_mesh.py \
      origin/Roe_Deer/origin.obj \
      --baseline-npz genzoo/output/data/Roe_deer.npz \
      --shape-only \
      --output-dir genzoo/smal_fit_roe_deer \
      --iterations 200 \
      --samples 4096
  ```
  - yaw 보정량은 ≈ +4°, 전방 축 정렬이 origin.obj와 일치
  - Chamfer: 0.1479 → 0.0921 (shape 단계)
  - scale prior를 bbox 기반으로 초기화하고 유지(≈0.39)하여 원본과 크기 차이를 해소
  - 결과물은 `genzoo/smal_fit_roe_deer/shape_fit_unity.obj` 등으로 저장
- 포즈를 선행 최적화한 뒤 shape를 보정하도록 파이프라인을 확장했다. 귀/다리 영역에 높은 가중치를 주는 anchor loss(관절 카테고리 기반)와 좌우 대칭 규제(SYMMETRIC_JOINT_PAIRS)를 추가했다.  
  ```
  python genzoo/fit_smal_pose_from_mesh.py \
      origin/Roe_Deer/origin.obj \
      --baseline-npz genzoo/output/data/Roe_deer.npz \
      --shape-only \
      --output-dir genzoo/smal_fit_roe_deer \
      --iterations 100 \
      --samples 2048
  ```
  - 유사변환 정합: Chamfer 0.051 → 포즈 단계 0.0327 → shape 단계 0.0260
  - 익힌 가중치 덕분에 귀·발목 인근 오차가 감소, `pose_fit_unity.obj`, `shape_fit_unity.obj` 등 추가 산출
  - 여전히 귀 끝/발굽 세부는 SMAL shape subspace 한계로 완전 일치하진 않음 → 추가 anchor 또는 개별 파라미터 필요
- `hmr2/utils/pose_utils.compute_similarity_transform`를 활용해 anchor joint(머리/귀/앞·뒷다리) 중심으로 추정한 대응점에 대해 Procrustes 기반 초깃값(scale/rotation/translation)을 계산, bbox 기반 초기화의 불안정을 해소했다. `hmr2/utils/geometry`의 회전 변환과 기존 `batch_rodrigues`를 조합해 포즈·형상 업데이트를 수행했다.
