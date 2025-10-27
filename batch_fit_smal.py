"""
SMAL 메시 피팅 배치 처리 스크립트

텍스트 파일에서 여러 작업을 읽어서 자동으로 처리합니다.

=== 사용법 ===

1. 작업 목록 파일 작성 (예: tasks.txt):
   target1.obj baseline1.npz
   target2.obj baseline2.npz
   target3.obj baseline3.npz

2. 배치 실행:
   python genzoo/batch_fit_smal.py tasks.txt

각 작업의 출력은 baseline npz 파일과 같은 디렉토리에 저장됩니다.
예: baseline1.npz가 output/animal1/baseline.npz라면
    결과는 output/animal1/ 에 저장됩니다.

=== 추가 옵션 ===

--iterations N       최적화 반복 횟수 (기본: 400)
--samples N          메시 샘플링 포인트 수 (기본: 4096)
--device cuda/cpu    사용할 디바이스 (기본: cuda)
--shape-only         Shape만 최적화 (Pose 고정)
--skip-existing      이미 처리된 작업 건너뛰기

=== 예제 ===

python genzoo/batch_fit_smal.py tasks.txt --iterations 300 --skip-existing
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import subprocess


def parse_task_file(task_file: Path) -> List[Tuple[str, str]]:
    """
    작업 파일 파싱

    Args:
        task_file: 작업 목록 파일 경로

    Returns:
        [(target_obj, baseline_npz), ...] 리스트

    파일 형식:
        각 줄에 "target.obj baseline.npz" 형식으로 작성
        빈 줄과 #으로 시작하는 주석은 무시
    """
    tasks = []

    with open(task_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # 빈 줄이나 주석 무시
            if not line or line.startswith('#'):
                continue

            # 공백으로 분리
            parts = line.split()

            if len(parts) != 2:
                print(f"⚠️  Line {line_num}: Invalid format (expected 'target.obj baseline.npz')")
                print(f"    Skipping: {line}")
                continue

            target_obj, baseline_npz = parts

            # 파일 존재 확인
            if not Path(target_obj).exists():
                print(f"⚠️  Line {line_num}: Target mesh not found: {target_obj}")
                print(f"    Skipping this task")
                continue

            if not Path(baseline_npz).exists():
                print(f"⚠️  Line {line_num}: Baseline NPZ not found: {baseline_npz}")
                print(f"    Skipping this task")
                continue

            tasks.append((target_obj, baseline_npz))

    return tasks


def get_output_dir_from_npz(npz_path: str) -> Path:
    """
    NPZ 파일 경로에서 출력 디렉토리 결정

    NPZ 파일과 같은 디렉토리 안에 NPZ 파일명으로 새 폴더 생성
    예: output/animal1/baseline.npz -> output/animal1/baseline/
        output/animal1/smal_parameters.npz -> output/animal1/smal_parameters/
    """
    npz_file = Path(npz_path)
    npz_stem = npz_file.stem  # 확장자 제외한 파일명
    return npz_file.parent / npz_stem


def check_task_completed(output_dir: Path) -> bool:
    """
    작업이 이미 완료되었는지 확인

    smal_fit_unity.obj 파일이 존재하면 완료된 것으로 간주
    """
    return (output_dir / "smal_fit_unity.obj").exists()


def run_fit_task(
    target_obj: str,
    baseline_npz: str,
    output_dir: Path,
    iterations: int,
    samples: int,
    device: str,
    shape_only: bool,
) -> bool:
    """
    단일 피팅 작업 실행

    Returns:
        bool: 성공 여부
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # fit_smal_pose_from_mesh.py 실행
    cmd = [
        sys.executable,
        "genzoo/fit_smal_pose_from_mesh.py",
        target_obj,
        "--baseline-npz", baseline_npz,
        "--output-dir", str(output_dir),
        "--iterations", str(iterations),
        "--samples", str(samples),
        "--device", device,
    ]

    if shape_only:
        cmd.append("--shape-only")

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Task failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Task failed with exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch process SMAL mesh fitting from task file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "task_file",
        type=str,
        help="Text file with tasks (each line: target.obj baseline.npz)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=400,
        help="Number of optimization iterations (default: 400)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4096,
        help="Number of surface samples (default: 4096)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for optimization (default: cuda)"
    )
    parser.add_argument(
        "--shape-only",
        action="store_true",
        help="Freeze pose and optimize shape only"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks that are already completed"
    )

    args = parser.parse_args()

    task_file = Path(args.task_file)
    if not task_file.exists():
        print(f"❌ Task file not found: {task_file}")
        sys.exit(1)

    # 작업 목록 파싱
    print(f"📋 Reading tasks from: {task_file}")
    tasks = parse_task_file(task_file)

    if not tasks:
        print("❌ No valid tasks found in the file")
        sys.exit(1)

    print(f"\n✅ Found {len(tasks)} valid tasks\n")

    # 작업 실행
    completed = 0
    skipped = 0
    failed = 0

    for i, (target_obj, baseline_npz) in enumerate(tasks, 1):
        output_dir = get_output_dir_from_npz(baseline_npz)

        print(f"\n{'#'*80}")
        print(f"Task {i}/{len(tasks)}")
        print(f"  Target:   {target_obj}")
        print(f"  Baseline: {baseline_npz}")
        print(f"  Output:   {output_dir}")
        print(f"{'#'*80}")

        # 이미 완료된 작업 확인
        if args.skip_existing and check_task_completed(output_dir):
            print(f"⏭️  Task already completed, skipping...")
            skipped += 1
            continue

        # 작업 실행
        success = run_fit_task(
            target_obj=target_obj,
            baseline_npz=baseline_npz,
            output_dir=output_dir,
            iterations=args.iterations,
            samples=args.samples,
            device=args.device,
            shape_only=args.shape_only,
        )

        if success:
            print(f"✅ Task {i}/{len(tasks)} completed successfully")
            completed += 1
        else:
            print(f"❌ Task {i}/{len(tasks)} failed")
            failed += 1

    # 최종 요약
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks:      {len(tasks)}")
    print(f"✅ Completed:     {completed}")
    print(f"⏭️  Skipped:       {skipped}")
    print(f"❌ Failed:        {failed}")
    print(f"{'='*80}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
