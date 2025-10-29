#!/usr/bin/env python3
"""
타겟 메시의 구멍을 메우는 전처리 스크립트

Usage:
    python genzoo/fix_mesh_holes.py input.obj output.obj
"""

import argparse
import trimesh
from pathlib import Path


def fix_mesh_holes(input_path: str, output_path: str):
    """
    메시의 구멍을 메우고 정리

    Args:
        input_path: 입력 OBJ 파일
        output_path: 출력 OBJ 파일
    """
    print(f"Loading mesh from {input_path}...")
    mesh = trimesh.load_mesh(input_path, process=False)

    print(f"Original mesh:")
    print(f"  - Vertices: {len(mesh.vertices)}")
    print(f"  - Faces: {len(mesh.faces)}")
    print(f"  - Is watertight: {mesh.is_watertight}")

    # 구멍 메우기
    if not mesh.is_watertight:
        print("\nFilling holes...")
        try:
            mesh.fill_holes()
            print(f"  ✓ Holes filled")
        except Exception as e:
            print(f"  ⚠ Could not fill all holes: {e}")

    # 불필요한 vertices 제거
    mesh.remove_unreferenced_vertices()

    # Degenerate faces 제거
    mesh.remove_degenerate_faces()

    # Duplicate faces 제거
    mesh.remove_duplicate_faces()

    print(f"\nCleaned mesh:")
    print(f"  - Vertices: {len(mesh.vertices)}")
    print(f"  - Faces: {len(mesh.faces)}")
    print(f"  - Is watertight: {mesh.is_watertight}")

    # 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    print(f"\n✓ Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fix mesh holes and clean geometry")
    parser.add_argument("input", help="Input OBJ file")
    parser.add_argument("output", help="Output OBJ file")
    args = parser.parse_args()

    fix_mesh_holes(args.input, args.output)


if __name__ == "__main__":
    main()
