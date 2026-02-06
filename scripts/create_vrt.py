#!/usr/bin/env python3
"""TIF画像のディレクトリからVRTファイルを作成するスクリプト"""

import argparse
import subprocess
import sys
from pathlib import Path


def create_vrt(input_dir: Path, output_path: Path | None = None) -> Path:
    """指定ディレクトリ内のTIFファイルからVRTを作成する

    Args:
        input_dir: TIFファイルが含まれるディレクトリ
        output_path: 出力VRTファイルのパス（省略時はディレクトリ名.vrt）

    Returns:
        作成されたVRTファイルのパス
    """
    input_dir = Path(input_dir).resolve()

    if not input_dir.is_dir():
        raise ValueError(f"ディレクトリが存在しません: {input_dir}")

    tif_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.TIF"))

    if not tif_files:
        raise ValueError(f"TIFファイルが見つかりません: {input_dir}")

    if output_path is None:
        output_path = input_dir / f"{input_dir.name}.vrt"
    else:
        output_path = Path(output_path).resolve()

    cmd = [
        "gdalbuildvrt",
        str(output_path),
        *[str(f) for f in sorted(tif_files)],
    ]

    print(f"TIFファイル数: {len(tif_files)}")
    print(f"出力先: {output_path}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"エラー: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"VRTファイルを作成しました: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="TIF画像のディレクトリからVRTファイルを作成"
    )
    parser.add_argument("input_dir", type=Path, help="TIFファイルが含まれるディレクトリ")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="出力VRTファイルのパス"
    )

    args = parser.parse_args()
    create_vrt(args.input_dir, args.output)


if __name__ == "__main__":
    main()
