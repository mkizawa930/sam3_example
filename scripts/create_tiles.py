#!/usr/bin/env python3
"""VRTまたはラスター画像をタイルに分割するスクリプト"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()


def create_tile(
    input_path: str,
    output_path: str,
    x_offset: int,
    y_offset: int,
    tile_width: int,
    tile_height: int,
) -> str | None:
    """1つのタイルを切り出す"""
    try:
        ds = gdal.Open(input_path)
        gdal.Translate(
            output_path,
            ds,
            srcWin=[x_offset, y_offset, tile_width, tile_height],
            creationOptions=["COMPRESS=LZW", "TILED=YES"],
        )
        ds = None
        return output_path
    except Exception:
        return None


def create_tiles(
    input_path: Path,
    output_dir: Path,
    tile_size: int = 1000,
    workers: int = 4,
) -> list[Path]:
    """ラスター画像をタイルに分割する

    Args:
        input_path: 入力ファイル（VRT, GeoTIFF等）
        output_dir: 出力ディレクトリ
        tile_size: タイルサイズ（ピクセル）
        workers: 並列処理数

    Returns:
        作成されたタイルファイルのリスト
    """
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = gdal.Open(str(input_path))
    width = ds.RasterXSize
    height = ds.RasterYSize
    ds = None

    print(f"入力サイズ: {width} x {height}")
    print(f"タイルサイズ: {tile_size} x {tile_size}")

    tasks = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile_w = min(tile_size, width - x)
            tile_h = min(tile_size, height - y)
            output_path = output_dir / f"tile_{x:06d}_{y:06d}.tif"
            tasks.append((x, y, tile_w, tile_h, output_path))

    print(f"タイル数: {len(tasks)}")

    created = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                create_tile,
                str(input_path),
                str(output_path),
                x,
                y,
                tile_w,
                tile_h,
            ): output_path
            for x, y, tile_w, tile_h, output_path in tasks
        }

        for i, future in enumerate(as_completed(futures), 1):
            output_path = futures[future]
            result = future.result()
            if result:
                created.append(Path(result))
            print(f"\r進捗: {i}/{len(tasks)}", end="", flush=True)

    print(f"\n完了: {len(created)} タイルを作成しました")
    return created


def main():
    parser = argparse.ArgumentParser(description="ラスター画像をタイルに分割")
    parser.add_argument("input", type=Path, help="入力ファイル（VRT, GeoTIFF等）")
    parser.add_argument("output_dir", type=Path, help="出力ディレクトリ")
    parser.add_argument(
        "-s", "--size", type=int, default=1000, help="タイルサイズ（デフォルト: 1000）"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=4, help="並列処理数（デフォルト: 4）"
    )

    args = parser.parse_args()
    create_tiles(args.input, args.output_dir, args.size, args.workers)


if __name__ == "__main__":
    main()
