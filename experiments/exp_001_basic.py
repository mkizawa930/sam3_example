"""基本的なSAM2セグメンテーション実験

使用方法:
    uv run python experiments/exp_001_basic.py --image data/sample.jpg

Note:
    SAM2はポイントプロンプトまたはボックスプロンプトを使用します。
    テキストプロンプトはSAM3で対応予定。
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from sam3_example import get_device, load_sam2_model, save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 Basic Segmentation")
    parser.add_argument(
        "--image", type=str, required=True,
        help="入力画像のパス"
    )
    parser.add_argument(
        "--point", type=str, default=None,
        help="セグメンテーションポイント (x,y 形式)"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/exp_001",
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--model", type=str, default="facebook/sam2-hiera-large",
        choices=[
            "facebook/sam2-hiera-tiny",
            "facebook/sam2-hiera-small",
            "facebook/sam2-hiera-base-plus",
            "facebook/sam2-hiera-large",
        ],
        help="使用するモデル"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # デバイス確認
    device = get_device()
    print(f"Using device: {device}")

    # 画像読み込み
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image_path} ({image.size})")

    # モデル読み込み
    print(f"Loading model: {args.model}")
    model, processor = load_sam2_model(model_name=args.model, device=device)
    print("Model loaded.")

    # ポイントプロンプトの準備
    if args.point:
        x, y = map(int, args.point.split(","))
        input_points = [[[x, y]]]
        input_labels = [[1]]  # 1 = foreground
    else:
        # デフォルト: 画像中央
        w, h = image.size
        input_points = [[[w // 2, h // 2]]]
        input_labels = [[1]]
        print(f"Using center point: ({w // 2}, {h // 2})")

    # 推論
    print("Running inference...")
    inputs = processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # 後処理
    masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
    )

    masks_np = masks[0].squeeze().cpu().numpy()
    if masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis, ...]

    scores_np = outputs.iou_scores[0].cpu().numpy().flatten()

    print(f"Found {len(masks_np)} masks")
    for i, score in enumerate(scores_np):
        print(f"  Mask {i}: IoU score = {score:.3f}")

    # 結果保存
    output_dir = Path(args.output)
    output_path = output_dir / f"{image_path.stem}_segmented.png"

    image_np = np.array(image)
    # RGBからBGRに変換（visualization.pyがBGRを期待）
    import cv2
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    save_result(
        image_bgr,
        output_path,
        masks=masks_np,
        scores=scores_np,
    )
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
