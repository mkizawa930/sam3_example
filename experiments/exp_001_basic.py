"""基本的なSAM3セグメンテーション実験

使用方法:
    uv run python experiments/exp_001_basic.py --image data/sample.jpg --prompt "dog"

Note:
    SAM3はテキストプロンプトを使用してセグメンテーションします。
    CUDA環境が必要です。
"""

from pathlib import Path

import click
import numpy as np
from PIL import Image

from sam3_example import get_device, load_sam3_model, save_result


@click.command()
@click.option("--image", type=str, required=True, help="入力画像のパス")
@click.option(
    "--prompt", type=str, required=True, help="セグメンテーション対象のテキスト"
)
@click.option("--output", type=str, default="outputs/exp_001", help="出力ディレクトリ")
@click.option(
    "--confidence",
    type=float,
    default=0.5,
    help="信頼度閾値 (0.0-1.0)",
)
def main(image: str, prompt: str, output: str, confidence: float):

    # デバイス確認
    device = get_device()
    print(f"Using device: {device}")

    # 画像読み込み
    image_path = Path(image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image_path} ({img.size})")

    # 画像サイズが512より大きい場合、中央から512x512に切り取り
    crop_size = 256
    w, h = img.size
    if w > crop_size or h > crop_size:
        left = max(0, (w - crop_size) // 2)
        top = max(0, (h - crop_size) // 2)
        right = min(w, left + crop_size)
        bottom = min(h, top + crop_size)
        img = img.crop((left, top, right, bottom))
        print(f"Cropped to: {img.size}")

    # モデル読み込み
    print("Loading SAM3 model...")
    sam_model, processor = load_sam3_model(device=device)
    processor.confidence_threshold = confidence
    print("Model loaded.")

    # 推論
    print(f"Running inference with prompt: '{prompt}'")
    state = processor.set_image(img)
    state = processor.set_text_prompt(prompt=prompt, state=state)

    # 結果取得
    masks_tensor = state["masks"]
    scores_tensor = state["scores"]

    if masks_tensor is None or len(masks_tensor) == 0:
        print("No objects detected.")
        return

    masks_np = masks_tensor.cpu().numpy()
    scores_np = scores_tensor.cpu().numpy()

    # マスク形状を (N, H, W) に正規化
    if masks_np.ndim == 4:
        masks_np = masks_np.squeeze(1)  # (N, 1, H, W) -> (N, H, W)

    print(f"Found {len(masks_np)} objects, mask shape: {masks_np.shape}")
    for i, score in enumerate(scores_np):
        print(f"  Object {i}: confidence = {score:.3f}")

    # 結果保存
    output_dir = Path(output)
    output_path = output_dir / f"{image_path.stem}_segmented.png"

    image_np = np.array(img)
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
