"""可視化ユーティリティ"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def overlay_mask(
    image: np.ndarray | Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """画像にマスクをオーバーレイする

    Args:
        image: 元画像 (H, W, 3) BGR or PIL Image
        mask: バイナリマスク (H, W)
        color: マスクの色 (B, G, R)
        alpha: 透明度 (0-1)

    Returns:
        np.ndarray: マスクがオーバーレイされた画像
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    result = image.copy()
    mask_bool = mask.astype(bool)

    overlay = result.copy()
    overlay[mask_bool] = color

    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
    return result


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """バウンディングボックスを描画する

    Args:
        image: 画像 (H, W, 3) BGR
        boxes: バウンディングボックス (N, 4) [x1, y1, x2, y2]
        scores: 信頼度スコア (N,)
        color: ボックスの色 (B, G, R)
        thickness: 線の太さ

    Returns:
        np.ndarray: ボックスが描画された画像
    """
    result = image.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        if scores is not None:
            label = f"{scores[i]:.2f}"
            cv2.putText(
                result, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

    return result


def save_result(
    image: np.ndarray,
    output_path: str | Path,
    masks: np.ndarray | None = None,
    boxes: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """結果を保存する

    Args:
        image: 元画像 (H, W, 3) BGR
        output_path: 出力パス
        masks: マスク配列 (N, H, W)
        boxes: バウンディングボックス (N, 4)
        scores: 信頼度スコア (N,)
        figsize: Figure サイズ
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = image.copy()

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]

    if masks is not None:
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            result = overlay_mask(result, mask, color=color)

    if boxes is not None:
        result = draw_boxes(result, boxes, scores=scores)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=figsize)
    plt.imshow(result_rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
