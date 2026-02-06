"""SAMモデルのロードとデバイス管理

SAM3はCUDA/Linux環境が必要です（tritonライブラリの制約）。
macOSではSAM2またはtransformers経由のモデルを使用してください。
"""

from __future__ import annotations

import torch

from transformers import Sam2Processor, Sam2Model


# SAM3は環境に応じて動的にインポート
_SAM3_AVAILABLE = False
try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    _SAM3_AVAILABLE = True
except ImportError:
    pass


def get_device() -> torch.device:
    """利用可能な最適なデバイスを取得する

    Returns:
        torch.device: CUDA > MPS > CPU の優先順位で選択
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_sam3_available() -> bool:
    """SAM3が利用可能かどうかを返す"""
    return _SAM3_AVAILABLE and torch.cuda.is_available()


def load_sam3_model(
    model_name: str = "facebook/sam3-hiera-large",
    device: torch.device | None = None,
):
    """SAM3モデルとプロセッサをロードする

    Args:
        model_name: モデル名（現在は未使用、将来の拡張用）
        device: 使用するデバイス（Noneの場合は自動選択）

    Returns:
        tuple: (model, processor)

    Raises:
        RuntimeError: SAM3が利用できない場合

    Note:
        SAM3はCUDA環境が必要です。macOSではload_sam2_modelを使用してください。
        初回実行時はHugging Faceへのログインが必要です: `huggingface-cli login`
    """
    if not _SAM3_AVAILABLE:
        raise RuntimeError(
            "SAM3 is not available. Install with: "
            "uv add 'sam3 @ git+https://github.com/facebookresearch/sam3.git'\n"
            "Note: SAM3 requires CUDA/Linux (triton library)."
        )

    if device is None:
        device = get_device()

    if device.type != "cuda":
        raise RuntimeError(
            f"SAM3 requires CUDA but got device: {device}. "
            "Use load_sam2_model() for non-CUDA environments."
        )

    model = build_sam3_image_model()
    model = model.to(device)
    model.eval()

    processor = Sam3Processor(model)

    return model, processor


def load_sam2_model(
    model_name: str = "facebook/sam2-hiera-large",
    device: torch.device | None = None,
) -> tuple[Sam2Model, Sam2Processor]:
    """SAM2モデルとプロセッサをロードする（macOS/MPS対応）

    Args:
        model_name: Hugging Face上のモデル名
            - "facebook/sam2-hiera-tiny"
            - "facebook/sam2-hiera-small"
            - "facebook/sam2-hiera-base-plus"
            - "facebook/sam2-hiera-large"
        device: 使用するデバイス（Noneの場合は自動選択）

    Returns:
        tuple[Sam2Model, Sam2Processor]: モデルとプロセッサのタプル
    """
    if device is None:
        device = get_device()

    processor = Sam2Processor.from_pretrained(model_name)
    model = Sam2Model.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    return model, processor


def load_model(
    prefer_sam3: bool = True,
    device: torch.device | None = None,
):
    """環境に応じて最適なモデルを自動選択してロードする

    Args:
        prefer_sam3: SAM3を優先するかどうか（CUDAが利用可能な場合）
        device: 使用するデバイス

    Returns:
        tuple: (model, processor, model_type)
            model_type は "sam3" または "sam2"
    """
    if device is None:
        device = get_device()

    if prefer_sam3 and is_sam3_available():
        model, processor = load_sam3_model(device=device)
        return model, processor, "sam3"
    else:
        model, processor = load_sam2_model(device=device)
        return model, processor, "sam2"
