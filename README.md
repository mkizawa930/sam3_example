# SAM3 Example

## インストール

```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

git clone https://github.com/facebookresearch/sam3.git
cd sam3
uv pip install -e .
```

## 画像セグメンテーション

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)

image = Image.open("image.jpg")
state = processor.set_image(image)

# テキストで指定
output = processor.set_text_prompt(state=state, prompt="person")
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```

## 動画トラッキング

```python
from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor()

# セッション開始
res = predictor.handle_request({
    "type": "start_session",
    "resource_path": "video.mp4"
})

# プロンプト追加
res = predictor.handle_request({
    "type": "add_prompt",
    "session_id": res["session_id"],
    "frame_index": 0,
    "text": "person"
})
```

## リンク

- https://github.com/facebookresearch/sam3
- https://ai.meta.com/sam3/
