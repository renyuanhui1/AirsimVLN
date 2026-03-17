"""
测试不同模型对红车的检测能力
用法: python test_model_detection.py [图像路径]
"""
import base64
import json
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

QWEN_API_KEY = os.getenv('QWEN_API_KEY', 'sk-b988e0fff98740ef90a8915d3b77dc11')
BASE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

MODELS = [
    "qwen-vl-plus-2025-05-07",
    "qwen-vl-plus-latest",
    "qwen3-vl-flash-2025-10-15",
    "qwen-vl-max-2025-08-13",
]

PROMPT = """请仔细观察这张无人机俯视图像，回答以下问题：

1. 图像中是否有红色车辆？（是/否）
2. 如果有，描述红车的位置（画面哪个区域：左上/左中/左下/中上/中央/中下/右上/右中/右下）
3. 红车大概占画面面积的百分比？
4. 你对检测结果的置信度？（高/中/低）

只返回 JSON：
{
  "target_visible": true/false,
  "position": "位置描述或null",
  "area_percent": 数字或null,
  "confidence": "高/中/低",
  "reasoning": "简短描述你看到了什么"
}"""


def image_to_base64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_model(model: str, image_base64: str) -> dict:
    payload = {
        "model": model,
        "input": {
            "messages": [{
                "role": "user",
                "content": [
                    {"image": f"data:image/jpeg;base64,{image_base64}"},
                    {"text": PROMPT},
                ]
            }]
        },
        "parameters": {"temperature": 0.1}
    }
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(BASE_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        content = result['output']['choices'][0]['message']['content']
        if isinstance(content, list):
            text = next((c['text'] for c in content if 'text' in c), '')
        else:
            text = content

        # 尝试解析 JSON
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
        return {"raw": text}
    except Exception as e:
        return {"error": str(e)}


if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else input("请输入图像路径: ").strip()

    print(f"图像: {image_path}\n{'='*60}")
    image_b64 = image_to_base64(image_path)

    for model in MODELS:
        print(f"\n模型: {model}")
        result = test_model(model, image_b64)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print('-' * 40)
