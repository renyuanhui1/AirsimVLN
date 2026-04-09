import base64
import json
import re
from typing import Optional

import cv2
import numpy as np
import requests


DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_LOCAL_MODEL = "qwen3.5:9b"

AREA_SEARCH_PROMPT = """你是一位无人机低空侦察分析员。当前输入是一张无人机俯视图像，无人机高度约为150米。

任务：
1. 判断画面中是否存在“具有黑色弹孔的地面轰炸区域”。
2. 如果存在，判断该地面轰炸区域在画面中的相对位置，并给出无人机下一步动作建议。
3. 只输出一个 JSON 对象，不要输出额外说明。

轰炸区域的重点静态迹象包括但不限于：
- 地面出现明显黑色或深色弹孔、坑洞、圆形或不规则坑斑
- 弹孔周边存在黑褐色烧蚀痕迹
- 多个弹孔集中分布，形成成片异常区域
- 与周边完整地面相比明显异常的受击区域

JSON 字段要求：
- action: move_forward / move_left / move_right / move_backward / move_down / rotate_left / rotate_right / hover / arrived
- parameters: 动作参数对象。移动使用 distance（米），旋转使用 angle（度），悬停使用 duration（秒）
- destroyed_area_visible: true/false
- target_offset: left / right / upper / lower / upper_left / upper_right / lower_left / lower_right / center / unknown
- centered: true/false
- confidence: high / mid / low
- should_descend: true/false
- reasoning: 一句简短依据

判定规则：
- 这里的“轰炸区域”优先指地面具有黑色弹孔的区域，而不是普通阴影、深色建筑顶面或正常路面污渍。
- 如果画面没有明显黑色弹孔区域，优先建议 move_forward 或小角度 rotate。
- 如果轰炸区域已发现但不在中心，应建议平移让其居中。
- 如果轰炸区域已比较清晰且接近中心，可输出 should_descend=true。
- centered 只能在目标主体非常接近画面几何中心时输出 true；只要目标明显偏上、偏下、偏左或偏右，就必须输出 false。
- 如果 target_offset 不是 center，则 centered 必须为 false。
- action 应尽量与 target_offset 对齐；如果目标主要偏上/偏下/偏左/偏右，优先先做平移或前后移动，不要优先下降。
"""

PLANE_SEARCH_PROMPT = """你是一位无人机目标锁定分析员。当前输入是一张无人机俯视图像，无人机正在炸毁区域上空下降搜索飞机目标。

任务：
1. 判断画面中是否存在飞机目标，重点寻找炸毁区域中的飞机或飞机残骸。
2. 如果存在，判断飞机在画面中的位置，并给出无人机下一步动作建议。
3. 只输出一个 JSON 对象，不要输出额外说明。

识别重点：
- 飞机机身、机翼、尾翼、发动机区域的平面轮廓
- 完整飞机、受损飞机、残骸化飞机均可算作飞机目标
- 优先锁定画面中最清晰、最适合继续评估的一架飞机

JSON 字段要求：
- action: move_forward / move_left / move_right / move_backward / move_down / rotate_left / rotate_right / hover / arrived
- parameters: 动作参数对象。移动使用 distance（米），旋转使用 angle（度），悬停使用 duration（秒）
- plane_visible: true/false
- target_offset: left / right / upper / lower / upper_left / upper_right / lower_left / lower_right / center / unknown
- centered: true/false
- confidence: high / mid / low
- should_descend: true/false
- reasoning: 一句简短依据

判定规则：
- 如果没看到飞机，优先小步平移搜索或轻微旋转。
- 如果看到飞机但不居中，优先输出平移动作。
- 如果飞机已清晰且接近中心，可输出 should_descend=true。
- 如果飞机已清晰且中心合适，可输出 arrived。
- centered 只能在飞机主体非常接近画面几何中心时输出 true；只要飞机明显偏上、偏下、偏左或偏右，就必须输出 false。
- 如果 target_offset 不是 center，则 centered 必须为 false。
- 当飞机不在中心时，action 应优先与 target_offset 对齐，用于先把飞机移回中心；不要在明显未居中时优先输出 move_down。
- 只有当飞机已经基本居中，才适合输出 should_descend=true 或 move_down / arrived。
"""

PLANE_EVAL_JSON_PROMPT = """你是一位资深战场毁伤评估专家。请基于输入的无人机俯视图像，只分析画面中最主要的一架飞机。

只输出一个 JSON 对象，不要输出额外说明，格式如下：
{
  "target_type": "飞机",
  "damage_level": "A/B/C",
  "visual_features": "客观描述主要可见损伤特征",
  "damage_estimate": "受损面积百分比单值或区间",
  "functional_assessment": "根据受损部位判断剩余作战或使用能力",
  "uncertainty": "若信息不足，说明不确定性来源；若较确定，可写 无"
}

判定要求：
1. 若飞机已碎裂或仅剩残骸，仍按飞机处理。
2. 优先根据机翼、尾翼、发动机、机身中段等核心部位判断。
3. 避免夸大，没有把握的细节不要臆测。
"""

DEFAULT_PROMPT = PLANE_EVAL_JSON_PROMPT


class LocalVlmAgent:
    def __init__(
        self,
        model: str = DEFAULT_LOCAL_MODEL,
        base_url: str = DEFAULT_LOCAL_BASE_URL,
        timeout: int = 120,
        think: bool = False,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.think = think

    @staticmethod
    def _robust_parse_json(text: str) -> Optional[dict]:
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "")
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None

        raw = text[start : end + 1]
        raw = raw.replace("：", ":").replace("，", ",")
        raw = re.sub(r"\bTrue\b", "true", raw)
        raw = re.sub(r"\bFalse\b", "false", raw)
        raw = re.sub(r"\bNone\b", "null", raw)
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        raw = re.sub(r"//[^\n]*", "", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def image_to_base64(image: np.ndarray) -> str:
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("图像编码失败")
        return base64.b64encode(buffer).decode("utf-8")

    def chat_with_image(self, image: np.ndarray, prompt: str, max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [self.image_to_base64(image)],
                }
            ],
            "stream": False,
            "think": self.think,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.2,
                "top_p": 0.7,
            },
        }
        response = requests.post(
            self.base_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        message = result.get("message", {})
        return str(message.get("content", ""))

    def decide_destroyed_area(self, image: np.ndarray, altitude: float, last_action: str) -> dict:
        prompt = (
            f"{AREA_SEARCH_PROMPT}\n\n"
            f"当前高度：{altitude:.1f} 米\n"
            f"上一步动作：{last_action}\n"
            "请返回唯一 JSON。"
        )
        raw_text = self.chat_with_image(image, prompt, max_tokens=512)
        parsed = self._robust_parse_json(raw_text)
        if parsed is None:
            return {
                "action": "hover",
                "parameters": {"duration": 1.0},
                "destroyed_area_visible": False,
                "target_offset": "unknown",
                "centered": False,
                "confidence": "low",
                "should_descend": False,
                "reasoning": f"炸毁区域 JSON 解析失败: {raw_text[:160]}",
            }

        return {
            "action": str(parsed.get("action", "hover")).strip().lower(),
            "parameters": parsed.get("parameters", {}) or {},
            "destroyed_area_visible": bool(parsed.get("destroyed_area_visible", False)),
            "target_offset": str(parsed.get("target_offset", "unknown")).strip().lower(),
            "centered": bool(parsed.get("centered", False)),
            "confidence": str(parsed.get("confidence", "low")).strip().lower(),
            "should_descend": bool(parsed.get("should_descend", False)),
            "reasoning": str(parsed.get("reasoning", "")).strip(),
        }

    def decide_plane(self, image: np.ndarray, altitude: float, last_action: str) -> dict:
        prompt = (
            f"{PLANE_SEARCH_PROMPT}\n\n"
            f"当前高度：{altitude:.1f} 米\n"
            f"上一步动作：{last_action}\n"
            "补充约束：如果当前已经处于较低评估高度附近，则不要继续下降，优先输出平移、前后移动或悬停来完成居中。\n"
            "请返回唯一 JSON。"
        )
        raw_text = self.chat_with_image(image, prompt, max_tokens=512)
        parsed = self._robust_parse_json(raw_text)
        if parsed is None:
            return {
                "action": "hover",
                "parameters": {"duration": 1.0},
                "plane_visible": False,
                "target_offset": "unknown",
                "centered": False,
                "confidence": "low",
                "should_descend": False,
                "reasoning": f"飞机搜索 JSON 解析失败: {raw_text[:160]}",
            }

        return {
            "action": str(parsed.get("action", "hover")).strip().lower(),
            "parameters": parsed.get("parameters", {}) or {},
            "plane_visible": bool(parsed.get("plane_visible", False)),
            "target_offset": str(parsed.get("target_offset", "unknown")).strip().lower(),
            "centered": bool(parsed.get("centered", False)),
            "confidence": str(parsed.get("confidence", "low")).strip().lower(),
            "should_descend": bool(parsed.get("should_descend", False)),
            "reasoning": str(parsed.get("reasoning", "")).strip(),
        }

    def evaluate_plane_damage(self, image: np.ndarray) -> str:
        return self.chat_with_image(image, PLANE_EVAL_JSON_PROMPT, max_tokens=768)
