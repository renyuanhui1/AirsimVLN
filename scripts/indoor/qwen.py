# scripts/qwen.py
import base64
import os
import re
import requests
import json
import logging
from typing import Dict, List, Optional
import numpy as np
import cv2

class QwenVisionAgent:
    def __init__(self, api_key: str, model: str = "qwen3-vl-flash", output_dir: str = None):
        """
        初始化 Qwen 视觉智能体
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        self.logger = logging.getLogger(__name__)
        self._image_counter = 0

        # 设置图像保存目录
        if output_dir is None:
            scene_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.getenv("INDOOR_OUTPUT_DIR", os.path.join(scene_dir, "output"))
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _save_image(self, image: np.ndarray, label: str) -> str:
        """保存图像到 output 目录，返回保存路径"""
        import datetime
        self._image_counter += 1
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._image_counter:04d}_{label}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.logger.info(f"图像已保存: {filepath}")
        return filepath

    @staticmethod
    def _robust_parse_json(text: str) -> Optional[Dict]:
        """从模型返回的文本中尽力提取合法 JSON 对象"""
        # 1. 去掉 markdown 代码块包裹
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = text.replace('```', '')

        # 2. 尝试直接找 { ... }
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end <= start:
            return None

        raw = text[start:end + 1]

        # 3. 修复常见格式问题
        # 中文冒号 → 英文冒号
        raw = raw.replace('：', ':')
        # 中文逗号 → 英文逗号
        raw = raw.replace('，', ',')
        # true/false 大小写容错
        raw = re.sub(r'\bTrue\b', 'true', raw)
        raw = re.sub(r'\bFalse\b', 'false', raw)
        raw = re.sub(r'\bNone\b', 'null', raw)
        # 去掉 trailing comma（如 "key": value, }）
        raw = re.sub(r',\s*([}\]])', r'\1', raw)
        # 去掉行内注释 // ...
        raw = re.sub(r'//[^\n]*', '', raw)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def image_to_base64(self, image: np.ndarray) -> str:
        """将numpy图像转换为base64"""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode('utf-8')

    def _request_json_decision(self, content: List[Dict], fallback: Dict, log_prefix: str) -> Dict:
        """向多模态模型发起请求，并尽量解析出 JSON 决策。"""
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            },
            "parameters": {
                "top_p": 0.7,
                "temperature": 0.2,
            }
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(self.base_url, json=payload, headers=headers, timeout=120)
                response.raise_for_status()

                result = response.json()
                msg_content = result['output']['choices'][0]['message']['content']
                if isinstance(msg_content, list):
                    raw_text = next((item['text'] for item in msg_content if 'text' in item), '')
                else:
                    raw_text = msg_content

                decision = self._robust_parse_json(raw_text)
                if decision is not None and 'action' in decision:
                    merged = dict(fallback)
                    merged.update(decision)
                    self.logger.info(f"{log_prefix}(尝试{attempt}): {merged}")
                    return merged

                last_error = f"JSON解析失败或缺少action字段，原始内容: {raw_text[:300]}"
                self.logger.warning(f"第{attempt}次尝试解析失败: {last_error}")
            except requests.exceptions.RequestException as exc:
                last_error = str(exc)
                self.logger.warning(f"第{attempt}次API请求失败: {exc}")
            except Exception as exc:
                last_error = str(exc)
                self.logger.warning(f"第{attempt}次决策异常: {exc}")

            if attempt < max_retries:
                import time
                time.sleep(1)

        fallback_response = dict(fallback)
        fallback_response["reasoning"] = f"决策连续{max_retries}次失败，使用兜底动作: {last_error}"
        self.logger.error(f"{log_prefix} 连续{max_retries}次失败: {last_error}")
        return fallback_response
    
    def decide_action_from_scene(self, image: np.ndarray, instruction: str, nav_state: Optional[Dict] = None) -> Dict:
        """
        基于当前画面和自然语言指令，让大模型直接输出下一步导航动作。

        Returns:
            {
                "action": "move_left|move_right|move_forward|move_backward|move_up|move_down|hover|arrived",
                "parameters": {
                    "distance": float,   # 米
                    "duration": float    # 秒（仅hover可选）
                },
                "reasoning": str
            }
        """
        self._save_image(image, "scene_nav")
        image_base64 = self.image_to_base64(image)

        nav_state = nav_state or {}
        state_text = (
            f"- 上一步动作: {nav_state.get('last_action', 'unknown')}\n"
            f"- 连续判定到达次数: {nav_state.get('arrived_streak', 0)}\n"
            f"- 距离上次看到目标的步数: {nav_state.get('steps_since_seen', 'unknown')}\n"
            f"- 最近是否发生过冲(看到后又丢失): {nav_state.get('recent_overshoot', False)}\n"
            f"- 连续前进次数: {nav_state.get('consecutive_forward', 0)}\n"
            f"- 上次侧向搜索动作: {nav_state.get('last_lateral_search_action', 'none')}"
        )

        prompt = f"""

你是低空室外住宅场景中的无人机前视视觉导航控制器。

场景背景：
- 无人机飞行高度约 8 米，使用前视摄像头（水平视角）观察场景
- 场景为独立民居建筑外围区域，包含建筑墙面、院落、车道、门道、草坪等元素
- 任务目标：绕过房屋建筑，在院落/车道/门口等室外区域找到一辆停放的红色轿车，飞到其旁边悬停

用户指令：{instruction}

短期导航状态（用于记忆）：
{state_text}

请只基于当前图像，输出"下一步"最合适的动作，不要一次规划多步。

动作集合只能是：
- move_left
- move_right
- move_forward
- move_backward
- move_up
- move_down
- hover
- arrived   （已到达目标位置时使用）

参数规则：
- move_* 使用 parameters.distance（单位米）
- hover 使用 parameters.duration（单位秒）
- arrived 参数可为空对象

室外住宅场景距离原则（场景尺度较小，不要大步）：
- 前进搜索时，建议 6 到 15 米
- 接近目标时，建议 3 到 8 米
- 侧向绕行或大范围搜索时，建议 5 到 12 米
- 精细对准时，建议 2 到 5 米
- 高度微调时，建议 2 到 5 米
- 不要输出超过 20 米的单步移动，场景范围有限

绕行建筑策略（必须遵守）：
- 若前方画面中央被建筑墙面占据（灰色/白色砖墙、门窗立面等），禁止 move_forward
  → 判断建筑更偏向画面左侧还是右侧：向视野更开阔的一侧横移绕行
  → 若建筑占据左侧更多 → 向右绕；若占据右侧更多 → 向左绕
- 若能看到建筑边角/墙角，向开口方向横移 5 到 10 米，然后继续前进
- 每次绕行侧移量够过建筑边缘即可，不要过度偏移

到达标准（宽松判定）：
- 红色轿车在画面中清晰可见且占比明显（near），立即返回 arrived
- 不需要对准车头/车门/特定角度，看到车且已接近即可
- 车辆偏左/偏右但已很近时，直接返回 arrived，不要再横移对中

丢失目标恢复策略：
- steps_since_seen ≤ 3（刚丢失）：优先 move_backward 3 到 5 米，避免冲过目标
- 连续多步未见目标：向上次侧移的反方向横移 5 到 10 米重找
- 同方向侧移最多连续 2 步；若仍不可见则换反方向继续搜索

输出要求：
- 只返回一个 JSON 对象，不要附加解释文本
- reasoning 简短说明判断依据（如：前方是建筑角落，向右绕行）

额外字段要求：
- target_visible: true/false（画面中红车是否可见）
- distance_bucket: "near|mid|far|unknown"
- centered: true/false（红车是否在画面中心附近）
- obstacle_ahead: true/false（前方是否有建筑/墙体等明显障碍）
- bypass_direction: "left|right|either|none"（如需绕行，建议方向）

distance_bucket 判定：
- near: 车体占比明显，已经接近到车门旁边的距离
- mid: 能清楚看到整车但仍有距离，需要继续接近
- far: 车体很小或仅局部可见，明显还远
- unknown: 无法判断

只返回 JSON：
{{
    "action": "move_right",
    "parameters": {{"distance": 8.0}},
    "target_visible": false,
    "distance_bucket": "unknown",
    "centered": false,
    "obstacle_ahead": true,
    "bypass_direction": "right",
    "reasoning": "前方是建筑墙面，左侧占比更大，向右横移绕过建筑角落"
}}
"""

        return self._request_json_decision(
            content=[
                {"image": f"data:image/jpeg;base64,{image_base64}"},
                {"text": prompt},
            ],
            fallback={
            "action": "hover",
            "parameters": {"duration": 1.5},
            "target_visible": False,
            "distance_bucket": "unknown",
            "centered": False,
            "obstacle_ahead": False,
            "bypass_direction": "either",
            "reasoning": "场景决策失败，悬停等待",
            },
            log_prefix="场景动作决策",
        )
