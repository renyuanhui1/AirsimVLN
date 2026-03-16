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
    def __init__(self, api_key: str, model: str = "qwen2.5-vl-72b-instruct", output_dir: str = None):
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
            output_dir = os.getenv("TOKYO_OUTPUT_DIR", os.path.join(scene_dir, "output"))
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
    
    def decide_action_from_aerial_scene(
        self,
        image: np.ndarray,
        instruction: str,
        nav_state: Optional[Dict] = None,
        front_image: Optional[np.ndarray] = None,
    ) -> Dict:
        """基于俯视图执行道路跟随与红车搜索。"""
        self._save_image(image, "aerial_nav")
        bottom_image_base64 = self.image_to_base64(image)

        nav_state = nav_state or {}
        altitude = float(nav_state.get('altitude', 0.0))
        state_text = (
            f"- 当前高度: {altitude:.1f}m\n"
            f"- 搜索高度目标: {nav_state.get('search_altitude', altitude):.1f}m\n"
            f"- 上一步动作: {nav_state.get('last_action', 'unknown')}\n"
            f"- 距离上次看到红车的步数: {nav_state.get('steps_since_seen', 'unknown')}\n"
            f"- 连续看见红车的帧数: {nav_state.get('target_lock_frames', 0)}\n"
            f"- 连续丢失道路的步数: {nav_state.get('road_lost_streak', 0)}\n"
            f"- 最近一次旋转方向: {nav_state.get('last_rotation', 'none')}"
        )

        prompt = f"""

你是东京城市三维场景中的无人机俯视视觉导航控制器。

场景背景：
- 无人机飞行高度约 100 米（搜索阶段），使用底部俯视摄像头（垂直朝下）
- 场景为东京城市卫星级三维地图：密集街道网络、高楼楼顶、十字路口、斑马线、绿化带等
- 红色车辆停放或行驶在道路上，从 100 米高空俯视时车辆呈小红色矩形
- 任务：沿道路网络搜索红色车辆，找到后飞到车辆正上方附近悬停

任务指令：{instruction}

短期状态：
{state_text}

请只基于当前俯视图像，输出"下一步"最合适的动作，不要一次规划多步。

动作集合：
- move_forward
- move_left
- move_right
- move_up
- move_down
- rotate_left
- rotate_right
- hover
- arrived

参数规则：
- move_* 使用 parameters.distance（米）
- rotate_* 使用 parameters.angle（度）
- hover 使用 parameters.duration（秒）
- arrived 参数可为空对象

俯视导航核心原则：
【道路跟随】
- 优先在道路上方飞行，不要穿越建筑楼顶（楼顶通常表现为大块灰色/棕色矩形区域）
- 未见红车时，沿道路方向前进，单步建议 14 到 28 米
- 无论道路是否弯折，始终输出 move_forward，不要输出 rotate_left 或 rotate_right
- 转向由上层逻辑控制，你只需判断道路和目标状态

【road_direction 判定规则（必须严格执行）】
- 观察画面中道路的延伸方向，以画面中心为基准：
  - 道路在画面下半部分向左偏转超过 15 度 → road_direction = "left_curve"
  - 道路在画面下半部分向右偏转超过 15 度 → road_direction = "right_curve"
  - 道路基本笔直延伸到画面上方 → road_direction = "forward"
  - 出现明显十字路口或多叉路 → road_direction = "intersection"
  - 看不到道路 → road_direction = "lost"
- road_direction 只用于上报道路状态，不影响 action 输出

【十字路口识别规则（必须严格执行）】
- 只有当无人机已经飞到路口正上方或路口中心区域时，才报告 intersection：
  - 路口交叉区域占据画面中央大部分，能清晰看到多个方向道路从画面中心向四周延伸
  - 画面中心是开阔的路口地面，而不是某条单一道路
  - 路口斑马线、停止线或路口地面必须出现在画面正中央
- 路口还在画面前方远处（路口在画面上半部分或边缘）→ 继续报 forward，前进接近
- 路口刚出现在画面中但还未到中心 → 继续报 forward，继续前进直到路口占据画面中央
- 禁止在路口还未到达时就报告 intersection，这会导致无人机在路口前方就开始转弯
- 在十字路口时 action 输出 “move_forward” 并将 road_direction 设为 “intersection”，让上层逻辑决定转弯方向

【红车搜索与锁定】
- 从高空俯视时，红车呈较小的鲜红色矩形，通常出现在道路或停车场上
- 发现红车后，只需前进接近，禁止使用 move_left 或 move_right 平移对准
- distance_bucket 为 far 时，前进 12 到 20 米
- distance_bucket 为 mid 时，前进 6 到 12 米
- distance_bucket 为 near 且 centered=true 时，输出 arrived

【高度管理】
- 搜索阶段保持 90 到 110 米高度，视野广，便于找到红车


【动作输出强约束（必须遵守）】
- 禁止输出 rotate_left 或 rotate_right，转向由上层逻辑控制
- 禁止连续输出 hover；默认应输出 move_forward / move_left / move_right 之一
- 只有在以下情况才允许输出 hover：
    1) target_visible=true 且 centered=true，正在等待到达判定
    2) 当前帧严重模糊、无法判断道路和目标（road_visible=false 且 road_follow_confidence=low）
- 若 road_visible=true 且 target_visible=false，优先输出 move_forward（建议 12 到 24 米），不要输出 hover
- 若 road_visible=false 且 target_visible=false，优先输出 move_forward（建议 12 到 20 米），不要输出 hover 或 rotate

到达标准（必须严格执行）：
- 只有同时满足以下所有条件，才能返回 arrived：
  1) target_visible=true
  2) centered=true（红车在画面正中央，偏移不超过画面宽度的 15%）
  3) distance_bucket 为 near 或 mid
- 未同时满足以上三条，禁止输出 arrived，继续平移或前进对准
- arrived 表示已到达车辆正上方附近，不需要贴地降落
- 禁止在看到红车但未居中时返回 arrived

输出要求：
- 只返回一个 JSON 对象，不要附加解释
- reasoning 简短说明本步依据

必须额外输出以下字段：
- road_visible: true/false
- road_follow_confidence: "high|mid|low|unknown"
- road_direction: "forward|left_curve|right_curve|intersection|lost"
- target_visible: true/false
- target_offset: "left|right|center|upper|lower|unknown"
- distance_bucket: "near|mid|far|unknown"
- centered: true/false
- should_descend: true/false

distance_bucket 从高空俯视判定：
- near: 红车在画面中清晰可见，占比明显，位置居中或接近居中
- mid: 能看到红色矩形但体积仍小，还需接近或平移对准
- far: 红色矩形很小或需要仔细辨认，明显还远
- unknown: 看不到或无法判断

只返回 JSON，例如：
{{
    "action": "move_forward",
    "parameters": {{"distance": 20.0}},
    "road_visible": true,
    "road_follow_confidence": "high",
    "road_direction": "forward",
    "target_visible": false,
    "target_offset": "unknown",
    "distance_bucket": "unknown",
    "centered": false,
    "should_descend": false,
    "reasoning": "道路清晰可见，继续沿道路前进扩大搜索范围"
}}
"""

        content = [
            {"image": f"data:image/jpeg;base64,{bottom_image_base64}"},
            {"text": prompt},
        ]

        if front_image is not None:
            self._save_image(front_image, "front_context")
            front_image_base64 = self.image_to_base64(front_image)
            content.insert(1, {"image": f"data:image/jpeg;base64,{front_image_base64}"})

        return self._request_json_decision(
            content=content,
            fallback={
                "action": "move_forward",
                "parameters": {"distance": 16.0},
                "road_visible": True,
                "road_follow_confidence": "low",
                "road_direction": "forward",
                "target_visible": False,
                "target_offset": "unknown",
                "distance_bucket": "unknown",
                "centered": False,
                "should_descend": False,
                "reasoning": "俯视导航决策失败，使用兜底前进动作保持搜索推进",
            },
            log_prefix="俯视导航动作决策",
        )
