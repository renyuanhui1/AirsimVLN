import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from dotenv import load_dotenv

from airsim_controller import AirSimController
from qwen_agent import DEFAULT_LOCAL_BASE_URL, DEFAULT_LOCAL_MODEL, LocalVlmAgent


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GROUNDING_ROOT = PROJECT_ROOT / "Grounding-Dino-Sam"
SAM3_ROOT = GROUNDING_ROOT / "models" / "sam3"
GROUNDING_DINO_ROOT = GROUNDING_ROOT / "models" / "grounding_dino"


load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


SEARCH_ALTITUDE = 150.0
PLANE_LOCK_ALTITUDE = 120.0
PLANE_EVAL_ALTITUDE = 70.0
ALTITUDE_TOLERANCE = 2.0
CAPTURE_STABILIZE_SECONDS = 1.2
MAX_LOCK_CENTER_STEPS = 6
MAX_EVAL_CENTER_STEPS = 4

VALID_ACTIONS = {
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "move_down",
    "rotate_left",
    "rotate_right",
    "hover",
    "arrived",
}
VALID_OFFSETS = {
    "left",
    "right",
    "upper",
    "lower",
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
    "center",
    "unknown",
}
VALID_CONFIDENCE = {"high", "mid", "low"}


class GroundingSamRunner:
    """封装 Grounding DINO + SAM3 的初始化、提示词生成与推理流程。"""

    def __init__(self, device: str, output_dir: str, grounding_root: Path):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.grounding_root = grounding_root
        self.grounding_model = None
        self.sam3_model = None
        self.sam3_processor = None
        self._build_sam3_image_model = None
        self._sam3_processor_cls = None
        self._load_image = None
        self._load_grounding_model = None
        self._predict = None

    def _ensure_imports(self) -> None:
        sam3_root = self.grounding_root / "models" / "sam3"
        grounding_dino_root = self.grounding_root / "models" / "grounding_dino"
        for path in (sam3_root, grounding_dino_root.parent, self.grounding_root):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from grounding_dino.groundingdino.util.inference import (
            load_image,
            load_model as load_grounding_model,
            predict,
        )

        self._build_sam3_image_model = build_sam3_image_model
        self._sam3_processor_cls = Sam3Processor
        self._load_image = load_image
        self._load_grounding_model = load_grounding_model
        self._predict = predict

    def init_models(self) -> None:
        """按需懒加载模型，避免任务刚开始时占满初始化开销。"""
        if self.grounding_model is not None and self.sam3_processor is not None:
            logger.debug("Grounding-SAM 模型已完成初始化，跳过重复加载")
            return

        sam3_checkpoint = self.grounding_root / "checkpoints" / "sam3.pt"
        gdino_checkpoint = self.grounding_root / "checkpoints" / "groundingdino_swint_ogc.pth"
        gdino_config = (
            self.grounding_root
            / "models"
            / "grounding_dino"
            / "groundingdino"
            / "config"
            / "GroundingDINO_SwinT_OGC.py"
        )

        logger.info("检查 Grounding-SAM 模型文件")
        for path in (sam3_checkpoint, gdino_checkpoint, gdino_config):
            if not path.exists():
                raise FileNotFoundError(f"缺失模型文件: {path}")
            logger.debug("模型文件已就绪: %s", path)

        self._ensure_imports()

        logger.info("初始化 SAM3, device=%s", self.device)
        self.sam3_model = self._build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            checkpoint_path=str(sam3_checkpoint),
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        self.sam3_processor = self._sam3_processor_cls(self.sam3_model, device=self.device)

        logger.info("初始化 Grounding DINO, device=%s", self.device)
        self.grounding_model = self._load_grounding_model(
            model_config_path=str(gdino_config),
            model_checkpoint_path=str(gdino_checkpoint),
            device=self.device,
        )
        logger.info("Grounding-SAM 模型初始化完成")

    @staticmethod
    def _generate_prompt(parsed_report: Optional[Dict[str, Any]]) -> str:
        if not isinstance(parsed_report, dict):
            logger.warning("毁伤评估结果不是字典，跳过 Grounding-SAM")
            return ""

        target_type = str(parsed_report.get("target_type", "")).strip().lower()
        category_map = {
            "飞机": "airplane .",
            "airplane": "airplane .",
            "plane": "airplane .",
            "aircraft": "airplane .",
            "坦克": "tank .",
            "tank": "tank .",
            "建筑": "building .",
            "building": "building .",
            "house": "building .",
        }
        prompt = category_map.get(target_type, "")
        if not prompt:
            logger.warning("目标类别 %r 未映射到 Grounding 提示词", target_type)
        return prompt

    def run(
        self,
        image_path: str,
        parsed_report: Optional[Dict[str, Any]],
        box_threshold: float,
        text_threshold: float,
    ) -> Dict[str, Any]:
        logger.info("开始执行 Grounding-SAM, image=%s", image_path)
        self.init_models()

        prompt = self._generate_prompt(parsed_report)
        if not prompt:
            return {
                "detected": False,
                "prompt": "",
                "num_detections": 0,
                "labels": [],
                "confidences": [],
                "boxes": [],
                "boxes_image": None,
                "masks_image": None,
                "reason": "未识别到可用目标类别，跳过 Grounding-SAM",
            }

        image_source, image = self._load_image(image_path)
        h, w, _ = image_source.shape
        logger.info(
            "运行 Grounding DINO, prompt=%s box_threshold=%.2f text_threshold=%.2f image=%dx%d",
            prompt,
            box_threshold,
            text_threshold,
            w,
            h,
        )
        boxes, confidences, labels = self._predict(
            model=self.grounding_model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        if isinstance(confidences, torch.Tensor):
            confidences = confidences.detach().cpu().numpy()

        boxes = np.asarray(boxes)
        confidences = np.asarray(confidences)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        if confidences.ndim == 0:
            confidences = confidences.reshape(1)
        else:
            confidences = confidences.reshape(-1)
        if isinstance(labels, str):
            labels = [labels]
        else:
            labels = [str(label) for label in labels]

        if len(boxes) == 0:
            logger.info("Grounding DINO 未检测到目标")
            return {
                "detected": False,
                "prompt": prompt,
                "num_detections": 0,
                "labels": [],
                "confidences": [],
                "boxes": [],
                "boxes_image": None,
                "masks_image": None,
            }

        input_boxes = boxes.copy()
        input_boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
        input_boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
        input_boxes[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
        input_boxes[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
        input_boxes[:, [0, 2]] = np.clip(input_boxes[:, [0, 2]], 0, w)
        input_boxes[:, [1, 3]] = np.clip(input_boxes[:, [1, 3]], 0, h)

        from PIL import Image
        import supervision as sv

        image_pil = Image.open(image_path).convert("RGB")
        state = self.sam3_processor.set_image(image_pil)
        state = self.sam3_processor.set_text_prompt(prompt, state)
        # Sam3Processor expects normalized cxcywh boxes here.
        for box in boxes:
            state = self.sam3_processor.add_geometric_prompt(box=box, label=True, state=state)

        masks = state.get("masks")
        if masks is None:
            logger.warning("SAM3 返回空掩码，回退为全零掩码")
            masks = np.zeros((len(boxes), h, w), dtype=bool)
        else:
            if isinstance(masks, torch.Tensor):
                masks = masks.detach().cpu().numpy()
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            if masks.ndim == 2:
                masks = masks[np.newaxis, ...]

        count_candidates = [len(boxes), len(confidences), len(labels), len(masks)]
        detection_count = min(count_candidates)
        if detection_count <= 0:
            logger.warning(
                "Grounding-SAM 输出数量异常, boxes=%d confidences=%d labels=%d masks=%d",
                len(boxes),
                len(confidences),
                len(labels),
                len(masks),
            )
            return {
                "detected": False,
                "prompt": prompt,
                "num_detections": 0,
                "labels": [],
                "confidences": [],
                "boxes": [],
                "boxes_image": None,
                "masks_image": None,
                "reason": "Grounding-SAM 输出数量异常",
            }

        if len(set(count_candidates)) != 1:
            logger.warning(
                "Grounding-SAM 输出数量不一致, boxes=%d confidences=%d labels=%d masks=%d, 将统一裁剪为 %d",
                len(boxes),
                len(confidences),
                len(labels),
                len(masks),
                detection_count,
            )

        input_boxes = input_boxes[:detection_count]
        confidences = confidences[:detection_count]
        labels = labels[:detection_count]
        masks = masks[:detection_count]

        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=np.arange(len(labels), dtype=np.int32),
        )
        labels_display = [f"{label} {float(conf):.2f}" for label, conf in zip(labels, confidences)]

        image_bgr = cv2.imread(image_path)
        boxed = sv.BoxAnnotator().annotate(scene=image_bgr.copy(), detections=detections)
        boxed = sv.LabelAnnotator().annotate(scene=boxed, detections=detections, labels=labels_display)
        masked = sv.MaskAnnotator().annotate(scene=boxed.copy(), detections=detections)

        boxes_output = self.output_dir / "grounding_dino_boxes.jpg"
        masks_output = self.output_dir / "grounding_dino_sam3_masks.jpg"
        cv2.imwrite(str(boxes_output), boxed)
        cv2.imwrite(str(masks_output), masked)

        logger.info(
            "Grounding-SAM 完成: detections=%d labels=%s",
            len(labels),
            list(labels),
        )
        return {
            "detected": True,
            "prompt": prompt,
            "num_detections": int(len(labels)),
            "labels": [str(x) for x in labels],
            "confidences": [float(x) for x in confidences],
            "boxes": input_boxes.tolist(),
            "boxes_image": str(boxes_output),
            "masks_image": str(masks_output),
        }


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _phase_banner(title: str) -> None:
    logger.info("=" * 24 + " %s " + "=" * 24, title)


def _save_image(image: np.ndarray, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, filename)
    if not cv2.imwrite(image_path, image):
        raise RuntimeError(f"保存图像失败: {image_path}")
    logger.info("已保存图像: %s", image_path)
    return image_path


def _log_table(title: str, rows: list[tuple[str, str]]) -> None:
    key_width = max(len("字段"), max((len(key) for key, _ in rows), default=2))
    value_width = max(len("内容"), max((len(value) for _, value in rows), default=2))
    border = f"+-{'-' * key_width}-+-{'-' * value_width}-+"

    logger.info(title)
    logger.info(border)
    logger.info("| %s | %s |", "字段".ljust(key_width), "内容".ljust(value_width))
    logger.info(border)
    for key, value in rows:
        logger.info("| %s | %s |", key.ljust(key_width), value.ljust(value_width))
    logger.info(border)


def _write_text_report(path: str, title: str, rows: list[tuple[str, str]], raw_report: str) -> None:
    lines = [title, "=" * len(title), ""]
    for key, value in rows:
        lines.append(f"{key}: {value}")
    lines.extend(["", "原始输出", "----", raw_report.strip() or "(空)", ""])
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    logger.info("已保存评估报告: %s", path)


def _is_at_or_below(altitude: float, target: float) -> bool:
    return altitude <= target + ALTITUDE_TOLERANCE


def _normalize_decision(decision: Optional[dict], visible_key: str) -> dict:
    raw = decision or {}
    action = str(raw.get("action", "hover")).strip().lower()
    if action not in VALID_ACTIONS:
        action = "hover"
    if action == "arrived":
        action = "hover"

    offset = str(raw.get("target_offset", "unknown")).strip().lower()
    if offset not in VALID_OFFSETS:
        offset = "unknown"

    confidence = str(raw.get("confidence", "low")).strip().lower()
    if confidence not in VALID_CONFIDENCE:
        confidence = "low"

    params = raw.get("parameters", {})
    if not isinstance(params, dict):
        params = {}

    visible = bool(raw.get(visible_key, False))
    centered = bool(raw.get("centered", False))
    should_descend = bool(raw.get("should_descend", False))

    if not visible:
        centered = False
        should_descend = False
        if offset == "center":
            offset = "unknown"

    if offset != "center":
        centered = False
    elif visible:
        centered = True

    return {
        "action": action,
        "parameters": params,
        visible_key: visible,
        "target_offset": offset,
        "centered": centered,
        "confidence": confidence,
        "should_descend": should_descend,
        "reasoning": str(raw.get("reasoning", "")).strip(),
    }


def _get_flight_snapshot(ctrl: AirSimController) -> dict:
    x, y, z = ctrl.get_position()
    altitude = ctrl.get_altitude()
    yaw_deg = np.degrees(ctrl.get_yaw())
    return {
        "x": round(x, 1),
        "y": round(y, 1),
        "z": round(z, 1),
        "altitude": round(altitude, 1),
        "yaw_deg": round(float(yaw_deg), 1),
    }


def _log_decision(stage: str, step: int, decision: dict, visible_key: str, snapshot: dict) -> None:
    if stage == "area":
        logger.info(
            "[area %02d] action=%s visible=%s offset=%s centered=%s descend=%s conf=%s reason=%s",
            step,
            decision["action"],
            decision[visible_key],
            decision["target_offset"],
            decision["centered"],
            decision["should_descend"],
            decision["confidence"],
            decision["reasoning"] or "-",
        )
    elif stage == "plane":
        logger.info(
            "[plane %02d] alt=%.1f action=%s visible=%s offset=%s centered=%s descend=%s conf=%s reason=%s",
            step,
            snapshot["altitude"],
            decision["action"],
            decision[visible_key],
            decision["target_offset"],
            decision["centered"],
            decision["should_descend"],
            decision["confidence"],
            decision["reasoning"] or "-",
        )
    else:
        logger.info(
            "[%s %02d] alt=%.1f action=%s visible=%s offset=%s centered=%s descend=%s conf=%s reason=%s",
            stage,
            step,
            snapshot["altitude"],
            decision["action"],
            decision[visible_key],
            decision["target_offset"],
            decision["centered"],
            decision["should_descend"],
            decision["confidence"],
            decision["reasoning"] or "-",
        )

    logger.debug(
        "[%s %02d] pos=(%.1f, %.1f, %.1f) yaw=%.1f",
        stage,
        step,
        snapshot["x"],
        snapshot["y"],
        snapshot["z"],
        snapshot["yaw_deg"],
    )


def _sleep_between_steps(duration: float = 0.8) -> None:
    time.sleep(duration)


def _stabilize_before_capture(ctrl: AirSimController, duration: float = CAPTURE_STABILIZE_SECONDS) -> None:
    ctrl.hover(duration=1.0)


def _descend_to(ctrl: AirSimController, target_altitude: float, current_altitude: float) -> bool:
    if _is_at_or_below(current_altitude, target_altitude):
        logger.info("当前高度 %.1f 米已满足目标高度 %.1f 米", current_altitude, target_altitude)
        return False

    distance = max(0.0, current_altitude - target_altitude)
    logger.info("下降到目标高度 %.1f 米, 当前高度 %.1f 米, 预计下降 %.1f 米", target_altitude, current_altitude, distance)
    ctrl.move_down(distance=distance, speed=2.0)
    return True


def _apply_action(ctrl: AirSimController, decision: dict, altitude: float, min_altitude: float) -> str:
    action = str(decision.get("action", "hover")).strip().lower()
    params = decision.get("parameters", {}) or {}

    if action == "move_forward":
        ctrl.move_forward(distance=_clamp(_safe_float(params.get("distance"), 35.0), 25.0, 80.0), speed=3.2)
    elif action == "move_backward":
        ctrl.move_backward(distance=_clamp(_safe_float(params.get("distance"), 25.0), 25.0, 60.0), speed=2.8)
    elif action == "move_left":
        ctrl.move_left(distance=_clamp(_safe_float(params.get("distance"), 25.0), 25.0, 60.0), speed=2.8)
    elif action == "move_right":
        ctrl.move_right(distance=_clamp(_safe_float(params.get("distance"), 25.0), 25.0, 60.0), speed=2.8)
    elif action == "move_down":
        max_drop = max(0.0, altitude - min_altitude)
        distance = min(_clamp(_safe_float(params.get("distance"), 15.0), 8.0, 40.0), max_drop)
        if distance > 0:
            ctrl.move_down(distance=distance, speed=2.0)
        else:
            ctrl.hover(duration=1.0)
            action = "hover"
    elif action == "rotate_left":
        ctrl.rotate_yaw(angle=_clamp(_safe_float(params.get("angle"), 15.0), 8.0, 25.0), speed=20.0)
    elif action == "rotate_right":
        ctrl.rotate_yaw(angle=-_clamp(_safe_float(params.get("angle"), 15.0), 8.0, 25.0), speed=20.0)
    else:
        ctrl.hover(duration=1.0)
        action = "hover"

    return action


def _center_target(ctrl: AirSimController, offset: str) -> str:
    if offset == "upper_left":
        ctrl.move_left(distance=30.0, speed=2.8)
        ctrl.move_forward(distance=30.0, speed=3.0)
        return "move_upper_left"
    if offset == "upper_right":
        ctrl.move_right(distance=30.0, speed=2.8)
        ctrl.move_forward(distance=30.0, speed=3.0)
        return "move_upper_right"
    if offset == "lower_left":
        ctrl.move_left(distance=30.0, speed=2.8)
        ctrl.move_backward(distance=30.0, speed=2.8)
        return "move_lower_left"
    if offset == "lower_right":
        ctrl.move_right(distance=30.0, speed=2.8)
        ctrl.move_backward(distance=30.0, speed=2.8)
        return "move_lower_right"
    if offset == "left":
        ctrl.move_left(distance=30.0, speed=2.8)
        return "move_left"
    if offset == "right":
        ctrl.move_right(distance=30.0, speed=2.8)
        return "move_right"
    if offset == "upper":
        ctrl.move_forward(distance=30.0, speed=3.0)
        return "move_forward"
    if offset == "lower":
        ctrl.move_backward(distance=30.0, speed=2.8)
        return "move_backward"

    ctrl.hover(duration=1.0)
    return "hover"


def _center_plane_before_next_stage(
    ctrl: AirSimController,
    agent: LocalVlmAgent,
    input_dir: Path,
    altitude_target: float,
    max_steps: int,
    image_prefix: str,
    stage_name: str,
    last_action: str,
) -> tuple[Optional[np.ndarray], str, bool]:
    plane_image = None

    for step in range(1, max_steps + 1):
        _stabilize_before_capture(ctrl)
        plane_image = ctrl.get_bottom_camera_image()
        if plane_image is None:
            logger.warning("[%s %02d] Failed to get bottom-view image; retrying after hover", stage_name, step)
            ctrl.hover(duration=1.0)
            continue

        _save_image(plane_image, str(input_dir), f"{image_prefix}_{step:02d}_input.jpg")
        snapshot = _get_flight_snapshot(ctrl)
        decision = _normalize_decision(
            agent.decide_plane(image=plane_image, altitude=snapshot["altitude"], last_action=last_action),
            "plane_visible",
        )
        _log_decision(stage_name, step, decision, "plane_visible", snapshot)

        if decision["plane_visible"] and decision["centered"]:
            logger.info("Target centered at altitude %.1f", snapshot["altitude"])
            return plane_image, "hover", True

        last_action = _apply_action(ctrl, decision, snapshot["altitude"], min_altitude=altitude_target)
        _sleep_between_steps()

    return plane_image, last_action, False


def _safe_land(ctrl: AirSimController) -> None:
    try:
        ctrl.land()
    except Exception as exc:
        logger.exception("降落失败: %s", exc)


def run_task_with_grounding_sam(
    max_area_steps: int,
    max_plane_steps: int,
    model: str,
    base_url: str,
    output_dir: str,
    think: bool,
    box_threshold: float,
    text_threshold: float,
    device: str,
    grounding_root: str,
) -> None:
    run_root = Path(output_dir)
    input_dir = run_root / "input"
    result_dir = run_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    agent = LocalVlmAgent(model=model, base_url=base_url, think=think)
    grounding_runner = GroundingSamRunner(
        device=device,
        output_dir=str(result_dir),
        grounding_root=Path(grounding_root).expanduser().resolve(),
    )

    airsim_ip = os.getenv("AIRSIM_IP", "192.168.31.178")
    airsim_port = int(os.getenv("AIRSIM_PORT", "41451"))
    vehicle_name = os.getenv("AIRSIM_VEHICLE_NAME", "Drone1")

    logger.info(
        "任务启动: vehicle=%s ip=%s port=%s model=%s output_dir=%s device=%s",
        vehicle_name,
        airsim_ip,
        airsim_port,
        model,
        str(run_root),
        device,
    )
    logger.info("Grounding-SAM 根目录: %s", grounding_runner.grounding_root)
    logger.info("输入图目录: %s", input_dir)
    logger.info("结果目录: %s", result_dir)

    ctrl = AirSimController(ip=airsim_ip, port=airsim_port, vehicle_name=vehicle_name)
    mission_started = False

    try:
        ctrl.arm_and_takeoff(altitude=SEARCH_ALTITUDE)
        mission_started = True
        if not ctrl.enable_topdown_only_mode():
            raise RuntimeError("俯视模式启用失败，无法继续执行任务。")
        ctrl.hover(duration=1.0)

        last_action = "hover"
        _phase_banner("阶段1/5：150米搜索轰炸区域")

        destroyed_area_image = None
        for step in range(1, max_area_steps + 1):
            if step > 1:
                _stabilize_before_capture(ctrl)
            image = ctrl.get_bottom_camera_image()
            if image is None:
                logger.warning("[area %02d] 获取俯视图失败，执行短暂悬停后重试", step)
                ctrl.hover(duration=1.0)
                continue

            _save_image(image, str(input_dir), f"area_step_{step:02d}_input.jpg")
            snapshot = _get_flight_snapshot(ctrl)
            decision = _normalize_decision(
                agent.decide_destroyed_area(image=image, altitude=snapshot["altitude"], last_action=last_action),
                "destroyed_area_visible",
            )
            _log_decision("area", step, decision, "destroyed_area_visible", snapshot)

            if decision["destroyed_area_visible"] and decision["centered"]:
                destroyed_area_image = image
                logger.info("已锁定轰炸区域")
                logger.info("保持 150 米高度，切换到飞机搜索阶段")
                last_action = "hover"
                break

            last_action = _apply_action(ctrl, decision, snapshot["altitude"], min_altitude=PLANE_LOCK_ALTITUDE)
            _sleep_between_steps()
        else:
            raise RuntimeError("在 150 米高度未找到可信的轰炸区域。")

        _phase_banner("阶段2/5：150米搜索飞机")
        plane_image = None

        for step in range(1, max_plane_steps + 1):
            _stabilize_before_capture(ctrl)
            image = ctrl.get_bottom_camera_image()
            if image is None:
                logger.warning("[plane %02d] 获取俯视图失败，执行短暂悬停后重试", step)
                ctrl.hover(duration=1.0)
                continue

            _save_image(image, str(input_dir), f"target_step_{step:02d}_input.jpg")
            snapshot = _get_flight_snapshot(ctrl)
            altitude = snapshot["altitude"]
            decision = _normalize_decision(
                agent.decide_plane(image=image, altitude=altitude, last_action=last_action),
                "plane_visible",
            )
            _log_decision("plane", step, decision, "plane_visible", snapshot)

            if decision["plane_visible"]:
                plane_image = image
                logger.info("Aircraft found; descend to %.1f m and run a dedicated centering step", PLANE_LOCK_ALTITUDE)
                break

            last_action = _apply_action(ctrl, decision, altitude, min_altitude=PLANE_LOCK_ALTITUDE)
            _sleep_between_steps()
        else:
            raise RuntimeError("Aircraft search stage failed to find a stable target.")

        _phase_banner("阶段3/5：120米锁定居中")
        current_altitude = ctrl.get_altitude()
        if _descend_to(ctrl, PLANE_LOCK_ALTITUDE, current_altitude):
            last_action = "move_down"
            _sleep_between_steps(1.0)
        else:
            logger.info("Already near %.1f m; no additional descent needed", PLANE_LOCK_ALTITUDE)

        logger.info("开始在 %.1f 米执行锁定居中", PLANE_LOCK_ALTITUDE)
        plane_image, last_action, lock_centered = _center_plane_before_next_stage(
            ctrl=ctrl,
            agent=agent,
            input_dir=input_dir,
            altitude_target=PLANE_LOCK_ALTITUDE,
            max_steps=MAX_LOCK_CENTER_STEPS,
            image_prefix="target_lock_step",
            stage_name="plane-lock",
            last_action=last_action,
        )
        if not lock_centered:
            raise RuntimeError("Aircraft found, but centering failed at the 120m lock stage.")

        current_altitude = ctrl.get_altitude()
        logger.info("锁定完成，开始下降至 %.1f 米评估高度", PLANE_EVAL_ALTITUDE)
        if _descend_to(ctrl, PLANE_EVAL_ALTITUDE, current_altitude):
            last_action = "move_down"
            _sleep_between_steps(1.0)
        else:
            logger.info("Already near evaluation altitude; no additional descent needed")

        logger.info("开始在 %.1f 米评估高度进行评估前复核", PLANE_EVAL_ALTITUDE)
        eval_centered = False
        for step in range(1, MAX_EVAL_CENTER_STEPS + 1):
            _stabilize_before_capture(ctrl)
            latest_eval_view = ctrl.get_bottom_camera_image()
            if latest_eval_view is None:
                logger.warning("[eval %02d] 获取俯视图失败，执行短暂悬停后重试", step)
                ctrl.hover(duration=1.0)
                continue

            plane_image = latest_eval_view
            _save_image(plane_image, str(input_dir), f"target_eval_step_{step:02d}_input.jpg")
            snapshot = _get_flight_snapshot(ctrl)
            eval_decision = _normalize_decision(
                agent.decide_plane(image=plane_image, altitude=snapshot["altitude"], last_action=last_action),
                "plane_visible",
            )
            if eval_decision["action"] == "move_down":
                eval_decision["action"] = "hover"
                eval_decision["parameters"] = {"duration": 1.0}
                eval_decision["should_descend"] = False
            _log_decision("plane", step, eval_decision, "plane_visible", snapshot)

            if eval_decision["plane_visible"] and eval_decision["centered"]:
                eval_centered = True
                logger.info("Target re-centered at altitude %.1f", snapshot["altitude"])
                break

            last_action = _apply_action(ctrl, eval_decision, snapshot["altitude"], min_altitude=PLANE_EVAL_ALTITUDE)
            _sleep_between_steps()

        if not eval_centered:
            logger.warning("在 %.1f 米未能再次确认目标居中，将使用最后一帧执行评估", PLANE_EVAL_ALTITUDE)
            if plane_image is None:
                raise RuntimeError("Missing aircraft image; cannot perform damage evaluation.")

        _phase_banner("阶段4/5：70米执行飞机毁伤评估")
        plane_eval_image_path = _save_image(plane_image, str(input_dir), "target_eval_input.jpg")
        eval_image_output_path = str(result_dir / "target_damage_eval_image.jpg")
        shutil.copy2(plane_eval_image_path, eval_image_output_path)
        logger.info("已复制评估图像到输出目录: %s", eval_image_output_path)
        report = agent.evaluate_plane_damage(plane_image)
        parsed_report = agent._robust_parse_json(report)
        report_rows = [
            ("目标类别", str((parsed_report or {}).get("target_type", "飞机"))),
            ("毁伤等级", str((parsed_report or {}).get("damage_level", "未知"))),
            ("视觉特征", str((parsed_report or {}).get("visual_features", "无"))),
            ("受损估算", str((parsed_report or {}).get("damage_estimate", "无"))),
            ("功能判断", str((parsed_report or {}).get("functional_assessment", "无"))),
            ("不确定性", str((parsed_report or {}).get("uncertainty", "无"))),
        ]
        if parsed_report is None:
            logger.warning("飞机毁伤评估 JSON 解析失败，保留原始文本")
            logger.info("飞机毁伤评估原始输出: %s", report)
        else:
            _log_table("飞机毁伤评估结果", report_rows)

        eval_json_path = str(result_dir / "target_damage_eval.json")
        with open(eval_json_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "eval_image": plane_eval_image_path,
                    "eval_image_output": eval_image_output_path,
                    "raw_report": report,
                    "parsed_report": parsed_report,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("已保存评估结果 JSON: %s", eval_json_path)
        eval_report_path = str(result_dir / "target_damage_eval_report.txt")
        _write_text_report(eval_report_path, "飞机毁伤评估报告", report_rows, report)

        _phase_banner("阶段5/5：Grounding DINO + SAM3 检测分割")
        grounding_result = grounding_runner.run(
            image_path=plane_eval_image_path,
            parsed_report=parsed_report,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        mission_result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_eval_image": plane_eval_image_path,
            "target_eval_image_output": eval_image_output_path,
            "eval_json": eval_json_path,
            "eval_report": eval_report_path,
            "grounding_sam": grounding_result,
        }
        mission_json_path = str(result_dir / "mission_with_grounding_sam.json")
        with open(mission_json_path, "w", encoding="utf-8") as file:
            json.dump(mission_result, file, ensure_ascii=False, indent=2)
        logger.info("已保存任务汇总 JSON: %s", mission_json_path)
        if grounding_result.get("detected"):
            logger.info(
                "Grounding-SAM 检测完成: detections=%d boxes=%s masks=%s",
                grounding_result.get("num_detections", 0),
                grounding_result.get("boxes_image"),
                grounding_result.get("masks_image"),
            )
        else:
            logger.info("Grounding-SAM 未检测到目标: %s", grounding_result.get("reason", "可调整提示词或阈值后重试"))

        if destroyed_area_image is not None:
            logger.info("炸毁区域图像已保存，可用于后续复核")
        ctrl.hover(duration=1.0)
    finally:
        if mission_started:
            _safe_land(ctrl)


def main() -> None:
    parser = argparse.ArgumentParser(description="在 VLN 基础任务上追加 Grounding DINO + SAM3 检测分割")
    parser.add_argument("--max-area-steps", type=int, default=18, help="150米搜索轰炸区域最大步数")
    parser.add_argument("--max-plane-steps", type=int, default=18, help="飞机搜索最大步数")
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL, help="本地多模态模型名，如 qwen3.5:9b")
    parser.add_argument("--base-url", default=DEFAULT_LOCAL_BASE_URL, help="本地模型接口地址")
    parser.add_argument("--think", action="store_true", help="启用模型思考模式")
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(__file__),
        help="输入/输出根目录，脚本会在该目录下使用 input/ 和 output/",
    )
    parser.add_argument("--box-threshold", type=float, default=0.4, help="Grounding DINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Grounding DINO text threshold")
    parser.add_argument(
        "--grounding-root",
        default=os.getenv("GROUNDING_SAM_ROOT", str(GROUNDING_ROOT)),
        help="Grounding-Dino-Sam 根目录，可通过 GROUNDING_SAM_ROOT 环境变量覆盖",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备",
    )

    args = parser.parse_args()
    run_task_with_grounding_sam(
        max_area_steps=args.max_area_steps,
        max_plane_steps=args.max_plane_steps,
        model=args.model,
        base_url=args.base_url,
        output_dir=args.output_dir,
        think=args.think,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        grounding_root=args.grounding_root,
    )


if __name__ == "__main__":
    main()
