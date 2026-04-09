import argparse
import logging
import os
import time

import cv2
import numpy as np
from dotenv import load_dotenv

from airsim_controller import AirSimController
from qwen_agent import DEFAULT_LOCAL_BASE_URL, DEFAULT_LOCAL_MODEL, LocalVlmAgent


load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


SEARCH_ALTITUDE = 150.0
PLANE_LOCK_ALTITUDE = 100.0
PLANE_EVAL_ALTITUDE = 50.0


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _apply_action(ctrl: AirSimController, decision: dict, altitude: float, min_altitude: float) -> str:
    action = str(decision.get("action", "hover")).strip().lower()
    params = decision.get("parameters", {}) or {}

    if action == "move_forward":
        ctrl.move_forward(distance=_clamp(_safe_float(params.get("distance"), 30.0), 20.0, 50.0), speed=3.0)
    elif action == "move_backward":
        ctrl.move_backward(distance=_clamp(_safe_float(params.get("distance"), 20.0), 20.0, 35.0), speed=2.5)
    elif action == "move_left":
        ctrl.move_left(distance=_clamp(_safe_float(params.get("distance"), 20.0), 20.0, 35.0), speed=2.5)
    elif action == "move_right":
        ctrl.move_right(distance=_clamp(_safe_float(params.get("distance"), 20.0), 20.0, 35.0), speed=2.5)
    elif action == "move_down":
        max_drop = max(0.0, altitude - min_altitude)
        distance = min(_clamp(_safe_float(params.get("distance"), 10.0), 5.0, 30.0), max_drop)
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
        ctrl.hover(duration=_clamp(_safe_float(params.get("duration"), 1.0), 0.5, 2.0))
        action = "hover"

    return action


def _center_target(ctrl: AirSimController, offset: str) -> str:
    if offset in {"left", "upper_left", "lower_left"}:
        ctrl.move_left(distance=20.0, speed=2.5)
        return "move_left"
    if offset in {"right", "upper_right", "lower_right"}:
        ctrl.move_right(distance=20.0, speed=2.5)
        return "move_right"
    if offset == "upper":
        ctrl.move_forward(distance=20.0, speed=2.5)
        return "move_forward"
    if offset == "lower":
        ctrl.move_backward(distance=20.0, speed=2.5)
        return "move_backward"

    ctrl.hover(duration=0.8)
    return "hover"


def _save_image(image: np.ndarray, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, filename)
    if not cv2.imwrite(image_path, image):
        raise RuntimeError(f"保存图像失败: {image_path}")
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


def run_task(
    max_area_steps: int,
    max_plane_steps: int,
    model: str,
    base_url: str,
    output_dir: str,
    think: bool,
) -> None:
    agent = LocalVlmAgent(model=model, base_url=base_url, think=think)

    airsim_ip = os.getenv("AIRSIM_IP", "192.168.31.178")
    airsim_port = int(os.getenv("AIRSIM_PORT", "41451"))
    vehicle_name = os.getenv("AIRSIM_VEHICLE_NAME", "Drone1")

    ctrl = AirSimController(ip=airsim_ip, port=airsim_port, vehicle_name=vehicle_name)

    try:
        ctrl.arm_and_takeoff(altitude=SEARCH_ALTITUDE)
        ctrl.enable_topdown_only_mode()
        ctrl.hover(duration=1.0)

        last_action = "hover"
        logger.info("阶段1/4：150米高度搜索轰炸区域")

        destroyed_area_image = None
        for step in range(1, max_area_steps + 1):
            image = ctrl.get_bottom_camera_image()
            if image is None:
                ctrl.hover(duration=1.0)
                continue

            _save_image(image, output_dir, f"area_step_{step:02d}_input.jpg")

            altitude = ctrl.get_altitude()
            decision = agent.decide_destroyed_area(image=image, altitude=altitude, last_action=last_action)
            logger.info(
                "[area %02d] action=%s visible=%s offset=%s centered=%s descend=%s conf=%s reason=%s",
                step,
                decision["action"],
                decision["destroyed_area_visible"],
                decision["target_offset"],
                decision["centered"],
                decision["should_descend"],
                decision["confidence"],
                decision["reasoning"],
            )

            if decision["destroyed_area_visible"] and not decision["centered"]:
                last_action = _center_target(ctrl, decision["target_offset"])
                time.sleep(0.8)
                continue

            if decision["destroyed_area_visible"] and decision["centered"]:
                destroyed_area_image = image
                area_image_path = _save_image(image, output_dir, "destroyed_area_topdown.jpg")
                logger.info("已锁定炸毁区域并保存俯视图: %s", area_image_path)
                logger.info("保持当前150米高度，进入飞机搜索阶段")
                last_action = "hover"
                break

            last_action = _apply_action(ctrl, decision, altitude, min_altitude=PLANE_LOCK_ALTITUDE)
            time.sleep(0.8)
        else:
            raise RuntimeError("在150米高度未找到可信的轰炸区域。")

        logger.info("阶段2/4：保持150米高度，在轰炸区域内搜索一架飞机")
        plane_image = None
        plane_found = False
        for step in range(1, max_plane_steps + 1):
            image = ctrl.get_bottom_camera_image()
            if image is None:
                ctrl.hover(duration=1.0)
                continue

            _save_image(image, output_dir, f"plane_step_{step:02d}_input.jpg")

            altitude = ctrl.get_altitude()
            decision = agent.decide_plane(image=image, altitude=altitude, last_action=last_action)
            logger.info(
                "[plane %02d] alt=%.1f action=%s visible=%s offset=%s centered=%s descend=%s conf=%s reason=%s",
                step,
                altitude,
                decision["action"],
                decision["plane_visible"],
                decision["target_offset"],
                decision["centered"],
                decision["should_descend"],
                decision["confidence"],
                decision["reasoning"],
            )

            if decision["plane_visible"] and not plane_found:
                plane_found = True
                logger.info("已发现飞机，先下降到100米后再进行居中处理")
                if altitude > PLANE_LOCK_ALTITUDE:
                    ctrl.move_down(distance=altitude - PLANE_LOCK_ALTITUDE, speed=2.0)
                    last_action = "move_down"
                    time.sleep(1.0)
                    continue

            if decision["plane_visible"] and not decision["centered"]:
                if altitude > PLANE_LOCK_ALTITUDE:
                    ctrl.move_down(distance=altitude - PLANE_LOCK_ALTITUDE, speed=2.0)
                    last_action = "move_down"
                    time.sleep(1.0)
                    continue
                last_action = _center_target(ctrl, decision["target_offset"])
                time.sleep(0.8)
                continue

            if decision["plane_visible"] and decision["centered"] and altitude > PLANE_EVAL_ALTITUDE:
                logger.info("阶段3/4：飞机已在100米附近居中，开始下降到50米进行评估")
                ctrl.move_down(distance=altitude - PLANE_EVAL_ALTITUDE, speed=2.0)
                last_action = "move_down"
                time.sleep(1.0)
                continue

            if decision["plane_visible"] and decision["centered"]:
                plane_image = image
                plane_image_path = _save_image(image, output_dir, "plane_topdown.jpg")
                logger.info("已锁定飞机并保存俯视图: %s", plane_image_path)
                break

            filtered_decision = dict(decision)
            if filtered_decision["action"] == "move_down":
                filtered_decision["action"] = "hover"
                filtered_decision["parameters"] = {"duration": 1.0}
            last_action = _apply_action(ctrl, filtered_decision, altitude, min_altitude=PLANE_LOCK_ALTITUDE)
            time.sleep(0.8)
        else:
            raise RuntimeError("飞机搜索与100米居中阶段未能稳定锁定目标。")

        logger.info("阶段4/4：50米高度执行飞机毁伤评估")
        if plane_image is None:
            raise RuntimeError("缺少飞机图像，无法执行毁伤评估。")

        _save_image(plane_image, output_dir, "plane_eval_input.jpg")
        report = agent.evaluate_plane_damage(plane_image)
        parsed_report = agent._robust_parse_json(report)
        if parsed_report is None:
            logger.info("飞机毁伤评估原始输出: %s", report)
        else:
            _log_table(
                "飞机毁伤评估结果",
                [
                    ("目标类别", str(parsed_report.get("target_type", "飞机"))),
                    ("毁伤等级", str(parsed_report.get("damage_level", "未知"))),
                    ("视觉特征", str(parsed_report.get("visual_features", "无"))),
                    ("受损估算", str(parsed_report.get("damage_estimate", "无"))),
                    ("功能判断", str(parsed_report.get("functional_assessment", "无"))),
                    ("不确定性", str(parsed_report.get("uncertainty", "无"))),
                ],
            )

        if destroyed_area_image is not None:
            logger.info("炸毁区域图像已保存，可用于后续复核。")
        ctrl.hover(duration=2.0)
    finally:
        ctrl.land()


def main() -> None:
    parser = argparse.ArgumentParser(description="使用本地多模态大模型执行炸毁区域搜索与飞机毁伤评估")
    parser.add_argument("--max-area-steps", type=int, default=18, help="100米搜索炸毁区域的最大步数")
    parser.add_argument("--max-plane-steps", type=int, default=18, help="下降后搜索飞机的最大步数")
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL, help="本地多模态模型名，例如 qwen3.5:9b")
    parser.add_argument("--base-url", default=DEFAULT_LOCAL_BASE_URL, help="本地模型接口地址")
    parser.add_argument("--think", action="store_true", help="启用模型思考模式，默认关闭")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "output", "local_damage_mission"),
        help="过程图像与报告输出目录",
    )
    args = parser.parse_args()

    run_task(
        max_area_steps=args.max_area_steps,
        max_plane_steps=args.max_plane_steps,
        model=args.model,
        base_url=args.base_url,
        output_dir=args.output_dir,
        think=args.think,
    )


if __name__ == "__main__":
    main()
