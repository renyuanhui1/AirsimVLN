"""
结构化任务序列执行器
支持指令格式：
  steps:
    - follow_road          : 沿道路前进，直到检测到十字路口或达到步数上限
    - turn_at_intersection : 在十字路口按指定方向转弯（direction: left/right）
    - find_target          : 沿路搜索目标并悬停到正上方
"""
import logging
import os
import sys
import time
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(__file__))

from airsim_controller import AirSimController
from qwen import QwenVisionAgent
import web_monitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── 环境变量配置 ──────────────────────────────────────────────
QWEN_API_KEY        = os.getenv('QWEN_API_KEY', 'sk-b988e0fff98740ef90a8915d3b77dc11')
AIRSIM_IP           = os.getenv('AIRSIM_IP', '172.21.192.1')
AIRSIM_PORT         = int(os.getenv('AIRSIM_PORT', '41451'))
SEARCH_ALTITUDE     = float(os.getenv('VLN_SEARCH_ALTITUDE', '100'))
TRACK_ALTITUDE      = float(os.getenv('VLN_TRACK_ALTITUDE', '28'))
MIN_ALTITUDE        = float(os.getenv('VLN_MIN_ALTITUDE', '18'))
MAX_ALTITUDE        = float(os.getenv('VLN_MAX_ALTITUDE', '130'))
ROAD_TURN_ANGLE     = float(os.getenv('VLN_ROAD_TURN_ANGLE', '24'))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _safe_float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── 各阶段执行函数 ────────────────────────────────────────────

def phase_follow_road(ctrl: AirSimController, qwen: QwenVisionAgent,
                      max_steps: int = 20) -> str:
    """
    沿道路前进，直到 VLM 报告 intersection 或步数耗尽。
    返回退出原因: 'intersection' | 'steps_exhausted'
    """
    logger.info('▶ 阶段: 沿道路前进')
    last_action = 'hover'
    last_rotation = 'none'
    road_lost_streak = 0

    for step in range(1, max_steps + 1):
        img = ctrl.get_bottom_camera_image()
        if img is None:
            ctrl.hover(duration=1.0)
            continue

        altitude = ctrl.get_altitude()
        # 保持搜索高度
        if altitude < SEARCH_ALTITUDE - 2.0:
            climb = _clamp(min(6.0, SEARCH_ALTITUDE - altitude), 1.5, 6.0)
            ctrl.move_up(distance=climb, speed=1.5)
            continue

        nav_state = {
            'altitude': altitude,
            'search_altitude': SEARCH_ALTITUDE,
            'last_action': last_action,
            'steps_since_seen': 999,
            'target_lock_frames': 0,
            'road_lost_streak': road_lost_streak,
            'last_rotation': last_rotation,
        }
        decision = qwen.decide_action_from_aerial_scene(img, '沿道路前进，注意识别十字路口', nav_state)

        road_direction = str(decision.get('road_direction', 'forward')).strip().lower()
        road_visible   = bool(decision.get('road_visible', False))

        logger.info(f'[follow {step:02d}] road={road_visible} dir={road_direction}')
        web_monitor.update_state(
            image=img,
            step=step, phase='沿道路前进',
            altitude=altitude,
            action=decision.get('action', 'move_forward'),
            reasoning=decision.get('reasoning', ''),
            road_visible=road_visible,
            road_direction=road_direction,
            target_visible=False,
            target_offset='unknown',
        )

        # 检测到十字路口，先前进到路口中央再退出
        if road_direction == 'intersection':
            logger.info('检测到十字路口，前进到路口中央')
            ctrl.move_forward(distance=30.0, speed=2.0)
            time.sleep(0.5)
            logger.info('已到达路口中央，退出 follow_road 阶段')
            return 'intersection'

        if road_visible:
            road_lost_streak = 0
        else:
            road_lost_streak += 1

        # 弯道跟随
        if road_visible and road_direction in {'left_curve', 'right_curve'}:
            if last_action not in {'rotate_left', 'rotate_right'}:
                angle = ROAD_TURN_ANGLE
                if road_direction == 'left_curve':
                    ctrl.rotate_yaw(angle=angle, speed=20)
                    last_rotation = 'left'
                    last_action = 'rotate_left'
                else:
                    ctrl.rotate_yaw(angle=-angle, speed=20)
                    last_rotation = 'right'
                    last_action = 'rotate_right'
                time.sleep(0.35)
                continue

        # 道路丢失兜底旋转
        if not road_visible and road_lost_streak >= 2:
            if last_rotation != 'left':
                ctrl.rotate_yaw(angle=20.0, speed=20)
                last_rotation = 'left'
                last_action = 'rotate_left'
            else:
                ctrl.rotate_yaw(angle=-20.0, speed=20)
                last_rotation = 'right'
                last_action = 'rotate_right'
            time.sleep(0.35)
            continue

        # 默认前进
        ctrl.move_forward(distance=16.0, speed=2.5)
        last_action = 'move_forward'
        if road_direction not in {'left_curve', 'right_curve'}:
            last_rotation = 'none'
        time.sleep(0.4)

    return 'steps_exhausted'


def phase_turn_at_intersection(ctrl: AirSimController, direction: str,
                                angle: float = 85.0):
    """在十字路口执行大角度转弯，然后小步前进对齐新道路。"""
    logger.info(f'▶ 阶段: 十字路口{direction}转 {angle}°')
    if direction == 'left':
        ctrl.rotate_yaw(angle=angle, speed=20)
    else:
        ctrl.rotate_yaw(angle=-angle, speed=20)
    time.sleep(0.5)
    # 转弯后前进一小步对齐新道路
    ctrl.move_forward(distance=10.0, speed=2.0)
    time.sleep(0.4)


def phase_find_target(ctrl: AirSimController, qwen: QwenVisionAgent,
                      target_desc: str, max_steps: int = 40) -> bool:
    """
    沿路搜索目标，找到后悬停到正上方。
    返回 True 表示成功找到目标。
    """
    logger.info(f'▶ 阶段: 搜索目标 [{target_desc}]')
    last_action = 'hover'
    last_rotation = 'none'
    last_avoid_direction = 'left'
    last_seen_step = None
    target_lock_frames = 0
    descend_count = 0
    road_lost_streak = 0
    consecutive_hover = 0

    instruction = f'沿道路搜索{target_desc}，找到后飞到正上方悬停。'

    for step in range(1, max_steps + 1):
        img = ctrl.get_bottom_camera_image()
        if img is None:
            ctrl.hover(duration=1.0)
            continue

        altitude = ctrl.get_altitude()
        if not (last_seen_step and step - last_seen_step <= 3) and altitude < SEARCH_ALTITUDE - 2.0:
            climb = _clamp(min(6.0, SEARCH_ALTITUDE - altitude), 1.5, 6.0)
            ctrl.move_up(distance=climb, speed=1.5)
            last_action = 'move_up'
            continue

        steps_since_seen = (step - last_seen_step) if last_seen_step else 999
        nav_state = {
            'altitude': altitude,
            'search_altitude': SEARCH_ALTITUDE,
            'last_action': last_action,
            'steps_since_seen': steps_since_seen,
            'target_lock_frames': target_lock_frames,
            'road_lost_streak': road_lost_streak,
            'last_rotation': last_rotation,
        }
        decision = qwen.decide_action_from_aerial_scene(img, instruction, nav_state)

        action          = str(decision.get('action', 'hover')).strip().lower()
        params          = decision.get('parameters', {}) or {}
        road_visible    = bool(decision.get('road_visible', False))
        road_direction  = str(decision.get('road_direction', 'lost')).strip().lower()
        target_visible  = bool(decision.get('target_visible', False))
        target_offset   = str(decision.get('target_offset', 'unknown')).strip().lower()
        distance_bucket = str(decision.get('distance_bucket', 'unknown')).strip().lower()
        centered        = bool(decision.get('centered', False))
        should_descend  = bool(decision.get('should_descend', False))

        web_monitor.update_state(
            image=img,
            step=step, phase='搜索目标',
            altitude=altitude,
            action=action,
            reasoning=decision.get('reasoning', ''),
            road_visible=road_visible,
            road_direction=road_direction,
            target_visible=target_visible,
            target_offset=target_offset,
            distance_bucket=distance_bucket,
            centered=centered,
        )

        if road_visible:
            road_lost_streak = 0
        else:
            road_lost_streak += 1

        if target_visible:
            last_seen_step = step
            target_lock_frames += 1
        else:
            target_lock_frames = 0

        logger.info(
            f'[find {step:02d}] action={action} alt={altitude:.1f}m '
            f'road={road_visible}/{road_direction} target={target_visible} '
            f'offset={target_offset} dist={distance_bucket} centered={centered}'
        )

        # 到达判定
        if (action == 'arrived' or (target_visible and centered)) and distance_bucket in {'near', 'mid'} and descend_count >= 1:
            logger.info('已到达目标正上方，任务完成')
            ctrl.hover(duration=2.0)
            return True

        # 连续 hover 兜底
        if action == 'hover' and not target_visible:
            consecutive_hover += 1
            if road_visible:
                action = 'move_forward'
                params = {'distance': 14.0}
            else:
                action = 'rotate_left' if last_rotation != 'left' else 'rotate_right'
                params = {'angle': 18.0}
        else:
            consecutive_hover = 0

        # 目标锁定后下降，最多下降2次
        if target_visible and should_descend and altitude > TRACK_ALTITUDE and descend_count < 2:
            descend = _clamp(min(6.0, altitude - TRACK_ALTITUDE), 1.5, 6.0)
            if altitude - descend >= MIN_ALTITUDE:
                logger.info(f'目标已锁定，下降 {descend:.1f}m（第{descend_count+1}次）')
                ctrl.move_down(distance=descend, speed=1.5)
                descend_count += 1
                last_action = 'move_down'
                time.sleep(0.4)
                continue

        # 道路丢失兜底
        if not road_visible and not target_visible and road_lost_streak >= 2:
            if last_rotation != 'left':
                ctrl.rotate_yaw(angle=20.0, speed=20)
                last_rotation = 'left'
                last_action = 'rotate_left'
            else:
                ctrl.rotate_yaw(angle=-20.0, speed=20)
                last_rotation = 'right'
                last_action = 'rotate_right'
            time.sleep(0.4)
            continue

        # 执行动作
        if action == 'move_forward':
            raw = _safe_float(params.get('distance'), 16.0)
            if target_visible and distance_bucket == 'far':
                dist = _clamp(raw, 8.0, 15.0)
            elif target_visible and distance_bucket == 'mid':
                dist = _clamp(raw, 3.0, 8.0)
            elif road_visible:
                dist = _clamp(raw, 12.0, 30.0)
            else:
                dist = _clamp(raw, 4.0, 10.0)
            ctrl.move_forward(distance=dist, speed=2.5)
        elif action == 'move_left':
            dist = _clamp(_safe_float(params.get('distance'), 6.0), 2.0 if target_visible else 4.0, 8.0 if target_visible else 12.0)
            ctrl.move_left(distance=dist, speed=2.0)
            last_avoid_direction = 'left'
        elif action == 'move_right':
            dist = _clamp(_safe_float(params.get('distance'), 6.0), 2.0 if target_visible else 4.0, 8.0 if target_visible else 12.0)
            ctrl.move_right(distance=dist, speed=2.0)
            last_avoid_direction = 'right'
        elif action == 'move_up':
            dist = _clamp(_safe_float(params.get('distance'), 4.0), 1.0, min(8.0, MAX_ALTITUDE - altitude))
            ctrl.move_up(distance=dist, speed=1.5) if dist > 0 else ctrl.hover(duration=1.0)
        elif action == 'move_down':
            max_desc = max(0.0, altitude - MIN_ALTITUDE)
            dist = _clamp(_safe_float(params.get('distance'), 3.0), 1.0, min(8.0, max_desc)) if max_desc > 0 else 0.0
            ctrl.move_down(distance=dist, speed=1.5) if dist > 0 else ctrl.hover(duration=1.0)
        elif action == 'rotate_left':
            angle = _clamp(_safe_float(params.get('angle'), 20.0), 8.0, 45.0)
            ctrl.rotate_yaw(angle=angle, speed=20)
            last_rotation = 'left'
        elif action == 'rotate_right':
            angle = _clamp(_safe_float(params.get('angle'), 20.0), 8.0, 45.0)
            ctrl.rotate_yaw(angle=-angle, speed=20)
            last_rotation = 'right'
        else:
            ctrl.hover(duration=_clamp(_safe_float(params.get('duration'), 1.5), 0.5, 3.0))
            action = 'hover'

        if action == 'move_forward' and road_direction not in {'left_curve', 'right_curve'}:
            last_rotation = 'none'

        last_action = action
        time.sleep(0.4)

    logger.warning('达到最大步数，未找到目标')
    return False


# ── 主入口 ────────────────────────────────────────────────────

def run_task_sequence(steps: List[Dict[str, Any]]):
    """
    执行结构化任务序列。

    示例 steps:
    [
        {"type": "follow_road", "max_steps": 20},
        {"type": "turn_at_intersection", "direction": "left"},
        {"type": "find_target", "target": "红色车辆", "max_steps": 40},
    ]
    """
    if not QWEN_API_KEY:
        raise RuntimeError('缺少 QWEN_API_KEY')

    web_monitor.start_server(port=5000)
    time.sleep(1)

    ctrl = AirSimController(ip=AIRSIM_IP, port=AIRSIM_PORT)
    qwen = QwenVisionAgent(api_key=QWEN_API_KEY)

    try:
        ctrl.arm_and_takeoff(altitude=SEARCH_ALTITUDE)
        ctrl.enable_topdown_only_mode()
        ctrl.hover(duration=1.0)

        for i, step in enumerate(steps):
            stype = step.get('type', '')
            logger.info(f'═══ 任务步骤 {i+1}/{len(steps)}: {stype} ═══')

            if stype == 'follow_road':
                result = phase_follow_road(ctrl, qwen, max_steps=step.get('max_steps', 20))
                logger.info(f'follow_road 结束，原因: {result}')

            elif stype == 'turn_at_intersection':
                direction = step.get('direction', 'left')
                angle = float(step.get('angle', 85.0))
                phase_turn_at_intersection(ctrl, direction, angle)

            elif stype == 'find_target':
                target = step.get('target', '红色车辆')
                success = phase_find_target(ctrl, qwen, target, max_steps=step.get('max_steps', 40))
                if success:
                    logger.info('目标找到，任务完成')
                    break
                else:
                    logger.warning('未找到目标，继续下一步骤')

            else:
                logger.warning(f'未知步骤类型: {stype}，跳过')

    except KeyboardInterrupt:
        logger.warning('用户中断')
    except Exception as exc:
        logger.error(f'任务异常: {exc}', exc_info=True)
    finally:
        ctrl.land()
        logger.info('任务结束')


if __name__ == '__main__':
    # 示例：沿道路飞行 → 第一个十字路口左转 → 搜索红色车辆
    task = [
        {"type": "follow_road",           "max_steps": 20},
        {"type": "turn_at_intersection",  "direction": "left"},
        {"type": "find_target",           "target": "红色车辆", "max_steps": 40},
    ]
    run_task_sequence(task)
