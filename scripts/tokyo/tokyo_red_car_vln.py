import logging
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(__file__))

from airsim_controller import AirSimController
from qwen import QwenVisionAgent


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


QWEN_API_KEY = os.getenv('QWEN_API_KEY', 'sk-b988e0fff98740ef90a8915d3b77dc11')
AIRSIM_IP = os.getenv('AIRSIM_IP', '172.21.192.1')
AIRSIM_PORT = int(os.getenv('AIRSIM_PORT', '41451'))

DEFAULT_INSTRUCTION = '在东京三维卫星地图中起飞到较高高度，用俯视相机沿道路搜索一辆红色车辆，找到后飞到它的正上方附近悬停。'
SEARCH_ALTITUDE = float(os.getenv('VLN_SEARCH_ALTITUDE', '100'))
TRACK_ALTITUDE = float(os.getenv('VLN_TRACK_ALTITUDE', '28'))
MIN_ALTITUDE = float(os.getenv('VLN_MIN_ALTITUDE', '18'))
MAX_ALTITUDE = float(os.getenv('VLN_MAX_ALTITUDE', '130'))
MAX_STEPS = int(os.getenv('VLN_MAX_STEPS', '40'))
FORWARD_SAFE_DISTANCE = float(os.getenv('VLN_FORWARD_SAFE_DISTANCE', '14'))
ROAD_TURN_ANGLE = float(os.getenv('VLN_ROAD_TURN_ANGLE', '24'))
MAX_CONSECUTIVE_HOVER = int(os.getenv('VLN_MAX_CONSECUTIVE_HOVER', '3'))
ENABLE_DEPTH_AVOIDANCE = os.getenv('VLN_ENABLE_DEPTH_AVOIDANCE', '1').strip().lower() in {'1', 'true', 'yes', 'on'}


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _check_forward_clearance(ctrl: AirSimController, safe_distance: float):
    """根据开关决定是否使用深度避障；默认由VLM负责视觉避障。"""
    if not ENABLE_DEPTH_AVOIDANCE:
        return True, None
    return ctrl.is_forward_path_clear(safe_distance_m=safe_distance)



def run_tokyo_red_car_vln(instruction: str = DEFAULT_INSTRUCTION):
    if not QWEN_API_KEY:
        raise RuntimeError('缺少 QWEN_API_KEY，请先在环境变量中配置。')

    logger.info('=' * 70)
    logger.info('东京道路红车俯视视觉语言导航')
    logger.info('=' * 70)
    logger.info(f'指令: {instruction}')
    logger.info(f'深度避障开关: {ENABLE_DEPTH_AVOIDANCE}')

    ctrl = AirSimController(ip=AIRSIM_IP, port=AIRSIM_PORT)
    qwen = QwenVisionAgent(api_key=QWEN_API_KEY)

    last_action = 'hover'
    last_rotation = 'none'
    last_avoid_direction = 'left'
    last_seen_step = None
    target_lock_frames = 0
    road_lost_streak = 0
    consecutive_hover = 0

    try:
        ctrl.arm_and_takeoff(altitude=SEARCH_ALTITUDE)
        ctrl.enable_topdown_only_mode()
        ctrl.hover(duration=1.0)

        for step in range(1, MAX_STEPS + 1):
            bottom_image = ctrl.get_bottom_camera_image()
            if bottom_image is None:
                logger.warning('无法获取底视相机图像，悬停后重试')
                ctrl.hover(duration=1.0)
                continue

            altitude = ctrl.get_altitude()
            steps_since_seen = (step - last_seen_step) if last_seen_step is not None else 999
            nav_state = {
                'altitude': altitude,
                'search_altitude': SEARCH_ALTITUDE,
                'last_action': last_action,
                'steps_since_seen': steps_since_seen,
                'target_lock_frames': target_lock_frames,
                'road_lost_streak': road_lost_streak,
                'last_rotation': last_rotation,
            }

            decision = qwen.decide_action_from_aerial_scene(
                image=bottom_image,
                instruction=instruction,
                nav_state=nav_state,
            )

            action = str(decision.get('action', 'hover')).strip().lower()
            params = decision.get('parameters', {}) or {}
            road_visible = bool(decision.get('road_visible', False))
            road_direction = str(decision.get('road_direction', 'lost')).strip().lower()
            target_visible = bool(decision.get('target_visible', False))
            target_offset = str(decision.get('target_offset', 'unknown')).strip().lower()
            distance_bucket = str(decision.get('distance_bucket', 'unknown')).strip().lower()
            centered = bool(decision.get('centered', False))
            should_descend = bool(decision.get('should_descend', False))
            reasoning = str(decision.get('reasoning', '')).strip()

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
                f'[{step:02d}] 动作={action} 高度={altitude:.1f}m 道路={road_visible}/{road_direction} '
                f'红车={target_visible} 偏移={target_offset} 距离={distance_bucket} '
                f'居中={centered} 下降={should_descend} 理由={reasoning}'
            )

            if action == 'hover' and not target_visible:
                consecutive_hover += 1
            else:
                consecutive_hover = 0

            # 模型给出 hover 但目标未出现时，立即触发本地改写，避免原地悬停。
            if action == 'hover' and not target_visible:
                if road_visible:
                    logger.warning('模型返回 hover，按道路规则改写为 move_forward')
                    action = 'move_forward'
                    params = {'distance': 14.0}
                else:
                    fallback_rotate = 'rotate_left' if last_rotation != 'left' else 'rotate_right'
                    logger.warning(f'模型返回 hover 且道路不可见，改写为 {fallback_rotate}')
                    action = fallback_rotate
                    params = {'angle': 18.0}

            # 默认保持高空俯视，避免在复杂城区贴楼飞行。
            if not target_visible and altitude < SEARCH_ALTITUDE - 2.0:
                climb = _clamp(min(6.0, SEARCH_ALTITUDE - altitude), 1.5, 6.0)
                logger.info(f'当前低于搜索高度，先上升 {climb:.1f}m 保持俯视搜索')
                ctrl.move_up(distance=climb, speed=1.5)
                last_action = 'move_up'
                time.sleep(0.4)
                continue

            if action == 'arrived' and target_visible and centered and distance_bucket in {'near', 'mid'}:
                logger.info('已到达红车正上方附近，任务完成')
                ctrl.hover(duration=2.0)
                break

            if target_visible and should_descend and altitude > TRACK_ALTITUDE:
                descend = _clamp(min(6.0, altitude - TRACK_ALTITUDE), 1.5, 6.0)
                if altitude - descend >= MIN_ALTITUDE:
                    logger.info(f'目标已锁定，下降到跟踪高度，下降 {descend:.1f}m')
                    ctrl.move_down(distance=descend, speed=1.5)
                    last_action = 'move_down'
                    time.sleep(0.4)
                    continue

            if not road_visible and not target_visible and road_lost_streak >= 2:
                # 正角度=左转，负角度=右转，direction 与实际旋转方向保持一致
                if last_rotation != 'left':
                    fallback_angle = 20.0
                    direction = 'rotate_left'
                    next_rotation = 'left'
                else:
                    fallback_angle = -20.0
                    direction = 'rotate_right'
                    next_rotation = 'right'
                logger.warning(f'连续丢失道路，执行兜底旋转: {direction}')
                ctrl.rotate_yaw(angle=fallback_angle, speed=20)
                last_rotation = next_rotation
                last_action = direction
                time.sleep(0.4)
                continue

            if action == 'hover' and not target_visible and consecutive_hover >= MAX_CONSECUTIVE_HOVER:
                if road_visible:
                    safe_distance = max(10.0, FORWARD_SAFE_DISTANCE - 1.0)
                    is_clear, clearance = ctrl.is_forward_path_clear(safe_distance_m=safe_distance)
                    if is_clear:
                        logger.warning('模型连续悬停，触发本地兜底: 沿道路小步前进')
                        ctrl.move_forward(distance=12.0, speed=2.2)
                        last_action = 'move_forward'
                        consecutive_hover = 0
                        time.sleep(0.35)
                        continue

                    clearance_text = 'unknown' if clearance is None else f'{clearance:.1f}m'
                    logger.warning(f'模型连续悬停且前方净空不足({clearance_text})，触发旋转重定位')

                fallback_angle = 18.0 if last_rotation != 'left' else -18.0
                direction = 'rotate_right' if fallback_angle < 0 else 'rotate_left'
                logger.warning(f'模型连续悬停，执行兜底旋转: {direction}')
                ctrl.rotate_yaw(angle=fallback_angle, speed=20)
                last_rotation = 'right' if fallback_angle < 0 else 'left'
                last_action = direction
                consecutive_hover = 0
                time.sleep(0.35)
                continue

            # 本地道路跟随硬规则：看到道路弯折时，先执行转向再前进，避免撞楼或偏离道路。
            if not target_visible and road_visible and road_direction in {'left_curve', 'right_curve'}:
                turn_side = 'left' if road_direction == 'left_curve' else 'right'
                if last_action not in {'rotate_left', 'rotate_right'}:
                    angle = _clamp(_safe_float(params.get('angle'), ROAD_TURN_ANGLE), 10.0, 38.0)
                    if turn_side == 'left':
                        logger.info(f'检测到左弯道路，优先左转 {angle:.1f} 度')
                        ctrl.rotate_yaw(angle=angle, speed=20)
                        last_rotation = 'left'
                        last_action = 'rotate_left'
                    else:
                        logger.info(f'检测到右弯道路，优先右转 {angle:.1f} 度')
                        ctrl.rotate_yaw(angle=-angle, speed=20)
                        last_rotation = 'right'
                        last_action = 'rotate_right'
                    time.sleep(0.35)
                    continue
                else:
                    follow_dist = _clamp(_safe_float(params.get('distance'), 12.0), 8.0, 16.0)
                    safe_distance = max(10.0, FORWARD_SAFE_DISTANCE - 2.0)
                    is_clear, clearance = _check_forward_clearance(ctrl, safe_distance)
                    if is_clear:
                        logger.info(f'弯道转向后沿路前进 {follow_dist:.1f}m')
                        ctrl.move_forward(distance=follow_dist, speed=2.2)
                        last_action = 'move_forward'
                        time.sleep(0.35)
                        continue
                    clearance_text = 'unknown' if clearance is None else f'{clearance:.1f}m'
                    logger.warning(f'弯道前方净空不足({clearance_text})，改为上升绕行')
                    climb = _clamp(min(4.0, MAX_ALTITUDE - altitude), 1.5, 4.0)
                    if climb > 0:
                        ctrl.move_up(distance=climb, speed=1.5)
                        last_action = 'move_up'
                        time.sleep(0.35)
                        continue

            # 红车可见但未居中时，强制平移对准，不允许前进
            if target_visible and not centered and action == 'move_forward':
                if target_offset in {'left', 'upper_left', 'lower_left'}:
                    logger.info('红车可见但偏左，强制改为 move_left')
                    action = 'move_left'
                    params = {'distance': 6.0}
                elif target_offset in {'right', 'upper_right', 'lower_right'}:
                    logger.info('红车可见但偏右，强制改为 move_right')
                    action = 'move_right'
                    params = {'distance': 6.0}

            if action == 'move_forward':
                safe_distance = FORWARD_SAFE_DISTANCE
                if altitude <= TRACK_ALTITUDE + 5.0:
                    safe_distance += 3.0
                is_clear, clearance = _check_forward_clearance(ctrl, safe_distance)
                if not is_clear:
                    clearance_text = 'unknown' if clearance is None else f'{clearance:.1f}m'
                    logger.warning(
                        f'前向路径疑似被建筑阻挡(净空={clearance_text}, 阈值={safe_distance:.1f}m)，执行强制避障'
                    )

                    up_room = MAX_ALTITUDE - altitude
                    if up_room > 2.0:
                        climb = _clamp(min(5.0, up_room), 2.0, 5.0)
                        ctrl.move_up(distance=climb, speed=1.5)
                        last_action = 'move_up'
                    else:
                        avoid_dir = 'left' if last_avoid_direction == 'right' else 'right'
                        shift = 8.0 if not target_visible else 4.0
                        if avoid_dir == 'left':
                            ctrl.move_left(distance=shift, speed=2.0)
                        else:
                            ctrl.move_right(distance=shift, speed=2.0)
                        last_avoid_direction = avoid_dir
                        last_action = f'move_{avoid_dir}'
                    time.sleep(0.4)
                    continue

                raw_dist = _safe_float(params.get('distance'), 16.0)
                if target_visible and distance_bucket == 'far':
                    distance = _clamp(raw_dist, 8.0, 15.0)
                elif target_visible and distance_bucket == 'mid':
                    distance = _clamp(raw_dist, 3.0, 8.0)
                elif road_visible:
                    distance = _clamp(raw_dist, 12.0, 30.0)
                else:
                    distance = _clamp(raw_dist, 4.0, 10.0)
                ctrl.move_forward(distance=distance, speed=2.5)
            elif action == 'move_left':
                raw_dist = _safe_float(params.get('distance'), 6.0)
                if target_visible:
                    distance = _clamp(raw_dist, 2.0, 8.0)
                else:
                    distance = _clamp(raw_dist, 4.0, 12.0)
                ctrl.move_left(distance=distance, speed=2.0)
                last_avoid_direction = 'left'
            elif action == 'move_right':
                raw_dist = _safe_float(params.get('distance'), 6.0)
                if target_visible:
                    distance = _clamp(raw_dist, 2.0, 8.0)
                else:
                    distance = _clamp(raw_dist, 4.0, 12.0)
                ctrl.move_right(distance=distance, speed=2.0)
                last_avoid_direction = 'right'
            elif action == 'move_up':
                raw_dist = _safe_float(params.get('distance'), 4.0)
                distance = _clamp(raw_dist, 1.0, min(8.0, MAX_ALTITUDE - altitude))
                if distance > 0:
                    ctrl.move_up(distance=distance, speed=1.5)
                else:
                    ctrl.hover(duration=1.0)
                    action = 'hover'
            elif action == 'move_down':
                raw_dist = _safe_float(params.get('distance'), 3.0)
                max_descent = max(0.0, altitude - MIN_ALTITUDE)
                distance = _clamp(raw_dist, 1.0, min(8.0, max_descent)) if max_descent > 0 else 0.0
                if distance > 0:
                    ctrl.move_down(distance=distance, speed=1.5)
                else:
                    ctrl.hover(duration=1.0)
                    action = 'hover'
            elif action == 'rotate_left':
                angle = _clamp(_safe_float(params.get('angle'), 20.0), 8.0, 45.0)
                ctrl.rotate_yaw(angle=angle, speed=20)
                last_rotation = 'left'
            elif action == 'rotate_right':
                angle = _clamp(_safe_float(params.get('angle'), 20.0), 8.0, 45.0)
                ctrl.rotate_yaw(angle=-angle, speed=20)
                last_rotation = 'right'
            else:
                duration = _clamp(_safe_float(params.get('duration'), 1.5), 0.5, 3.0)
                ctrl.hover(duration=duration)
                action = 'hover'

            if action == 'move_forward' and road_direction not in {'left_curve', 'right_curve'}:
                last_rotation = 'none'

            last_action = action
            time.sleep(0.4)
        else:
            logger.warning('达到最大步数，未成功锁定红车，任务结束')
    except KeyboardInterrupt:
        logger.warning('用户中断任务')
    except Exception as exc:
        logger.error(f'任务异常终止: {exc}', exc_info=True)
    finally:
        ctrl.land()
        logger.info('任务结束')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        task_instruction = ' '.join(sys.argv[1:]).strip()
    else:
        task_instruction = DEFAULT_INSTRUCTION
    run_tokyo_red_car_vln(task_instruction)