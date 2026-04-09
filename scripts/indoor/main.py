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
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

QWEN_API_KEY = os.getenv('QWEN_API_KEY', 'sk-b988e0fff98740ef90a8915d3b77dc11')
AIRSIM_IP = os.getenv('AIRSIM_IP', '172.21.192.1')
AIRSIM_PORT = int(os.getenv('AIRSIM_PORT', '41451'))

DEFAULT_INSTRUCTION = '绕过房子，寻找一个红色的汽车，飞到他的旁边停下来'
MAX_STEPS = 30
MAX_CONSECUTIVE_FORWARD = 5
MAX_SAME_SIDE_LATERAL_ATTEMPTS = 2


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def run_red_car_mission(instruction: str = DEFAULT_INSTRUCTION):
    if not QWEN_API_KEY:
        raise RuntimeError('缺少 QWEN_API_KEY，请先在环境变量中配置。')

    logger.info('=' * 70)
    logger.info('导航测试')
    logger.info('=' * 70)
    logger.info(f'指令: {instruction}')

    ctrl = AirSimController(ip=AIRSIM_IP, port=AIRSIM_PORT)
    qwen = QwenVisionAgent(api_key=QWEN_API_KEY)

    try:
        ctrl.arm_and_takeoff(altitude=8)
        last_action = 'hover'
        last_seen_step = None
        seen_once = False
        consecutive_forward = 0
        last_bypass_direction = 'right'
        last_lateral_search_action = None
        same_side_lateral_attempts = 0

        for step in range(1, MAX_STEPS + 1):
            image = ctrl.get_front_camera_image()
            if image is None:
                logger.warning('无法获取图像，悬停后重试')
                ctrl.hover(duration=1.0)
                continue

            steps_since_seen = (step - last_seen_step) if last_seen_step is not None else 999
            nav_state = {
                'last_action': last_action,
                'arrived_streak': 0,
                'steps_since_seen': steps_since_seen,
                'recent_overshoot': bool(seen_once and steps_since_seen <= 3),
                'consecutive_forward': consecutive_forward,
                'last_lateral_search_action': last_lateral_search_action or 'none',
            }

            decision = qwen.decide_action_from_scene(
                image=image,
                instruction=instruction,
                nav_state=nav_state,
            )
            action = str(decision.get('action', 'hover')).strip().lower()
            params = decision.get('parameters', {}) or {}
            reasoning = decision.get('reasoning', '')
            target_visible = bool(decision.get('target_visible', False))
            distance_bucket = str(decision.get('distance_bucket', 'unknown')).strip().lower()
            obstacle_ahead = bool(decision.get('obstacle_ahead', False))
            bypass_direction = str(decision.get('bypass_direction', 'either')).strip().lower()

            if target_visible:
                seen_once = True
                last_seen_step = step
                last_lateral_search_action = None
                same_side_lateral_attempts = 0

            logger.info(
                f'[{step:02d}] 动作={action} 可见={target_visible} '
                f'距离={distance_bucket} 障碍={obstacle_ahead} 绕行={bypass_direction} 理由={reasoning}'
            )

            if action == 'arrived' and target_visible and distance_bucket == 'near':
                logger.info('已到达红车附近，任务完成')
                ctrl.hover(duration=1.5)
                break

            if action == 'arrived' and target_visible and distance_bucket == 'mid':
                logger.warning('判定到达但距离仍为 mid，先小步前进再继续判断')
                ctrl.move_forward(distance=5.0, speed=1.5)
                consecutive_forward = 0
                last_action = 'move_forward'
                time.sleep(0.3)
                continue

            if seen_once and not target_visible and steps_since_seen <= 3:
                recover_dist = 6.0 if steps_since_seen <= 1 else 4.0
                logger.warning(f'目标疑似丢失，执行回退重捕获 {recover_dist:.1f}m')
                ctrl.move_backward(distance=recover_dist, speed=1.8)
                consecutive_forward = 0
                last_action = 'move_backward'
                time.sleep(0.3)
                continue

            if action == 'move_forward' and obstacle_ahead and not target_visible:
                if bypass_direction in {'left', 'right'}:
                    avoid_dir = bypass_direction
                else:
                    avoid_dir = 'left' if last_bypass_direction == 'right' else 'right'

                if last_lateral_search_action == f'move_{avoid_dir}':
                    same_side_lateral_attempts += 1
                else:
                    same_side_lateral_attempts = 1

                if same_side_lateral_attempts > MAX_SAME_SIDE_LATERAL_ATTEMPTS:
                    avoid_dir = 'right' if avoid_dir == 'left' else 'left'
                    same_side_lateral_attempts = 1
                    logger.warning(f'同向绕行尝试过多，切换绕行方向: {avoid_dir}')

                logger.warning(f'前方存在障碍，改为侧向绕行: {avoid_dir}')
                if avoid_dir == 'left':
                    ctrl.move_left(distance=10.0, speed=2.0)
                else:
                    ctrl.move_right(distance=10.0, speed=2.0)
                last_bypass_direction = avoid_dir
                last_lateral_search_action = f'move_{avoid_dir}'
                consecutive_forward = 0
                last_action = f'move_{avoid_dir}'
                time.sleep(0.3)
                continue

            if action == 'move_left':
                if not target_visible:
                    if last_lateral_search_action == 'move_left':
                        same_side_lateral_attempts += 1
                    else:
                        same_side_lateral_attempts = 1
                    if same_side_lateral_attempts > MAX_SAME_SIDE_LATERAL_ATTEMPTS:
                        logger.warning('向左连续搜索未发现目标，改为向右搜索')
                        action = 'move_right'
                        same_side_lateral_attempts = 1
                dist = _clamp(float(params.get('distance', 8.0)), 1.0, 25.0)
                if action == 'move_left':
                    ctrl.move_left(distance=dist, speed=2.0)
                    last_lateral_search_action = 'move_left'
                else:
                    ctrl.move_right(distance=dist, speed=2.0)
                    last_lateral_search_action = 'move_right'
                consecutive_forward = 0
            elif action == 'move_right':
                if not target_visible:
                    if last_lateral_search_action == 'move_right':
                        same_side_lateral_attempts += 1
                    else:
                        same_side_lateral_attempts = 1
                    if same_side_lateral_attempts > MAX_SAME_SIDE_LATERAL_ATTEMPTS:
                        logger.warning('向右连续搜索未发现目标，改为向左搜索')
                        action = 'move_left'
                        same_side_lateral_attempts = 1
                dist = _clamp(float(params.get('distance', 8.0)), 1.0, 25.0)
                if action == 'move_right':
                    ctrl.move_right(distance=dist, speed=2.0)
                    last_lateral_search_action = 'move_right'
                else:
                    ctrl.move_left(distance=dist, speed=2.0)
                    last_lateral_search_action = 'move_left'
                consecutive_forward = 0
            elif action == 'move_forward':
                raw_dist = float(params.get('distance', 15.0))
                if not target_visible and not obstacle_ahead:
                    dist = _clamp(max(raw_dist, 25.0), 25.0, 45.0)
                elif target_visible and distance_bucket == 'far':
                    dist = _clamp(max(raw_dist, 15.0), 15.0, 35.0)
                elif target_visible and distance_bucket == 'mid':
                    dist = _clamp(raw_dist, 5.0, 15.0)
                else:
                    dist = _clamp(raw_dist, 1.0, 45.0)
                ctrl.move_forward(distance=dist, speed=2.0)
                consecutive_forward += 1
                if consecutive_forward > MAX_CONSECUTIVE_FORWARD:
                    fallback_dir = 'left' if last_bypass_direction == 'right' else 'right'
                    logger.warning(f'连续前进过多，执行侧向绕行: {fallback_dir}')
                    if fallback_dir == 'left':
                        ctrl.move_left(distance=4.0, speed=2.0)
                    else:
                        ctrl.move_right(distance=4.0, speed=2.0)
                    last_bypass_direction = fallback_dir
                    last_lateral_search_action = f'move_{fallback_dir}'
                    consecutive_forward = 0
            elif action == 'move_backward':
                dist = _clamp(float(params.get('distance', 3.0)), 0.5, 12.0)
                ctrl.move_backward(distance=dist, speed=2.0)
                consecutive_forward = 0
            elif action == 'move_up':
                dist = _clamp(float(params.get('distance', 2.0)), 0.5, 8.0)
                ctrl.move_up(distance=dist, speed=1.5)
                consecutive_forward = 0
            elif action == 'move_down':
                dist = _clamp(float(params.get('distance', 2.0)), 0.5, 8.0)
                ctrl.move_down(distance=dist, speed=1.5)
                consecutive_forward = 0
            else:
                dur = _clamp(float(params.get('duration', 1.5)), 0.5, 3.0)
                ctrl.hover(duration=dur)
                consecutive_forward = 0

            last_action = action
            time.sleep(0.3)
        else:
            logger.warning('达到最大步数，未明确到达目标，结束任务')

    except KeyboardInterrupt:
        logger.warning('用户中断任务')
    finally:
        ctrl.land()
        logger.info('任务结束')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cmd = ' '.join(sys.argv[1:]).strip()
    else:
        cmd = DEFAULT_INSTRUCTION
    run_red_car_mission(cmd)