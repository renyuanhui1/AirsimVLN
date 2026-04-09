# scripts/tokyo/airsim_controller.py
import math
import airsim
import numpy as np
import time
from typing import Tuple, Iterable, Optional, Dict
import logging

class AirSimController:
    FRONT_CAMERA_CANDIDATES = ("front_center", "front", "0")
    BOTTOM_CAMERA_CANDIDATES = ("bottom_center", "downward_center", "bottom", "down", "1")

    def __init__(self, ip: str = "localhost", port: int = 41451):
        self.client = airsim.MultirotorClient(ip=ip, port=port)
        self.client.confirmConnection()
        self.vehicle_name = "Drone1"
        self.logger = logging.getLogger(__name__)
        self.home_position = None
        self._resolved_cameras = {}
        self._front_camera_name = None
        self._topdown_camera_name = None
        self._topdown_mode_enabled = False
        self._FRONT_POSE = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))
        self._DOWN_POSE = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(-90), 0, 0))
        
    def arm_and_takeoff(self, altitude: float = 5):
        """武装飞机并起飞到初始高度"""
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        
        # 记录起始位置
        self.home_position = self.client.getMultirotorState(self.vehicle_name).kinematics_estimated.position
        
        # 起飞
        self.logger.info("起飞中...")
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()

        # 等待姿态稳定
        time.sleep(2)

        # 基于起始地面位置计算目标z（世界坐标）
        target_z = self.home_position.z_val - altitude
        self.client.moveToZAsync(
            z=target_z,
            velocity=2,
            vehicle_name=self.vehicle_name
        ).join()

        # 到达后再等姿态完全稳定
        time.sleep(2)

        current_z = -self.client.getMultirotorState(self.vehicle_name).kinematics_estimated.position.z_val
        self.logger.info(f"起飞完成，当前高度: {current_z:.1f}m")

    def _fetch_scene_response(self, camera_name: str):
        responses = self.client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
        ], vehicle_name=self.vehicle_name)
        if not responses or len(responses[0].image_data_uint8) == 0:
            return None
        return responses[0]

    def _decode_scene_response(self, response) -> np.ndarray:
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        return img1d.reshape(response.height, response.width, 3)

    def get_named_camera_image(self, camera_name: str) -> Optional[np.ndarray]:
        try:
            response = self._fetch_scene_response(camera_name)
            if response is None:
                return None
            return self._decode_scene_response(response)
        except Exception as exc:
            self.logger.warning(f"获取相机 {camera_name} 图像异常: {exc}")
            return None

    def probe_camera_candidates(self) -> Dict[str, np.ndarray]:
        images: Dict[str, np.ndarray] = {}
        ordered = []
        for name in self.BOTTOM_CAMERA_CANDIDATES + self.FRONT_CAMERA_CANDIDATES:
            if name not in ordered:
                ordered.append(name)
        for name in ordered:
            image = self.get_named_camera_image(name)
            if image is not None:
                images[name] = image
        return images

    def _resolve_front_camera(self) -> Optional[str]:
        """找到第一个可用的前视相机名称并缓存。"""
        if self._front_camera_name:
            return self._front_camera_name
        for name in self.FRONT_CAMERA_CANDIDATES:
            image = self.get_named_camera_image(name)
            if image is not None:
                self._front_camera_name = name
                self.logger.info(f"前视相机已解析为: {name}")
                return name
        return None

    def _resolve_bottom_camera(self) -> Optional[str]:
        cached = self._resolved_cameras.get("bottom")
        if cached:
            return cached
        for name in self.BOTTOM_CAMERA_CANDIDATES:
            image = self.get_named_camera_image(name)
            if image is not None:
                self._resolved_cameras["bottom"] = name
                self.logger.info(f"底视相机已解析为: {name}")
                return name
        return None

    def enable_topdown_only_mode(self) -> bool:
        """优先使用真实底视相机；若不存在，则将前视相机旋转为俯视。"""
        native_bottom = self._resolve_bottom_camera()
        if native_bottom is not None:
            self._topdown_camera_name = native_bottom
            self._topdown_mode_enabled = True
            self.logger.info(f"已启用底视模式，使用原生底视相机: {native_bottom}")
            for window_id in (0, 1, 2):
                try:
                    self.client.simSetSubwindow(window_id, native_bottom, airsim.ImageType.Scene, vehicle_name=self.vehicle_name)
                except Exception:
                    pass
            return True

        cam = self._resolve_front_camera()
        if cam is None:
            self.logger.error("未找到可用相机，无法启用俯视模式")
            return False

        try:
            self.client.simSetCameraPose(cam, self._DOWN_POSE, vehicle_name=self.vehicle_name)
            self._topdown_camera_name = cam
            self._topdown_mode_enabled = True
            self.logger.info(f"未找到原生底视相机，已退回前视相机俯视方案: {cam} (pitch=-90)")

            for window_id in (0, 1, 2):
                try:
                    self.client.simSetSubwindow(window_id, cam, airsim.ImageType.Scene, vehicle_name=self.vehicle_name)
                except Exception:
                    pass
            return True
        except Exception as exc:
            self.logger.error(f"启用仅俯视模式失败: {exc}")
            return False

    def _get_scene_image_from_candidates(self, role: str, camera_names: Iterable[str]) -> Optional[np.ndarray]:
        """按候选名称依次尝试获取场景图像，兼容不同插件/蓝图中的相机命名。"""
        candidates = list(camera_names)
        preferred = self._resolved_cameras.get(role)
        if preferred:
            candidates = [preferred] + [name for name in candidates if name != preferred]

        last_error = None
        for camera_name in candidates:
            try:
                response = self._fetch_scene_response(camera_name)
                if response is None:
                    continue

                img_rgb = self._decode_scene_response(response)

                if self._resolved_cameras.get(role) != camera_name:
                    self.logger.info(f"{role} 相机已解析为: {camera_name}")
                self._resolved_cameras[role] = camera_name
                return img_rgb
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            self.logger.error(f"获取 {role} 相机图像异常: {last_error}")
        else:
            self.logger.error(f"获取 {role} 相机图像失败，已尝试: {candidates}")
        return None
    
    def get_front_camera_image(self, width: int = 640, height: int = 480) -> Optional[np.ndarray]:
        """获取前置相机图像，优先使用已解析成功的相机名称。"""
        del width, height
        if self._topdown_mode_enabled:
            self.logger.info("仅俯视模式已启用，跳过前视图像采集")
            return None
        return self._get_scene_image_from_candidates("front", self.FRONT_CAMERA_CANDIDATES)

    def get_front_depth_map(self) -> Optional[np.ndarray]:
        """获取前置深度图（米），用于本地硬规则避障。"""
        candidates = list(self.FRONT_CAMERA_CANDIDATES)
        preferred = self._resolved_cameras.get("front")
        if preferred:
            candidates = [preferred] + [name for name in candidates if name != preferred]

        last_error = None
        for camera_name in candidates:
            try:
                responses = self.client.simGetImages([
                    airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False)
                ], vehicle_name=self.vehicle_name)

                if not responses:
                    continue

                response = responses[0]
                if response.width == 0 or response.height == 0 or not response.image_data_float:
                    continue

                depth = np.array(response.image_data_float, dtype=np.float32).reshape(response.height, response.width)
                depth[~np.isfinite(depth)] = np.nan
                depth[depth <= 0.05] = np.nan

                if self._resolved_cameras.get("front") != camera_name:
                    self.logger.info(f"front 深度相机已解析为: {camera_name}")
                    self._resolved_cameras["front"] = camera_name
                return depth
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            self.logger.warning(f"获取前向深度图异常: {last_error}")
        return None

    def estimate_forward_clearance(self, roi_width_ratio: float = 0.5, roi_height_ratio: float = 0.45) -> Optional[float]:
        """估计前方通道净空距离（米），取前视中部 ROI 的低分位数。"""
        depth = self.get_front_depth_map()
        if depth is None:
            return None

        h, w = depth.shape
        roi_w = max(20, int(w * roi_width_ratio))
        roi_h = max(20, int(h * roi_height_ratio))
        x0 = max(0, (w - roi_w) // 2)
        x1 = min(w, x0 + roi_w)
        y0 = max(0, int(h * 0.30))
        y1 = min(h, y0 + roi_h)

        roi = depth[y0:y1, x0:x1]
        valid = roi[np.isfinite(roi)]
        if valid.size < 150:
            return None

        return float(np.percentile(valid, 12))

    def is_forward_path_clear(self, safe_distance_m: float = 12.0) -> tuple:
        """判断前向路径是否安全，返回 (is_clear, clearance_m)。"""
        clearance = self.estimate_forward_clearance()
        if clearance is None:
            return False, None
        return clearance >= safe_distance_m, clearance

    def get_bottom_camera_image(self) -> Optional[np.ndarray]:
        """获取俯视图像。启用仅俯视模式后，不再恢复到前视姿态。"""
        if not self._topdown_mode_enabled:
            self.enable_topdown_only_mode()

        cam = self._topdown_camera_name or self._resolve_bottom_camera() or self._resolve_front_camera()
        if cam is None:
            self.logger.error("无可用相机，无法获取俯视图")
            return None

        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False)
            ], vehicle_name=self.vehicle_name)

            if not responses or len(responses[0].image_data_uint8) == 0:
                self.logger.error("俯视图拍摄失败，图像数据为空")
                return None

            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            return img1d.reshape(response.height, response.width, 3)
        except Exception as exc:
            self.logger.error(f"获取俯视图异常: {exc}")
            return None

    def get_dual_camera_images(self):
        """同时获取前视和俯视图像。"""
        front = self.get_front_camera_image()
        bottom = self.get_bottom_camera_image()
        return front, bottom
    
    def move_up(self, distance: float, speed: float = 1):
        """向上移动"""
        self.logger.info(f"上升 {distance}m")
        self.client.moveByVelocityAsync(
            vx=0, vy=0, vz=-speed,
            duration=distance/speed,
            vehicle_name=self.vehicle_name
        ).join()
    
    def move_down(self, distance: float, speed: float = 1):
        """向下移动"""
        self.logger.info(f"下降 {distance}m")
        self.client.moveByVelocityAsync(
            vx=0, vy=0, vz=speed,
            duration=distance/speed,
            vehicle_name=self.vehicle_name
        ).join()
    
    def rotate_yaw(self, angle: float, speed: float = 30):
        """旋转偏航角（正角度=左转，负角度=右转，NED坐标系修正）"""
        self.logger.info(f"旋转 {angle}度")
        duration = abs(angle) / speed
        yaw_rate = -speed if angle > 0 else speed
        
        self.client.rotateByYawRateAsync(
            yaw_rate=yaw_rate,
            duration=duration,
            vehicle_name=self.vehicle_name
        ).join()
    
    def hover(self, duration: float = 2):
        """悬停，锁定当前位置防止漂移"""
        self.logger.info(f"悬停 {duration}秒")
        pos = self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.position
        self.client.moveToPositionAsync(
            pos.x_val, pos.y_val, pos.z_val, velocity=0.5,
            vehicle_name=self.vehicle_name
        ).join()
        time.sleep(duration)
    
    def get_position(self) -> Tuple[float, float, float]:
        """获取当前位置 (x, y, z)"""
        state = self.client.getMultirotorState(self.vehicle_name)
        pos = state.kinematics_estimated.position
        return pos.x_val, pos.y_val, pos.z_val
    
    def get_altitude(self) -> float:
        """获取当前相对地面高度（米）"""
        pos = self.get_position()
        if self.home_position is not None:
            return self.home_position.z_val - pos[2]
        return -pos[2]
    
    def release_package(self):
        """释放货物"""
        self.logger.info("📦 货物已释放")
        # 可以在这里添加UE4蓝图调用或物理对象生成
        time.sleep(0.5)
    
    def land(self):
        """着陆"""
        self.logger.info("着陆中...")
        self.client.landAsync(vehicle_name=self.vehicle_name).join()
        self.client.armDisarm(False, self.vehicle_name)
        self.logger.info("着陆完成")
    
    def move_to_position(self, x: float, y: float, z: float, velocity: float = 2):
        """移动到绝对世界坐标位置"""
        self.logger.info(f"移动到位置 x={x:.1f} y={y:.1f} z={z:.1f}")
        self.client.moveToPositionAsync(
            x=x, y=y, z=z,
            velocity=velocity,
            vehicle_name=self.vehicle_name
        ).join()

    def get_yaw(self) -> float:
        """获取当前偏航角（弧度）"""
        state = self.client.getMultirotorState(self.vehicle_name)
        q = state.kinematics_estimated.orientation
        # 四元数转偏航角
        siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
        return math.atan2(siny_cosp, cosy_cosp)

    def move_left(self, distance: float, speed: float = 2):
        """向左移动指定距离（相对于无人机朝向）"""
        self.logger.info(f"向左移动 {distance}m")
        x, y, z = self.get_position()
        yaw = self.get_yaw()
        target_x = x + distance * math.sin(yaw)
        target_y = y - distance * math.cos(yaw)
        self.client.moveToPositionAsync(
            x=target_x, y=target_y, z=z,
            velocity=speed,
            vehicle_name=self.vehicle_name
        ).join()

    def move_right(self, distance: float, speed: float = 2):
        """向右移动指定距离（相对于无人机朝向）"""
        self.logger.info(f"向右移动 {distance}m")
        x, y, z = self.get_position()
        yaw = self.get_yaw()
        target_x = x - distance * math.sin(yaw)
        target_y = y + distance * math.cos(yaw)
        self.client.moveToPositionAsync(
            x=target_x, y=target_y, z=z,
            velocity=speed,
            vehicle_name=self.vehicle_name
        ).join()

    def move_forward(self, distance: float, speed: float = 2):
        """向前移动指定距离（相对于无人机朝向）"""
        self.logger.info(f"向前移动 {distance}m")
        x, y, z = self.get_position()
        yaw = self.get_yaw()
        target_x = x + distance * math.cos(yaw)
        target_y = y + distance * math.sin(yaw)
        self.client.moveToPositionAsync(
            x=target_x, y=target_y, z=z,
            velocity=speed,
            vehicle_name=self.vehicle_name
        ).join()

    def move_backward(self, distance: float, speed: float = 2):
        """向后移动指定距离（相对于无人机朝向）"""
        self.logger.info(f"向后移动 {distance}m")
        x, y, z = self.get_position()
        yaw = self.get_yaw()
        target_x = x - distance * math.cos(yaw)
        target_y = y - distance * math.sin(yaw)
        self.client.moveToPositionAsync(
            x=target_x, y=target_y, z=z,
            velocity=speed,
            vehicle_name=self.vehicle_name
        ).join()

    def return_to_home(self):
        """返回起始位置（保持当前高度，不俯冲到地面）"""
        if self.home_position:
            self.logger.info("返回起始点...")
            _, _, current_z = self.get_position()
            self.client.moveToPositionAsync(
                x=self.home_position.x_val,
                y=self.home_position.y_val,
                z=current_z,
                velocity=3,
                vehicle_name=self.vehicle_name
            ).join()