import logging
import math
import time
from typing import Dict, Iterable, Optional, Tuple

import airsim
import numpy as np


class AirSimController:
    FRONT_CAMERA_CANDIDATES = ("front_center", "front", "0")
    BOTTOM_CAMERA_CANDIDATES = ("bottom_center", "downward_center", "bottom", "down", "1")

    def __init__(self, ip: str = "192.168.31.178", port: int = 41451, vehicle_name: str = "Drone1"):
        self.client = airsim.MultirotorClient(ip=ip, port=port)
        self.client.confirmConnection()
        self.vehicle_name = vehicle_name
        self.logger = logging.getLogger(__name__)
        self.home_position = None
        self._resolved_cameras: Dict[str, str] = {}
        self._front_camera_name = None
        self._topdown_camera_name = None
        self._topdown_mode_enabled = False
        self._front_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))
        self._down_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(-90), 0, 0))

    def arm_and_takeoff(self, altitude: float = 5.0) -> None:
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        self.home_position = self.client.getMultirotorState(self.vehicle_name).kinematics_estimated.position

        self.logger.info("起飞中...")
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(2)

        target_z = self.home_position.z_val - altitude
        self.client.moveToZAsync(z=target_z, velocity=2, vehicle_name=self.vehicle_name).join()
        time.sleep(2)

        current_z = -self.client.getMultirotorState(self.vehicle_name).kinematics_estimated.position.z_val
        self.logger.info("起飞完成，当前高度: %.1fm", current_z)

    def _fetch_scene_response(self, camera_name: str):
        responses = self.client.simGetImages(
            [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)],
            vehicle_name=self.vehicle_name,
        )
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
            self.logger.warning("获取相机 %s 图像异常: %s", camera_name, exc)
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
        if self._front_camera_name:
            return self._front_camera_name
        for name in self.FRONT_CAMERA_CANDIDATES:
            image = self.get_named_camera_image(name)
            if image is not None:
                self._front_camera_name = name
                self.logger.info("前视相机已解析为: %s", name)
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
                self.logger.info("底视相机已解析为: %s", name)
                return name
        return None

    def enable_topdown_only_mode(self) -> bool:
        native_bottom = self._resolve_bottom_camera()
        if native_bottom is not None:
            self._topdown_camera_name = native_bottom
            self._topdown_mode_enabled = True
            self.logger.info("已启用底视模式，使用原生底视相机: %s", native_bottom)
            return True

        cam = self._resolve_front_camera()
        if cam is None:
            self.logger.error("未找到可用相机，无法启用俯视模式")
            return False

        try:
            self.client.simSetCameraPose(cam, self._down_pose, vehicle_name=self.vehicle_name)
            self._topdown_camera_name = cam
            self._topdown_mode_enabled = True
            self.logger.info("未找到原生底视相机，已退回前视相机俯视方案: %s", cam)
            return True
        except Exception as exc:
            self.logger.error("启用仅俯视模式失败: %s", exc)
            return False

    def _get_scene_image_from_candidates(self, role: str, camera_names: Iterable[str]) -> Optional[np.ndarray]:
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
                    self.logger.info("%s 相机已解析为: %s", role, camera_name)
                self._resolved_cameras[role] = camera_name
                return img_rgb
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            self.logger.error("获取 %s 相机图像异常: %s", role, last_error)
        else:
            self.logger.error("获取 %s 相机图像失败，已尝试: %s", role, candidates)
        return None

    def get_front_camera_image(self) -> Optional[np.ndarray]:
        if self._topdown_mode_enabled:
            self.logger.info("仅俯视模式已启用，跳过前视图像采集")
            return None
        return self._get_scene_image_from_candidates("front", self.FRONT_CAMERA_CANDIDATES)

    def get_bottom_camera_image(self) -> Optional[np.ndarray]:
        if not self._topdown_mode_enabled:
            if not self.enable_topdown_only_mode():
                self.logger.error("俯视模式启用失败，拒绝返回非俯视图像")
                return None

        cam = self._topdown_camera_name
        if cam is None:
            self.logger.error("无可用相机，无法获取俯视图")
            return None

        try:
            response = self._fetch_scene_response(cam)
            if response is None:
                self.logger.error("俯视图拍摄失败，图像数据为空")
                return None
            return self._decode_scene_response(response)
        except Exception as exc:
            self.logger.error("获取俯视图异常: %s", exc)
            return None

    def move_up(self, distance: float, speed: float = 1.0) -> None:
        self.logger.info("上升 %.1fm", distance)
        self.client.moveByVelocityAsync(
            vx=0,
            vy=0,
            vz=-speed,
            duration=distance / speed,
            vehicle_name=self.vehicle_name,
        ).join()

    def move_down(self, distance: float, speed: float = 1.0) -> None:
        self.logger.info("下降 %.1fm", distance)
        self.client.moveByVelocityAsync(
            vx=0,
            vy=0,
            vz=speed,
            duration=distance / speed,
            vehicle_name=self.vehicle_name,
        ).join()

    def rotate_yaw(self, angle: float, speed: float = 30.0) -> None:
        self.logger.info("旋转 %.1f度", angle)
        duration = abs(angle) / speed
        yaw_rate = -speed if angle > 0 else speed
        self.client.rotateByYawRateAsync(
            yaw_rate=yaw_rate,
            duration=duration,
            vehicle_name=self.vehicle_name,
        ).join()

    def hover(self, duration: float = 2.0) -> None:
        self.logger.info("悬停 %.1f秒", duration)
        pos = self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.position
        self.client.moveToPositionAsync(
            pos.x_val,
            pos.y_val,
            pos.z_val,
            velocity=0.5,
            vehicle_name=self.vehicle_name,
        ).join()
        time.sleep(duration)

    def get_position(self) -> Tuple[float, float, float]:
        state = self.client.getMultirotorState(self.vehicle_name)
        pos = state.kinematics_estimated.position
        return pos.x_val, pos.y_val, pos.z_val

    def get_altitude(self) -> float:
        pos = self.get_position()
        if self.home_position is not None:
            return self.home_position.z_val - pos[2]
        return -pos[2]

    def land(self) -> None:
        self.logger.info("着陆中...")
        self.client.landAsync(vehicle_name=self.vehicle_name).join()
        self.client.armDisarm(False, self.vehicle_name)
        self.logger.info("着陆完成")

    def move_to_position(self, x: float, y: float, z: float, velocity: float = 2.0) -> None:
        self.logger.info("移动到位置 x=%.1f y=%.1f z=%.1f", x, y, z)
        self.client.moveToPositionAsync(
            x=x,
            y=y,
            z=z,
            velocity=velocity,
            vehicle_name=self.vehicle_name,
        ).join()

    def get_yaw(self) -> float:
        state = self.client.getMultirotorState(self.vehicle_name)
        q = state.kinematics_estimated.orientation
        siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
        return math.atan2(siny_cosp, cosy_cosp)

    def move_left(self, distance: float, speed: float = 2.0) -> None:
        self.logger.info("向左移动 %.1fm", distance)
        x, y, z = self.get_position()
        yaw = self.get_yaw()
        target_x = x + distance * math.sin(yaw)
        target_y = y - distance * math.cos(yaw)
        self.client.moveToPositionAsync(
            x=target_x,
            y=target_y,
            z=z,
            velocity=speed,
            vehicle_name=self.vehicle_name,
        ).join()

    def move_right(self, distance: float, speed: float = 2.0) -> None:
        self.logger.info("向右移动 %.1fm", distance)
        x, y, z = self.get_position()
        yaw = self.get_yaw()
        target_x = x - distance * math.sin(yaw)
        target_y = y + distance * math.cos(yaw)
        self.client.moveToPositionAsync(
            x=target_x,
            y=target_y,
            z=z,
            velocity=speed,
            vehicle_name=self.vehicle_name,
        ).join()

    def move_forward(self, distance: float, speed: float = 2.0) -> None:
        self.logger.info("向前移动 %.1fm", distance)
        x, y, z = self.get_position()
        yaw = self.get_yaw()
        target_x = x + distance * math.cos(yaw)
        target_y = y + distance * math.sin(yaw)
        self.client.moveToPositionAsync(
            x=target_x,
            y=target_y,
            z=z,
            velocity=speed,
            vehicle_name=self.vehicle_name,
        ).join()

    def move_backward(self, distance: float, speed: float = 2.0) -> None:
        self.logger.info("向后移动 %.1fm", distance)
        x, y, z = self.get_position()
        yaw = self.get_yaw()
        target_x = x - distance * math.cos(yaw)
        target_y = y - distance * math.sin(yaw)
        self.client.moveToPositionAsync(
            x=target_x,
            y=target_y,
            z=z,
            velocity=speed,
            vehicle_name=self.vehicle_name,
        ).join()
