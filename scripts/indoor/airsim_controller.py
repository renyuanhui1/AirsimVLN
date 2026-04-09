# scripts/airsim.py
import math
import airsim
import numpy as np
import time
from typing import Tuple, Dict, Optional
import logging

class AirSimController:
    def __init__(self, ip: str = "localhost", port: int = 41451):
        """
        连接到 Windows 中的 AirSim
        """
        self.client = airsim.MultirotorClient(ip=ip, port=port)
        self.client.confirmConnection()
        self.vehicle_name = "Drone1"  # 楼宇送货用多旋翼更合适
        self.logger = logging.getLogger(__name__)
        self.home_position = None
        
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
    
    def get_front_camera_image(self, width: int = 640, height: int = 480) -> Optional[np.ndarray]:
        """获取前置相机RGB图像"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
            ], vehicle_name=self.vehicle_name)
            
            if not responses or len(responses[0].image_data_uint8) == 0:
                self.logger.error("获取图像失败")
                return None
            
            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            
            return img_rgb
        except Exception as e:
            self.logger.error(f"获取相机图像异常: {e}")
            return None
    
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
        """旋转偏航角"""
        self.logger.info(f"旋转 {angle}度")
        duration = abs(angle) / speed
        yaw_rate = speed if angle > 0 else -speed
        
        self.client.rotateByYawRateAsync(
            yaw_rate=yaw_rate,
            duration=duration,
            vehicle_name=self.vehicle_name
        ).join()
    
    def hover(self, duration: float = 2):
        """悬停"""
        self.logger.info(f"悬停 {duration}秒")
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
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
        """返回起始位置"""
        if self.home_position:
            self.logger.info("返回起始点...")
            self.client.moveToPositionAsync(
                x=self.home_position.x_val,
                y=self.home_position.y_val,
                z=self.home_position.z_val,
                velocity=3,
                vehicle_name=self.vehicle_name
            ).join()