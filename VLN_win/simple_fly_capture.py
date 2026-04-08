import argparse
import logging
import os

import cv2
from dotenv import load_dotenv

from airsim_controller import AirSimController


load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _save_image(image, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"保存图像失败: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="起飞到20米并保存一张俯视图")
    parser.add_argument("--altitude", type=float, default=20.0, help="起飞高度（米）")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "output", "simple_fly_capture"),
        help="图像输出目录",
    )
    args = parser.parse_args()

    airsim_ip = os.getenv("AIRSIM_IP", "127.0.0.1")
    airsim_port = int(os.getenv("AIRSIM_PORT", "41451"))
    vehicle_name = os.getenv("AIRSIM_VEHICLE_NAME", "Drone1")

    ctrl = AirSimController(ip=airsim_ip, port=airsim_port, vehicle_name=vehicle_name)

    try:
        ctrl.arm_and_takeoff(altitude=args.altitude)
        ctrl.enable_topdown_only_mode()
        ctrl.hover(duration=1.0)

        image = ctrl.get_bottom_camera_image()
        if image is not None:
            path = _save_image(image, args.output_dir, "topdown_20m.jpg")
            logger.info("20米高度俯视图已保存: %s", path)
        else:
            logger.warning("20米高度未获取到俯视图")

        ctrl.hover(duration=1.0)
    finally:
        ctrl.land()


if __name__ == "__main__":
    main()
