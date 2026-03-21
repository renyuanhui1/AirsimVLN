import argparse
import os
import time
from typing import Tuple

import requests
import airsim

from geo_coordinate_mapper import GeoCoordinateMapper


def _cjk_score(text: str) -> int:
    score = 0
    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            score += 2
        elif 0x3040 <= code <= 0x30FF:
            score += 1
    return score


def reverse_geocode(lat: float, lon: float, api_key: str, language: str = "zh-CN") -> Tuple[str, str]:
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    try:
        resp = requests.get(
            url,
            params={
                "latlng": f"{lat:.10f},{lon:.10f}",
                "language": language,
                "key": api_key,
            },
            timeout=10,
        )
        data = resp.json()
    except Exception as exc:
        return "REQUEST_EXCEPTION", f"请求异常: {exc}"

    status = data.get("status", "UNKNOWN")
    if status != "OK":
        err = data.get("error_message", "")
        return status, err or "查询失败"

    results = data.get("results", [])
    if not results:
        return "OK", "无结果"

    # 优先选择中文信息更丰富的结果，避免总是命中英文第一条。
    best = max(results, key=lambda r: _cjk_score(r.get("formatted_address", "")))
    return "OK", best.get("formatted_address", "无格式化地址")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read UE position from AirSim, convert to lat/lon, and query Google reverse geocoding."
    )
    parser.add_argument(
        "--calibration",
        default="map_location/geo_calibration_5points.json",
        help="Path to calibration json file",
    )
    parser.add_argument("--ip", default=os.getenv("AIRSIM_IP", "localhost"), help="AirSim RPC IP")
    parser.add_argument("--port", type=int, default=int(os.getenv("AIRSIM_PORT", "41451")), help="AirSim RPC port")
    parser.add_argument("--vehicle", default="Drone1", help="Vehicle name in AirSim")
    parser.add_argument("--interval", type=float, default=8.0, help="Polling interval seconds")
    parser.add_argument("--language", default="zh-CN", help="Google geocode language")
    parser.add_argument(
        "--output",
        choices=["minimal", "full"],
        default="minimal",
        help="minimal: only lat/lon + place text; full: include UE xyz and status",
    )
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 GOOGLE_MAPS_API_KEY，请先 export GOOGLE_MAPS_API_KEY=AIza...")

    mapper = GeoCoordinateMapper(args.calibration)

    client = airsim.MultirotorClient(ip=args.ip, port=args.port)
    client.confirmConnection()

    try:
        while True:
            state = client.getMultirotorState(vehicle_name=args.vehicle)
            pos = state.kinematics_estimated.position
            ue_x, ue_y, ue_z = pos.x_val, pos.y_val, pos.z_val

            lat, lon = mapper.ue_xy_to_latlon(ue_x, ue_y)
            status, place_text = reverse_geocode(lat, lon, api_key=api_key, language=args.language)

            if args.output == "minimal":
                print(f"经纬度: {lat:.8f}, {lon:.8f} | 位置: {place_text}")
            else:
                print(
                    f"UE坐标=({ue_x:10.3f}, {ue_y:10.3f}, {ue_z:10.3f}) | "
                    f"经纬度=({lat:.8f}, {lon:.8f}) | 状态={status} | 位置={place_text}"
                )

            time.sleep(max(0.2, args.interval))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
