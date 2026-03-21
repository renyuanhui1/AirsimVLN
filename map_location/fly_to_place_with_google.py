import argparse
import json
import math
import os
import time
from typing import Dict, Tuple

import airsim
import requests

EARTH_RADIUS_M = 6378137.0
UE_UNITS_PER_METER = 100.0  # UE坐标常用厘米，AirSim运动控制/状态是米
DEFAULT_PLACE = "东京国际论坛"

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def cjk_score(text: str) -> int:
    score = 0
    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            score += 2
        elif 0x3040 <= code <= 0x30FF:
            score += 1
    return score


def is_valid_building_name(name: str) -> bool:
    if not name:
        return False
    t = name.strip()
    if not t:
        return False

    # 过滤纯数字/门牌号样式（例如 "1", "2-3"）
    compact = t.replace("-", "").replace("_", "").replace(" ", "")
    if compact.isdigit():
        return False

    # 过滤 Plus Code 这类占位名称（例如 "MQH5+G9"）
    if "+" in t and len(t) <= 12:
        return False

    return True


class AffineGeoMapper:
    """UE XY <-> lat/lon mapper from affine calibration json."""

    def __init__(self, calibration_file: str):
        with open(calibration_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        ref = cfg["reference"]
        self.ref_lat = float(ref["lat"])
        self.ref_lon = float(ref["lon"])

        p = cfg["affine_ue_xy_to_enu_m"]
        self.east_ax = float(p["east_ax"])
        self.east_by = float(p["east_by"])
        self.east_c = float(p["east_c"])
        self.north_ax = float(p["north_ax"])
        self.north_by = float(p["north_by"])
        self.north_c = float(p["north_c"])

        det = self.east_ax * self.north_by - self.east_by * self.north_ax
        if abs(det) < 1e-12:
            raise ValueError("标定矩阵不可逆，无法做 lat/lon -> UE 反算")
        self._inv_det = 1.0 / det

    def ue_xy_to_latlon(self, ue_x: float, ue_y: float) -> Tuple[float, float]:
        east_m = self.east_ax * ue_x + self.east_by * ue_y + self.east_c
        north_m = self.north_ax * ue_x + self.north_by * ue_y + self.north_c
        lat = self.ref_lat + math.degrees(north_m / EARTH_RADIUS_M)
        lon = self.ref_lon + math.degrees(east_m / (EARTH_RADIUS_M * math.cos(math.radians(self.ref_lat))))
        return lat, lon

    def latlon_to_ue_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        dlat = math.radians(lat - self.ref_lat)
        dlon = math.radians(lon - self.ref_lon)
        east_m = dlon * EARTH_RADIUS_M * math.cos(math.radians(self.ref_lat))
        north_m = dlat * EARTH_RADIUS_M

        rhs_e = east_m - self.east_c
        rhs_n = north_m - self.north_c

        ue_x = (self.north_by * rhs_e - self.east_by * rhs_n) * self._inv_det
        ue_y = (-self.north_ax * rhs_e + self.east_ax * rhs_n) * self._inv_det
        return ue_x, ue_y


def google_geocode_address(place_text: str, api_key: str, language: str = "zh-CN") -> Dict:
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    resp = requests.get(
        url,
        params={"address": place_text, "language": language, "key": api_key},
        timeout=12,
    )
    data = resp.json()
    status = data.get("status", "UNKNOWN")
    if status != "OK" or not data.get("results"):
        raise RuntimeError(f"地址解析失败: status={status}, error={data.get('error_message', '')}")

    best = max(data["results"], key=lambda r: cjk_score(r.get("formatted_address", "")))
    loc = best["geometry"]["location"]
    return {
        "lat": float(loc["lat"]),
        "lon": float(loc["lng"]),
        "formatted_address": best.get("formatted_address", ""),
        "place_id": best.get("place_id", ""),
        "types": best.get("types", []),
    }


def google_reverse_geocode(lat: float, lon: float, api_key: str, language: str = "zh-CN") -> Dict:
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    resp = requests.get(
        url,
        params={"latlng": f"{lat:.10f},{lon:.10f}", "language": language, "key": api_key},
        timeout=12,
    )
    data = resp.json()
    status = data.get("status", "UNKNOWN")
    if status != "OK" or not data.get("results"):
        return {
            "status": status,
            "address": data.get("error_message", "无结果"),
            "building_name": "未知",
            "types": [],
        }

    results = data["results"]

    def rank(r: Dict) -> Tuple[int, int]:
        types = set(r.get("types", []))
        if "point_of_interest" in types:
            pri = 6
        elif "premise" in types:
            pri = 5
        elif "street_address" in types:
            pri = 4
        elif "route" in types:
            pri = 3
        elif "plus_code" in types:
            pri = 1
        else:
            pri = 0
        return pri, cjk_score(r.get("formatted_address", ""))

    ordered = sorted(results, key=rank, reverse=True)
    best = ordered[0]

    # 地址输出优先具体地点（楼宇/POI/街道），同优先级再偏中文
    def address_rank(r: Dict) -> Tuple[int, int]:
        types = set(r.get("types", []))

        # 抑制过于泛化的结果
        if "plus_code" in types:
            pri = -2
        elif "postal_code" in types:
            pri = -1
        elif "administrative_area_level_1" in types or "locality" in types:
            pri = 0
        elif "route" in types:
            pri = 3
        elif "street_address" in types:
            pri = 4
        elif "premise" in types:
            pri = 5
        elif "point_of_interest" in types:
            pri = 6
        else:
            pri = 1

        return pri, cjk_score(r.get("formatted_address", ""))

    best_address_result = max(results, key=address_rank)

    building_name = "未知"
    # 在所有候选里找最可靠名称，避免取到纯数字门牌号
    wanted_types = ["point_of_interest", "establishment", "premise", "subpremise"]
    for r in ordered:
        components = r.get("address_components", [])
        for wt in wanted_types:
            for c in components:
                c_types = set(c.get("types", []))
                if wt in c_types:
                    candidate = c.get("long_name", "").strip()
                    if is_valid_building_name(candidate):
                        building_name = candidate
                        break
            if building_name != "未知":
                break
        if building_name != "未知":
            break

    if building_name == "未知":
        formatted = best.get("formatted_address", "")
        head = formatted.split(",")[0].strip() if formatted else ""
        if is_valid_building_name(head):
            building_name = head

    return {
        "status": "OK",
        "address": best_address_result.get("formatted_address", "无格式化地址"),
        "building_name": building_name,
        "types": best.get("types", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Google地点 -> 经纬度 -> UE坐标飞行，并输出过程地理信息")
    parser.add_argument("--place", default=DEFAULT_PLACE, help="目标地点文本（默认使用脚本内置地点）")
    parser.add_argument("--calibration", default="map_location/geo_calibration_5points.json", help="标定文件路径")
    parser.add_argument("--ip", default=os.getenv("AIRSIM_IP", "localhost"), help="AirSim RPC IP")
    parser.add_argument("--port", type=int, default=int(os.getenv("AIRSIM_PORT", "41451")), help="AirSim RPC端口")
    parser.add_argument("--vehicle", default="Drone1", help="车辆名称")
    parser.add_argument("--velocity", type=float, default=6.0, help="moveToPosition 速度")
    parser.add_argument("--takeoff-altitude", type=float, default=200.0, help="全程保持飞行高度（米）")
    parser.add_argument("--poll-interval", type=float, default=8.0, help="过程信息刷新间隔")
    parser.add_argument("--arrive-threshold-m", type=float, default=8.0, help="到达判定（经纬度球面距离，米）")
    parser.add_argument("--language", default="zh-CN", help="Google API 语言")
    parser.add_argument("--dry-run", action="store_true", help="只计算目标并打印，不执行飞行")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 GOOGLE_MAPS_API_KEY，请先 export GOOGLE_MAPS_API_KEY=AIza...")

    mapper = AffineGeoMapper(args.calibration)

    target_geo = google_geocode_address(args.place, api_key, language=args.language)
    target_lat = target_geo["lat"]
    target_lon = target_geo["lon"]
    target_ue_x, target_ue_y = mapper.latlon_to_ue_xy(target_lat, target_lon)
    target_ned_x = target_ue_x / UE_UNITS_PER_METER
    target_ned_y = target_ue_y / UE_UNITS_PER_METER
    target_info = google_reverse_geocode(target_lat, target_lon, api_key, language=args.language)

    print("=" * 72)
    print(f"目标地点: {args.place}")
    print(f"Google解析: {target_geo['formatted_address']}")
    print(f"目标经纬度: {target_lat:.8f}, {target_lon:.8f}")
    print(f"反算UE目标XY: {target_ue_x:.3f}, {target_ue_y:.3f}")
    print(f"目标NED(米): {target_ned_x:.3f}, {target_ned_y:.3f}")
    print(f"目标点地理信息: {target_info['address']}")
    print(f"目标点建筑名称: {target_info['building_name']}")
    print("=" * 72)

    if args.dry_run:
        print("dry-run模式：未执行飞行")
        return

    client = airsim.MultirotorClient(ip=args.ip, port=args.port)
    client.confirmConnection()
    client.enableApiControl(True, args.vehicle)
    client.armDisarm(True, args.vehicle)

    state = client.getMultirotorState(vehicle_name=args.vehicle)
    home_pos = state.kinematics_estimated.position

    takeoff_z = home_pos.z_val - args.takeoff_altitude
    print(f"先起飞到 {args.takeoff_altitude:.1f}m，高度目标z={takeoff_z:.3f}")
    client.takeoffAsync(vehicle_name=args.vehicle).join()
    client.moveToZAsync(takeoff_z, 2.5, vehicle_name=args.vehicle).join()

    print(f"直飞目标点: NED=({target_ned_x:.3f}, {target_ned_y:.3f}, {takeoff_z:.3f}), 速度={args.velocity}")
    client.moveToPositionAsync(
        target_ned_x,
        target_ned_y,
        takeoff_z,
        args.velocity,
        vehicle_name=args.vehicle,
    )

    step = 0
    while True:
        step += 1
        state = client.getMultirotorState(vehicle_name=args.vehicle)
        pos = state.kinematics_estimated.position

        cur_ue_x = pos.x_val * UE_UNITS_PER_METER
        cur_ue_y = pos.y_val * UE_UNITS_PER_METER
        cur_lat, cur_lon = mapper.ue_xy_to_latlon(cur_ue_x, cur_ue_y)
        geo_info = google_reverse_geocode(cur_lat, cur_lon, api_key, language=args.language)
        dist_m = haversine_m(cur_lat, cur_lon, target_lat, target_lon)

        print(
            f"[过程 {step:03d}] 经纬度={cur_lat:.8f}, {cur_lon:.8f} | "
            f"距目标={dist_m:.1f}m | 位置={geo_info['address']}"
        )

        if dist_m <= args.arrive_threshold_m:
            final_info = google_reverse_geocode(cur_lat, cur_lon, api_key, language=args.language)
            print("=" * 72)
            print("到达目标附近")
            print(f"最终经纬度: {cur_lat:.8f}, {cur_lon:.8f}")
            print(f"最终地址: {final_info['address']}")
            print(f"最终建筑名称: {final_info['building_name']}")
            print("=" * 72)
            client.hoverAsync(vehicle_name=args.vehicle).join()
            break

        time.sleep(max(0.5, args.poll_interval))


if __name__ == "__main__":
    main()
