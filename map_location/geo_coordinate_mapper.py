import argparse
import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple

EARTH_RADIUS_M = 6378137.0


@dataclass
class AffineParams:
    east_ax: float
    east_by: float
    east_c: float
    north_ax: float
    north_by: float
    north_c: float


class GeoCoordinateMapper:
    """Independent mapper: UE world XY -> WGS84 lat/lon using affine calibration."""

    def __init__(self, calibration_file: str):
        with open(calibration_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        ref = cfg["reference"]
        self.ref_lat = float(ref["lat"])
        self.ref_lon = float(ref["lon"])

        p = cfg["affine_ue_xy_to_enu_m"]
        self.params = AffineParams(
            east_ax=float(p["east_ax"]),
            east_by=float(p["east_by"]),
            east_c=float(p["east_c"]),
            north_ax=float(p["north_ax"]),
            north_by=float(p["north_by"]),
            north_c=float(p["north_c"]),
        )

    def ue_xy_to_enu_m(self, ue_x: float, ue_y: float) -> Tuple[float, float]:
        p = self.params
        east_m = p.east_ax * ue_x + p.east_by * ue_y + p.east_c
        north_m = p.north_ax * ue_x + p.north_by * ue_y + p.north_c
        return east_m, north_m

    def enu_m_to_latlon(self, east_m: float, north_m: float) -> Tuple[float, float]:
        lat = self.ref_lat + math.degrees(north_m / EARTH_RADIUS_M)
        lon = self.ref_lon + math.degrees(east_m / (EARTH_RADIUS_M * math.cos(math.radians(self.ref_lat))))
        return lat, lon

    def ue_xy_to_latlon(self, ue_x: float, ue_y: float) -> Tuple[float, float]:
        east_m, north_m = self.ue_xy_to_enu_m(ue_x, ue_y)
        return self.enu_m_to_latlon(east_m, north_m)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert UE XY to WGS84 lat/lon using calibration json.")
    parser.add_argument("--calibration", required=True, help="Path to calibration json")
    parser.add_argument("--ue-x", type=float, required=True, help="UE world X")
    parser.add_argument("--ue-y", type=float, required=True, help="UE world Y")
    parser.add_argument("--ue-z", type=float, default=None, help="UE world Z (optional passthrough)")
    args = parser.parse_args()

    mapper = GeoCoordinateMapper(args.calibration)
    lat, lon = mapper.ue_xy_to_latlon(args.ue_x, args.ue_y)

    if args.ue_z is None:
        print(f"lat={lat:.10f}, lon={lon:.10f}")
    else:
        print(f"lat={lat:.10f}, lon={lon:.10f}, ue_z={args.ue_z:.3f}")


if __name__ == "__main__":
    main()
