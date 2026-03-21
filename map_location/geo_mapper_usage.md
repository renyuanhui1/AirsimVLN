# 独立坐标映射工具使用说明

本工具不会修改你现有控制脚本，只是单独把 UE 坐标转换为经纬度。

## 文件

- `geo_coordinate_mapper.py`
- `geo_calibration_5points.json`

## 单次转换示例

```bash
python map_location/geo_coordinate_mapper.py \
  --calibration map_location/geo_calibration_5points.json \
  --ue-x 34203.0 --ue-y -4299.986554 --ue-z 18632.202573
```

输出示例：

```text
lat=35.xxxxx, lon=139.xxxxx, ue_z=xxxxx.xxx
```

## 在你自己的脚本中调用（可选）

```python
from geo_coordinate_mapper import GeoCoordinateMapper

mapper = GeoCoordinateMapper('map_location/geo_calibration_5points.json')
lat, lon = mapper.ue_xy_to_latlon(ue_x, ue_y)
```

## 说明

- 当前标定仅用于平面定位：`UE(X,Y) -> (lat, lon)`。
- `Z` 暂不参与拟合。
- 若你后续新增控制点，建议重拟合参数并替换 json。
