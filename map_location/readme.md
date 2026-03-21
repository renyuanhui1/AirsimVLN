# map_location 脚本说明

这个目录用于集中管理“UE 坐标 <-> 经纬度 <-> Google 地理信息”相关脚本。

## 目录文件

- `geo_calibration_5points.json`：坐标标定参数（核心数据文件）
- `geo_coordinate_mapper.py`：UE XY 转经纬度的独立转换工具
- `live_geo_lookup.py`：实时读取无人机位置并反查地理信息
- `fly_to_place_with_google.py`：输入地点名后飞行到目标，并输出过程地理信息
- `geo_mapper_usage.md`：坐标转换工具的补充示例

## 1) geo_coordinate_mapper.py

### 功能

- 将 UE 世界坐标 `X/Y` 转换为 `lat/lon`。
- 不连接 AirSim，不控制无人机，适合做离线换算和验证。

### 用法

```bash
python map_location/geo_coordinate_mapper.py \
  --calibration map_location/geo_calibration_5points.json \
  --ue-x 34203.0 --ue-y -4299.986554 --ue-z 18632.202573
```

## 2) live_geo_lookup.py

### 功能

- 连接 AirSim，周期性读取无人机位置。
- 将位置转换为经纬度，并调用 Google Reverse Geocoding 输出地址。

### 用法

```bash
export GOOGLE_MAPS_API_KEY='你的Key'
python map_location/live_geo_lookup.py \
  --ip localhost --port 41451 \
  --vehicle Drone1 \
  --interval 8 \
  --output minimal
```

### 常用参数

- `--interval`：刷新间隔（秒）
- `--output minimal|full`：精简/详细输出
- `--language`：Google 返回语言（默认 `zh-CN`）

## 3) fly_to_place_with_google.py

### 功能

- 输入地点名（默认“东京国际论坛”），通过 Google Geocoding 获取目标经纬度。
- 经纬度反算为 UE 目标点，再转换为 AirSim NED 米制坐标并飞行。
- 飞行过程中持续输出当前位置经纬度、距离目标、地理信息，最终输出到达信息。

### 用法

```bash
export GOOGLE_MAPS_API_KEY='你的Key'
python map_location/fly_to_place_with_google.py \
  --place "东京国际论坛" \
  --ip localhost --port 41451 \
  --vehicle Drone1 \
  --takeoff-altitude 200 \
  --velocity 6 \
  --poll-interval 8
```

### 常用参数

- `--place`：目标地点文本
- `--takeoff-altitude`：起飞并保持的飞行高度（米）
- `--arrive-threshold-m`：到达阈值（米）
- `--dry-run`：只解析目标并打印，不实际飞行

## 环境要求

- AirSim 可连接（默认 `AIRSIM_IP=localhost`, `AIRSIM_PORT=41451`）
- 已配置 `GOOGLE_MAPS_API_KEY`
- Python 环境安装依赖：`airsim`、`requests`

## 快速排查

- 报 `REQUEST_DENIED`：检查 Google API Key 是否有效、是否开启 Geocoding API。
- 一直连不上 AirSim：确认 UE/Colosseum 场景已完全加载，端口 `41451` 已监听。
- 路径不对：本目录脚本默认使用 `map_location/geo_calibration_5points.json`。
