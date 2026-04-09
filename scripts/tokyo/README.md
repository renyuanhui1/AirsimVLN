# 东京红车俯视视觉语言导航运行说明

## 功能说明

脚本入口：scripts/tokyo/task_runner.py

该脚本用于在 Colosseum / AirSim UE5 场景中控制无人机执行结构化任务：

- 起飞到较高搜索高度
- 使用底部俯视相机沿道路搜索
- 按预设步骤穿越路口或在路口转向
- 在目标区域搜索红色车辆
- 飞到红车正上方附近后悬停

## 运行前准备

确保以下条件满足：

1. UE5 中的 Colosseum 场景已经启动
2. 无人机启用了 API Control
3. Python 依赖已经安装
4. 已配置 Qwen API Key

## 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

如果缺少 dotenv，也可以单独安装：

```bash
pip install python-dotenv
```

## 环境变量配置

最少需要配置：

```bash
export QWEN_API_KEY=你的阿里云百炼Key
export AIRSIM_IP=172.21.192.1
export AIRSIM_PORT=41451
```

可选导航参数：

```bash
export VLN_SEARCH_ALTITUDE=35
export VLN_TRACK_ALTITUDE=18
export VLN_MIN_ALTITUDE=12
export VLN_MAX_ALTITUDE=45
export VLN_MAX_STEPS=40
```

参数说明：

- VLN_SEARCH_ALTITUDE：初始搜索高度，默认 35 米
- VLN_TRACK_ALTITUDE：发现目标后的跟踪高度，默认 18 米
- VLN_MIN_ALTITUDE：允许下降的最低高度，默认 12 米
- VLN_MAX_ALTITUDE：允许上升的最高高度，默认 45 米
- VLN_MAX_STEPS：最大决策步数，默认 40

## 运行命令

在项目根目录执行：

```bash
python scripts/tokyo/task_runner.py
```

## 运行流程

脚本会按以下逻辑执行：

1. 连接 Colosseum / AirSim
2. 起飞到搜索高度
3. 启动网页监控界面
4. 按任务步骤执行道路跟随、穿越路口、路口转向与目标搜索
5. 在每个阶段读取底视相机图像并调用 Qwen 进行局部决策
6. 锁定红车后下降到跟踪高度
7. 飞到红车正上方附近并悬停

## 相关代码位置

- 主任务脚本：scripts/tokyo/task_runner.py
- 飞行控制器：scripts/tokyo/airsim_controller.py
- Qwen 视觉决策：scripts/tokyo/qwen.py
- 网页监控：scripts/tokyo/web_monitor.py

## 常见问题排查

### 1. 连接不上仿真器

检查：

- AIRSIM_IP 是否正确
- AIRSIM_PORT 是否正确
- UE5 场景是否已经完全启动
- Windows 与 WSL / Linux 网络是否互通

可以直接运行主任务脚本；如果连接失败，按终端日志中的报错信息检查 `AIRSIM_IP`、`AIRSIM_PORT`、UE 场景启动状态以及 WSL/Windows 网络连通性。

### 2. 获取不到底视相机图像

当前代码会自动尝试以下底视相机名称：

- bottom_center
- downward_center
- bottom
- down
- 1

如果你的 Colosseum 蓝图里相机名不同，需要修改 scripts/tokyo/airsim_controller.py 中的 `BOTTOM_CAMERA_CANDIDATES`。

### 3. 没有 Qwen Key

如果没有配置 QWEN_API_KEY，脚本会直接报错并退出。

### 4. 无人机高度不合适

优先调整：

- VLN_SEARCH_ALTITUDE
- VLN_TRACK_ALTITUDE
- VLN_MIN_ALTITUDE

东京卫星地图楼体较高时，建议把搜索高度提高到 35 到 50 米之间再测试。

## 建议的首次运行方式

第一次运行建议使用下面这组参数：

```bash
export QWEN_API_KEY=你的阿里云百炼Key
export AIRSIM_IP=172.21.192.1
export AIRSIM_PORT=41451
export VLN_SEARCH_ALTITUDE=35
export VLN_TRACK_ALTITUDE=18
export VLN_MIN_ALTITUDE=12
export VLN_MAX_STEPS=40
python scripts/tokyo/task_runner.py
```

## 输出说明

运行过程中：

- 终端会输出每一步动作决策
- 图像会保存到 output 目录
- 如果成功，最后会在红车正上方附近悬停后降落

如果你希望保留悬停而不是自动降落，可以继续修改主脚本里的 finally 逻辑。
