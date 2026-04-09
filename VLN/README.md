# VLN Bundle

## 目录说明

- `airsim_controller.py`: AirSim 控制脚本，负责起飞、移动、取图、降落
- `qwen_agent.py`: 本地模型智能体，包含提示词、接口调用和结果解析
- `task.py`: 主任务脚本，执行炸毁区域搜索、飞机搜索和毁伤评估
- `simple_fly_capture.py`: 起飞到 20 米并保存一张俯视图的测试脚本
- `input/`: 输入目录
- `output/`: 输出目录
- `requirements.txt`: Python 依赖

## 环境变量

- `AIRSIM_IP`: AirSim 服务地址，默认 `192.168.31.178`
- `AIRSIM_PORT`: AirSim 端口，默认 `41451`
- `AIRSIM_VEHICLE_NAME`: 飞行器名称，默认 `Drone1`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行任务

```bash
python task.py
```

如需指定本地模型地址和模型名：

```bash
python task.py --base-url http://127.0.0.1:11434/api/chat --model qwen3.5:9b
```

服务器部署示例：

```bash
export AIRSIM_IP=192.168.31.178
export AIRSIM_PORT=41451
python task.py --base-url http://127.0.0.1:11434/api/chat --model qwen3.5:9b
```

## 查询 AirSim 主机 IP

如果后续更换电脑，需要先查询运行 AirSim 的那台主机 IP，再把它填到 `AIRSIM_IP`。

Linux:

```bash
hostname -I
```

或者：

```bash
ip addr
```

Windows:

```powershell
ipconfig
```

拿到主机 IP 后，可以在服务器上测试 AirSim 端口是否可达：

```bash
nc -vz 主机IP 41451
```

如果显示连接成功，这个 IP 就可以作为 `AIRSIM_IP` 使用。

## 运行相机测试

```bash
python simple_fly_capture.py
```
