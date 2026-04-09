# WSL 连接 Windows AirSim 指南（镜像网络 / 非镜像网络）

本文用于说明在 WSL 中运行控制脚本、在 Windows 中运行 UE/AirSim 时，如何正确配置连接参数。

## 1. 先判断你是否在使用 mirrored（镜像）网络

如果你在 `C:\Users\<你的用户名>\.wslconfig` 中配置了：

```ini
[wsl2]
networkingMode=mirrored
```

并执行过：

```powershell
wsl --shutdown
```

那么通常可视为 mirrored 网络模式。

## 2. mirrored 模式下如何连接 AirSim

mirrored 模式下，WSL 与 Windows 网络栈更接近，推荐优先用 `localhost` 连接：

```bash
export AIRSIM_IP=localhost
export AIRSIM_PORT=41451
python scripts/tokyo/task_runner.py
```

连通性快速检查：

```bash
timeout 2 bash -lc '</dev/tcp/localhost/41451' && echo "AirSim端口可达" || echo "AirSim端口不可达"
```

## 3. 非 mirrored（常见 NAT）模式下如何连接 AirSim

非 mirrored 时，不建议直接用 `localhost` 连接 Windows AirSim。应改为 Windows 主机 IP。

### 3.1 在 WSL 查询 Windows 主机 IP（常用方式）

方式 A（默认网关，最常用）：

```bash
ip route | awk '/default/ {print $3}'
```
ip route show | grep default | awk '{print $3}'

方式 B（DNS nameserver，很多环境也可用）：

```bash
awk '/nameserver/ {print $2; exit}' /etc/resolv.conf
```

拿到 IP（例如 `172.21.192.1`）后：

```bash
export AIRSIM_IP=172.21.192.1
export AIRSIM_PORT=41451
python scripts/tokyo/task_runner.py
```

### 3.2 端口探测（先探测再跑脚本）

```bash
timeout 2 bash -lc '</dev/tcp/$AIRSIM_IP/41451' && echo "AirSim端口可达" || echo "AirSim端口不可达"
```

## 4. 常见故障与处理

1. 端口不可达  
先确认 UE/Colosseum 已完全进入场景（不是加载中），并确认 AirSim RPC 端口确实为 `41451`。

2. 能连接但无法控制  
检查无人机是否启用 API Control，脚本连接的 vehicle 是否正确。

3. 改过网络模式后突然连不上  
很常见。切换 mirrored/NAT 后，`AIRSIM_IP` 需要同步切换：  
- mirrored：优先 `localhost`  
- NAT：优先 Windows 主机 IP（网关或 resolv.conf 中的地址）

## 5. 建议的日常使用流程

1. 每次开跑前先确认你当前是 mirrored 还是 NAT。  
2. 设置对应 `AIRSIM_IP`。  
3. 用 `timeout ... /dev/tcp/...` 先做端口探测。  
4. 再运行 `python scripts/tokyo/task_runner.py`。

这样可以把“连不上仿真器”的问题提前在启动前暴露出来。
