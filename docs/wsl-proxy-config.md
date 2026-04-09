# WSL2 代理配置指南

## 问题

WSL2 默认使用 NAT 网络模式，WSL 的 `127.0.0.1` 与 Windows 不互通，无法直接使用 Windows 上的代理（如 Clash Verge）。

## 解决方案：WSL2 镜像网络模式

编辑 `C:\Users\PC\.wslconfig`：

```ini
[wsl2]
networkingMode=mirrored
```

然后在 PowerShell 中重启 WSL：

```powershell
wsl --shutdown
```

重新打开 WSL 即可。

## 原理

- NAT 模式（默认）：WSL 有独立虚拟网卡，`127.0.0.1` 指向 WSL 自身
- mirrored 模式：WSL 共享 Windows 网络栈，`127.0.0.1` 两边互通

## 使用代理

mirrored 模式下，WSL 中可直接使用 Windows 代理：

```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
```

## 对 AirSim 的影响

本项目通过 `localhost:41451` 连接 Windows 上的 AirSim 模拟器，mirrored 模式下完全兼容，无需修改。

## 常用测试命令

### 测试代理是否可用

```bash
curl -v --proxy http://127.0.0.1:7897 https://www.google.com
```

### 测试直连（mirrored 模式 + TUN/全局代理）

```bash
curl -v https://www.google.com
```

### 测试 pip 是否能走代理

```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
pip install --dry-run requests
```

### 查看 WSL 网关 IP

```bash
ip route | grep default | awk '{print $3}'
```

### 查看 DNS 服务器（宿主机 IP）

```bash
cat /etc/resolv.conf | grep nameserver
```

### 测试 AirSim 端口是否可达

```bash
curl -v telnet://localhost:41451
```

### 测试 Windows 代理端口是否监听（PowerShell）

```powershell
netstat -ano | findstr 7897
```

## 恢复默认

直接删除 `C:\Users\PC\.wslconfig` 文件，然后 `wsl --shutdown` 重启即可。
