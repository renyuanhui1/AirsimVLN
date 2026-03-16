# AirsimVLN - 卫星地图视觉语言导航系统

基于 Colosseum + UE 仿真环境的无人机视觉语言导航系统。通过大模型解析自然语言指令，结合卫星地图模板匹配，实现无人机自主搜索与目标定位。

## 项目结构

```
AirsimVLN/
├── scripts/
│   ├── tokyo/          # 东京场景导航脚本
│   └── indoor/         # 室内场景导航脚本
├── docs/               # 项目文档
├── requirements.txt    # Python 依赖
└── claude.md           # Claude Code 配置
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export QWEN_API_KEY=你的阿里云百炼Key
export AIRSIM_IP=172.21.192.1
export AIRSIM_PORT=41451
```

### 3. 启动 UE 仿真场景，然后运行

```bash
python scripts/tokyo/tokyo_red_car_vln.py
```

## 文档

- [系统设计文档](docs/卫星地图视觉语言导航系统.md)
- [东京场景运行说明](docs/RUN_TOKYO_VLN.md)
- [UE 配置笔记](docs/ue.md)
