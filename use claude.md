# Claude Code 安装教程 (WSL)

## 前置要求

- 已安装 WSL（推荐 Ubuntu 20.04 或更高版本）
- 拥有有效的 Anthropic 账号

## 第一步：安装 Node.js（使用 nvm）

下载并安装 nvm：
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
```

激活环境：
```bash
source ~/.bashrc
```

安装 Node.js LTS：
```bash
nvm install --lts
```

验证版本：
```bash
node -v  # 期望输出：v20.x.x 或 v22.x.x
npm -v   # 期望输出：10.x.x 或更高
```

## 第二步：安装 Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```

## 第三步：配置环境变量

如果输入 `claude` 提示 command not found，需要将 npm 路径加入 PATH：
```bash
echo "export PATH=\"\$(npm config get prefix)/bin:\$PATH\"" >> ~/.bashrc
source ~/.bashrc
```

## 第四步：启动与登录

```bash
claude
```

首次启动会引导你登录 Anthropic 账号完成授权。
