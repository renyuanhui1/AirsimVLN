📋 前置要求已安装 WSL 
(推荐 Ubuntu 20.04 或更高版本)拥有有效的 Anthropic 账号（用于登录授权）
🛠️ 第一步：安装 Node.js (推荐使用 nvm)在 WSL 中直接使用 sudo apt install nodejs 往往版本过旧，且权限管理麻烦。推荐使用 nvm (Node Version Manager)。下载并安装 nvm：

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh

激活环境：Bashsource ~/.bashrc

安装 Node.js LTS (长期支持版)：Claude Code 
nvm install --lts
验证版本：
Bashnode -v  # 理想输出：v20.x.x 或 v22.x.x
npm -v   # 理想输出：10.x.x 或更高
🚀 第二步：安装 Claude Code使用 npm 全局安装 Anthropic 官方包：
npm install -g @anthropic-ai/claude-code
⚙️ 第三步：配置环境变量 (关键)如果在安装后输入 claude 提示 command not found，需要将 npm 的二进制路径加入到 $PATH 中。写入配置文件：
echo "export PATH=\"\$(npm config get prefix)/bin:\$PATH\"" >> ~/.bashrc
使配置生效：source ~/.bashrc

🔑 第四步：启动与身份验证初始化：在终端输入以下命令启动：Bashclaude