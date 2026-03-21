# Claude Code 使用学习笔记

## 1. Git 配置

配置 Git 用户名和邮箱（一次性操作）：
```bash
git config --global user.name "renyuanhui1"
git config --global user.email "renyuanhui@nuaa.edu.cn"
```

## 2. CLAUDE.md 文件

CLAUDE.md 是 Claude Code 的配置文件，每次启动对话时会自动读取，里面的内容会作为工作指令。

当前规则：
1. 先通读问题，阅读相关代码文件，再动手
2. 做重大改动前，先确认方案
3. 每一步给简要说明
4. 改动尽量简单，影响越少越好
5. 维护架构文档
6. 不猜测没看过的代码，先读再答
7. 执行命令前先告知
8. 文件写入不超过150行，大文件分多次操作

## 3. GitHub 仓库使用流程

### 创建仓库后克隆到本地
```bash
git clone https://github.com/renyuanhui1/study.git
```

### 日常推送代码流程
```bash
git status          # 查看改动
git add .           # 添加所有改动
git commit -m "提交说明"  # 提交
git push            # 推送到 GitHub
```

第一次 push 会弹窗要求登录 GitHub 授权。

## 4. 斜杠命令

在项目目录下创建 `.claude/commands/push.md` 文件，就可以用 `/push` 一键提交推送代码到 GitHub。

命令文件示例内容：
```markdown
请帮我完成以下操作：
1. 用 git status 查看当前改动
2. 用 git add 添加所有改动
3. 根据改动内容生成一个简洁的中文 commit message
4. git commit
5. git push 到远程仓库

每一步都告诉我执行了什么。
```

## 5. 其他常用命令

- `/model` - 切换 AI 模型
- `/context` - 查看上下文使用情况
- `/mcp` - 查看 MCP 服务器配置
- `/help` - 获取帮助
