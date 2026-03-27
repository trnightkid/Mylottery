# OpenClaw 完整安装指南

> 基于 2026年3月25-26日实机安装记录整理，涵盖所有关键步骤和踩坑点

---

## 环境信息

- **操作系统**: Linux (x64)
- **Node 版本**: v22.22.2
- **OpenClaw 版本**: 2026.3.24
- **已配置渠道**: Telegram

---

## 一、安装前准备

### 1.1 系统要求

- **Node**: 推荐 Node 24，或 Node 22.14+
- **操作系统**: macOS / Linux / Windows (WSL2) / Windows 原生
- **网络**: 能够访问 api.qnaigc.com（MiniMax API）、api.telegram.org

### 1.2 确认 Node 版本

```bash
node -v   # 确认已安装 Node
npm -v    # 确认 npm 可用
```

如果未安装 Node，推荐使用 installer 脚本自动安装。

---

## 二、安装 OpenClaw

### 方式一：installer 脚本（推荐，最简单）

**macOS / Linux / WSL2：**
```bash
curl -fsSL https://openclaw.ai/install.sh | bash
```

**Windows (PowerShell)：**
```powershell
iwr -useb https://openclaw.ai/install.ps1 | iex
```

inaller 脚本会自动：
1. 检测并安装 Node（如果需要）
2. 安装 OpenClaw
3. 运行引导式 onboarding

### 方式二：npm 全局安装

如果你已经管理自己的 Node：
```bash
npm install -g openclaw@latest
```

### 方式三：install-cli.sh（无需 root）

适合在共享环境或不想污染全局包的情况：
```bash
curl -fsSL https://openclaw.ai/install-cli.sh | bash
```

### ⚠️ 踩坑点：sharp 构建错误

如果在 npm 安装时遇到 `sharp` 模块构建失败，尝试：
```bash
SHARP_IGNORE_GLOBAL_LIBVIPS=1 npm install -g openclaw@latest
```

---

## 三、引导式初始化配置

安装完成后，运行：
```bash
openclaw onboard --install-daemon
```

这会启动一个交互式引导流程，帮你配置：
- Gateway 运行模式（local/remote）
- 认证方式（密码）
- 默认 AI 模型
- 渠道配置（Telegram/WhatsApp/Discord 等）

### 引导过程说明

引导过程会在 `~/.openclaw/openclaw.json` 生成配置文件。

**关键配置路径**（很多人搞错）：
- **Telegram 配置路径**: `channels.telegram` ❌ 不是 `plugins.entries.telegram`
- **模型配置路径**: `models.providers`
- **Gateway 配置路径**: `gateway`

---

## 四、Telegram Bot 配置（详细步骤）

这是最容易踩坑的部分，请严格按照以下步骤操作。

### 4.1 创建 Telegram Bot

1. 在 Telegram 搜索 **@BotFather**
2. 发送 `/newbot`
3. 按提示设置 bot 名称和用户名
4. 获得 Bot Token，格式类似：`123456789:ABCdefGHIjklMNOpqrSTUvwxyz`

**⚠️ 重要：妥善保管 Token，不要泄露**

### 4.2 配置 Bot Token

**错误做法**：在 `plugins.entries.telegram` 下配置（这是无效的路径）

**正确做法**：在 `channels.telegram` 下配置

```bash
# 查看当前配置
openclaw config get

# 或者直接编辑配置文件
nano ~/.openclaw/openclaw.json
```

正确的配置结构：
```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "botToken": "你的BotToken",
      "dmPolicy": "pairing"
    }
  }
}
```

### 4.3 通过 config.patch 配置（推荐）

```bash
openclaw config patch channels.telegram.enabled true
openclaw config patch channels.telegram.botToken "8744746577:AAEO5JFdQlMCZouR7mHmRYN_48i8rdPoBwU"
openclaw config patch channels.telegram.dmPolicy "pairing"
```

或者一步到位：
```bash
openclaw config patch '{"channels":{"telegram":{"enabled":true,"botToken":"8744746577:AAEO5JFdQlMCZouR7mHmRYN_48i8rdPoBwU","dmPolicy":"pairing"}}}'
```

### 4.4 重启 Gateway

修改配置后需要重启 Gateway：
```bash
openclaw gateway restart
```

或者给正在运行的 Gateway 发送信号：
```bash
# 找到 Gateway 进程
openclaw gateway status

# 重启
kill -SIGUSR1 <pid>
```

### 4.5 配对（Pairing）

Telegram 默认使用 `dmPolicy: "pairing"`，新用户首次使用需要配对批准。

1. 在 Telegram 向你的 Bot 发送任意消息
2. 查看配对请求：
   ```bash
   openclaw pairing list telegram
   ```
3. 批准配对：
   ```bash
   openclaw pairing approve telegram <配对码>
   ```

### 4.6 dmPolicy 说明

| 值 | 说明 |
|---|---|
| `pairing` | 默认，需要配对批准 |
| `allowlist` | 只允许白名单用户，需要在 `allowFrom` 中指定用户 ID |
| `open` | 开放访问（需配合 `allowFrom: ["*"]`） |
| `disabled` | 禁用 DM |

### 4.7 查找自己的 Telegram User ID

方法一：通过 Bot 日志
1. 向 Bot 发送消息
2. 查看日志：`openclaw logs --follow`
3. 在日志中找到 `from.id`

方法二：使用官方 API
```bash
curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates"
```

方法三：使用第三方 Bot（不推荐）
- 搜索 `@userinfobot` 或 `@getidsbot`

---

## 五、验证安装

### 5.1 检查 CLI 是否可用
```bash
openclaw --version
```

### 5.2 运行诊断
```bash
openclaw doctor
```

### 5.3 检查 Gateway 状态
```bash
openclaw gateway status
```

### 5.4 打开 Control UI
```bash
openclaw dashboard
```
默认地址：http://127.0.0.1:18789/

---

## 六、常见问题与解决

### Q1: `openclaw: command not found`

```bash
# 检查 Node 全局包路径
npm prefix -g

# 确认 PATH 包含全局 bin 目录
echo $PATH

# 如果没有，添加到 ~/.bashrc 或 ~/.zshrc
export PATH="$(npm prefix -g)/bin:$PATH"
source ~/.bashrc   # 或 source ~/.zshrc
```

### Q2: Telegram Bot 没有响应

1. 检查 Bot Token 是否正确配置在 `channels.telegram.botToken`
2. 检查 Gateway 是否重启
3. 检查日志：
   ```bash
   openclaw logs --follow
   ```
4. 确认已批准配对：
   ```bash
   openclaw pairing list telegram
   ```

### Q3: 配对码过期

配对码有效期为 1 小时。让用户在 Telegram 重新发送消息以生成新码。

### Q4: Gateway 无法启动，端口被占用

```bash
# 查看 18789 端口占用
lsof -i :18789

# 或者修改配置文件中的端口
openclaw config patch gateway.port 18790
```

### Q5: 配置文件格式错误

编辑 `~/.openclaw/openclaw.json` 时确保 JSON 格式正确：
- 键名必须用双引号
- 不能有尾随逗号
- 可以用在线 JSON 验证器检查

---

## 七、配置文件结构参考

完整的 `~/.openclaw/openclaw.json` 结构：

```json
{
  "meta": {
    "lastTouchedVersion": "2026.3.24",
    "lastTouchedAt": "2026-03-26T07:11:51.496Z"
  },
  "env": {
    "QINIU_API_KEY": "你的API密钥"
  },
  "models": {
    "mode": "merge",
    "providers": {
      "qiniu": {
        "baseUrl": "https://api.qnaigc.com/v1",
        "apiKey": "你的API密钥",
        "api": "openai-completions",
        "models": [
          {
            "id": "minimax/minimax-m2.7",
            "name": "MiniMax M2.7",
            "input": ["text"],
            "contextWindow": 128000,
            "maxTokens": 8192
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "qiniu/minimax/minimax-m2.7"
      }
    }
  },
  "channels": {
    "telegram": {
      "enabled": true,
      "botToken": "你的BotToken",
      "dmPolicy": "pairing"
    }
  },
  "gateway": {
    "port": 18789,
    "mode": "local",
    "bind": "auto",
    "auth": {
      "mode": "password",
      "password": "你的密码"
    }
  }
}
```

---

## 八、Telegram 隐私模式注意事项

如果 Bot 在群里看不到消息：

1. 在 BotFather 发送 `/setprivacy`
2. 选择 **Disable**（允许 Bot 看到所有消息）
3. 将 Bot 从群中移除再重新加入

---

## 九、后续维护

### 更新 OpenClaw
```bash
npm update -g openclaw@latest
```

### 查看日志
```bash
openclaw logs              # 最近日志
openclaw logs --follow     # 实时跟踪
```

### 备份配置
```bash
cp ~/.openclaw/openclaw.json ~/.openclaw/openclaw.json.backup
```

---

## 附录：本文涉及的关键路径

| 功能 | 配置路径 |
|---|---|
| Telegram Bot | `channels.telegram.botToken` |
| Telegram 开关 | `channels.telegram.enabled` |
| DM 策略 | `channels.telegram.dmPolicy` |
| 模型供应商 | `models.providers` |
| 默认模型 | `agents.defaults.model.primary` |
| Gateway 端口 | `gateway.port` |
| Gateway 密码 | `gateway.auth.password` |

---

*文档生成时间：2026-03-26*
*整理自实机安装过程，版本 2026.3.24*
