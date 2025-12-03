# AstrBot AI生图插件

基于 OpenAI 兼容接口的 AI 图像生成插件，支持文生图和图生图功能。

## ✨ 功能特性

- **统一命令**：使用 `/AI生图` 命令，自动检测是否有图片输入
  - 无图片 → 文生图模式
  - 有图片 → 图生图模式（支持回复图片或直接发送图片）
- **多模型支持**：Gemini 2.5 Flash、Gemini 3.0 Pro、Imagen 4.0
- **智能方向选择**：在提示词中包含"横屏"或"竖屏"自动选择图片方向
- **智能代理**：自动判断内网直连、外网走代理

## 📦 安装

将插件文件夹放置到 AstrBot 的插件目录下：

```
AstrBot/data/plugins/astrbot_plugin_ai_image/
├── main.py
├── metadata.yaml
├── _conf_schema.json
└── README.md
```

重启 AstrBot 即可自动加载插件。

## ⚙️ 配置

在 AstrBot 管理面板中配置以下参数：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `api_url` | API 服务地址 | `http://localhost:8000/v1/chat/completions` |
| `api_key` | API 密钥 | `password` |
| `proxy_url` | 代理服务器地址（用于下载外网图片） | `http://ip:port` |
| `default_model` | 默认模型 | `gemini-3.0-pro-image-portrait` |
| `request_timeout` | 请求超时时间（秒） | `120` |
| `max_retry_attempts` | 最大重试次数 | `3` |

## 🎨 使用方法

### 文生图

```
/AI生图 一只可爱的猫咪在花园里玩耍
/AI生图 横屏 宇宙星空背景的科幻城市
/AI生图 竖屏 古风美女水墨画
```

### 图生图

1. 发送图片，然后输入命令：
```
/AI生图 把图片转换成水彩画风格
```

2. 或者回复一张图片并输入命令：
```
（回复某张图片）/AI生图 给这张图片添加下雪效果
```

### 其他命令

```
/AI生图帮助    # 显示帮助信息
/AI模型列表    # 查看所有可用模型
```

## 🤖 支持的模型

| 模型 | 方向 | 支持图生图 |
|------|------|------------|
| Gemini 2.5 Flash | 横屏/竖屏 | ✅ |
| Gemini 3.0 Pro | 横屏/竖屏 | ✅ |
| Imagen 4.0 | 横屏/竖屏 | ❌ |

> 💡 使用图生图时，如果当前模型不支持，会自动切换到 Gemini 模型

## 🌐 代理说明

插件会自动判断是否需要使用代理：

- **直连**：内网地址（10.x、127.x、192.168.x、172.16-31.x）、QQ 相关域名
- **代理**：外网资源（如 Google Cloud Storage 图片）

## 📝 更新日志

### v1.0.0
- 初始版本发布
- 支持文生图和图生图
- 支持多种 Gemini/Imagen 模型
- 智能代理判断

## 📄 许可证

MIT License

## 🙏 致谢

- [AstrBot](https://github.com/Soulter/AstrBot) - 多平台 LLM 聊天机器人框架

