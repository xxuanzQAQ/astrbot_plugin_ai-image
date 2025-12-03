# -*- coding: utf-8 -*-
"""
AstrBot AI生图插件
基于OpenAI兼容接口的文生图和图生图功能
自动检测图片：有图片则图生图，无图片则文生图
使用AstrBot框架内置的系统代理
"""

import asyncio
import aiohttp
import aiofiles
import base64
import json
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.all import Image, Plain
from astrbot.core.message.components import Reply


# 图片模型配置
IMAGE_MODELS = {
    # Gemini 图/文生图（支持图生图）
    "gemini-2.5-flash-image-landscape": {
        "name": "Gemini 2.5 Flash",
        "size": "横屏",
        "support_i2i": True,
    },
    "gemini-2.5-flash-image-portrait": {
        "name": "Gemini 2.5 Flash",
        "size": "竖屏",
        "support_i2i": True,
    },
    "gemini-3.0-pro-image-landscape": {
        "name": "Gemini 3.0 Pro",
        "size": "横屏",
        "support_i2i": True,
    },
    "gemini-3.0-pro-image-portrait": {
        "name": "Gemini 3.0 Pro",
        "size": "竖屏",
        "support_i2i": True,
    },
    # Imagen 文生图（不支持图生图）
    "imagen-4.0-generate-preview-landscape": {
        "name": "Imagen 4.0",
        "size": "横屏",
        "support_i2i": False,
    },
    "imagen-4.0-generate-preview-portrait": {
        "name": "Imagen 4.0",
        "size": "竖屏",
        "support_i2i": False,
    },
}

# 响应文本中图片信息的匹配模式
_DATA_URL_PATTERN = re.compile(r"(data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+)")
_HTTP_URL_PATTERN = re.compile(r"(https?://[^\s\)\]\"'<>]+)")
_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[.*?\]\((https?://[^\s\)]+)\)")

# ========== 代理配置 ==========
# 不需要代理的域名（QQ相关域名需要直连）
NO_PROXY_DOMAINS = [".qq.com", ".gtimg.cn", ".qpic.cn", ".qlogo.cn"]


def should_use_proxy(url: str) -> bool:
    """判断该URL是否需要使用代理（仅外网资源如googleapis.com需要代理）"""
    if not url:
        return False
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()

        # data: URL不需要代理
        if parsed.scheme == "data":
            return False

        # 本地地址不需要代理
        if host in ["localhost", "127.0.0.1", "::1"]:
            return False

        # QQ相关域名直连
        for domain in NO_PROXY_DOMAINS:
            if host == domain[1:] or host.endswith(domain):
                return False

        # 内网网段直连（10.x.x.x, 127.x.x.x, 192.168.x.x, 169.254.x.x, 172.16-31.x.x）
        if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host):
            if (
                host.startswith("10.")
                or host.startswith("127.")
                or host.startswith("192.168.")
                or host.startswith("169.254.")
            ):
                return False
            parts = host.split(".")
            if len(parts) >= 2:
                a, b = int(parts[0]), int(parts[1])
                if a == 172 and 16 <= b <= 31:
                    return False

        # 其他情况需要代理（如 storage.googleapis.com）
        return True
    except Exception:
        return False


@register(
    "astrbot_plugin_ai_image",
    "XuXuan",
    "AI生图插件 - 基于OpenAI兼容接口的文生图/图生图功能",
    "1.0.0",
)
class AIImagePlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)

        # API 配置
        self.api_url = config.get(
            "api_url", "http://localhost:8000/v1/chat/completions"
        ).strip()
        self.api_key = config.get("api_key", "").strip()

        # 模型配置
        self.default_model = config.get(
            "default_model", "gemini-3.0-pro-image-portrait"
        ).strip()
        if self.default_model not in IMAGE_MODELS:
            self.default_model = "gemini-3.0-pro-image-portrait"

        # 超时配置
        self.request_timeout = int(config.get("request_timeout", 120))
        self.max_retry_attempts = int(config.get("max_retry_attempts", 3))

        # 代理配置（用于下载外网资源如Google Cloud Storage）
        self.proxy_url = config.get("proxy_url", "http://192.168.100.2:7890").strip()

        # 输出目录
        self.data_dir = Path(__file__).parent / "output"
        self.data_dir.mkdir(exist_ok=True)

    def _get_proxy_for_url(self, url: str) -> str | None:
        """根据URL判断是否需要使用代理"""
        if should_use_proxy(url):
            logger.debug(f"URL需要代理: {url[:50]}...")
            return self.proxy_url
        logger.debug(f"URL直连: {url[:50]}...")
        return None

    async def _cleanup_old_files(self, minutes: int = 15):
        """清理超过指定时间的临时文件"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(minutes=minutes)

            for pattern in ["ai_image_*.png", "ai_image_*.jpg", "ai_image_*.jpeg"]:
                for file_path in self.data_dir.glob(pattern):
                    try:
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_mtime < cutoff_time:
                            file_path.unlink()
                            logger.debug(f"已清理过期图像: {file_path}")
                    except Exception as e:
                        logger.warning(f"清理文件 {file_path} 时出错: {e}")
        except Exception as e:
            logger.error(f"图像清理过程出错: {e}")

    async def _save_base64_image(
        self, base64_string: str, image_format: str = "png"
    ) -> str | None:
        """保存 base64 图像到文件"""
        try:
            await self._cleanup_old_files()

            image_data = base64.b64decode(base64_string)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            image_path = (
                self.data_dir / f"ai_image_{timestamp}_{unique_id}.{image_format}"
            )

            async with aiofiles.open(image_path, "wb") as f:
                await f.write(image_data)

            logger.info(f"图像已保存到: {image_path.absolute()}")
            return str(image_path)
        except Exception as e:
            logger.error(f"保存图像文件失败: {e}")
            return None

    async def _download_image(
        self, url: str, proxy_url: str | None = None
    ) -> str | None:
        """下载图片并保存到文件"""
        try:
            await self._cleanup_old_files()

            # 创建新的session来下载图片，避免使用已关闭的session
            timeout = aiohttp.ClientTimeout(total=60)
            connector = aiohttp.TCPConnector(ssl=False)  # 跳过SSL验证

            async with aiohttp.ClientSession(
                timeout=timeout, connector=connector
            ) as session:
                logger.debug(f"正在下载图片: {url[:100]}...")
                async with session.get(url, proxy=proxy_url) as response:
                    if response.status != 200:
                        logger.error(f"下载图片失败，状态码: {response.status}")
                        return None

                    content = await response.read()

                    # 根据Content-Type确定扩展名
                    content_type = response.headers.get("Content-Type", "image/png")
                    if "jpeg" in content_type or "jpg" in content_type:
                        ext = "jpg"
                    elif "gif" in content_type:
                        ext = "gif"
                    elif "webp" in content_type:
                        ext = "webp"
                    else:
                        ext = "png"

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]
                    image_path = (
                        self.data_dir / f"ai_image_{timestamp}_{unique_id}.{ext}"
                    )

                    async with aiofiles.open(image_path, "wb") as f:
                        await f.write(content)

                    logger.info(f"图像已下载并保存到: {image_path.absolute()}")
                    return str(image_path)
        except aiohttp.ClientError as e:
            logger.error(f"下载图片网络错误: {type(e).__name__}: {e}")
            return None
        except Exception as e:
            logger.error(f"下载图片失败: {type(e).__name__}: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return None

    def _extract_image_data(self, content: str) -> tuple[str | None, str | None]:
        """从响应内容中提取图片数据（base64或URL）"""
        if not content:
            return None, None

        # 1. 尝试提取 base64 数据
        match = _DATA_URL_PATTERN.search(content)
        if match:
            data_url = match.group(1)
            try:
                header, base64_part = data_url.split(",", 1)
                fmt = header.split("/")[1].split(";")[0]
                return base64_part, fmt
            except Exception:
                pass

        # 2. 尝试提取 Markdown 图片 URL
        match = _MARKDOWN_IMAGE_PATTERN.search(content)
        if match:
            return match.group(1), None

        # 3. 尝试提取普通 URL
        match = _HTTP_URL_PATTERN.search(content)
        if match:
            url = match.group(1)
            # 检查是否是图片URL
            if any(
                ext in url.lower()
                for ext in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".webp",
                    "image",
                    "storage.googleapis.com",
                ]
            ):
                return url, None

        return None, None

    async def _call_api_stream(
        self, model: str, messages: list
    ) -> tuple[str | None, str | None]:
        """
        调用 API 的流式接口生成图片。
        API调用不使用代理（内网地址），下载图片时根据URL判断是否需要代理。

        Returns:
            tuple: (base64_or_url, image_path) 或 (None, None) 表示失败
        """
        payload = {"model": model, "messages": messages, "stream": True}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        client_timeout = aiohttp.ClientTimeout(total=self.request_timeout)

        # API调用不使用代理（根据shouldUseProxy判断，内网地址直连）
        api_proxy = self._get_proxy_for_url(self.api_url)

        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.post(
                    self.api_url, json=payload, headers=headers, proxy=api_proxy
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"API 请求失败: HTTP {response.status}, {error_text}"
                        )
                        return None, None

                    collected_content = ""
                    raw_buffer = ""

                    # 处理流式响应
                    async for raw_line in response.content:
                        raw_buffer += raw_line.decode("utf-8")

                        while "\n" in raw_buffer:
                            line, raw_buffer = raw_buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue

                            if line.startswith("data: "):
                                line = line[6:]

                            if line == "[DONE]":
                                break

                            try:
                                chunk = json.loads(line)

                                if "error" in chunk:
                                    error_message = chunk["error"].get(
                                        "message", str(chunk["error"])
                                    )
                                    logger.error(f"API 返回错误: {error_message}")
                                    continue

                                if "choices" in chunk and chunk["choices"]:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta and delta["content"]:
                                        collected_content += delta["content"]
                            except json.JSONDecodeError:
                                if "data:image" in line or "https://" in line:
                                    collected_content += line
                                continue

                    # 处理剩余的buffer
                    if raw_buffer.strip():
                        line = raw_buffer.strip()
                        if line.startswith("data: "):
                            line = line[6:]
                        if line and line != "[DONE]":
                            try:
                                chunk = json.loads(line)
                                if "choices" in chunk and chunk["choices"]:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta and delta["content"]:
                                        collected_content += delta["content"]
                            except json.JSONDecodeError:
                                pass

                    logger.debug(f"收集到的内容长度: {len(collected_content)}")

                    # 提取图片数据
                    image_data, fmt = self._extract_image_data(collected_content)

                    if image_data:
                        # 如果是 base64 数据
                        if not image_data.startswith("http"):
                            file_path = await self._save_base64_image(
                                image_data, fmt or "png"
                            )
                            if file_path:
                                return image_data, file_path
                        else:
                            # 是 URL，根据URL判断是否需要代理下载图片
                            download_proxy = self._get_proxy_for_url(image_data)
                            file_path = await self._download_image(
                                image_data, download_proxy
                            )
                            if file_path:
                                return image_data, file_path

                    logger.warning(
                        f"未能从响应中提取图片: {collected_content[:500] if collected_content else '(空)'}"
                    )
                    return None, None

        except asyncio.TimeoutError:
            logger.error(f"API 请求超时 (>{self.request_timeout}s)")
            return None, None
        except aiohttp.ClientError as e:
            logger.error(f"API 网络请求失败: {e}")
            return None, None
        except Exception as e:
            logger.error(f"API 调用异常: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None, None

    async def _collect_input_images(self, event: AstrMessageEvent) -> list[str]:
        """收集输入图片（当前消息和引用消息中的图片）"""
        images: list[str] = []
        if (
            hasattr(event, "message_obj")
            and event.message_obj
            and hasattr(event.message_obj, "message")
        ):
            for comp in event.message_obj.message:
                if isinstance(comp, Image):
                    try:
                        base64_data = await comp.convert_to_base64()
                        images.append(base64_data)
                    except Exception as e:
                        logger.warning(f"转换图片到base64失败: {e}")
                elif isinstance(comp, Reply) and comp.chain:
                    for reply_comp in comp.chain:
                        if isinstance(reply_comp, Image):
                            try:
                                base64_data = await reply_comp.convert_to_base64()
                                images.append(base64_data)
                                logger.info("从引用消息中获取到图片")
                            except Exception as e:
                                logger.warning(f"转换引用消息中的图片失败: {e}")
        return images

    def _select_model(self, prompt: str, has_images: bool) -> str:
        """根据提示词和是否有图片选择合适的模型"""
        base_model = self.default_model.replace("-landscape", "").replace(
            "-portrait", ""
        )

        # 根据提示词中的关键字选择横屏或竖屏
        if "横屏" in prompt:
            model = f"{base_model}-landscape"
        elif "竖屏" in prompt:
            model = f"{base_model}-portrait"
        else:
            model = self.default_model

        # 确保模型存在
        if model not in IMAGE_MODELS:
            model = self.default_model

        # 如果有图片但当前模型不支持图生图，切换到支持的模型
        if has_images and not IMAGE_MODELS[model].get("support_i2i", False):
            # 切换到 Gemini 3.0 Pro
            if "landscape" in model:
                model = "gemini-3.0-pro-image-landscape"
            else:
                model = "gemini-3.0-pro-image-portrait"
            logger.info(f"当前模型不支持图生图，已切换到: {model}")

        return model

    async def _send_image_result(
        self, event: AstrMessageEvent, file_path: str
    ) -> Image:
        """发送图片结果"""
        try:
            callback_api_base = self.context.get_config().get("callback_api_base")
            image_component = Image.fromFileSystem(file_path)
            if callback_api_base:
                try:
                    download_url = await image_component.convert_to_web_link()
                    return Image.fromURL(download_url)
                except Exception as e:
                    logger.warning(f"转换为web链接失败: {e}")
            return image_component
        except Exception as e:
            logger.warning(f"发送图片失败: {e}")
            return Image.fromFileSystem(file_path)

    @filter.command("AI生图")
    async def generate_image_command(self, event: AstrMessageEvent, prompt: str = ""):
        """
        AI生图指令 - 自动检测是文生图还是图生图

        使用方法：
        - /AI生图 <描述> - 文生图
        - 发送图片后 /AI生图 <描述> - 图生图
        - 回复图片消息 /AI生图 <描述> - 图生图
        """
        # 提取提示词
        if not prompt:
            raw = getattr(event, "message_str", "") or ""
            parts = raw.strip().split(" ", 1)
            if len(parts) == 2:
                prompt = parts[1].strip()

        if not prompt:
            yield event.plain_result(
                "请提供图片描述，例如：/AI生图 一只可爱的猫咪在花园里玩耍\n"
                "也可以发送图片后使用此命令进行图生图。"
            )
            return

        # 收集输入图片
        input_images = await self._collect_input_images(event)
        has_images = len(input_images) > 0

        # 选择模型
        model = self._select_model(prompt, has_images)
        model_info = IMAGE_MODELS.get(model, {})

        # 构建提示信息
        mode_text = "图生图" if has_images else "文生图"
        yield event.plain_result(
            f"🎨 正在{mode_text}，请稍候...\n"
            f"模型：{model_info.get('name', model)} ({model_info.get('size', '默认')})"
        )

        # 构建消息内容
        if input_images:
            # 图生图模式
            content = [{"type": "text", "text": prompt}]
            for img_base64 in input_images:
                if not img_base64.startswith("data:image/"):
                    img_base64 = f"data:image/png;base64,{img_base64}"
                content.append({"type": "image_url", "image_url": {"url": img_base64}})
        else:
            # 文生图模式
            content = prompt

        messages = [{"role": "user", "content": content}]

        # 调用 API（API调用和图片下载的代理由内部自动判断）
        try:
            start_time = datetime.now()
            _, file_path = await self._call_api_stream(model, messages)

            if not file_path:
                yield event.chain_result(
                    [Plain("❌ 图片生成失败，请检查服务配置或稍后重试。")]
                )
                return

            elapsed = (datetime.now() - start_time).total_seconds()
            image_component = await self._send_image_result(event, file_path)
            yield event.chain_result(
                [Plain(f"✨ {mode_text}完成！({elapsed:.1f}s)\n"), image_component]
            )

        except Exception as e:
            logger.error(f"生成图片失败: {e}")
            yield event.chain_result([Plain(f"❌ 生成失败: {str(e)}")])

    @filter.command("AI生图帮助")
    async def help_command(self, event: AstrMessageEvent):
        """显示帮助信息"""
        model_info = IMAGE_MODELS.get(self.default_model, {})
        help_text = f"""🎨 AI生图插件帮助

📌 使用方法：
• /AI生图 <描述> - 根据文字生成图片
• 发送图片后 /AI生图 <描述> - 基于图片进行修改
• 回复图片消息 /AI生图 <描述> - 基于图片进行修改

📌 屏幕方向：
在描述中包含"竖屏"或"横屏"来选择生成方向
例如：/AI生图 横屏 城市夜景

📌 当前配置：
• 默认模型：{model_info.get('name', self.default_model)} ({model_info.get('size', '默认')})
• 图生图支持：{'是' if model_info.get('support_i2i', False) else '否'}

📌 使用示例：
• /AI生图 一只可爱的猫咪在花园里玩耍
• /AI生图 横屏 山水画风格的风景
• [发送图片] /AI生图 变成水彩画风格

📌 注意事项：
• 生图约需10-30秒
• Imagen模型不支持图生图，会自动切换到Gemini
• 请避免生成违规内容"""

        yield event.plain_result(help_text)

    @filter.command("AI模型列表")
    async def list_models_command(self, event: AstrMessageEvent):
        """列出所有可用的模型"""
        model_list = "📋 AI生图可用模型列表\n\n"

        for model_id, info in IMAGE_MODELS.items():
            i2i_tag = "✅图生图" if info.get("support_i2i", False) else "❌仅文生图"
            current_tag = " 👈当前" if model_id == self.default_model else ""
            model_list += f"• {info['name']} ({info['size']}) {i2i_tag}{current_tag}\n  {model_id}\n\n"

        yield event.plain_result(model_list)
