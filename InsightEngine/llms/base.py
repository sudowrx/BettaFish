"""
Unified OpenAI-compatible LLM client for the Insight Engine, with retry support.
"""

import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Iterator, Generator
from loguru import logger

from openai import OpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
utils_dir = os.path.join(project_root, "utils")
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

try:
    from retry_helper import with_retry, LLM_RETRY_CONFIG
except ImportError:
    def with_retry(config=None):
        def decorator(func):
            return func
        return decorator

    LLM_RETRY_CONFIG = None


class LLMClient:
    """Minimal wrapper around the OpenAI-compatible chat completion API."""

    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        if not api_key:
            raise ValueError("Insight Engine INSIGHT_ENGINE_API_KEY is required.")
        if not model_name:
            raise ValueError("Insight Engine INSIGHT_ENGINE_MODEL_NAME is required.")

        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.provider = model_name
        # 超时上限。豆包/推理模型单次 15-25s，300s 已足够宽松。
        # 原值 1800(30min) 的问题: 服务端半开连接(CLOSE-WAIT)时要挂 30 分钟
        # 才抛异常，再叠加 with_retry 的指数退避(6 次, 首次 60s)，
        # 最坏情况一个请求能拖 3.5 小时，表现为"任务卡住不动"。
        timeout_fallback = os.getenv("LLM_REQUEST_TIMEOUT") or os.getenv("INSIGHT_ENGINE_REQUEST_TIMEOUT") or "300"
        try:
            self.timeout = float(timeout_fallback)
        except ValueError:
            self.timeout = 300.0

        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            # 连接级重试: httpx 层遇到 socket 挂起/连接重置时立即换新连接，
            # 比等 with_retry 的 60s 起步退避快得多。
            "max_retries": 2,
            "timeout": self.timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    @with_retry(LLM_RETRY_CONFIG)
    def invoke(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        current_time = datetime.now().strftime("%Y年%m月%d日%H时%M分")
        time_prefix = f"今天的实际时间是{current_time}"
        if user_prompt:
            user_prompt = f"{time_prefix}\n{user_prompt}"
        else:
            user_prompt = time_prefix
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        allowed_keys = {"temperature", "top_p", "presence_penalty", "frequency_penalty", "stream"}
        extra_params = {key: value for key, value in kwargs.items() if key in allowed_keys and value is not None}

        timeout = kwargs.pop("timeout", self.timeout)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            timeout=timeout,
            **extra_params,
        )

        if response.choices and response.choices[0].message:
            return self.validate_response(response.choices[0].message.content)
        return ""

    def stream_invoke(self, system_prompt: str, user_prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        流式调用LLM，逐步返回响应内容
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            **kwargs: 额外参数（temperature, top_p等）
            
        Yields:
            响应文本块（str）
        """
        current_time = datetime.now().strftime("%Y年%m月%d日%H时%M分")
        time_prefix = f"今天的实际时间是{current_time}"
        if user_prompt:
            user_prompt = f"{time_prefix}\n{user_prompt}"
        else:
            user_prompt = time_prefix
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        allowed_keys = {"temperature", "top_p", "presence_penalty", "frequency_penalty"}
        extra_params = {key: value for key, value in kwargs.items() if key in allowed_keys and value is not None}
        # 强制使用流式
        extra_params["stream"] = True

        timeout = kwargs.pop("timeout", self.timeout)

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                timeout=timeout,
                **extra_params,
            )
            
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
        except Exception as e:
            logger.error(f"流式请求失败: {str(e)}")
            raise e
    
    @with_retry(LLM_RETRY_CONFIG)
    def stream_invoke_to_string(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        流式调用LLM并安全地拼接为完整字符串（避免UTF-8多字节字符截断）

        墙钟超时说明：
        httpx 的 read timeout 在流式模式下只计算「两个 chunk 之间」的间隔，
        **首字节到达之前完全不计时**。推理模型（豆包 V4 等）在生成前有很长的
        静默思考期，如果服务端迟迟不发第一个 chunk，httpx 的 timeout 永远不会触发，
        请求就无限期挂着 —— 表现为「日志不推进、CPU 不动、任务像死了」。

        实测（2026-10-29）：一个流式请求建立连接后 20 分钟没有任何 chunk，
        TCP 层只有 20 秒一次的 keepalive，lastsnd/lastrcv 持续推进但内容为 0。

        所以这里自己实现墙钟上限：超过 deadline 就主动关闭 stream 并抛异常，
        让上层的 with_retry 以「快速失败」的方式重试。

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            **kwargs: 额外参数（temperature, top_p等）

        Returns:
            完整的响应字符串
        """
        # 以字节形式收集所有块
        byte_chunks = []

        timeout = kwargs.pop("timeout", self.timeout)
        # 流式墙钟上限：默认取 timeout 的 2 倍（timeout 本身是单次 chunk 间隔上限），
        # 可用 STREAM_WALL_CLOCK_TIMEOUT 单独覆盖。
        wall_clock = float(
            os.getenv("STREAM_WALL_CLOCK_TIMEOUT") or (timeout * 2)
        )

        stream_iter = self.stream_invoke(system_prompt, user_prompt, timeout=timeout, **kwargs)
        try:
            # 为什么用「工作线程 + queue」而不是直接 for 循环：
            #
            # httpx 的 read timeout 在流式模式下只计算「两个 chunk 之间」的间隔，
            # **首字节到达之前完全不计时**。推理模型（豆包 V4 等）在生成前有很长的
            # 静默思考期，服务端迟迟不发第一个 chunk 时 httpx 的 timeout 永远不会触发。
            #
            # 实测（2026-10-29）：一个流式请求建立连接后 20 分钟没有任何 chunk，
            # TCP 层只有 20 秒一次的 keepalive（lastsnd/lastrcv 持续推进但内容为 0）。
            #
            # 而生成器的 close() 也救不了：GeneratorExit 无法打断阻塞在 socket read
            # 或 time.sleep() 上的迭代器。所以必须把迭代放到独立线程，
            # 主线程用 queue.get(timeout=...) 实现真正的墙钟上限。
            #
            # 超过上限 → 主线程抛 TimeoutError，后台线程被设为 daemon 随进程退出，
            # 上层的 with_retry 会以「快速失败」方式重试，而不是永远挂着。
            import queue
            import threading

            q: "queue.Queue" = queue.Queue(maxsize=256)
            _sentinel = object()

            def _pump():
                try:
                    for chunk in stream_iter:
                        q.put(chunk.encode('utf-8'))
                    q.put(_sentinel)
                except BaseException as exc:  # noqa: BLE001
                    try:
                        q.put(exc)
                    except Exception:
                        pass

            pump = threading.Thread(target=_pump, daemon=True)
            pump.start()

            while True:
                try:
                    item = q.get(timeout=wall_clock)
                except queue.Empty:
                    raise TimeoutError(
                        f"流式请求超过墙钟上限 {wall_clock:.0f}s（未收到完整响应）"
                    ) from None
                if item is _sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                byte_chunks.append(item)
        finally:
            # 主动关闭底层连接，避免半开的 socket 继续挂着
            close = getattr(stream_iter, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass

        # 拼接所有字节，然后一次性解码
        if byte_chunks:
            return b''.join(byte_chunks).decode('utf-8', errors='replace')
        return ""

    @staticmethod
    def validate_response(response: Optional[str]) -> str:
        if response is None:
            return ""
        return response.strip()

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model_name,
            "api_base": self.base_url or "default",
        }
