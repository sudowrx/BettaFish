"""
重试机制工具模块
提供通用的网络请求重试功能，增强系统健壮性
"""

import time
from functools import wraps
from typing import Any, Callable, Optional
import requests
from loguru import logger


RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429}


def _get_http_status_code(exc: Exception) -> Optional[int]:
    """Extract an HTTP status code from requests/OpenAI-style exceptions."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)

    try:
        return int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return None


def _should_retry_exception(exc: Exception) -> bool:
    """Return whether an exception represents a transient failure."""
    status_code = _get_http_status_code(exc)
    if status_code is None:
        return True
    return status_code in RETRYABLE_HTTP_STATUS_CODES or 500 <= status_code <= 599


def _emit_retry_log(func_name: str, attempt: int, error: str, delay: float, final: bool = False) -> None:
    """把重试事件同时打到标准 logging。

    loguru 的 sink 不保证落进引擎自己的日志文件（insight.log 等），
    结果是「任务静默」时完全看不到重试发生过 —— 排查时只能靠猜。
    这里额外用标准 logging 打一条，让根因在日志里可见。
    """
    import logging

    level = logging.ERROR if final else logging.WARNING
    msg = (
        f"[RETRY] {func_name} 第 {attempt} 次尝试失败: {error[:200]}"
        + (f" | 最终失败" if final else f" | {delay:.1f}s 后重试")
    )
    logging.getLogger("bettafish.retry").log(level, msg)


# 配置日志
class RetryConfig:
    """重试配置类"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        retry_on_exceptions: tuple = None
    ):
        """
        初始化重试配置
        
        Args:
            max_retries: 最大重试次数
            initial_delay: 初始延迟秒数
            backoff_factor: 退避因子（每次重试延迟翻倍）
            max_delay: 最大延迟秒数
            retry_on_exceptions: 需要重试的异常类型元组
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        
        # 默认需要重试的异常类型
        if retry_on_exceptions is None:
            self.retry_on_exceptions = (
                requests.exceptions.RequestException,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                requests.exceptions.Timeout,
                requests.exceptions.TooManyRedirects,
                ConnectionError,
                TimeoutError,
                Exception  # OpenAI和其他API可能抛出的一般异常
            )
        else:
            self.retry_on_exceptions = retry_on_exceptions

# 默认配置
DEFAULT_RETRY_CONFIG = RetryConfig()

def with_retry(config: RetryConfig = None):
    """
    重试装饰器
    
    Args:
        config: 重试配置，如果不提供则使用默认配置
    
    Returns:
        装饰器函数
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):  # +1 因为第一次不算重试
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"函数 {func.__name__} 在第 {attempt + 1} 次尝试后成功")
                    return result
                    
                except config.retry_on_exceptions as e:
                    last_exception = e

                    if not _should_retry_exception(e):
                        status_code = _get_http_status_code(e)
                        logger.error(
                            f"函数 {func.__name__} 遇到不可重试的 HTTP {status_code} 错误: {e}"
                        )
                        raise

                    if attempt == config.max_retries:
                        # 最后一次尝试也失败了
                        logger.error(f"函数 {func.__name__} 在 {config.max_retries + 1} 次尝试后仍然失败")
                        logger.error(f"最终错误: {str(e)}")
                        _emit_retry_log(
                            func.__name__, attempt + 1, str(e), 0.0, final=True
                        )
                        raise e

                    # 计算延迟时间
                    delay = min(
                        config.initial_delay * (config.backoff_factor ** attempt),
                        config.max_delay
                    )

                    logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {str(e)}")
                    logger.info(f"将在 {delay:.1f} 秒后进行第 {attempt + 2} 次尝试...")
                    # loguru 的 sink 不一定落进引擎日志文件，这里额外用标准 logging
                    # 打一条，保证「任务为什么静默」在 insight.log 里看得见。
                    _emit_retry_log(func.__name__, attempt + 1, str(e), delay)

                    time.sleep(delay)
                
                except Exception as e:
                    # 不在重试列表中的异常，直接抛出
                    logger.error(f"函数 {func.__name__} 遇到不可重试的异常: {str(e)}")
                    raise e
            
            # 这里不应该到达，但作为安全网
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator

def retry_on_network_error(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """
    专门用于网络错误的重试装饰器（简化版）
    
    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟秒数
        backoff_factor: 退避因子
    
    Returns:
        装饰器函数
    """
    config = RetryConfig(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor
    )
    return with_retry(config)

class RetryableError(Exception):
    """自定义的可重试异常"""
    pass

def with_graceful_retry(config: RetryConfig = None, default_return=None):
    """
    优雅重试装饰器 - 用于非关键API调用
    失败后不会抛出异常，而是返回默认值，保证系统继续运行
    
    Args:
        config: 重试配置，如果不提供则使用默认配置
        default_return: 所有重试失败后返回的默认值
    
    Returns:
        装饰器函数
    """
    if config is None:
        config = SEARCH_API_RETRY_CONFIG
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):  # +1 因为第一次不算重试
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"非关键API {func.__name__} 在第 {attempt + 1} 次尝试后成功")
                    return result
                    
                except config.retry_on_exceptions as e:
                    last_exception = e

                    if not _should_retry_exception(e):
                        status_code = _get_http_status_code(e)
                        logger.warning(
                            f"非关键API {func.__name__} 遇到不可重试的 HTTP {status_code} 错误: {e}"
                        )
                        return default_return
                    
                    if attempt == config.max_retries:
                        # 最后一次尝试也失败了，返回默认值而不抛出异常
                        logger.warning(f"非关键API {func.__name__} 在 {config.max_retries + 1} 次尝试后仍然失败")
                        logger.warning(f"最终错误: {str(e)}")
                        logger.info(f"返回默认值以保证系统继续运行: {default_return}")
                        return default_return
                    
                    # 计算延迟时间
                    delay = min(
                        config.initial_delay * (config.backoff_factor ** attempt),
                        config.max_delay
                    )
                    
                    logger.warning(f"非关键API {func.__name__} 第 {attempt + 1} 次尝试失败: {str(e)}")
                    logger.info(f"将在 {delay:.1f} 秒后进行第 {attempt + 2} 次尝试...")
                    
                    time.sleep(delay)
                
                except Exception as e:
                    # 不在重试列表中的异常，返回默认值
                    logger.warning(f"非关键API {func.__name__} 遇到不可重试的异常: {str(e)}")
                    logger.info(f"返回默认值以保证系统继续运行: {default_return}")
                    return default_return
            
            # 这里不应该到达，但作为安全网
            return default_return
            
        return wrapper
    return decorator

def make_retryable_request(
    request_func: Callable,
    *args,
    max_retries: int = 5,
    **kwargs
) -> Any:
    """
    直接执行可重试的请求（不使用装饰器）
    
    Args:
        request_func: 要执行的请求函数
        *args: 传递给请求函数的位置参数
        max_retries: 最大重试次数
        **kwargs: 传递给请求函数的关键字参数
    
    Returns:
        请求函数的返回值
    """
    config = RetryConfig(max_retries=max_retries)
    
    @with_retry(config)
    def _execute():
        return request_func(*args, **kwargs)
    
    return _execute()

# 预定义一些常用的重试配置
# LLM 调用专用重试配置。
#
# 原来的配置(max_retries=6, initial_delay=60, backoff=2, max_delay=600)
# 纯退避总时长 = 60+120+240+480+600+600 = 2100s = 35 分钟。
# 叠加 300s 的单次请求超时，一条卡死的请求能让整个报告任务静默 40 分钟以上，
# 表现为「日志不推进、CPU 不动、任务像挂了」。
#
# 豆包/推理模型单次正常耗时 15-60s，短暂失败重试 2-3 次足够；
# 真失败应该尽快抛出让上层感知，而不是在退避里睡半小时。
LLM_RETRY_CONFIG = RetryConfig(
    max_retries=3,        # 3 次足够覆盖瞬时网络抖动
    initial_delay=5.0,    # 快速起步，不让用户干等一分钟
    backoff_factor=2.0,   # 指数退避
    max_delay=60.0        # 单次等待最长 1 分钟
)

SEARCH_API_RETRY_CONFIG = RetryConfig(
    max_retries=5,        # 增加到5次重试
    initial_delay=2.0,    # 增加初始延迟
    backoff_factor=1.6,   # 调整退避因子
    max_delay=25.0        # 增加最大延迟
)

DB_RETRY_CONFIG = RetryConfig(
    max_retries=5,        # 增加到5次重试
    initial_delay=1.0,    # 保持较短的数据库重试延迟
    backoff_factor=1.5,
    max_delay=10.0
)
